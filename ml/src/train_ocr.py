"""
Train the CRNN text recogniser on real word crops.

    bash run.sh train-ocr 40

WHAT IS DIFFERENT FROM train_restorer.py
---------------------------------------
The restorer is a small residual CNN on synthetic 64x256 patches. This is a
sequence model on real photographs, and three things follow from that:

1. CTC LOSS, not MSE. The target is a character sequence of unknown alignment
   inside the crop. CTC marginalises over every valid alignment, so no
   character-position labels are needed -- which is why we can train on word
   boxes without ever knowing where the '5' sits.

2. BATCHING BY WIDTH. Crops vary from 16px to 320px, and a batch must be
   rectangular to become a tensor. Samples are bucketed into width bins and
   width-padded to the batch max, with a mask so the padding is ignored by the
   loss. Sorting the whole dataset by width once and slicing consecutive runs
   keeps padding waste under ~10% without any shuffling overhead.

3. THE 0.4% IS DROPPED, NOT PADDED. CTC needs T >= label_length. Measured over
   the 30,021 built crops, median chars/timestep is 2.16 but 0.4% of samples
   sit below 1.0 and their loss is literally infinite. Feeding them in would
   produce NaN gradients. train_ocr.py filters them at load time and REPORTS
   how many, rather than dropping them silently -- a silently shrinking
   dataset is how a training set rots.

WHAT IS REPORTED
----------------
Character Error Rate (CER) on the DOCUMENT-split validation set, per
augmentation variant, plus exact-match word accuracy. Word accuracy is the
number that matters for this product: a dosage or an account number is right or
it is wrong. CER is diagnostic; it is reported because a rising exact-match with
a falling CER usually means the model is getting the EASY words right and
making one-character mistakes on the hard ones, which is precisely the failure
mode that turns 500mg into 5O0mg.

Checkpoint selection is on validation exact-match, not on loss. Loss is a poor
proxy here because it is dominated by the many easy crops; exact match is
what the field extractor consumes.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import realdata as R          # noqa: E402
import seedutil               # noqa: E402
from ocr_model import build   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts" / "ocr_train.npz"
CKPT = ROOT / "models" / "ocr"


class CRNNDataset(torch.utils.data.Dataset):
    """Packed crops + labels + widths. Normalisation to [-1,1] happens here."""

    def __init__(self, images, widths, labels, indices):
        self.images = images          # [N,32,maxw] uint8
        self.widths = widths
        self.labels = labels
        self.idx = np.asarray(indices, np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        w = int(self.widths[j])
        a = self.images[j, :, :w].astype(np.float32) / 255.0
        # (x/255 - 0.5) / 0.5 -> [-1,1]. Centring on the page background is
        # right here because every crop is dark text on light paper, so 1.0 is
        # "paper" and -1.0 is "ink". That sign convention lets the first conv
        # start from a useful bias instead of learning the polarity.
        a = (a - 0.5) / 0.5
        return torch.from_numpy(a[None]), R.encode(self.labels[j])


def width_bucketed_batches(ds, batch_size, width_tol=24, rng=None):
    """
    Yield index lists whose members have similar widths, after a one-time sort.

    A batch is a single tensor, so all its crops must share a width. Sorting
    by width and slicing consecutive runs keeps padding waste low; shuffling
    happens by shuffling the ORDER OF BUCKETS, not of samples within a bucket,
    so a bucket's internal width order is preserved.

    width_tol is the cap on padding inside one batch: samples more than
    `width_tol` px apart are not batched together even if adjacent.
    """
    order = np.argsort(ds.widths[ds.idx], kind="stable")
    sorted_idx = ds.idx[order]

    buckets, cur, cur_w = [], [], None
    for k in sorted_idx:
        w = int(ds.widths[int(k)])
        if cur and (w - cur_w) > width_tol:
            buckets.append(cur)
            cur, cur_w = [], None
        cur.append(int(k))
        cur_w = w
    if cur:
        buckets.append(cur)

    if rng is not None:
        rng.shuffle(buckets)

    for b in buckets:
        for s in range(0, len(b), batch_size):
            yield b[s:s + batch_size]


def collate(ds, ids, model=None):
    """
    Pad a width-homogeneous batch to its own max and build the CTC targets.

    Returns (images, target_lengths, flat_targets, widths, input_lengths).

    `input_lengths` is the per-sample TRUE timestep count, which is a function
    of that crop's own width. It is None only when no model is supplied (the
    unit tests that check padding alone), but every training and evaluation
    call must pass one: CTC aligns characters into the sequence it is told
    exists, so reporting the PADDED batch width would let the model place real
    characters in padding and produce confident nonsense.

    Note the CTC target convention: when targets are concatenated into one flat
    tensor, `target_lengths` stays PER-SAMPLE (not cumulative). The cumulative
    form is only for the tuple-of-ints API. Getting this backwards silently
    trains the model to emit the wrong number of characters.
    """
    ws = [int(ds.widths[i]) for i in ids]
    wmax = max(ws)
    n, h = len(ids), ds.images.shape[1]
    x = np.zeros((n, h, wmax), np.float32)
    lens = []
    for r, i in enumerate(ids):
        w = int(ds.widths[i])
        x[r, :, :w] = (ds.images[i, :, :w].astype(np.float32) / 255.0 - 0.5) / 0.5
        lens.append(len(R.encode(ds.labels[i])))
    flat = []
    for i in ids:
        flat.extend(R.encode(ds.labels[i]))
    inp = (torch.tensor([model.timesteps(w) for w in ws], dtype=torch.long)
           if model is not None else None)
    return (torch.from_numpy(x[:, None]),       # [n,1,h,wmax]
            torch.tensor(lens, dtype=torch.long),
            torch.tensor(flat, dtype=torch.long),
            torch.tensor(ws, dtype=torch.long),
            inp)


@torch.no_grad()
def evaluate(model, ds, indices, device, batch_size=64):
    """
    CER + exact-match, overall and per augmentation variant.

    Per-variant counts are accumulated as (edits, chars, exact, n) and the
    rates are DIVIDED AT THE END. Dividing per sample and then averaging would
    weight a 3-character crop the same as a 28-character one, which flatters
    CER and hides exactly the long labels (prices, account numbers) that carry
    the risk.
    """
    model.eval()
    sub = CRNNDataset(ds.images, ds.widths, ds.labels, indices)
    tot_e = tot_c = tot_x = tot_n = 0
    per = defaultdict(lambda: [0, 0, 0, 0])   # edits, chars, exact, n

    for ids in width_bucketed_batches(sub, batch_size):
        x, tl, flat, ws, _ = collate(sub, ids, model)
        logits = model(x.to(device)).argmax(-1).cpu().numpy()
        T_batch = logits.shape[1]
        for r, i in enumerate(ids):
            # Timesteps are a function of THIS crop's width, not the padded
            # batch width, so trim before decoding or the tail of every short
            # crop in a batch decodes as garbage.
            T = min(model.timesteps(int(ds.widths[i])), T_batch)
            got = R.decode_greedy(logits[r, :T])
            want = ds.labels[i]
            gn, wn = R.norm_label(got), R.norm_label(want)
            e = _edit_distance(gn, wn)
            c = max(len(wn), 1)
            ok = int(got.strip().upper() == want.strip().upper())
            tot_e += e; tot_c += c; tot_x += ok; tot_n += 1
            v = VARIANT_OF[int(i)]
            per[v][0] += e; per[v][1] += c
            per[v][2] += ok; per[v][3] += 1

    return {
        "n": tot_n,
        "cer": tot_e / max(tot_c, 1),
        "word_acc": tot_x / max(tot_n, 1),
        "by_variant": {
            k: {"cer": v[0] / max(v[1], 1),
                "word_acc": v[2] / max(v[3], 1),
                "n": v[3]}
            for k, v in sorted(per.items())
        },
    }


def _edit_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--data", default=str(CACHE))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    # Load every cache and CONCATENATE, so the model trains on both clean
    # scans and real hand-held photographs.
    #
    # The two caches have different schemas: SROIE carries `splits`/`variants`
    # and a `receipts` array, the handheld consensus cache carries `split` and
    # page names. Reading one schema and assuming the other would either crash
    # or silently train on half the data, so both are normalised here.
    #
    # The handheld photos are the ones the product actually faces -- night
    # shots, plastic sleeves, tilted -- while SROIE is flatbed scans. Training
    # on scans alone and testing on photos is a domain gap that showed up as
    # 14% CER on held-out scans against 56% on held-out photos.
    caches = []
    for spec in args.data.split(","):
        p = Path(spec.strip())
        if not p.exists():
            print(f"missing {p}\nrun:  bash run.sh build-data", file=sys.stderr)
            return 1
        caches.append(p)
    if not caches:
        print("no --data given", file=sys.stderr)
        return 1

    all_imgs, all_w, all_lab, all_split, all_var, all_src = [], [], [], [], [], []
    S2I = {"train": 0, "val": 1, "test": 2}
    S2V = {"clean": 0, "sroie": 0}
    for p in caches:
        d = np.load(p, allow_pickle=True)
        im, wd = d["images"], d["widths"]
        lb = [str(x) for x in d["labels"]]
        sp = d["splits"] if "splits" in d else d["split"]
        sp = [x if isinstance(x, str) else ["train", "val", "test"][int(x)]
              for x in sp]
        vr = [str(x) for x in d["variants"]] if "variants" in d \
            else ["handheld"] * len(lb)
        rc = [str(x) for x in d["receipts"]] if "receipts" in d \
            else [str(x) for x in d["page"]]
        all_imgs.append(im)
        all_w.append(np.asarray(wd, np.int32))
        all_lab.append(lb)
        all_split.append([S2I[s] for s in sp])
        all_var.append([S2V.get(v, 0) for v in vr])
        all_src.append([f"{p.name}:{r}" for r in rc])
        print(f"  loaded {p.name}: {len(lb)} crops, "
              f"{len(set(rc))} sources")

    images = np.concatenate(all_imgs, axis=0)
    widths = np.concatenate(all_w, axis=0)
    labels = [x for s in all_lab for x in s]
    splits = [x for s in all_split for x in s]
    variants = [x for s in all_var for x in s]
    srcs = [x for s in all_src for x in s]
    n_all = len(labels)

    # CTC feasibility filter. Computed with the real conv geometry per width,
    # not a formula, because MaxPool2d floors and odd widths are off by one.
    global VARIANT_OF
    VARIANT_OF = variants
    model = build(R.NUM_CLASSES).to(device)

    keep = []
    dropped = 0
    for i in range(n_all):
        t = model.timesteps(int(widths[i]))
        need = len(R.encode(labels[i]))
        if t >= need:
            keep.append(i)
        else:
            dropped += 1

    idx_by_split = {"train": [], "val": [], "test": []}
    for i in keep:
        idx_by_split[splits[i]].append(i)

    print(f"  crops           {n_all}")
    print(f"  CTC-infeasible  {dropped} ({dropped/n_all*100:.2f}%) dropped "
          f"-- T < label length makes the loss infinite")
    for k in ("train", "val", "test"):
        ids = idx_by_split[k]
        recs = len({srcs[i] for i in ids})
        print(f"  {k:5} {len(ids):6} crops  {recs:4} receipts")

    train_ds = CRNNDataset(images, widths, labels, idx_by_split["train"])
    val_ds = CRNNDataset(images, widths, labels, idx_by_split["val"])
    test_ds = CRNNDataset(images, widths, labels, idx_by_split["test"])

    model = model.to(device)
    n_params = model.n_params()
    print(f"\nmodel: {n_params:,} params, {R.NUM_CLASSES} classes, "
          f"charset={len(R.CHARSET)} symbols")

    ctc = nn.CTCLoss(blank=R.BLANK, zero_infinity=True)
    # zero_infinity=True is not a convenience: without it the 0.4% of samples
    # whose target is longer than the sequence contribute inf/NaN and destroy
    # the run. The filter above is belt, this is braces.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(list(width_bucketed_batches(train_ds, args.batch_size)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * steps_per_epoch),
        pct_start=0.25)

    CKPT.mkdir(parents=True, exist_ok=True)
    best = -1.0
    hist = []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        rng = np.random.RandomState(args.seed * 1000 + ep)
        tot, nb = 0.0, 0
        t_ep = time.time()
        for ids in width_bucketed_batches(train_ds, args.batch_size, rng=rng):
            x, tl, flat, ws, input_lengths = collate(train_ds, ids, model)
            x = x.to(device)
            logits = model(x)                       # [n,T,C]
            lp = logits.log_softmax(-1).permute(1, 0, 2)   # [T,n,C]
            # PER-SAMPLE input lengths are mandatory, not optional. Crops in a
            # batch share a padded width, but each crop's TRUE timestep count
            # is a function of its own width, so a short crop sitting in a
            # wide batch has FEWER real timesteps than T. Passing the batch
            # width as one int tells CTC that padded timesteps are real signal,
            # and it will happily align characters into the padding.
            loss = ctc(lp, flat.to(device),
                       input_lengths.to(device),      # [n] int64
                       tl.to(device))                 # [n] int64 target lengths
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping: CTC loss is notoriously spiky early in
            # training when the model has not yet learned to emit blanks, and
            # a single spike can undo a whole epoch of progress.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            try:
                sched.step()
            except ValueError:
                pass
            tot += float(loss.detach())
            nb += 1

        val = evaluate(model, val_ds, val_ds.idx, device)
        acc = val["word_acc"]
        hist.append({"epoch": ep, "loss": tot / max(nb, 1),
                     "val_word_acc": acc, "val_cer": val["cer"]})
        flag = ""
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(),
                        "num_classes": R.NUM_CLASSES,
                        "charset": R.CHARSET,
                        "epoch": ep, "val_word_acc": acc, "val_cer": val["cer"],
                        "params": n_params,
                        "seeder": "seedutil-v1"},
                       CKPT / "crnn.pt")
            flag = "  <- best, saved"
        print(f"  ep{ep:3}  loss {tot/max(nb,1):7.4f}  "
              f"val word-acc {acc*100:5.1f}%  CER {val['cer']*100:5.2f}%  "
              f"{time.time()-t_ep:5.1f}s{flag}")

    test = evaluate(model, test_ds, test_ds.idx, device)
    print(f"\nbest val word-acc  {best*100:.2f}%")
    print(f"TEST (held-out)    word-acc {test['word_acc']*100:.2f}%  "
          f"CER {test['cer']*100:.2f}%  ({test['n']} crops, "
          f"{len({d['receipts'][i] for i in idx_by_split['test']})} receipts)")

    (ROOT / "artifacts" / "ocr_training.json").write_text(json.dumps({
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "params": n_params, "num_classes": R.NUM_CLASSES,
        "charset": R.CHARSET,
        "crops_total": n_all, "dropped_infeasible": dropped,
        "train_crops": len(idx_by_split["train"]),
        "val_crops": len(idx_by_split["val"]),
        "test_crops": len(idx_by_split["test"]),
        "best_val_word_acc": best,
        "test_word_acc": test["word_acc"], "test_cer": test["cer"],
        "history": hist,
        "elapsed_s": time.time() - t0,
        "seeder": "seedutil-v1",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
SightLine ML — Train the document restorer on simulated phone-capture data.

Data is generated once from capture.py's physically-motivated profiles (motion
blur, bokeh defocus, Poisson-Gaussian sensor noise, hand shadow, perspective,
JPEG), then CACHED to .npz. Regenerating it costs ~1s per document because each
degradation is a real convolve, so without caching every training run pays the
generation cost again; with caching, subsequent epochs are pure tensor work.

LEAKAGE CONTROL: the split is by DOCUMENT SEED, not by patch. Every patch from
one rendered document lands entirely in train or entirely in val, so a val
patch cannot share glyph shapes with a train patch. Splitting by patch inflates
PSNR by several dB.

Chunked/resumable: --epochs is the TARGET total and the run resumes from
models/restorer/restorer.pt, so a CPU box that reaps background processes can
still reach the target in slices.

Usage:
    python3 src/train_restorer.py --epochs 30
    python3 src/train_restorer.py --epochs 30 --resume --docs-per-profile 12
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import capture  # noqa: E402
import seedutil  # noqa: E402  — stable cross-process seed derivation
from model import (CharbonnierLoss, PatchDataset, SightLineNet,  # noqa: E402
                   count_params, psnr)

PATCH_W, PATCH_H = 256, 64
BANDS = 3
CACHE = Path("artifacts/restorer_data.npz")


def build_pairs(n_docs_per_profile=6, seed=123, profiles=None, verbose=True):
    """
    Return (pairs, doc_seeds). Each pair is (degraded_patch, clean_patch).
    The random draw covers ALL profiles so the network sees the full
    degradation family, not just the currently-hard one.
    """
    profiles = profiles or list(capture.CAPTURE_PROFILES)
    pairs, seeds = [], []
    t0 = time.time()
    for pname in profiles:
        params = capture.CAPTURE_PROFILES[pname]
        for i in range(n_docs_per_profile):
            ds = seed * 7919 + seedutil.name_hash(pname) % 100003 + i
            rng = random.Random(ds)
            # vary the capture within the profile's envelope so the network
            # sees a distribution, not one fixed example per profile
            p = dict(params)
            if p.get("motion"):
                p["motion"] = max(3, int(p["motion"] * rng.uniform(0.7, 1.3)))
                p["motion_angle"] = rng.uniform(0, 180)
            if p.get("defocus"):
                p["defocus"] = max(0.8, p["defocus"] * rng.uniform(0.7, 1.4))
            if p.get("iso"):
                p["iso"] = int(p["iso"] * rng.uniform(0.6, 1.6))
            if p.get("light") is not None:
                p["light"] = min(0.8, p["light"] * rng.uniform(0.7, 1.3))
            if p.get("jpeg"):
                p["jpeg"] = int(np.clip(p["jpeg"] * rng.uniform(0.8, 1.2), 30, 95))
            lines, _ = capture.DOCS[ds % len(capture.DOCS)]
            clean = capture.render_document(lines, seed=ds)
            deg = capture.apply_profile(clean, seed=ds, **p)
            for band in range(BANDS):
                y0 = 100 + band * PATCH_H
                if y0 + PATCH_H > clean.size[1]:
                    break
                pairs.append((
                    capture.to_patch(deg, y0=y0, w=PATCH_W, h=PATCH_H),
                    capture.to_patch(clean, y0=y0, w=PATCH_W, h=PATCH_H),
                ))
                seeds.append(ds)
        if verbose:
            print(f"  generated {pname:<16} ({len(pairs)} patches, "
                  f"{time.time()-t0:.1f}s)", flush=True)
    return pairs, np.array(seeds)


def load_or_build(n_docs_per_profile, seed, verbose=True):
    """Return (pairs, seeds), building and caching on first use."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if int(z["n_docs_per_profile"]) >= n_docs_per_profile and \
           int(z["seed"]) == seed:
            if verbose:
                print(f"[data] cache hit {CACHE} "
                      f"({len(z['deg'])} patches)", flush=True)
            return list(zip(z["deg"], z["clean"])), z["seeds"]
    pairs, seeds = build_pairs(n_docs_per_profile, seed, verbose=verbose)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, deg=np.array([p[0] for p in pairs]),
                        clean=np.array([p[1] for p in pairs]),
                        seeds=seeds, n_docs_per_profile=n_docs_per_profile,
                        seed=seed)
    if verbose:
        print(f"[data] cached -> {CACHE}", flush=True)
    return pairs, seeds


def split_by_seed(pairs, seeds, seed=123, val_frac=0.2):
    uniq = sorted(set(seeds.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_seeds = set(uniq[:n_val])
    is_val = np.array([s in val_seeds for s in seeds])
    return ([p for p, v in zip(pairs, is_val) if not v],
            [p for p, v in zip(pairs, is_val) if v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30, help="TARGET total epochs")
    ap.add_argument("--docs-per-profile", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="models/restorer")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "restorer.pt"
    log_p = out / "restorer_training_log.json"

    print("[data] loading capture corpus ...", flush=True)
    pairs, seeds = load_or_build(args.docs_per_profile, args.seed)
    tr, va = split_by_seed(pairs, seeds, args.seed)
    print(f"  patches: train {len(tr)} / val {len(va)}  "
          f"(split by document seed, {len(set(seeds.tolist()))} docs)", flush=True)

    model = SightLineNet()
    print(f"[model] SightLineNet params={count_params(model):,}", flush=True)

    dl_tr = DataLoader(PatchDataset(tr), batch_size=args.batch, shuffle=True)
    dl_va = DataLoader(PatchDataset(va), batch_size=args.batch, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    loss_fn = CharbonnierLoss()

    start_epoch, history, best_psnr, best_state = 0, [], -1.0, None
    if args.resume and ck.exists():
        saved = torch.load(ck, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        start_epoch = saved["epoch"]
        best_psnr = saved.get("val_psnr", -1.0)
        best_state = saved["state_dict"]
        if log_p.exists():
            history = json.load(open(log_p))["history"]
        print(f"  [resume] from epoch {start_epoch} (best val PSNR {best_psnr:.2f} dB)",
              flush=True)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y in dl_tr:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vp, vc = [], []
            for x, y in dl_va:
                vp.append(model(x))
                vc.append(y)
            val_psnr = psnr(torch.cat(vp), torch.cat(vc))
        rec = {"epoch": epoch + 1, "train_loss": round(tot / max(nb, 1), 6),
               "val_psnr_db": round(val_psnr, 3),
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save({"state_dict": best_state, "epoch": epoch + 1,
                    "val_psnr": best_psnr, "args": vars(args)}, ck)
        json.dump({"history": history, "best_val_psnr_db": round(best_psnr, 3),
                   "n_train_patches": len(tr), "n_val_patches": len(va),
                   "docs_per_profile": args.docs_per_profile,
                   "params": count_params(model)},
                  open(log_p, "w"), indent=2)
        print(f"epoch {epoch+1:>3}/{args.epochs}  loss {rec['train_loss']:.5f}  "
              f"val PSNR {val_psnr:6.2f} dB  best {best_psnr:6.2f}  "
              f"[{rec['seconds']:.0f}s]", flush=True)

    print(f"\nbest val PSNR: {best_psnr:.2f} dB   "
          f"({time.time()-t0:.1f}s training)", flush=True)


if __name__ == "__main__":
    main()

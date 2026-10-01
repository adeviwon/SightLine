"""Benchmark: find a network/threading config that trains in a sane time."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from torch.utils.data import DataLoader
import numpy as np
from model import SightLineNet, CharbonnierLoss, PatchDataset

print(f"torch {torch.__version__}  threads={torch.get_num_threads()}")
z = np.load("artifacts/restorer_data.npz")
pairs = list(zip(z["deg"], z["clean"]))[:336]
print(f"{len(pairs)} patches")

for threads in (4,):
    torch.set_num_threads(threads)
    for ch, depth in ((48, 8), (32, 6), (24, 6)):
        m = SightLineNet(channels=ch, depth=depth)
        n = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        lf = CharbonnierLoss()
        dl = DataLoader(PatchDataset(pairs), batch_size=32, shuffle=True)
        t0 = time.time()
        nb = 0
        for x, y in dl:
            opt.zero_grad(); lf(m(x), y).backward(); opt.step(); nb += 1
            if nb == 3:
                break  # warmup
        t0 = time.time()
        nb = 0
        for x, y in dl:
            opt.zero_grad(); lf(m(x), y).backward(); opt.step(); nb += 1
        dt = time.time() - t0
        print(f"  ch={ch} depth={depth} params={n:>8,}  "
              f"{dt:.1f}s/epoch ({nb} batches)  -> 40 epochs = {dt*40/60:.1f} min")

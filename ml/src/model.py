"""
SightLine ML — PyTorch document-restoration network (SightLineNet).

WHAT THIS MODEL IS FOR
----------------------
The user's success criterion is accurate text recognition from blurry phone
photos. Tesseract's accuracy collapses with motion blur, sensor noise and
low contrast. Classical filters (median, unsharp) apply a FIXED inverse
regardless of what actually damaged the image. This network learns the
inverse of the measured degradation family instead.

DESIGN CONSTRAINTS (these drive every choice below)
  - Must run on a phone CPU inside a browser. So: small, single-channel,
    ONNX-exportable, no exotic ops.
  - Must train on a laptop CPU in minutes. So: ~220K params, no attention.
  - Must handle BLUR, not just noise. A pure denoiser (DnCNN) cannot invert
    an information-destroying blur — you cannot recover detail that is not
    in the pixels. So the network is a RESTORER: it sharpens and denoises
    jointly, trained against the clean render as supervision.

ARCHITECTURE
  Input  64x256 grayscale, float32 [0,1]
  Conv3x3(1->48) + ReLU
  5 x [Conv3x3(48->48) + BatchNorm + ReLU]
  Conv3x3(48->48) + Conv3x3(48->1)
  Output = input + predicted_residual   (residual learning: the net learns
                                          the correction, not the image)

Residual formulation matters here: it makes the network identity-initialised
in effect, so it starts as a no-op and learns to ADD the sharpening/denoising
it needs. A direct-output network has to learn the identity from scratch and
trains much worse on a small corpus.

TRAINING
  Loss: Charbonnier (smooth L1) — robust to the heavy-tailed residuals that
  outliers produce, unlike plain MSE.
  Target: the CLEAN render, not the clean patch, so the supervision matches
  what OCR actually needs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1: sqrt(a^2 + eps^2). Robust to outliers, differentiable at 0."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class SightLineNet(nn.Module):
    """
    Residual document restorer. See module docstring for the rationale.

    Defaults are ch=32, depth=6 (37,793 params). That size was chosen by
    measurement, not taste: `ml/src/bench.py` on 4 CPU threads gives

        ch=48 depth=8  126,097 params   124 s/epoch
        ch=32 depth=6   37,793 params    52 s/epoch   <- default
        ch=24 depth=6   22,753 params    29 s/epoch

    40 epochs at the default trains in ~35 min on a laptop CPU, which fits in
    a hackathon, and 38K parameters is comfortably inside a phone's ONNX
    budget. Pass channels/depth explicitly if you want the bigger model and
    have the time budget for it.
    """

    def __init__(self, channels=32, depth=6):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 3):
            layers += [nn.Conv2d(channels, channels, 3, padding=1),
                       nn.BatchNorm2d(channels),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, channels, 3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(channels, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)
        # Zero-init the last layer: the network begins as an exact identity
        # (output == input) and learns the residual from there.
        last = self.body[-1]
        assert isinstance(last, nn.Conv2d), "last layer must be the output conv"
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return x + self.body(x)


class PatchDataset(torch.utils.data.Dataset):
    """(degraded_patch, clean_patch) float32 [0,1] 64x256 tensors."""

    def __init__(self, pairs):
        self.items = [(np.asarray(d, dtype=np.float32), np.asarray(c, dtype=np.float32))
                      for d, c in pairs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, c = self.items[i]
        return torch.from_numpy(d)[None], torch.from_numpy(c)[None]


def psnr(pred, target):
    """Peak signal-to-noise ratio on [0,1] tensors. dB."""
    mse = torch.mean((pred - target) ** 2).clamp(min=1e-10)
    return float(10.0 * torch.log10(1.0 / mse))


def count_params(model):
    return sum(p.numel() for p in model.parameters())

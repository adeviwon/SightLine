"""
test_model.py — SightLineNet's architectural invariants.

Three properties justify the network's design and each one is a real
regression risk if it silently breaks:

  1. IDENTITY AT INIT. The last conv is zero-initialised, so forward(x) == x
     exactly before training. This is what lets a small corpus train a
     restorer at all. If it breaks, the first training step is a large
     destructive gradient, not a small corrective one.
  2. SHAPE. Output shape == input shape, or the tiled inference in
     evaluate.py cannot reassemble an image.
  3. MOBILE BUDGET. The network must run in a browser on a phone CPU, so the
     parameter count is a hard design constraint, not a preference.

Also: gradients must be finite, PSNR must behave, and tiled inference must
agree with direct inference away from the tile seams.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Project modules live in ml/src as plain modules (not an installed package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from model import (  # noqa: E402
    CharbonnierLoss, PatchDataset, SightLineNet, count_params, psnr)

PATCH_H, PATCH_W = 64, 256
MOBILE_PARAM_BUDGET = 500_000


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return SightLineNet()


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(1)
    return torch.rand(2, 1, PATCH_H, PATCH_W)


# ── 1. Shape ────────────────────────────────────────────────────────────

def test_output_shape_equals_input_shape(net, x):
    assert net(x).shape == x.shape


@pytest.mark.parametrize("shape", [(1, 1, 64, 256), (4, 1, 64, 256), (1, 1, 128, 512)])
def test_output_shape_across_batch_and_tile_sizes(net, shape):
    torch.manual_seed(2)
    t = torch.rand(*shape)
    assert net(t).shape == t.shape


# ── 2. Identity at init (the load-bearing invariant) ────────────────────

def test_network_is_exact_identity_at_init(net, x):
    """
    Zero-init of the final conv makes the residual exactly zero, so the
    network is a mathematical identity before any training. Asserted with
    torch.equal (bitwise), not allclose: 'close to identity' would hide a
    small non-zero init that still wrecks the first optimisation step.
    """
    net.eval()
    with torch.no_grad():
        y = net(x)
    assert torch.equal(y, x), (
        f"not an exact identity at init: max|diff| = "
        f"{(y - x).abs().max().item():.3e}")


def test_identity_at_init_in_train_mode(x):
    """BatchNorm updates running stats in train mode but the residual is
    still zero, so the output must be the identity there too."""
    torch.manual_seed(3)
    m = SightLineNet()
    m.train()
    with torch.no_grad():
        y = m(x)
    assert torch.equal(y, x)


def test_last_layer_is_zero_initialised(net):
    """The identity property comes from THIS initialisation; assert the
    mechanism so a future refactor cannot remove it silently."""
    last = net.body[-1]
    assert isinstance(last, torch.nn.Conv2d), "last layer must be the output conv"
    assert torch.count_nonzero(last.weight) == 0
    assert torch.count_nonzero(last.bias) == 0


def test_identity_holds_for_saturated_pixels():
    """Identity must hold for any input, including pure black/white — a
    BatchNorm division-by-zero would break it only on degenerate inputs."""
    torch.manual_seed(4)
    m = SightLineNet().eval()
    for val in (0.0, 1.0):
        t = torch.full((1, 1, PATCH_H, PATCH_W), val)
        with torch.no_grad():
            assert torch.equal(m(t), t), f"identity failed for input {val}"


# ── 3. Mobile budget ────────────────────────────────────────────────────

def test_parameter_count_within_mobile_budget(net):
    n = count_params(net)
    assert n < MOBILE_PARAM_BUDGET, (
        f"{n:,} parameters exceeds the {MOBILE_PARAM_BUDGET:,} phone-CPU budget")


def test_parameter_count_is_a_plausible_restorer(net):
    """
    Guard the budget from BOTH sides: the network must be small enough for a
    phone, but not so small that it has collapsed into a no-op. (The exact
    size is a design choice recorded in SightLineNet's docstring and may
    change; these bounds are what must not change.)
    """
    n = count_params(net)
    assert n < MOBILE_PARAM_BUDGET, (
        f"{n:,} parameters exceeds the {MOBILE_PARAM_BUDGET:,} phone-CPU budget")
    assert n > 10_000, (
        f"only {n:,} parameters — the restorer has collapsed toward a no-op")


def test_larger_explicit_configs_stay_within_budget():
    """
    The budget must hold for the biggest configuration the constructor
    advertises, not just the default: picking a bigger net must never be able
    to silently break the phone-CPU constraint.
    """
    for channels, depth in [(32, 6), (48, 8)]:
        torch.manual_seed(0)
        n = count_params(SightLineNet(channels=channels, depth=depth))
        assert n < MOBILE_PARAM_BUDGET, (
            f"ch={channels} depth={depth} has {n:,} parameters, over budget")


def test_model_is_the_same_size_as_its_input_at_the_documented_default():
    """The documented default config must actually be the default."""
    small = SightLineNet()
    assert count_params(small) == count_params(SightLineNet(channels=32, depth=6))


# ── 4. Backward pass ────────────────────────────────────────────────────

def test_backward_pass_produces_finite_gradients(net, x):
    """
    After one optimiser-free step, EVERY parameter must have a finite
    gradient and the output must be differentiable. The train loop clips at
    1.0, which silently papers over a NaN gradient (clip returns NaN), so
    finiteness is asserted here instead.
    """
    net.train()
    net.zero_grad(set_to_none=True)
    target = torch.rand_like(x)
    loss = CharbonnierLoss()(net(x), target)
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    n_grads = 0
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a NaN/Inf gradient"
        n_grads += 1
    assert n_grads == count_params(net) or n_grads == len(list(net.parameters()))


def test_gradient_reaches_first_layer_after_one_step(x):
    """
    At step 0 the last conv is zero-initialised, so d(loss)/d(anything upstream
    of it) is EXACTLY zero by construction — that is what makes the network an
    identity at init, not a bug. The property that must hold is that gradient
    reaches the first conv as soon as the head becomes non-zero, i.e. the stem
    is genuinely connected and can train.
    """
    torch.manual_seed(7)
    m = SightLineNet()
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    y = torch.rand_like(x)

    loss = CharbonnierLoss()(m(x), y)
    loss.backward()
    assert m.body[0].weight.grad is not None
    assert torch.count_nonzero(m.body[-1].weight.grad) > 0, (
        "the head got no gradient, so nothing can ever train")
    assert torch.count_nonzero(m.body[0].weight.grad) == 0, (
        "the stem is NOT upstream of the zero-init head — the net is detached")

    opt.step()          # the head is now non-zero
    m.zero_grad(set_to_none=True)
    CharbonnierLoss()(m(x), y).backward()
    assert torch.count_nonzero(m.body[0].weight.grad) > 0, (
        "after one optimiser step the first conv still gets no gradient — "
        "the stem is disconnected and only the head would ever train")


def test_relu_activations_are_not_all_dead(net, x):
    """Dead ReLUs would still 'work' but the net would be a 220K-param no-op."""
    net.train()
    h = x
    with torch.no_grad():
        for i, layer in enumerate(net.body):
            h = layer(h)
            if isinstance(layer, torch.nn.ReLU) and h.numel():
                assert (h > 0).any(), f"ReLU at body[{i}] is entirely dead"


# ── 5. PSNR ─────────────────────────────────────────────────────────────

def test_psnr_of_perfect_reconstruction_is_very_high(x):
    assert psnr(x, x) > 30.0, "a perfect reconstruction must score high"


def test_psnr_is_monotone_in_error():
    torch.manual_seed(5)
    target = torch.rand(1, 1, PATCH_H, PATCH_W)
    small = target + 0.01 * torch.randn_like(target)
    large = target + 0.20 * torch.randn_like(target)
    assert psnr(target, target) > psnr(small, target) > psnr(large, target)


def test_psnr_of_a_darkened_image_is_low():
    """PSNR must actually punish error, not always return a big number."""
    target = torch.full((1, 1, PATCH_H, PATCH_W), 0.9)
    assert psnr(target, torch.full_like(target, 0.3)) < 20.0


# ── 6. Charbonnier loss ─────────────────────────────────────────────────

def test_charbonnier_is_essentially_zero_at_perfect_prediction(x):
    """It is a smooth L1, so it bottoms out at eps rather than 0 — but it
    must be ~0, not ~the scale of the data."""
    v = CharbonnierLoss()(x, x).item()
    assert v < 1e-2, f"loss at pred==target is {v}"


def test_charbonnier_is_positive_and_grows_with_error():
    target = torch.zeros(1, 1, PATCH_H, PATCH_W)
    fn = CharbonnierLoss()
    small = fn(torch.full_like(target, 0.05), target).item()
    large = fn(torch.full_like(target, 0.50), target).item()
    assert 0 < small < large


def test_charbonnier_is_finite_at_zero_error_gradient():
    """The eps term exists so the gradient at 0 is defined; exercise it."""
    t = torch.zeros(1, 1, 4, 4, requires_grad=True)
    CharbonnierLoss()(t, torch.zeros(1, 1, 4, 4)).backward()
    assert torch.isfinite(t.grad).all(), "gradient at zero error is not finite"


def test_charbonnier_is_robust_to_outliers():
    """The stated reason for choosing Charbonnier over MSE: heavy-tailed
    residuals from bad captures must not dominate the batch."""
    base = torch.full((1, 1, 64, 256), 0.1)
    outlier = base.clone()
    outlier[0, 0, 0, 0] = 50.0          # one wildly wrong pixel
    fn = CharbonnierLoss()
    c_ratio = fn(outlier, torch.zeros_like(outlier)).item() / fn(
        base, torch.zeros_like(base)).item()
    mse_ratio = float((outlier ** 2).mean() / (base ** 2).mean())
    # Charbonnier is ~L1, so one huge pixel barely moves the mean; MSE squares
    # it and is dominated by that single pixel.
    assert c_ratio < 1.5, f"Charbonnier is not robust to an outlier: {c_ratio:.2f}x"
    assert mse_ratio > 5.0, f"the MSE control is not outlier-dominated: {mse_ratio:.2f}x"


# ── 7. Tiled inference (the evaluate.py pattern) ────────────────────────

def _tiled(model, arr_, th=PATCH_H, tw=PATCH_W):
    """Reimplementation of evaluate.arm_restorer's tiling, on float [0,1]."""
    H, W = arr_.shape
    pad_h, pad_w = (th - H % th) % th, (tw - W % tw) % tw
    gp = np.pad(arr_, ((0, pad_h), (0, pad_w)), mode="reflect")
    Hp, Wp = gp.shape
    out = np.zeros((Hp, Wp), np.float32)
    cnt = np.zeros((Hp, Wp), np.float32)
    with torch.no_grad():
        for y in range(0, Hp, th):
            for zz in range(0, Wp, tw):
                tile = torch.from_numpy(gp[y:y + th, zz:zz + tw])[None, None]
                out[y:y + th, zz:zz + tw] += model(tile)[0, 0].numpy()
                cnt[y:y + th, zz:zz + tw] += 1
    return out[:H, :W] / np.maximum(cnt, 1.0)[:H, :W]


def test_tiled_inference_equals_direct_inference_no_padding(net):
    """
    On a size that needs no padding the tiling is a pure loop, so it must
    agree with the direct forward pass EXACTLY (bitwise). This catches an
    off-by-one in the tile loop or a wrong stride.
    """
    net.eval()
    g = torch.rand(1, 1, 2 * PATCH_H, 2 * PATCH_W)
    with torch.no_grad():
        direct = net(g)[0, 0].numpy()
    assert np.array_equal(_tiled(net, g[0, 0].numpy()), direct)


def test_tiled_inference_with_padding_preserves_shape(net):
    """A non-multiple size must be padded, tiled, cropped — and come back at
    exactly the input size. This is the shape contract evaluate.py relies on."""
    net.eval()
    g = torch.rand(PATCH_H + 13, PATCH_W + 37)
    tiled = _tiled(net, g.numpy())
    assert tiled.shape == g.shape


def test_tiled_inference_agrees_away_from_seams(net, x):
    """
    Honest statement of what tiling guarantees: a 3x3 conv sees zero padding
    at a tile border, so pixels within the receptive-field radius of a seam
    can differ. Away from those seams the two must agree closely. (At init
    the net is an exact identity, so this is a tight test.)
    """
    net.eval()
    g = torch.rand(1, 1, 2 * PATCH_H, 2 * PATCH_W)
    with torch.no_grad():
        direct = net(g)[0, 0].numpy()
    d = np.abs(_tiled(net, g[0, 0].numpy()) - direct)
    mask = np.ones(d.shape, bool)
    for y in (0, PATCH_H, 2 * PATCH_H):
        mask[max(0, y - 2):y + 2, :] = False
    for zz in (0, PATCH_W, 2 * PATCH_W):
        mask[:, max(0, zz - 2):zz + 2] = False
    assert d[mask].max() < 1e-5, (
        f"tiling disagrees away from the seams by {d[mask].max():.3e}")


def test_tiled_inference_error_is_confined_to_seams(x):
    """Perturb the head so the net is NOT an identity, then confirm the
    disagreement is bounded and local — not a wholesale misalignment."""
    torch.manual_seed(6)
    m = SightLineNet().eval()
    with torch.no_grad():
        m.body[-1].weight.normal_(0, 0.1)
        m.body[-1].bias.normal_(0, 0.1)
    g = torch.rand(1, 1, 2 * PATCH_H, 2 * PATCH_W)
    with torch.no_grad():
        direct = m(g)[0, 0].numpy()
    d = np.abs(_tiled(m, g[0, 0].numpy()) - direct)
    assert d.max() < 0.5, f"tiled output diverges wildly: {d.max():.3f}"
    # the receptive field of 8 stacked 3x3 convs spans +/-8 px of a seam
    mask = np.ones(d.shape, bool)
    for y in (0, PATCH_H, 2 * PATCH_H):
        mask[max(0, y - 8):y + 8, :] = False
    for zz in (0, PATCH_W, 2 * PATCH_W):
        mask[:, max(0, zz - 8):zz + 8] = False
    assert d[mask].max() < 1e-4, (
        f"tiling error is not confined to seams: {d[mask].max():.3e}")


# ── 8. PatchDataset ─────────────────────────────────────────────────────

def test_patch_dataset_yields_channel_first_tensors():
    d = np.random.rand(64, 256).astype(np.float32)
    c = np.random.rand(64, 256).astype(np.float32)
    ds = PatchDataset([(d, c)])
    xd, xc = ds[0]
    assert xd.shape == (1, 64, 256) == xc.shape, "expected NCHW patches"
    assert xd.dtype == torch.float32
    assert torch.equal(xd[0], torch.from_numpy(d))


def test_patch_dataset_preserves_pair_order_and_count():
    pairs = [(np.full((64, 256), i, np.float32), np.full((64, 256), i + 100, np.float32))
             for i in range(5)]
    ds = PatchDataset(pairs)
    assert len(ds) == 5
    for i in range(5):
        d, c = ds[i]
        assert float(d[0, 0, 0]) == i
        assert float(c[0, 0, 0]) == i + 100


def test_dataset_feeds_the_model(net):
    """End-to-end shape check: dataset -> batch -> forward -> loss."""
    pairs = [(np.random.rand(64, 256).astype(np.float32),
              np.random.rand(64, 256).astype(np.float32)) for _ in range(3)]
    ds = PatchDataset(pairs)
    batch = torch.stack([ds[i][0] for i in range(len(ds))])
    out = net(batch)
    assert out.shape == batch.shape
    assert torch.isfinite(CharbonnierLoss()(out, torch.stack([ds[i][1] for i in range(len(ds))])))
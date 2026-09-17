import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jepa_model.losses import (
    make_base_grid,
    photometric_loss,
    smoothness_loss,
    tokens_to_grid,
    warp_image_with_flow,
)

B, C, H, W, D, P = 2, 3, 8, 8, 4, 16
GRID = 4  # sqrt(P)


def test_tokens_to_grid_roundtrip():
    tokens = torch.arange(B * P * D, dtype=torch.float32).reshape(B, P, D)
    grid = tokens_to_grid(tokens)
    assert grid.shape == (B, D, GRID, GRID)
    recovered = grid.reshape(B, D, P).transpose(1, 2)
    assert torch.allclose(recovered, tokens)


def test_tokens_to_grid_rejects_non_square():
    tokens = torch.zeros(1, 12, 4)
    with pytest.raises(ValueError):
        tokens_to_grid(tokens)


def test_make_base_grid_shape_and_range():
    grid = make_base_grid(3, H, W, torch.device("cpu"))
    assert grid.shape == (3, H, W, 2)
    assert -1.0 <= grid.abs().max() <= 1.0 + 1e-6
    assert torch.equal(grid[0, 1:, :, 0], grid[0, :-1, :, 0])


def test_zero_flow_is_identity():
    img = torch.rand(B, C, H, W)
    flow = torch.zeros(B, 2, H, W)
    warped = warp_image_with_flow(img, flow)
    assert torch.allclose(warped, img, atol=1e-4)


def test_translation_flow_shifts_image():
    img = torch.zeros(1, 1, H, W)
    img[:, :, 2, 2] = 1.0
    flow = torch.zeros(1, 2, H, W)
    flow[:, 0] = 2.0
    flow[:, 1] = 0.0
    warped = warp_image_with_flow(img, flow)
    # grid_sample warps backward: sampling input x+2 maps content two pixels left.
    peak = warped[0, 0].argmax().item()
    row, col = divmod(peak, W)
    assert row == 2 and col == 0


def test_photometric_loss_prefers_true_flow():
    frame_t = torch.rand(1, 1, H, W)
    true_flow = torch.zeros(1, 2, H, W)
    true_flow[:, 1] = 3.0
    frame_t1 = warp_image_with_flow(frame_t, true_flow).detach()
    wrong_flow = torch.full((1, 2, H, W), 5.0)
    assert photometric_loss(frame_t, frame_t1, true_flow) < photometric_loss(
        frame_t, frame_t1, wrong_flow
    )


def test_photometric_loss_zero_on_identical_frames():
    img = torch.rand(1, 1, H, W)
    loss = photometric_loss(img, img, torch.zeros(1, 2, H, W))
    assert loss.item() < 1e-5


def test_smoothness_loss_zero_for_constant_flow():
    flow = torch.full((1, 2, H, W), np.pi)
    assert smoothness_loss(flow).item() == pytest.approx(0.0, abs=1e-6)


def test_smoothness_loss_positive_for_rough_flow():
    flow = torch.rand(1, 2, H, W)
    assert smoothness_loss(flow) > 0

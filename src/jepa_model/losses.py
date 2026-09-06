from __future__ import annotations

import torch
import torch.nn.functional as F


def tokens_to_grid(tokens: torch.Tensor) -> torch.Tensor:
    """Convert [B, P, D] tokens to [B, D, H, W] assuming square patch grid."""
    b, p, d = tokens.shape
    h = int(p**0.5)
    if h * h != p:
        raise ValueError(f"Expected square token count, got {p}")
    return tokens.transpose(1, 2).reshape(b, d, h, h)


def make_base_grid(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1)
    return grid.unsqueeze(0).repeat(b, 1, 1, 1)


def warp_image_with_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    img: [B, C, H, W], flow: [B, 2, H, W] in pixel units.
    """
    b, _c, h, w = img.shape
    base_grid = make_base_grid(b, h, w, img.device)
    # Normalize flow from pixels to [-1, 1] grid space.
    fx = 2.0 * flow[:, 0] / max(w - 1, 1)
    fy = 2.0 * flow[:, 1] / max(h - 1, 1)
    flow_grid = torch.stack([fx, fy], dim=-1)
    sample_grid = base_grid + flow_grid
    return F.grid_sample(img, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


def photometric_loss(frame_t: torch.Tensor, frame_t1: torch.Tensor, flow_t_to_t1: torch.Tensor) -> torch.Tensor:
    pred_t1 = warp_image_with_flow(frame_t, flow_t_to_t1)
    return F.l1_loss(pred_t1, frame_t1)


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    return dx + dy

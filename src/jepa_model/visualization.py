from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_hwc(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return arr


def flow_to_rgb(flow: torch.Tensor) -> np.ndarray:
    fx = flow[0].detach().cpu().numpy()
    fy = flow[1].detach().cpu().numpy()
    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)
    h = (ang + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = mag / (mag.max() + 1e-6)

    import colorsys

    rgb = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float32)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            rgb[i, j] = colorsys.hsv_to_rgb(float(h[i, j]), float(s[i, j]), float(v[i, j]))
    return rgb


def save_training_curves(path: str | Path, losses: dict[str, list[float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, values in losses.items():
        if values:
            plt.plot(values, label=name)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("MC-JEPA Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_triplet(path: str | Path, frame_t: torch.Tensor, frame_t1: torch.Tensor, warped: torch.Tensor, flow: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb_t = _to_hwc(frame_t)
    rgb_t1 = _to_hwc(frame_t1)
    rgb_w = _to_hwc(warped)
    rgb_f = flow_to_rgb(flow)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(rgb_t)
    axs[0].set_title("Frame t")
    axs[1].imshow(rgb_t1)
    axs[1].set_title("Frame t+1")
    axs[2].imshow(rgb_w)
    axs[2].set_title("Warp(frame t)")
    axs[3].imshow(rgb_f)
    axs[3].set_title("Predicted flow")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)

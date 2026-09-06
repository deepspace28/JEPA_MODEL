from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LocalVideoDataset(Dataset):
    """
    Expects .npy clips with shape (T, H, W, C) uint8.
    """

    def __init__(self, root: str, clip_len: int, size: int):
        self.root = Path(root)
        self.clip_len = clip_len
        self.size = size
        self.files = sorted(self.root.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy clips found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        clip = np.load(self.files[idx])
        if clip.shape[0] < self.clip_len:
            raise ValueError(f"Clip too short: {self.files[idx]}")
        clip = clip[: self.clip_len]
        clip = torch.from_numpy(clip).float() / 255.0
        clip = clip.permute(0, 3, 1, 2)
        return clip

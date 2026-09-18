import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jepa_model.data.video_stream_dataset import LocalVideoDataset


def _make_clip(path, t=5, h=8, w=8, c=3, seed=0):
    rng = np.random.default_rng(seed)
    np.save(path, rng.integers(0, 256, size=(t, h, w, c), dtype=np.uint8))


def test_raises_when_no_clips(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalVideoDataset(tmp_path, clip_len=2, size=8)


def test_discovers_and_sorts_files(tmp_path):
    for name in ["b.npy", "a.npy", "c.npy"]:
        _make_clip(tmp_path / name, seed=hash(name) % 10)
    ds = LocalVideoDataset(tmp_path, clip_len=2, size=8)
    assert [Path(f).name for f in ds.files] == ["a.npy", "b.npy", "c.npy"]
    assert len(ds) == 3


def test_getitem_shape_dtype_and_scale(tmp_path):
    clip = np.full((5, 8, 8, 3), 255, dtype=np.uint8)
    np.save(tmp_path / "clip.npy", clip)
    ds = LocalVideoDataset(tmp_path, clip_len=5, size=8)
    out = ds[0]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (5, 3, 8, 8)
    assert out.max() == 1.0 and out.dtype == torch.float32


def test_truncates_to_clip_len(tmp_path):
    np.save(tmp_path / "long.npy", np.zeros((10, 8, 8, 3), dtype=np.uint8))
    ds = LocalVideoDataset(tmp_path, clip_len=4, size=8)
    assert ds[0].shape[0] == 4


def test_rejects_too_short_clip(tmp_path):
    np.save(tmp_path / "short.npy", np.zeros((2, 8, 8, 3), dtype=np.uint8))
    ds = LocalVideoDataset(tmp_path, clip_len=4, size=8)
    with pytest.raises(ValueError, match="too short"):
        ds[0]

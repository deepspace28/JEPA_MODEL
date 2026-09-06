from pathlib import Path

import numpy as np


def test_dataset_layout_smoke():
    train = Path("data/clips/train")
    val = Path("data/clips/val")
    train.mkdir(parents=True, exist_ok=True)
    val.mkdir(parents=True, exist_ok=True)

    fake = np.zeros((16, 224, 224, 3), dtype=np.uint8)
    np.save(train / "clip_000000.npy", fake)
    np.save(val / "clip_000000.npy", fake)

    assert (train / "clip_000000.npy").exists()
    assert (val / "clip_000000.npy").exists()

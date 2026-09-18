import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jepa_model.utils import choose_device, save_json, set_seed


def test_choose_device_honors_explicit_cfg():
    assert choose_device("cpu") == "cpu"


def test_choose_device_auto_never_crashes():
    # on CI there is no CUDA, but the call must stay valid either way
    assert choose_device("auto") in ("cuda", "cpu")


def test_set_seed_is_deterministic():
    import random

    import numpy as np
    import torch

    set_seed(7)
    a = (random.random(), np.random.rand(), torch.rand(3).sum().item())
    set_seed(7)
    b = (random.random(), np.random.rand(), torch.rand(3).sum().item())
    assert a == b


def test_save_json_creates_parents_and_roundtrips(tmp_path):
    import json

    target = tmp_path / "nested" / "dir" / "out.json"
    payload = {"step": 3, "loss": 0.25}
    save_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload

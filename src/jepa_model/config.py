from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        return Config(raw=yaml.safe_load(f))

# JEPA_MODEL

Professional, reproducible implementation of JEPA-style training with an MC-JEPA-inspired joint objective (content + motion).

Reference paper:
- MC-JEPA: *A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features* (arXiv:2307.12698v1, July 24, 2023)

## Installation

```bash
python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Dataset Presets

Available presets for preprocessing:
- `kinetics`
- `something-something`
- `ego4d`

## Real Data Pipeline

1) Put raw videos under one root, e.g. `data/raw_videos/`.

2) Build clips with a preset:

Kinetics-style:
```bash
python scripts/prepare_videos.py --preset kinetics --input-root data/raw_videos --train-out data/clips/train --val-out data/clips/val
```

Something-Something-style:
```bash
python scripts/prepare_videos.py --preset something-something --input-root data/raw_videos --train-out data/clips/train --val-out data/clips/val
```

Ego4D-style:
```bash
python scripts/prepare_videos.py --preset ego4d --input-root data/raw_videos --train-out data/clips/train --val-out data/clips/val
```

3) Train with matching config:

Kinetics:
```bash
python scripts/train.py train-mc-jepa --config configs/mc_jepa_kinetics.yaml
```

Something-Something:
```bash
python scripts/train.py train-mc-jepa --config configs/mc_jepa_something_something.yaml
```

Ego4D:
```bash
python scripts/train.py train-mc-jepa --config configs/mc_jepa_ego4d.yaml
```

4) Run ablation benchmark:

```bash
python scripts/train.py benchmark-mc-jepa --config configs/mc_jepa_kinetics.yaml
```

## Output Artifacts

Saved under `artifacts/mc_jepa/<preset>/`:
- `mcjepa_losses.png`
- `sample_step_*.png`
- `benchmark/benchmark_runs.csv`
- `benchmark/benchmark_summary.json`
- `benchmark/benchmark_ablation.png`
- `benchmark/REPORT.md`

## Optional Debug Source

Atari collector is kept only for debugging:
```bash
python scripts/collect_atari.py
```

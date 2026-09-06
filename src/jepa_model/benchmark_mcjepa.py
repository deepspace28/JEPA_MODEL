from __future__ import annotations

import copy
import csv
from pathlib import Path
from statistics import mean, stdev

from jepa_model.config import load_config
from jepa_model.train_mcjepa import train_mc_jepa
from jepa_model.utils import save_json


def _write_config(path: Path, cfg: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_benchmark(config_path: str = "configs/mc_jepa.yaml") -> dict:
    cfg = load_config(config_path).raw
    bench = cfg.get("benchmark", {})
    seeds = bench.get("seeds", [cfg["seed"]])
    ablations = bench.get("ablations", [{"name": "full", "lambda_content": 1.0, "lambda_photo": 1.0, "lambda_smooth": 0.1}])

    artifacts_root = Path(cfg["paths"].get("artifacts_dir", "artifacts/mc_jepa"))
    bench_dir = artifacts_root / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for ab in ablations:
        ab_name = ab["name"]
        vals = []
        for seed in seeds:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["seed"] = int(seed)
            run_cfg["training"]["lambda_content"] = float(ab["lambda_content"])
            run_cfg["training"]["lambda_photo"] = float(ab["lambda_photo"])
            run_cfg["training"]["lambda_smooth"] = float(ab["lambda_smooth"])
            run_cfg["paths"]["artifacts_dir"] = str(artifacts_root / f"{ab_name}_seed{seed}")
            run_cfg["paths"]["checkpoints_dir"] = str(Path(cfg["paths"]["checkpoints_dir"]) / f"{ab_name}_seed{seed}")

            run_cfg_path = bench_dir / f"config_{ab_name}_seed{seed}.yaml"
            _write_config(run_cfg_path, run_cfg)
            metrics = train_mc_jepa(str(run_cfg_path))
            vals.append(metrics["final_total"])
            runs.append({
                "ablation": ab_name,
                "seed": seed,
                "final_total": metrics["final_total"],
                "final_content": metrics["final_content"],
                "final_photo": metrics["final_photo"],
                "final_smooth": metrics["final_smooth"],
            })

    by_ablation = {}
    for ab in ablations:
        name = ab["name"]
        vals = [r["final_total"] for r in runs if r["ablation"] == name and r["final_total"] is not None]
        by_ablation[name] = {
            "mean_final_total": mean(vals) if vals else None,
            "std_final_total": stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    csv_path = bench_dir / "benchmark_runs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation", "seed", "final_total", "final_content", "final_photo", "final_smooth"])
        writer.writeheader()
        writer.writerows(runs)

    summary = {"runs": runs, "summary": by_ablation}
    save_json(bench_dir / "benchmark_summary.json", summary)

    # Plot summary bars.
    import matplotlib.pyplot as plt

    labels = list(by_ablation.keys())
    means = [by_ablation[k]["mean_final_total"] for k in labels]
    stds = [by_ablation[k]["std_final_total"] for k in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel("Final total loss (lower is better)")
    plt.title("MC-JEPA Ablation Benchmark")
    plt.tight_layout()
    plt.savefig(bench_dir / "benchmark_ablation.png", dpi=140)
    plt.close()

    # Markdown report.
    md = ["# MC-JEPA Benchmark Report", "", "| Ablation | Mean Final Total | Std | N |", "|---|---:|---:|---:|"]
    for k in labels:
        row = by_ablation[k]
        md.append(f"| {k} | {row['mean_final_total']:.6f} | {row['std_final_total']:.6f} | {row['n']} |")
    (bench_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    return summary

import argparse
import json

from jepa_model.benchmark_mcjepa import run_benchmark
from jepa_model.evaluate import evaluate
from jepa_model.train_decoder_probe import train_decoder_probe
from jepa_model.train_mcjepa import train_mc_jepa
from jepa_model.train_stage1 import train


def main() -> None:
    parser = argparse.ArgumentParser(description="JEPA_MODEL CLI")
    parser.add_argument(
        "command",
        choices=["train", "train-mc-jepa", "benchmark-mc-jepa", "probe", "eval", "benchmark"],
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    if args.command == "train":
        print(train(args.config))
        return
    if args.command == "train-mc-jepa":
        print(train_mc_jepa(args.config))
        return
    if args.command == "benchmark-mc-jepa":
        print(json.dumps(run_benchmark(args.config), indent=2))
        return
    if args.command == "probe":
        print(train_decoder_probe(args.config))
        return
    if args.command == "eval":
        print(evaluate(args.config))
        return

    results = []
    for seed in range(args.runs):
        train_metrics = train(args.config)
        eval_metrics = evaluate(args.config)
        results.append({"run": seed, "train": train_metrics, "eval": eval_metrics})

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

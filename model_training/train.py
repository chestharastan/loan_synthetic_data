"""Command-line entry point for model training.

Usage
-----
    python -m model_training.train               # train on data/loan_applications.csv
    python -m model_training.train --data data/sample.csv
"""

from __future__ import annotations

import argparse
import logging

from common import load_config

from .trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train loan risk & approval models.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--data", default=None, help="Override input CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_config(args.config)
    if args.data is not None:
        config.training["input_path"] = args.data

    trainer = ModelTrainer(config)
    results = trainer.run()

    for result in results.values():
        print(result.report)

    print("\nModels saved to:", config.path(config.training["model_dir"]))
    print("Metrics saved to:", config.path(config.training["metrics_dir"]) / "metrics.json")


if __name__ == "__main__":
    main()

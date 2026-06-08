"""Command-line entry point for synthetic data generation.

Usage
-----
    python -m data_generation.run                 # use config.yaml defaults
    python -m data_generation.run --n 5000        # override sample count
    python -m data_generation.run --out data/sample.csv
"""

from __future__ import annotations

import argparse
import logging

from common import load_config

from .generator import LoanDataGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic loan-application data.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--n", type=int, default=None, help="Override number of samples")
    parser.add_argument("--out", default=None, help="Override output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_config(args.config)
    if args.n is not None:
        config.generation["n_samples"] = args.n

    generator = LoanDataGenerator(config)
    df = generator.generate()
    path = generator.save(df, args.out)

    print("\n" + generator.summarise(df))
    print(f"\nDataset written to: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Starter script for FL-RoBERTa transformer training."""
from __future__ import annotations

import argparse

from fl_roberta.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Starter FL-RoBERTa training entrypoint.")
    parser.add_argument("--data", required=True, help="Path to CSV with text/label columns")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--model", choices=["roberta", "fl-roberta", "bert"], default="fl-roberta")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    print("FL-RoBERTa starter scaffold")
    print(f"Model: {args.model}")
    print(f"Seed: {cfg.seed}")
    print(f"LR: {cfg.learning_rate}, batch: {cfg.batch_size}, epochs: {cfg.epochs}")
    print(f"Focal parameters alpha={cfg.focal_alpha}, gamma={cfg.focal_gamma}")
    print("Next step: plug this config into a HuggingFace Trainer loop using the focal loss module.")


if __name__ == "__main__":
    main()

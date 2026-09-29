"""
Master Command-Line Entrypoint for Part 9: Classical vs Hybrid Comparison.

Usage:
    python scripts/run_comparison.py --config configs/comparison.yaml
    python scripts/run_comparison.py --quick
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import PROJECT_ROOT
from src.comparison.config import ComparisonConfig
from src.comparison.runner import ComparisonRunner


def main():
    parser = argparse.ArgumentParser(description="Part 9: Classical vs Hybrid Scientific Comparison")
    parser.add_argument("--config", type=str, default="configs/comparison.yaml", help="Path to comparison YAML config")
    parser.add_argument("--quick", action="store_true", help="Run rapid verification mode with 1 seed and fewer epochs")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config

    if os.path.exists(config_path):
        print(f"[CONFIG] Loading comparison configuration from: {config_path}")
        cfg = ComparisonConfig.from_yaml(config_path)
    else:
        print(f"[CONFIG] Warning: {config_path} not found. Using default ComparisonConfig.")
        cfg = ComparisonConfig()

    if args.quick:
        cfg.quick_mode = True
        print("[CONFIG] Running in --quick mode.")

    runner = ComparisonRunner(cfg)
    summary = runner.run()

    print(f"\n[DONE] Comparative analysis completed with status: {summary['status']}")
    print(f"[DONE] Full report available at: {summary['report_path']}")


if __name__ == "__main__":
    main()

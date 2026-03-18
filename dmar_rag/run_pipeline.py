"""
Convenience launcher to start DMAR training or evaluation with one command.

Examples:
- Train all models then evaluate:
    python run_pipeline.py --mode all --config my_pipeline.json
- Only evaluate with existing weights:
    python run_pipeline.py --mode eval --config my_pipeline.json
"""

import argparse
import sys

import pipeline


def run(steps, config_path=None):
    """Call pipeline.main with constructed argv."""
    argv = ["--steps", ",".join(steps)]
    if config_path:
        argv += ["--config", config_path]
    prev = sys.argv
    try:
        sys.argv = ["pipeline"] + argv
        pipeline.main()
    finally:
        sys.argv = prev


def main():
    ap = argparse.ArgumentParser(description="Launch DMAR training/evaluation pipeline")
    ap.add_argument(
        "--mode",
        choices=["train", "eval", "all"],
        default="all",
        help="train: convert+train models; eval: convert+evaluate; all: full pipeline",
    )
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config; if omitted, built-in defaults are used")
    args = ap.parse_args()

    if args.mode == "train":
        steps = ["convert", "train_gating", "train_gnn", "train_cross", "train_joint"]
    elif args.mode == "eval":
        steps = ["convert", "eval"]
    else:
        steps = ["convert", "train_gating", "train_gnn", "train_cross", "train_joint", "eval"]

    run(steps, args.config)


if __name__ == "__main__":
    main()

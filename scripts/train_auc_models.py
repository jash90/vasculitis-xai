#!/usr/bin/env python3
"""
Train the full-feature AUC-first model comparison.

This script is separate from scripts/retrain_aligned_models.py. The aligned
script keeps the API's 20-feature patient form usable; this one optimizes the
offline AUC table on the richer clinical dataset and saves its own artifacts.
"""

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.auc_training import DEFAULT_N_FEATURES, train_auc_models


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full-feature AUC models.")
    parser.add_argument(
        "--data",
        default=PROJECT_ROOT / "data" / "raw" / "aktualne_dane.csv",
        type=Path,
        help="Path to the raw patient CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "models" / "saved" / "auc",
        type=Path,
        help="Directory for AUC-first artifacts.",
    )
    parser.add_argument(
        "--n-features",
        default=DEFAULT_N_FEATURES,
        type=int,
        help="Number of selected features.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model keys to train.",
    )
    parser.add_argument(
        "--skip-stacking",
        action="store_true",
        help="Skip the slower stacking model.",
    )
    parser.add_argument(
        "--random-state",
        default=42,
        type=int,
        help="Random seed for split, feature selection, and models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("AUC-first training")
    print(f"  data:       {args.data}")
    print(f"  output:     {args.output_dir}")
    print(f"  n_features: {args.n_features}")

    result = train_auc_models(
        args.data,
        args.output_dir,
        model_keys=args.models,
        n_features=args.n_features,
        random_state=args.random_state,
        include_stacking=not args.skip_stacking,
    )

    print("\nModel comparison:")
    print("Model                 AUC    AP     Accuracy Recall Specificity Precision F1")
    print("-" * 78)
    for row in result["comparison"]:
        print(
            f"{row['model']:<20} "
            f"{row['auc']:.3f}  "
            f"{row['ap']:.3f}  "
            f"{row['accuracy']:.3f}    "
            f"{row['recall']:.3f}  "
            f"{row['specificity']:.3f}       "
            f"{row['precision']:.3f}     "
            f"{row['f1']:.3f}"
        )

    best = result["metadata"]["best_model"]
    print(f"\nBest model: {best}")
    print(f"Artifacts saved in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

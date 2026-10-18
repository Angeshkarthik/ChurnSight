"""CLI script to run model training, cross-validation benchmarking, and artifact export."""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.trainer import run_model_benchmarking


def main():
    parser = argparse.ArgumentParser(
        description="Customer Intelligence Platform - Model Benchmarking CLI"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to cleaned dataset CSV (default: data/processed/telco_churn_cleaned.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save serialized model and metadata (default: models/)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size fraction (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable artifact serialization",
    )

    args = parser.parse_args()

    result = run_model_benchmarking(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.seed,
        export_artifacts=not args.no_export,
        verbose=True,
    )

    print("\nBenchmarking complete!")
    if result.artifact_path:
        print(f"Champion model artifact saved to: {result.artifact_path}")
        print(f"Model metadata saved to: {result.metadata_path}\n")


if __name__ == "__main__":
    main()

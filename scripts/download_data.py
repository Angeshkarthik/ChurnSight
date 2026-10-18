"""CLI script to download the raw Telco Customer Churn dataset."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so modules can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_telco_churn_dataset, DATASET_METADATA


def main():
    parser = argparse.ArgumentParser(
        description="Download raw Telco Customer Churn dataset from verified public repository."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom destination path for raw CSV file (default: data/raw/Telco-Customer-Churn.csv)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if target file already exists",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset provenance and licensing metadata and exit",
    )

    args = parser.parse_args()

    if args.info:
        print("\n--- Telco Customer Churn Dataset Provenance ---")
        for key, value in DATASET_METADATA.items():
            print(f"{key:<20}: {value}")
        print("------------------------------------------------\n")
        return

    dest = Path(args.output) if args.output else None
    print(f"Executing dataset download to {dest or 'data/raw/Telco-Customer-Churn.csv'}...")
    saved_path = download_telco_churn_dataset(target_path=dest, force_download=args.force)
    print(f"Dataset successfully saved to: {saved_path}")


if __name__ == "__main__":
    main()

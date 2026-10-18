"""Data Pipeline Orchestration Module.

Combines downloading, loading, validation, cleaning, and export into a reproducible,
single-command pipeline execution.

Pipeline Stages:
1. Ingestion: Ensures raw data exists at data/raw/Telco-Customer-Churn.csv.
2. Raw Validation: Audits schema conformance and reports initial data quality anomalies.
3. Transformation: Cleans whitespace, coerces numeric types, and documents decisions.
4. Clean Validation: Asserts that cleaned dataset conforms to strict post-cleaning criteria.
5. Export: Persists cleaned tabular data to data/processed/telco_churn_cleaned.csv.
"""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Optional, Union

import pandas as pd

# Add project root to sys.path if invoked directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_telco_churn_dataset
from src.data.loader import load_raw_data
from src.data.validator import validate_dataset, ValidationReport
from src.data.cleaner import clean_telco_churn_data, save_processed_data, CleaningReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for data pipeline execution outputs and audit summaries."""

    raw_shape: tuple
    cleaned_shape: tuple
    raw_validation_report: ValidationReport
    cleaning_report: CleaningReport
    cleaned_validation_report: ValidationReport
    output_path: Path
    cleaned_df: pd.DataFrame


def run_data_pipeline(
    raw_path: Optional[Union[str, Path]] = None,
    processed_path: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    cleaning_strategy: str = "impute_zero",
    verbose: bool = True,
) -> PipelineResult:
    """Execute the complete data pipeline from raw ingestion to cleaned export.

    Parameters
    ----------
    raw_path : Optional[Union[str, Path]], default None
        Path to raw CSV file.
    processed_path : Optional[Union[str, Path]], default None
        Path to save cleaned CSV file.
    force_download : bool, default False
        Force re-downloading the raw CSV even if it already exists.
    cleaning_strategy : str, default 'impute_zero'
        Strategy for TotalCharges blank entries ('impute_zero' or 'drop').
    verbose : bool, default True
        If True, prints formatted audit reports to stdout.

    Returns
    -------
    PipelineResult
        Encapsulates all reports, shapes, and the cleaned DataFrame.
    """
    logger.info(">>> Starting Customer Churn Data Pipeline Execution <<<")

    # Step 1: Ingestion / Download check
    if raw_path is None:
        raw_file = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
    else:
        raw_file = Path(raw_path)

    if not raw_file.exists() or force_download:
        logger.info("Raw file missing or force_download=True. Triggering download...")
        download_telco_churn_dataset(target_path=raw_file, force_download=force_download)

    # Step 2: Load Raw Dataset
    logger.info("Loading raw dataset from %s...", raw_file)
    raw_df = load_raw_data(file_path=raw_file)

    # Step 3: Raw Data Validation & Audit
    logger.info("Running raw data validation...")
    raw_validation = validate_dataset(raw_df, raise_on_error=True)

    if verbose:
        print("\n" + raw_validation.summary_text() + "\n")

    # Step 4: Data Cleaning & Normalization
    logger.info("Cleaning raw dataset using strategy: '%s'...", cleaning_strategy)
    cleaned_df, cleaning_report = clean_telco_churn_data(
        df=raw_df, total_charges_strategy=cleaning_strategy
    )

    # Step 5: Cleaned Data Validation
    logger.info("Validating cleaned dataset...")
    cleaned_validation = validate_dataset(cleaned_df, raise_on_error=True)

    # Step 6: Export Cleaned Dataset
    saved_path = save_processed_data(df=cleaned_df, destination_path=processed_path)
    cleaning_report.output_path = saved_path

    if verbose:
        print("\n" + cleaning_report.summary_text() + "\n")
        print("\n" + cleaned_validation.summary_text() + "\n")

    logger.info(">>> Data Pipeline Execution Completed Successfully! <<<")

    return PipelineResult(
        raw_shape=raw_df.shape,
        cleaned_shape=cleaned_df.shape,
        raw_validation_report=raw_validation,
        cleaning_report=cleaning_report,
        cleaned_validation_report=cleaned_validation,
        output_path=saved_path,
        cleaned_df=cleaned_df,
    )


def main():
    """Command-line entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Customer Intelligence Platform - Data Pipeline CLI"
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=None,
        help="Custom path to raw Telco Customer Churn CSV (default: data/raw/Telco-Customer-Churn.csv)",
    )
    parser.add_argument(
        "--processed-path",
        type=str,
        default=None,
        help="Custom path for cleaned output CSV (default: data/processed/telco_churn_cleaned.csv)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["impute_zero", "drop"],
        default="impute_zero",
        help="Strategy for blank TotalCharges entries (default: 'impute_zero')",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of raw CSV from source",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed textual summary reports",
    )

    args = parser.parse_args()

    result = run_data_pipeline(
        raw_path=args.raw_path,
        processed_path=args.processed_path,
        force_download=args.force_download,
        cleaning_strategy=args.strategy,
        verbose=not args.quiet,
    )

    print(f"\nFinal Summary: Cleaned data successfully written to: {result.output_path}")
    print(f"Original shape: {result.raw_shape} -> Cleaned shape: {result.cleaned_shape}\n")


if __name__ == "__main__":
    main()

"""Data ingestion, validation, and cleaning sub-package.

Exposes primary pipeline functions and reporting data structures.
"""

from src.data.download import download_telco_churn_dataset, DATASET_METADATA
from src.data.loader import load_raw_data, DataLoaderError
from src.data.validator import (
    validate_dataset,
    DataValidationError,
    ValidationReport,
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
)
from src.data.cleaner import (
    clean_telco_churn_data,
    save_processed_data,
    CleaningReport,
)
from src.data.pipeline import run_data_pipeline, PipelineResult

__all__ = [
    "download_telco_churn_dataset",
    "DATASET_METADATA",
    "load_raw_data",
    "DataLoaderError",
    "validate_dataset",
    "DataValidationError",
    "ValidationReport",
    "EXPECTED_COLUMNS",
    "TARGET_COLUMN",
    "ID_COLUMN",
    "clean_telco_churn_data",
    "save_processed_data",
    "CleaningReport",
    "run_data_pipeline",
    "PipelineResult",
]

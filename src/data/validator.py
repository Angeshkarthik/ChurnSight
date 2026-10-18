"""Data Validator Module.

Provides schema validation, data integrity audits, and diagnostic reporting
for both raw and processed datasets in the Customer Churn platform.

Design Philosophy:
- Fail-Fast: Detect schema drift, missing columns, or invalid targets immediately.
- Explicit Reporting: Generate structured validation reports instead of silent boolean checks.
- Auditability: Quantify missing values, whitespace anomalies, duplicates, and class balance.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Official schema specification for the Telco Customer Churn dataset
EXPECTED_COLUMNS: List[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"
VALID_TARGET_VALUES: Set[str] = {"Yes", "No"}


class DataValidationError(Exception):
    """Custom exception raised when critical data validation criteria fail."""
    pass


@dataclass
class ValidationReport:
    """Structured report encapsulating data health and schema audit metrics."""

    shape: Tuple[int, int]
    columns: List[str]
    column_types: Dict[str, str]
    missing_values: Dict[str, int]
    whitespace_counts: Dict[str, int]
    duplicate_rows_count: int
    duplicate_id_count: int
    target_distribution: Dict[str, int] = field(default_factory=dict)
    target_proportions: Dict[str, float] = field(default_factory=dict)
    validation_passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary_text(self) -> str:
        """Format the validation metrics into a human-readable diagnostic report."""
        lines = [
            "=" * 64,
            "DATA VALIDATION & INTEGRITY REPORT",
            "=" * 64,
            f"Dataset Shape          : {self.shape[0]} rows x {self.shape[1]} columns",
            f"Duplicate Total Rows   : {self.duplicate_rows_count}",
            f"Duplicate Customer IDs : {self.duplicate_id_count}",
            "-" * 64,
            "Target ('Churn') Class Distribution:",
        ]
        if self.target_distribution:
            for cls, count in self.target_distribution.items():
                prop = self.target_proportions.get(cls, 0.0) * 100
                lines.append(f"  - Class '{cls}': {count:,} samples ({prop:.2f}%)")
        else:
            lines.append("  - Target column not present or not evaluated.")

        lines.append("-" * 64)
        lines.append("Whitespace / Blank String Counts per Column:")
        whitespace_found = False
        for col, count in self.whitespace_counts.items():
            if count > 0:
                lines.append(f"  - {col:<22}: {count} blank/whitespace entries")
                whitespace_found = True
        if not whitespace_found:
            lines.append("  - None detected.")

        lines.append("-" * 64)
        lines.append("Missing Values (NaN/Null) per Column:")
        nulls_found = False
        for col, count in self.missing_values.items():
            if count > 0:
                lines.append(f"  - {col:<22}: {count} null entries")
                nulls_found = True
        if not nulls_found:
            lines.append("  - None detected.")

        if self.warnings:
            lines.append("-" * 64)
            lines.append("WARNINGS:")
            for warn in self.warnings:
                lines.append(f"  [!] {warn}")

        if self.errors:
            lines.append("-" * 64)
            lines.append("ERRORS:")
            for err in self.errors:
                lines.append(f"  [X] {err}")

        lines.append("=" * 64)
        lines.append(f"OVERALL STATUS: {'PASSED' if self.validation_passed else 'FAILED'}")
        lines.append("=" * 64)
        return "\n".join(lines)


def validate_dataset(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Audit a dataset for schema conformance, missing values, duplicates, and target distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.
    expected_columns : Optional[List[str]], default None
        List of required column names. If None, uses default EXPECTED_COLUMNS.
    target_column : str, default 'Churn'
        Name of the binary classification target column.
    id_column : str, default 'customerID'
        Name of the primary customer identifier column.
    raise_on_error : bool, default True
        If True, raises DataValidationError when critical validation checks fail.

    Returns
    -------
    ValidationReport
        Comprehensive audit report.

    Raises
    ------
    DataValidationError
        If raise_on_error is True and any critical check (missing required columns,
        missing target, invalid target values, or empty DataFrame) fails.
    """
    if expected_columns is None:
        expected_columns = EXPECTED_COLUMNS

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Check for empty dataframe
    if df.empty:
        errors.append("DataFrame is completely empty (0 rows).")

    # 2. Check for missing required columns
    current_columns = list(df.columns)
    missing_cols = [col for col in expected_columns if col not in current_columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 3. Target column validation
    target_dist: Dict[str, int] = {}
    target_props: Dict[str, float] = {}
    if target_column in df.columns:
        unique_targets = set(df[target_column].dropna().unique())
        invalid_targets = unique_targets - VALID_TARGET_VALUES
        if invalid_targets:
            errors.append(
                f"Target column '{target_column}' contains invalid classes: {invalid_targets}. "
                f"Expected subset of: {VALID_TARGET_VALUES}"
            )
        counts = df[target_column].value_counts(dropna=False).to_dict()
        props = df[target_column].value_counts(dropna=False, normalize=True).to_dict()
        target_dist = {str(k): int(v) for k, v in counts.items()}
        target_props = {str(k): float(v) for k, v in props.items()}
    else:
        errors.append(f"Target column '{target_column}' is absent from dataset.")

    # 4. Duplicate checks
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        warnings.append(f"Dataset contains {duplicate_rows} duplicate rows across all columns.")

    duplicate_ids = 0
    if id_column in df.columns:
        duplicate_ids = int(df[id_column].duplicated().sum())
        if duplicate_ids > 0:
            errors.append(f"Duplicate customer identifier '{id_column}' detected: {duplicate_ids} occurrences.")

    # 5. Missing values check (standard NaNs)
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}

    # 6. Whitespace / Blank string check across object columns
    whitespace_counts: Dict[str, int] = {}
    for col in df.columns:
        if df[col].dtype == "object":
            # Count string values that are empty or contain only whitespace characters
            blank_mask = df[col].astype(str).str.strip() == ""
            blank_count = int(blank_mask.sum())
            whitespace_counts[col] = blank_count
            if blank_count > 0:
                warnings.append(
                    f"Column '{col}' contains {blank_count} whitespace/blank string entries."
                )
        else:
            whitespace_counts[col] = 0

    # 7. Compile report
    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    passed = len(errors) == 0

    report = ValidationReport(
        shape=df.shape,
        columns=current_columns,
        column_types=column_types,
        missing_values=missing_values,
        whitespace_counts=whitespace_counts,
        duplicate_rows_count=duplicate_rows,
        duplicate_id_count=duplicate_ids,
        target_distribution=target_dist,
        target_proportions=target_props,
        validation_passed=passed,
        errors=errors,
        warnings=warnings,
    )

    if not passed and raise_on_error:
        error_msg = f"Data validation failed with {len(errors)} error(s):\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        logger.error(error_msg)
        raise DataValidationError(error_msg)

    return report

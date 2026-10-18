"""Data Cleaner Module.

Implements explicit, documented cleaning transformations for the Telco Customer Churn dataset.

Targeted Reliability Guarantees:
1. Validate Before Imputing `TotalCharges`:
   - For `strategy="impute_zero"`, strictly verifies that EVERY row with blank/whitespace
     `TotalCharges` has `tenure == 0`.
   - If any row with blank `TotalCharges` has positive tenure (> 0), missing tenure (NaN),
     or invalid tenure, raises a descriptive `DataCleaningError` instead of blindly assigning zero.
2. Preserve Genuine Missing Values:
   - When trimming leading/trailing whitespace across string columns, genuine nulls (None, np.nan, pd.NA)
     are preserved without ever being cast into literal strings like `"nan"` or `"None"`.
3. Strict Numeric Conversion:
   - Explicitly checks for unexpected values in `TotalCharges` (e.g. non-numeric corruption like "abc" or "$100").
   - Raises `DataCleaningError` with actionable customer information rather than silently coercing them to NaNs.
4. Granular Cleaning Audit Reporting:
   - Distinguishes entries imputed from entries dropped in `CleaningReport`.
   - Never drops rows or imputes values silently.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaningError(Exception):
    """Custom exception raised when data cleaning validation fails."""
    pass


@dataclass
class CleaningReport:
    """Detailed audit log capturing all transformations applied during cleaning."""

    initial_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    total_charges_strategy: str
    total_charges_blank_count: int
    entries_imputed: int = 0
    entries_dropped: int = 0
    affected_customer_ids: List[str] = field(default_factory=list)
    columns_trimmed: List[str] = field(default_factory=list)
    dtype_changes: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    unexpected_conversion_failures: int = 0
    output_path: Optional[Path] = None

    def summary_text(self) -> str:
        """Format cleaning actions into a clear audit log."""
        lines = [
            "=" * 64,
            "DATA CLEANING & TRANSFORMATION AUDIT REPORT",
            "=" * 64,
            f"Initial Dataset Shape       : {self.initial_shape[0]} rows x {self.initial_shape[1]} columns",
            f"Final Dataset Shape         : {self.final_shape[0]} rows x {self.final_shape[1]} columns",
            f"TotalCharges Strategy       : '{self.total_charges_strategy}'",
            f"TotalCharges Blanks Detected: {self.total_charges_blank_count}",
            f"  - Entries Imputed (to 0.0): {self.entries_imputed}",
            f"  - Entries Dropped         : {self.entries_dropped}",
        ]
        if self.affected_customer_ids:
            preview = ", ".join(self.affected_customer_ids[:5])
            if len(self.affected_customer_ids) > 5:
                preview += f" ... (+{len(self.affected_customer_ids) - 5} more)"
            lines.append(f"Affected Customer IDs ({len(self.affected_customer_ids)}): {preview}")

        lines.extend([
            "-" * 64,
            f"Columns Trimmed (Whitespace): {', '.join(self.columns_trimmed) if self.columns_trimmed else 'None'}",
            "-" * 64,
            "Data Type Casts Applied:",
        ])
        if self.dtype_changes:
            for col, (old_t, new_t) in self.dtype_changes.items():
                lines.append(f"  - {col:<20}: {old_t} -> {new_t}")
        else:
            lines.append("  - None")

        if self.unexpected_conversion_failures > 0:
            lines.append("-" * 64)
            lines.append(f"Conversion Failures         : {self.unexpected_conversion_failures}")

        if self.output_path:
            lines.append("-" * 64)
            lines.append(f"Cleaned Dataset Saved To    : {self.output_path}")

        lines.append("=" * 64)
        return "\n".join(lines)


def clean_telco_churn_data(
    df: pd.DataFrame,
    total_charges_strategy: str = "impute_zero",
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Clean the raw Telco Customer Churn dataset according to documented rules.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw DataFrame. The original DataFrame is not mutated.
    total_charges_strategy : str, default 'impute_zero'
        How to handle whitespace entries in TotalCharges:
        - 'impute_zero': Sets TotalCharges to 0.0 where tenure == 0 (domain-accurate).
        - 'drop': Drops rows where TotalCharges is blank.

    Returns
    -------
    Tuple[pd.DataFrame, CleaningReport]
        Cleaned DataFrame and the associated audit report.

    Raises
    ------
    ValueError
        If an unknown total_charges_strategy is passed.
    DataCleaningError
        If validation before imputation fails (e.g. positive or invalid tenure for blank TotalCharges)
        or if unexpected non-numeric values occur during type coercion.
    """
    if total_charges_strategy not in {"impute_zero", "drop"}:
        raise ValueError(
            f"Invalid total_charges_strategy: '{total_charges_strategy}'. "
            "Must be either 'impute_zero' or 'drop'."
        )

    # Defensive copy to guarantee immutability of the caller's DataFrame
    cleaned_df = df.copy()
    initial_shape = cleaned_df.shape
    dtype_changes: Dict[str, Tuple[str, str]] = {}
    columns_trimmed: List[str] = []

    # 1. Trim leading and trailing whitespace on string columns WITHOUT converting nulls to literal strings
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == "object":
            original_series = cleaned_df[col]
            # Strip whitespace ONLY if the element is an actual string, preserving None / np.nan
            trimmed_series = original_series.map(
                lambda val: val.strip() if isinstance(val, str) else val
            )

            # Check if any values were altered by trimming
            changed = False
            for old_val, new_val in zip(original_series, trimmed_series):
                if isinstance(old_val, str) and old_val != new_val:
                    changed = True
                    break

            if changed:
                columns_trimmed.append(col)
            cleaned_df[col] = trimmed_series

    # 2. Address TotalCharges validation, imputation, and numeric casting
    affected_ids: List[str] = []
    blank_tc_count = 0
    entries_imputed = 0
    entries_dropped = 0
    unexpected_conversion_failures = 0

    if "TotalCharges" in cleaned_df.columns:
        old_tc_dtype = str(cleaned_df["TotalCharges"].dtype)

        # Identify rows where TotalCharges is an empty or whitespace-only string
        is_string_tc = cleaned_df["TotalCharges"].apply(lambda x: isinstance(x, str))
        blank_mask = is_string_tc & (cleaned_df["TotalCharges"] == "")
        blank_tc_count = int(blank_mask.sum())

        if blank_tc_count > 0:
            if "customerID" in cleaned_df.columns:
                affected_ids = cleaned_df.loc[blank_mask, "customerID"].tolist()

            logger.info(
                "Found %d rows with blank 'TotalCharges'. Applying strategy: '%s'.",
                blank_tc_count,
                total_charges_strategy,
            )

            if total_charges_strategy == "impute_zero":
                # Strictest validation before imputing zero:
                # 1. 'tenure' column must exist
                if "tenure" not in cleaned_df.columns:
                    raise DataCleaningError(
                        "Cannot validate 'TotalCharges' imputation: 'tenure' column is missing from dataset."
                    )

                blank_tenures = cleaned_df.loc[blank_mask, "tenure"]

                # 2. Tenure values must not be missing (NaN)
                if blank_tenures.isna().any():
                    nan_indices = blank_tenures[blank_tenures.isna()].index.tolist()
                    raise DataCleaningError(
                        f"Validation failed before imputing 'TotalCharges': {len(nan_indices)} row(s) "
                        f"with blank 'TotalCharges' have missing (NaN) tenure at indices {nan_indices}."
                    )

                # 3. Tenure values must be numeric
                numeric_tenure = pd.to_numeric(blank_tenures, errors="coerce")
                if numeric_tenure.isna().any():
                    invalid_indices = blank_tenures[numeric_tenure.isna()].index.tolist()
                    raise DataCleaningError(
                        f"Validation failed before imputing 'TotalCharges': {len(invalid_indices)} row(s) "
                        f"with blank 'TotalCharges' have non-numeric tenure at indices {invalid_indices}."
                    )

                # 4. Tenure values must not be negative
                if (numeric_tenure < 0).any():
                    neg_indices = blank_tenures[numeric_tenure < 0].index.tolist()
                    raise DataCleaningError(
                        f"Validation failed before imputing 'TotalCharges': Found negative tenure at indices {neg_indices}."
                    )

                # 5. Tenure values MUST BE strictly 0 for zero imputation
                positive_tenure_mask = numeric_tenure > 0
                if positive_tenure_mask.any():
                    invalid_rows = cleaned_df.loc[blank_mask].loc[positive_tenure_mask]
                    invalid_descriptions = [
                        f"(CustomerID: {row.get('customerID', idx)}, tenure: {row['tenure']})"
                        for idx, row in invalid_rows.iterrows()
                    ]
                    raise DataCleaningError(
                        f"Validation failed before imputing 'TotalCharges': Found {len(invalid_rows)} row(s) "
                        f"with blank 'TotalCharges' but positive tenure: {', '.join(invalid_descriptions)}. "
                        "Zero imputation is only valid for newly joined customers with tenure == 0."
                    )

                # All checks passed: impute 0.0 safely
                cleaned_df.loc[blank_mask, "TotalCharges"] = 0.0
                entries_imputed = blank_tc_count

            elif total_charges_strategy == "drop":
                cleaned_df = cleaned_df.loc[~blank_mask].reset_index(drop=True)
                entries_dropped = blank_tc_count
                logger.info("Dropped %d rows with blank 'TotalCharges'.", blank_tc_count)

        # 3. Numeric conversion and strict validation against unexpected NaNs
        pre_existing_nans = int(cleaned_df["TotalCharges"].isna().sum())
        numeric_series = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")
        post_nans = int(numeric_series.isna().sum())

        unexpected_nans = post_nans - pre_existing_nans
        if unexpected_nans > 0:
            # Locate rows where conversion failed unexpectedly
            failed_mask = numeric_series.isna() & cleaned_df["TotalCharges"].notna()
            failed_rows = cleaned_df.loc[failed_mask]
            failed_samples = [
                f"(CustomerID: {row.get('customerID', idx)}, raw_value: {repr(row['TotalCharges'])})"
                for idx, row in failed_rows.head(5).iterrows()
            ]
            unexpected_conversion_failures = int(failed_mask.sum())
            raise DataCleaningError(
                f"Numeric conversion failed for 'TotalCharges': {unexpected_conversion_failures} value(s) "
                f"could not be parsed as float: {', '.join(failed_samples)}. "
                "Invalid values must be investigated rather than silently coerced to NaN."
            )

        cleaned_df["TotalCharges"] = numeric_series
        new_tc_dtype = str(cleaned_df["TotalCharges"].dtype)
        dtype_changes["TotalCharges"] = (old_tc_dtype, new_tc_dtype)

    report = CleaningReport(
        initial_shape=initial_shape,
        final_shape=cleaned_df.shape,
        total_charges_strategy=total_charges_strategy,
        total_charges_blank_count=blank_tc_count,
        entries_imputed=entries_imputed,
        entries_dropped=entries_dropped,
        affected_customer_ids=affected_ids,
        columns_trimmed=columns_trimmed,
        dtype_changes=dtype_changes,
        unexpected_conversion_failures=unexpected_conversion_failures,
    )

    return cleaned_df, report


def save_processed_data(
    df: pd.DataFrame,
    destination_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save cleaned DataFrame to the destination path.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    destination_path : Optional[Union[str, Path]], default None
        Destination path. Defaults to `data/processed/telco_churn_cleaned.csv`
        resolved relative to the repository root.

    Returns
    -------
    Path
        Resolved path where the processed CSV was saved.
    """
    if destination_path is None:
        project_root = Path(__file__).resolve().parents[2]
        dest_path = project_root / "data" / "processed" / "telco_churn_cleaned.csv"
    else:
        dest_path = Path(destination_path).resolve()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_path, index=False)
    logger.info("Saved processed dataset to: %s (%d rows x %d cols)", dest_path, df.shape[0], df.shape[1])
    return dest_path

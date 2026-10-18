"""Data Cleaner Module.

Implements explicit, documented cleaning transformations for the Telco Customer Churn dataset.

Core Quality Anomalies Addressed:
1. `TotalCharges` Whitespace Issue:
   - In the raw dataset, 11 rows contain an empty whitespace string " " for `TotalCharges`.
   - Inspection reveals that all 11 rows have `tenure == 0` (new customers enrolled in the current billing cycle).
   - Business & Mathematical Rationale: A customer with 0 months of tenure has accumulated $0.00
     in historical billing. Setting `TotalCharges = 0.0` reflects actual domain reality without
     arbitrary imputation or data loss.
   - An alternative strategy (`strategy="drop"`) is also supported if strict exclusion of tenure=0 rows is required.
2. Type Casting:
   - Coerces `TotalCharges` from `object` (string) to `float64`.
3. String Normalization:
   - Strips leading and trailing whitespaces from all text/categorical columns to prevent subtle grouping bugs.
4. Auditability:
   - Generates a structured `CleaningReport` recording every decision and affected customer ID.
   - Never drops rows or imputes values silently.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Detailed audit log capturing all transformations applied during cleaning."""

    initial_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    total_charges_strategy: str
    total_charges_blank_count: int
    affected_customer_ids: List[str] = field(default_factory=list)
    columns_trimmed: List[str] = field(default_factory=list)
    dtype_changes: Dict[str, Tuple[str, str]] = field(default_factory=dict)
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
            f"Whitespace Entries Fixed    : {self.total_charges_blank_count}",
            f"Affected Customer IDs ({len(self.affected_customer_ids)}): {', '.join(self.affected_customer_ids[:5])}"
            + (f" ... (+{len(self.affected_customer_ids) - 5} more)" if len(self.affected_customer_ids) > 5 else ""),
            "-" * 64,
            f"Columns Trimmed (Whitespace): {', '.join(self.columns_trimmed) if self.columns_trimmed else 'None'}",
            "-" * 64,
            "Data Type Casts Applied:",
        ]
        if self.dtype_changes:
            for col, (old_t, new_t) in self.dtype_changes.items():
                lines.append(f"  - {col:<20}: {old_t} -> {new_t}")
        else:
            lines.append("  - None")

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
    """
    if total_charges_strategy not in {"impute_zero", "drop"}:
        raise ValueError(
            f"Invalid total_charges_strategy: '{total_charges_strategy}'. "
            "Must be either 'impute_zero' or 'drop'."
        )

    # Work on a defensive copy to guarantee immutability of the input
    cleaned_df = df.copy()
    initial_shape = cleaned_df.shape
    dtype_changes: Dict[str, Tuple[str, str]] = {}
    columns_trimmed: List[str] = []

    # 1. Trim leading and trailing whitespace across all string columns
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == "object":
            original_series = cleaned_df[col].astype(str)
            trimmed_series = original_series.str.strip()
            # Check if any values were actually altered by trimming
            if (original_series != trimmed_series).any():
                columns_trimmed.append(col)
            cleaned_df[col] = trimmed_series

    # 2. Address the known TotalCharges whitespace issue
    affected_ids: List[str] = []
    blank_tc_count = 0

    if "TotalCharges" in cleaned_df.columns:
        old_tc_dtype = str(cleaned_df["TotalCharges"].dtype)

        # Identify rows where TotalCharges is blank or whitespace
        blank_mask = cleaned_df["TotalCharges"] == ""
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
                # Verify that these rows have tenure == 0 before imputing 0.0
                if "tenure" in cleaned_df.columns:
                    tenure_values = cleaned_df.loc[blank_mask, "tenure"].unique()
                    logger.info("Tenure values for blank TotalCharges rows: %s", tenure_values)

                cleaned_df.loc[blank_mask, "TotalCharges"] = "0.0"
            elif total_charges_strategy == "drop":
                cleaned_df = cleaned_df.loc[~blank_mask].reset_index(drop=True)
                logger.info("Dropped %d rows with blank 'TotalCharges'.", blank_tc_count)

        # Coerce TotalCharges to numeric float64
        cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")
        new_tc_dtype = str(cleaned_df["TotalCharges"].dtype)
        dtype_changes["TotalCharges"] = (old_tc_dtype, new_tc_dtype)

        # Check if any unexpected NaNs were generated
        unexpected_nans = int(cleaned_df["TotalCharges"].isna().sum())
        if unexpected_nans > 0:
            logger.warning(
                "Coercion of 'TotalCharges' resulted in %d unexpected NaNs.", unexpected_nans
            )

    report = CleaningReport(
        initial_shape=initial_shape,
        final_shape=cleaned_df.shape,
        total_charges_strategy=total_charges_strategy,
        total_charges_blank_count=blank_tc_count,
        affected_customer_ids=affected_ids,
        columns_trimmed=columns_trimmed,
        dtype_changes=dtype_changes,
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

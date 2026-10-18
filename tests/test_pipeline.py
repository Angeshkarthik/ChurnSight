"""Integration tests for the complete Data Pipeline."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.pipeline import run_data_pipeline, PipelineResult


def test_run_data_pipeline_end_to_end(tmp_path):
    """Test full execution of the data pipeline writing to a custom temporary destination."""
    custom_processed = tmp_path / "custom_cleaned.csv"

    result = run_data_pipeline(
        processed_path=custom_processed,
        cleaning_strategy="impute_zero",
        verbose=False,
    )

    assert isinstance(result, PipelineResult)
    assert result.raw_shape == (7043, 21)
    assert result.cleaned_shape == (7043, 21)
    assert result.raw_validation_report.validation_passed is True
    assert result.cleaning_report.total_charges_blank_count == 11
    assert result.cleaned_validation_report.validation_passed is True
    assert result.output_path == custom_processed
    assert custom_processed.exists()

    # Verify saved CSV properties
    saved_df = pd.read_csv(custom_processed)
    assert saved_df.shape == (7043, 21)
    assert saved_df["TotalCharges"].dtype == "float64"
    assert saved_df["TotalCharges"].isna().sum() == 0

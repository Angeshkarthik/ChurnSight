"""Unit tests for the Data Cleaner module."""

import pytest
import pandas as pd
from pathlib import Path

from src.data.cleaner import clean_telco_churn_data, save_processed_data, CleaningReport


@pytest.fixture
def dirty_sample_df():
    """Create a sample DataFrame with known quality issues."""
    return pd.DataFrame({
        "customerID": ["CUST-01", "CUST-02", "CUST-03", "CUST-04"],
        "gender": [" Male ", "Female", "Male", "Female "],
        "tenure": [0, 10, 0, 5],
        "MonthlyCharges": [25.0, 50.0, 30.0, 45.0],
        "TotalCharges": [" ", "500.0", "", "225.0"],  # Rows 0 and 2 have whitespace
        "Contract": [" Month-to-month ", "One year", "Two year", "Month-to-month"],
        "Churn": ["No", "No", "No", "Yes"],
    })


def test_clean_impute_zero_strategy(dirty_sample_df):
    """Verify whitespace in TotalCharges is imputed with 0.0 and cast to float64."""
    cleaned_df, report = clean_telco_churn_data(
        dirty_sample_df, total_charges_strategy="impute_zero"
    )

    assert isinstance(report, CleaningReport)
    assert report.total_charges_blank_count == 2
    assert report.affected_customer_ids == ["CUST-01", "CUST-03"]
    assert report.final_shape == (4, 7)

    # Check that values were converted to float 0.0
    assert cleaned_df.loc[0, "TotalCharges"] == 0.0
    assert cleaned_df.loc[2, "TotalCharges"] == 0.0
    assert cleaned_df["TotalCharges"].dtype == "float64"

    # Check whitespace stripping
    assert cleaned_df.loc[0, "gender"] == "Male"
    assert cleaned_df.loc[3, "gender"] == "Female"
    assert cleaned_df.loc[0, "Contract"] == "Month-to-month"


def test_clean_drop_strategy(dirty_sample_df):
    """Verify drop strategy removes rows with blank TotalCharges."""
    cleaned_df, report = clean_telco_churn_data(
        dirty_sample_df, total_charges_strategy="drop"
    )

    assert report.total_charges_blank_count == 2
    assert report.final_shape == (2, 7)
    assert "CUST-01" not in cleaned_df["customerID"].values
    assert "CUST-03" not in cleaned_df["customerID"].values


def test_clean_invalid_strategy(dirty_sample_df):
    """Verify ValueError is raised if an invalid strategy is passed."""
    with pytest.raises(ValueError, match="Invalid total_charges_strategy"):
        clean_telco_churn_data(dirty_sample_df, total_charges_strategy="invalid_mode")


def test_save_processed_data(tmp_path, dirty_sample_df):
    """Verify cleaned data is successfully saved and can be read back."""
    target_csv = tmp_path / "processed_test.csv"
    saved_path = save_processed_data(dirty_sample_df, destination_path=target_csv)

    assert saved_path.exists()
    reloaded_df = pd.read_csv(saved_path)
    assert reloaded_df.shape == dirty_sample_df.shape

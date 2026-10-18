"""Unit tests for the Data Cleaner module.

Includes regression tests for pre-imputation validation, null value preservation,
strict numeric conversions, and granular audit reporting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import (
    clean_telco_churn_data,
    save_processed_data,
    CleaningReport,
    DataCleaningError,
)


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
    assert report.entries_imputed == 2
    assert report.entries_dropped == 0
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
    """Verify drop strategy removes rows with blank TotalCharges and tracks counts."""
    cleaned_df, report = clean_telco_churn_data(
        dirty_sample_df, total_charges_strategy="drop"
    )

    assert report.total_charges_blank_count == 2
    assert report.entries_imputed == 0
    assert report.entries_dropped == 2
    assert report.final_shape == (2, 7)
    assert "CUST-01" not in cleaned_df["customerID"].values
    assert "CUST-03" not in cleaned_df["customerID"].values


def test_clean_invalid_strategy(dirty_sample_df):
    """Verify ValueError is raised if an invalid strategy is passed."""
    with pytest.raises(ValueError, match="Invalid total_charges_strategy"):
        clean_telco_churn_data(dirty_sample_df, total_charges_strategy="invalid_mode")


def test_preserve_genuine_nulls_during_string_trimming():
    """Verify genuine pandas nulls (None, np.nan) remain null and are not turned into 'nan' strings."""
    df_with_nulls = pd.DataFrame({
        "customerID": ["CUST-01", "CUST-02", "CUST-03", "CUST-04"],
        "gender": [None, np.nan, "  Male  ", "Female "],
        "tenure": [12, 24, 0, 5],
        "MonthlyCharges": [25.0, 50.0, 30.0, 45.0],
        "TotalCharges": ["300.0", "1200.0", " ", "225.0"],
        "Churn": ["No", "No", "No", "Yes"],
    })

    cleaned_df, report = clean_telco_churn_data(df_with_nulls, total_charges_strategy="impute_zero")

    # Critical assertions: row 0 and 1 must remain true nulls
    assert pd.isna(cleaned_df.loc[0, "gender"])
    assert pd.isna(cleaned_df.loc[1, "gender"])
    assert cleaned_df.loc[0, "gender"] != "None"
    assert cleaned_df.loc[1, "gender"] != "nan"
    assert cleaned_df.loc[2, "gender"] == "Male"
    assert cleaned_df.loc[3, "gender"] == "Female"


def test_blank_total_charges_with_positive_tenure_raises_error():
    """Verify DataCleaningError is raised if blank TotalCharges row has positive tenure."""
    corrupted_df = pd.DataFrame({
        "customerID": ["CUST-99"],
        "tenure": [12],  # Positive tenure with blank TotalCharges!
        "MonthlyCharges": [50.0],
        "TotalCharges": [" "],
        "Churn": ["No"],
    })

    with pytest.raises(DataCleaningError, match="positive tenure"):
        clean_telco_churn_data(corrupted_df, total_charges_strategy="impute_zero")


def test_blank_total_charges_with_missing_tenure_raises_error():
    """Verify DataCleaningError is raised if blank TotalCharges row has NaN tenure."""
    corrupted_df = pd.DataFrame({
        "customerID": ["CUST-99"],
        "tenure": [np.nan],  # Missing tenure
        "MonthlyCharges": [50.0],
        "TotalCharges": [" "],
        "Churn": ["No"],
    })

    with pytest.raises(DataCleaningError, match="missing \\(NaN\\) tenure"):
        clean_telco_churn_data(corrupted_df, total_charges_strategy="impute_zero")


def test_blank_total_charges_with_invalid_tenure_raises_error():
    """Verify DataCleaningError is raised if blank TotalCharges row has non-numeric tenure."""
    corrupted_df = pd.DataFrame({
        "customerID": ["CUST-99"],
        "tenure": ["invalid_tenure"],
        "MonthlyCharges": [50.0],
        "TotalCharges": [" "],
        "Churn": ["No"],
    })

    with pytest.raises(DataCleaningError, match="non-numeric tenure"):
        clean_telco_churn_data(corrupted_df, total_charges_strategy="impute_zero")


def test_blank_total_charges_without_tenure_column_raises_error():
    """Verify DataCleaningError is raised if tenure column is completely missing."""
    corrupted_df = pd.DataFrame({
        "customerID": ["CUST-99"],
        "MonthlyCharges": [50.0],
        "TotalCharges": [" "],
        "Churn": ["No"],
    })

    with pytest.raises(DataCleaningError, match="'tenure' column is missing"):
        clean_telco_churn_data(corrupted_df, total_charges_strategy="impute_zero")


def test_unexpected_invalid_numeric_values_raises_error():
    """Verify unexpected non-numeric corrupted strings in TotalCharges raise DataCleaningError."""
    corrupted_df = pd.DataFrame({
        "customerID": ["CUST-ERR"],
        "tenure": [5],
        "MonthlyCharges": [50.0],
        "TotalCharges": ["$125.50_corrupted"],  # Invalid numeric value
        "Churn": ["No"],
    })

    with pytest.raises(DataCleaningError, match="Numeric conversion failed for 'TotalCharges'"):
        clean_telco_churn_data(corrupted_df, total_charges_strategy="impute_zero")


def test_save_processed_data(tmp_path, dirty_sample_df):
    """Verify cleaned data is successfully saved and can be read back."""
    target_csv = tmp_path / "processed_test.csv"
    saved_path = save_processed_data(dirty_sample_df, destination_path=target_csv)

    assert saved_path.exists()
    reloaded_df = pd.read_csv(saved_path)
    assert reloaded_df.shape == dirty_sample_df.shape

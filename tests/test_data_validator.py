"""Unit tests for the Data Validator module."""

import pytest
import pandas as pd

from src.data.validator import (
    validate_dataset,
    DataValidationError,
    EXPECTED_COLUMNS,
    ValidationReport,
)


@pytest.fixture
def sample_valid_df():
    """Create a minimal valid DataFrame that matches the schema."""
    data = {col: ["No"] * 4 for col in EXPECTED_COLUMNS}
    data["customerID"] = ["ID-01", "ID-02", "ID-03", "ID-04"]
    data["SeniorCitizen"] = [0, 1, 0, 0]
    data["tenure"] = [12, 24, 0, 5]
    data["MonthlyCharges"] = [29.85, 56.95, 20.0, 70.35]
    data["TotalCharges"] = [350.0, 1360.0, 0.0, 350.0]
    data["Churn"] = ["No", "Yes", "No", "Yes"]
    return pd.DataFrame(data)


def test_validate_valid_dataset(sample_valid_df):
    """Verify validation passes on a compliant DataFrame."""
    report = validate_dataset(sample_valid_df, raise_on_error=True)
    assert isinstance(report, ValidationReport)
    assert report.validation_passed is True
    assert report.shape == (4, len(EXPECTED_COLUMNS))
    assert report.target_distribution == {"No": 2, "Yes": 2}
    assert report.target_proportions == {"No": 0.5, "Yes": 0.5}
    assert len(report.errors) == 0


def test_validate_missing_required_column(sample_valid_df):
    """Verify DataValidationError is raised when a required column is missing."""
    corrupted_df = sample_valid_df.drop(columns=["MonthlyCharges"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataset(corrupted_df, raise_on_error=True)


def test_validate_missing_target_column(sample_valid_df):
    """Verify DataValidationError is raised when the target 'Churn' column is missing."""
    no_target_df = sample_valid_df.drop(columns=["Churn"])
    with pytest.raises(DataValidationError, match="Target column 'Churn' is absent"):
        validate_dataset(no_target_df, raise_on_error=True)


def test_validate_invalid_target_values(sample_valid_df):
    """Verify DataValidationError is raised when target contains unknown categories."""
    invalid_target_df = sample_valid_df.copy()
    invalid_target_df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataValidationError, match="contains invalid classes"):
        validate_dataset(invalid_target_df, raise_on_error=True)


def test_validate_duplicate_customer_ids(sample_valid_df):
    """Verify DataValidationError is raised when duplicate customer IDs exist."""
    dup_df = sample_valid_df.copy()
    dup_df.loc[1, "customerID"] = dup_df.loc[0, "customerID"]
    with pytest.raises(DataValidationError, match="Duplicate customer identifier"):
        validate_dataset(dup_df, raise_on_error=True)


def test_validate_detects_whitespace(sample_valid_df):
    """Verify whitespace values in columns are properly detected and counted."""
    ws_df = sample_valid_df.copy()
    ws_df["TotalCharges"] = ws_df["TotalCharges"].astype(object)
    ws_df.loc[0, "TotalCharges"] = "   "
    report = validate_dataset(ws_df, raise_on_error=False)
    assert report.whitespace_counts["TotalCharges"] == 1
    assert any("whitespace/blank" in warn for warn in report.warnings)


def test_validate_empty_dataframe():
    """Verify validation fails when given an empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(DataValidationError, match="empty"):
        validate_dataset(empty_df, raise_on_error=True)

"""Unit tests for the Data Loader module."""

import pytest
from pathlib import Path
import pandas as pd

from src.data.loader import load_raw_data, DataLoaderError


def test_load_raw_data_success():
    """Verify loading the actual raw dataset returns a valid 21-column DataFrame."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[1] == 21
    assert "customerID" in df.columns
    assert "Churn" in df.columns


def test_load_raw_data_file_not_found(tmp_path):
    """Verify FileNotFoundError is raised when target CSV does not exist."""
    fake_path = tmp_path / "non_existent_churn_dataset.csv"
    with pytest.raises(FileNotFoundError, match="Raw dataset file not found"):
        load_raw_data(file_path=fake_path)


def test_load_raw_data_empty_file(tmp_path):
    """Verify ValueError is raised when CSV file has 0 bytes."""
    empty_file = tmp_path / "empty_file.csv"
    empty_file.touch()  # 0 bytes
    with pytest.raises(ValueError, match="empty"):
        load_raw_data(file_path=empty_file)

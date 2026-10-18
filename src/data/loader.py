"""Data Loader Module.

Responsible for safely loading the raw Telco Customer Churn dataset into a
pandas DataFrame, verifying file existence, non-emptiness, and handling file paths.

Design Philosophy:
- Pathlib over raw strings: Ensures seamless cross-platform path handling (Windows vs Unix).
- Safe file I/O: Validates that files exist and are non-empty before invoking Pandas parser.
- Immutability: Loads data into a DataFrame copy without mutating or modifying the underlying file.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Custom exception raised when data loading fails."""
    pass


def load_raw_data(
    file_path: Optional[Union[str, Path]] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : Optional[Union[str, Path]], default None
        Path to the raw CSV file. If None, defaults to `data/raw/Telco-Customer-Churn.csv`
        resolved relative to the repository root.
    encoding : str, default 'utf-8'
        Character encoding of the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame untouched.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the resolved path.
    ValueError
        If the file is empty or cannot be parsed as a tabular CSV.
    DataLoaderError
        If an unexpected error occurs during data loading.
    """
    if file_path is None:
        project_root = Path(__file__).resolve().parents[2]
        resolved_path = project_root / "data" / "raw" / "Telco-Customer-Churn.csv"
    else:
        resolved_path = Path(file_path).resolve()

    logger.info("Attempting to load raw dataset from: %s", resolved_path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at: {resolved_path}. "
            "Please run the download script (scripts/download_data.py) first."
        )

    if not resolved_path.is_file():
        raise DataLoaderError(f"Provided path is not a valid file: {resolved_path}")

    if resolved_path.stat().st_size == 0:
        raise ValueError(f"Raw dataset file is empty (0 bytes): {resolved_path}")

    try:
        # Load raw CSV without automatic NA coercion so validator can inspect raw whitespace
        df = pd.read_csv(resolved_path, encoding=encoding, keep_default_na=False)
        logger.info(
            "Successfully loaded raw dataset. Shape: %d rows x %d columns.",
            df.shape[0],
            df.shape[1],
        )
        return df
    except pd.errors.EmptyDataError as err:
        raise ValueError(f"The file contains no readable data: {resolved_path}") from err
    except Exception as err:
        raise DataLoaderError(f"Unexpected error loading dataset from {resolved_path}: {err}") from err

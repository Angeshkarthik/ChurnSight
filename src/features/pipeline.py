"""Feature Engineering and Preprocessing Pipeline Module.

Constructs reusable, production-ready feature preprocessing transformers
following Scikit-learn pipeline standards.

Data Leakage Prevention Guarantees:
1. Identifier Isolation: `customerID` is stripped before feature transformations to prevent
   overfitting and arbitrary identifier memorization.
2. Target Isolation: The target label `Churn` is never passed into the feature matrix X.
3. Strict Train-Set Fitting: Feature transformers (`StandardScaler`, `OneHotEncoder`) are fit
   strictly on training data (`X_train`) and only applied (`transform`) to test data (`X_test`)
   or unseen production inference requests.
4. Production-Resilient Encoding: `OneHotEncoder(handle_unknown='ignore')` gracefully ignores
   unexpected categorical values during real-time API requests instead of crashing with a ValueError.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.config import (
    ALL_PREDICTOR_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_column: Optional[str] = ID_COLUMN,
    drop_id: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target vector y from predictor matrix X and drop arbitrary identifier.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer churn DataFrame.
    target_column : str, default 'Churn'
        Name of the target column to extract.
    id_column : Optional[str], default 'customerID'
        Name of the primary key / identifier column.
    drop_id : bool, default True
        If True, drops the identifier column from X to prevent data leakage and memorization.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target series y.

    Raises
    ------
    KeyError
        If target_column is not present in df.
    """
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    if drop_id and id_column and id_column in X.columns:
        logger.info("Dropping identifier column '%s' from feature matrix X.", id_column)
        X = X.drop(columns=[id_column])

    return X, y


def encode_target(y: pd.Series) -> pd.Series:
    """Encode binary target values into standard numeric format (0 and 1).

    Parameters
    ----------
    y : pd.Series
        Target series containing 'Yes'/'No' or already numeric 1/0.

    Returns
    -------
    pd.Series
        Binary integer series where 'Yes' -> 1 and 'No' -> 0.

    Raises
    ------
    ValueError
        If y contains unrecognized target categories.
    """
    if pd.api.types.is_numeric_dtype(y):
        unique_vals = set(y.dropna().unique())
        if unique_vals.issubset({0, 1}):
            return y.astype(int)

    # Standardize string representation
    y_str = y.astype(str).str.strip()
    valid_classes = {"Yes", "No"}
    found_classes = set(y_str.unique())

    if not found_classes.issubset(valid_classes):
        invalid = found_classes - valid_classes
        raise ValueError(
            f"Target contains invalid classes: {invalid}. Expected subset of {valid_classes}."
        )

    mapping = {"No": 0, "Yes": 1}
    y_encoded = y_str.map(mapping).astype(int)
    return y_encoded


def build_feature_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Construct a Scikit-learn ColumnTransformer for numerical scaling and categorical encoding.

    Parameters
    ----------
    numerical_features : Optional[List[str]], default None
        List of continuous feature names to scale. If None, uses NUMERICAL_FEATURES.
    categorical_features : Optional[List[str]], default None
        List of categorical feature names to one-hot encode. If None, uses CATEGORICAL_FEATURES.

    Returns
    -------
    ColumnTransformer
        Configured, unfitted Scikit-learn ColumnTransformer.
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    # Numerical Transformer: Zero mean and unit variance standard scaling
    num_transformer = StandardScaler()

    # Categorical Transformer: One-hot encoding with ignore mode for unseen future categories
    cat_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",  # Drop any unlisted columns defensively
        verbose_feature_names_out=False,
    )

    return preprocessor


def create_train_test_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into stratified training and testing subsets.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor feature matrix.
    y : pd.Series
        Encoded target vector.
    test_size : float, default 0.2
        Fraction of samples allocated to the test split.
    random_state : int, default 42
        Seed ensuring deterministic reproducibility.
    stratify : bool, default True
        If True, preserves class balance distribution between train and test splits.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    logger.info(
        "Created train/test split: Train shape=%s, Test shape=%s, Test size=%.2f",
        X_train.shape,
        X_test.shape,
        test_size,
    )
    return X_train, X_test, y_train, y_test


def fit_and_transform_features(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Fit preprocessor strictly on X_train, and transform X_train and optional X_test.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Unfitted ColumnTransformer instance.
    X_train : pd.DataFrame
        Training feature matrix used to learn scaling parameters and category mappings.
    X_test : Optional[pd.DataFrame], default None
        Testing feature matrix. Transformed using parameters learned strictly from X_train.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray], List[str]]
        Transformed X_train array, transformed X_test array (or None), and list of feature names.
    """
    # 1. Fit and transform strictly on X_train (prevents data leakage)
    X_train_transformed = preprocessor.fit_transform(X_train)

    # 2. Extract clean output feature names
    feature_names = preprocessor.get_feature_names_out().tolist()

    # 3. Transform X_test if provided (NEVER call fit on test data!)
    X_test_transformed = None
    if X_test is not None:
        X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_test_transformed, feature_names

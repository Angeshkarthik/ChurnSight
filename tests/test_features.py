"""Unit tests for the Feature Engineering module.

Tests target separation, identifier exclusion, target encoding,
leak-free preprocessing, unseen category handling, and stratified splitting.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.features.config import (
    ALL_PREDICTOR_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.features.pipeline import (
    build_feature_preprocessor,
    create_train_test_splits,
    encode_target,
    fit_and_transform_features,
    split_features_and_target,
)


@pytest.fixture
def sample_churn_df():
    """Create a synthetic DataFrame matching the Telco Churn cleaned schema."""
    n_samples = 20
    np.random.seed(42)
    return pd.DataFrame({
        "customerID": [f"ID-{i:03d}" for i in range(n_samples)],
        "gender": ["Male", "Female"] * (n_samples // 2),
        "SeniorCitizen": [0, 1] * (n_samples // 2),
        "Partner": ["Yes", "No"] * (n_samples // 2),
        "Dependents": ["No", "Yes"] * (n_samples // 2),
        "tenure": np.random.randint(0, 72, size=n_samples),
        "PhoneService": ["Yes"] * n_samples,
        "MultipleLines": ["No", "Yes"] * (n_samples // 2),
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"] * (n_samples // 4),
        "OnlineSecurity": ["Yes", "No"] * (n_samples // 2),
        "OnlineBackup": ["No", "Yes"] * (n_samples // 2),
        "DeviceProtection": ["Yes", "No"] * (n_samples // 2),
        "TechSupport": ["No", "Yes"] * (n_samples // 2),
        "StreamingTV": ["No", "Yes"] * (n_samples // 2),
        "StreamingMovies": ["Yes", "No"] * (n_samples // 2),
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"] * (n_samples // 4),
        "PaperlessBilling": ["Yes", "No"] * (n_samples // 2),
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] * (n_samples // 4),
        "MonthlyCharges": np.random.uniform(20.0, 115.0, size=n_samples),
        "TotalCharges": np.random.uniform(20.0, 5000.0, size=n_samples),
        "Churn": ["No", "No", "No", "Yes"] * (n_samples // 4),  # 25% churn
    })


def test_split_features_and_target_drops_id(sample_churn_df):
    """Verify customerID is excluded and target is separated."""
    X, y = split_features_and_target(sample_churn_df, drop_id=True)

    assert ID_COLUMN not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert len(X.columns) == 19  # 21 original - target - customerID
    assert len(y) == len(sample_churn_df)
    assert y.name == TARGET_COLUMN


def test_split_features_missing_target(sample_churn_df):
    """Verify KeyError is raised when target column is missing."""
    corrupted_df = sample_churn_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(KeyError, match="Target column 'Churn' not found"):
        split_features_and_target(corrupted_df)


def test_encode_target_yes_no():
    """Verify target series encodes 'Yes' to 1 and 'No' to 0."""
    raw_y = pd.Series(["No", "Yes", "No", "Yes", "No"])
    encoded_y = encode_target(raw_y)

    assert encoded_y.tolist() == [0, 1, 0, 1, 0]
    assert np.issubdtype(encoded_y.dtype, np.integer)


def test_encode_target_invalid_classes():
    """Verify ValueError is raised if target contains unrecognized classes."""
    invalid_y = pd.Series(["No", "Yes", "Maybe"])
    with pytest.raises(ValueError, match="invalid classes"):
        encode_target(invalid_y)


def test_build_feature_preprocessor(sample_churn_df):
    """Verify ColumnTransformer handles numerical scaling and categorical encoding."""
    X, _ = split_features_and_target(sample_churn_df)
    preprocessor = build_feature_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)

    # Fit and transform
    X_transformed, _, feature_names = fit_and_transform_features(preprocessor, X)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(sample_churn_df)
    # Output columns should exceed input columns due to one-hot expansion
    assert X_transformed.shape[1] > len(X.columns)
    assert len(feature_names) == X_transformed.shape[1]


def test_leak_free_transformation(sample_churn_df):
    """Verify test data is transformed strictly using parameters learned from training data."""
    X, y = split_features_and_target(sample_churn_df)
    y_encoded = encode_target(y)

    X_train, X_test, _, _ = create_train_test_splits(
        X, y_encoded, test_size=0.25, random_state=42, stratify=False
    )

    preprocessor = build_feature_preprocessor()
    X_train_trans, X_test_trans, _ = fit_and_transform_features(preprocessor, X_train, X_test)

    # Scaler mean must match X_train mean, NOT X_test mean
    scaler = preprocessor.named_transformers_["num"]
    expected_means = X_train[NUMERICAL_FEATURES].mean().values
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)
    assert X_test_trans is not None
    assert X_test_trans.shape[0] == len(X_test)


def test_handle_unseen_categories_gracefully(sample_churn_df):
    """Verify OneHotEncoder handle_unknown='ignore' does not crash on unseen production categories."""
    X, y = split_features_and_target(sample_churn_df)
    y_encoded = encode_target(y)

    X_train, X_test, _, _ = create_train_test_splits(X, y_encoded, test_size=0.2, random_state=42)

    # Introduce unseen category in test set
    X_test_corrupted = X_test.copy()
    X_test_corrupted.iloc[0, X_test.columns.get_loc("PaymentMethod")] = "Crypto / Bitcoin"
    X_test_corrupted.iloc[0, X_test.columns.get_loc("InternetService")] = "Satellite Internet"

    preprocessor = build_feature_preprocessor()
    # Must not raise ValueError
    X_train_trans, X_test_trans, _ = fit_and_transform_features(
        preprocessor, X_train, X_test_corrupted
    )
    assert X_test_trans is not None
    assert not np.isnan(X_test_trans).any()


def test_stratified_train_test_split(sample_churn_df):
    """Verify stratified split preserves target class proportions in train and test sets."""
    X, y = split_features_and_target(sample_churn_df)
    y_encoded = encode_target(y)

    _, _, y_train, y_test = create_train_test_splits(
        X, y_encoded, test_size=0.3, random_state=42, stratify=True
    )

    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()

    # Proportions should match within reasonable tolerance for small synthetic fixture
    assert abs(train_churn_rate - test_churn_rate) < 0.15

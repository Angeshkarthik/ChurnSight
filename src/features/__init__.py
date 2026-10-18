"""Feature engineering sub-package for Customer Intelligence Platform."""

from src.features.config import (
    ID_COLUMN,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_PREDICTOR_FEATURES,
)
from src.features.pipeline import (
    split_features_and_target,
    encode_target,
    build_feature_preprocessor,
    create_train_test_splits,
    fit_and_transform_features,
)

__all__ = [
    "ID_COLUMN",
    "TARGET_COLUMN",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "ALL_PREDICTOR_FEATURES",
    "split_features_and_target",
    "encode_target",
    "build_feature_preprocessor",
    "create_train_test_splits",
    "fit_and_transform_features",
]

"""Model Registry and Pipeline Factory Module.

Instantiates machine learning model pipelines adhering to Scikit-learn Pipeline standards.
Each pipeline bundles:
1. Feature Preprocessing (`ColumnTransformer` from `src.features`)
2. Estimator (Dummy, Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting)

Design Philosophy:
- Encapsulation: Bundling preprocessor with the estimator ensures leak-free cross-validation
  and seamless deployment (one object to serialize and serve).
- Determinism: Random seeds are strictly fixed for reproducible benchmarking.
- Robust Stability: Uses battle-tested Scikit-learn native estimators without C++ library conflicts.
"""

from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.pipeline import build_feature_preprocessor

RANDOM_SEED: int = 42


def get_model_pipeline(
    model_name: str,
    random_state: int = RANDOM_SEED,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Create a complete Scikit-learn Pipeline combining feature preprocessor and estimator.

    Parameters
    ----------
    model_name : str
        Identifier for the model family:
        - 'dummy': Baseline predicting most frequent class
        - 'logistic_regression': Standard linear probabilistic classifier
        - 'logistic_regression_balanced': Cost-sensitive logistic regression for class imbalance
        - 'random_forest': Bagging ensemble of decision trees
        - 'gradient_boosting': Sequential residual boosting ensemble
        - 'hist_gradient_boosting': Histogram-based gradient boosting (fast, modern ensemble)
    random_state : int, default 42
        Seed ensuring deterministic reproducibility.
    custom_params : Optional[Dict[str, Any]], default None
        Optional dictionary of hyperparameter overrides for the classifier.

    Returns
    -------
    Pipeline
        Configured, unfitted Scikit-learn Pipeline with 'preprocessor' and 'classifier' steps.

    Raises
    ------
    ValueError
        If model_name is not recognized.
    """
    params = custom_params or {}

    if model_name == "dummy":
        default_params = {"strategy": "most_frequent", "random_state": random_state}
        default_params.update(params)
        estimator: BaseEstimator = DummyClassifier(**default_params)

    elif model_name == "logistic_regression":
        default_params = {
            "max_iter": 1000,
            "random_state": random_state,
            "solver": "lbfgs",
        }
        default_params.update(params)
        estimator = LogisticRegression(**default_params)

    elif model_name == "logistic_regression_balanced":
        default_params = {
            "max_iter": 1000,
            "random_state": random_state,
            "class_weight": "balanced",
            "solver": "lbfgs",
        }
        default_params.update(params)
        estimator = LogisticRegression(**default_params)

    elif model_name == "random_forest":
        default_params = {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_leaf": 4,
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(params)
        estimator = RandomForestClassifier(**default_params)

    elif model_name == "gradient_boosting":
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": random_state,
        }
        default_params.update(params)
        estimator = GradientBoostingClassifier(**default_params)

    elif model_name == "hist_gradient_boosting":
        default_params = {
            "max_iter": 100,
            "learning_rate": 0.08,
            "max_depth": 5,
            "min_samples_leaf": 20,
            "random_state": random_state,
        }
        default_params.update(params)
        estimator = HistGradientBoostingClassifier(**default_params)

    else:
        valid_models = [
            "dummy",
            "logistic_regression",
            "logistic_regression_balanced",
            "random_forest",
            "gradient_boosting",
            "hist_gradient_boosting",
        ]
        raise ValueError(
            f"Unknown model_name: '{model_name}'. Supported options are: {valid_models}"
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            ("classifier", estimator),
        ]
    )

    return pipeline


def get_all_candidate_models(random_state: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """Retrieve all candidate model pipelines for benchmark comparison.

    Parameters
    ----------
    random_state : int, default 42
        Seed ensuring deterministic reproducibility.

    Returns
    -------
    Dict[str, Pipeline]
        Dictionary mapping human-readable model names to configured Pipelines.
    """
    model_keys = [
        ("Dummy Baseline", "dummy"),
        ("Logistic Regression", "logistic_regression"),
        ("Logistic Regression (Balanced)", "logistic_regression_balanced"),
        ("Random Forest", "random_forest"),
        ("Gradient Boosting", "gradient_boosting"),
        ("Hist Gradient Boosting", "hist_gradient_boosting"),
    ]

    return {
        display_name: get_model_pipeline(key, random_state=random_state)
        for display_name, key in model_keys
    }

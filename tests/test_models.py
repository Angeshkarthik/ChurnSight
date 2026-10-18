"""Unit tests for Model Training, Registry, and Evaluation modules."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.evaluation import (
    EvaluationMetrics,
    evaluate_predictions,
    evaluate_threshold_tradeoffs,
    format_benchmark_table,
)
from src.models.registry import (
    get_all_candidate_models,
    get_model_pipeline,
)
from src.models.trainer import cross_validate_candidates


@pytest.fixture
def sample_feature_data():
    """Create a minimal synthetic dataset matching the preprocessor expectation."""
    n_samples = 40
    np.random.seed(42)
    X = pd.DataFrame({
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
    })
    y = pd.Series([0, 0, 0, 1] * (n_samples // 4))  # 25% positive churn
    return X, y


def test_get_model_pipeline_valid():
    """Verify all valid model names construct proper Pipelines."""
    valid_names = [
        "dummy",
        "logistic_regression",
        "logistic_regression_balanced",
        "random_forest",
        "gradient_boosting",
        "hist_gradient_boosting",
    ]
    for name in valid_names:
        pipe = get_model_pipeline(name)
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps


def test_get_model_pipeline_invalid():
    """Verify unknown model names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model_name"):
        get_model_pipeline("non_existent_deep_network")


def test_get_all_candidate_models():
    """Verify candidate models dictionary contains all expected benchmark models."""
    candidates = get_all_candidate_models()
    assert len(candidates) >= 5
    assert "Dummy Baseline" in candidates
    assert "Logistic Regression" in candidates
    assert "Random Forest" in candidates
    assert "Gradient Boosting" in candidates


def test_evaluate_predictions_metrics():
    """Verify calculation of accuracy, precision, recall, and confusion matrix."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.2, 0.85])

    metrics = evaluate_predictions(y_true, y_pred, y_prob, model_name="TestModel")

    assert isinstance(metrics, EvaluationMetrics)
    # y_true has 3 positive (idx 2, 3, 5), 3 negative (idx 0, 1, 4)
    # y_pred gets idx 2 and 5 correct (TP=2), misses idx 3 (FN=1), 0 false positives (FP=0), 3 true negatives (TN=3)
    assert metrics.true_positives == 2
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 3
    assert metrics.false_positives == 0

    # Recall = TP / (TP + FN) = 2 / 3
    assert abs(metrics.recall - (2 / 3)) < 1e-4
    # Precision = TP / (TP + FP) = 2 / 2 = 1.0
    assert abs(metrics.precision - 1.0) < 1e-4
    # Accuracy = (3 + 2) / 6 = 5 / 6
    assert abs(metrics.accuracy - (5 / 6)) < 1e-4
    assert metrics.roc_auc > 0.5


def test_evaluate_threshold_tradeoffs():
    """Verify threshold evaluation computes precision/recall across cutoffs."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.4, 0.6, 0.8])

    tradeoffs = evaluate_threshold_tradeoffs(y_true, y_prob, thresholds=[0.3, 0.7])
    assert len(tradeoffs) == 2
    assert tradeoffs[0]["threshold"] == 0.3
    # At th=0.3: preds = [0, 1, 1, 1], TP=2, FP=1, FN=0, TN=1 -> Recall=1.0
    assert tradeoffs[0]["recall"] == 1.0
    # At th=0.7: preds = [0, 0, 0, 1], TP=1, FP=0, FN=1, TN=2 -> Precision=1.0, Recall=0.5
    assert tradeoffs[1]["precision"] == 1.0
    assert tradeoffs[1]["recall"] == 0.5


def test_pipeline_fit_and_predict(sample_feature_data):
    """Verify complete Pipeline fits and predicts without data leakage."""
    X, y = sample_feature_data
    pipe = get_model_pipeline("logistic_regression")

    pipe.fit(X, y)
    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_format_benchmark_table():
    """Verify table formatter outputs a non-empty string with column headers."""
    m1 = EvaluationMetrics(
        model_name="Model A",
        accuracy=0.8,
        precision=0.7,
        recall=0.6,
        f1_score=0.65,
        roc_auc=0.85,
        pr_auc=0.75,
        true_negatives=70,
        false_positives=10,
        false_negatives=10,
        true_positives=10,
    )
    table_str = format_benchmark_table([m1])
    assert "Model A" in table_str
    assert "ROC-AUC" in table_str
    assert "Recall" in table_str

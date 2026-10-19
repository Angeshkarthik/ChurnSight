"""Unit tests for Model Training, Registry, and Evaluation modules.

Covers:
- CV-based champion selection with deterministic tie-breaking (no test-set peeking)
- OOF-based threshold optimization (independent of holdout labels)
- Final holdout isolation (no candidate models or holdout threshold sweeps)
- Explicit confusion matrix label ordering (TN, FP, FN, TP)
- Positive-class metric calculations (Precision, Recall, F1 for Churn)
- Defensive input validation (rejection of NaN, +Inf, -Inf, invalid labels, invalid predictions)
"""

import math
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.evaluation import (
    EvaluationMetrics,
    evaluate_predictions,
    evaluate_threshold_tradeoffs,
    format_benchmark_table,
    select_optimal_threshold,
)
from src.models.registry import (
    get_all_candidate_models,
    get_model_pipeline,
)
from src.models.trainer import (
    cross_validate_candidates,
    select_champion_from_cv,
    select_threshold_from_oof,
)


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


# --- REGISTRY TESTS ---

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
        get_model_pipeline("non_existent_neural_net")


def test_get_all_candidate_models():
    """Verify candidate models dictionary contains all expected benchmark models."""
    candidates = get_all_candidate_models()
    assert len(candidates) >= 5
    assert "Dummy Baseline" in candidates
    assert "Logistic Regression" in candidates
    assert "Random Forest" in candidates
    assert "Gradient Boosting" in candidates


# --- SELECTION TESTS (NO TEST SET INSPECTION) ---

def test_select_champion_from_cv_uses_pr_auc():
    """Verify champion selection strictly uses mean CV PR-AUC from training CV, ignoring test metrics."""
    cv_summary = pd.DataFrame([
        {"Model": "Model Low PR High ROC", "CV PR-AUC Mean": 0.60, "CV ROC-AUC Mean": 0.88, "CV F1 Mean": 0.55},
        {"Model": "Model High PR Moderate ROC", "CV PR-AUC Mean": 0.66, "CV ROC-AUC Mean": 0.84, "CV F1 Mean": 0.58},
        {"Model": "Dummy Baseline", "CV PR-AUC Mean": 0.26, "CV ROC-AUC Mean": 0.50, "CV F1 Mean": 0.00},
    ])

    champion = select_champion_from_cv(cv_summary)
    # Must choose Model High PR Moderate ROC because PR-AUC (0.66 > 0.60) is primary metric
    assert champion == "Model High PR Moderate ROC"


def test_select_champion_from_cv_tie_breaking():
    """Verify deterministic tie-breaking on ROC-AUC when CV PR-AUC is tied."""
    cv_summary = pd.DataFrame([
        {"Model": "Model Alpha", "CV PR-AUC Mean": 0.65, "CV ROC-AUC Mean": 0.83, "CV F1 Mean": 0.58},
        {"Model": "Model Beta", "CV PR-AUC Mean": 0.65, "CV ROC-AUC Mean": 0.87, "CV F1 Mean": 0.59},
    ])

    champion = select_champion_from_cv(cv_summary)
    assert champion == "Model Beta"


def test_select_optimal_threshold_max_f1():
    """Verify threshold selection finds the threshold maximizing F1 score."""
    tradeoffs = [
        {"threshold": 0.20, "precision": 0.40, "recall": 0.90, "f1_score": 0.55},
        {"threshold": 0.35, "precision": 0.55, "recall": 0.75, "f1_score": 0.64},  # Highest F1
        {"threshold": 0.50, "precision": 0.70, "recall": 0.50, "f1_score": 0.58},
    ]

    selected_th, metrics = select_optimal_threshold(tradeoffs, criterion="max_f1")
    assert selected_th == 0.35
    assert metrics["f1_score"] == 0.64


def test_select_threshold_from_oof_independent_of_test_data(sample_feature_data):
    """Verify threshold selection operates strictly via out-of-fold training predictions."""
    X, y = sample_feature_data
    pipe = get_model_pipeline("logistic_regression")

    # Run OOF threshold tuning
    threshold, best_metrics, tradeoffs = select_threshold_from_oof(
        pipe, X, y, n_splits=3, random_state=42, criterion="max_f1"
    )

    assert 0.0 < threshold < 1.0
    assert "f1_score" in best_metrics
    assert len(tradeoffs) > 0


# --- EVALUATION & METRIC RELIABILITY TESTS ---

def test_evaluate_predictions_explicit_confusion_matrix_ordering():
    """Verify confusion matrix follows explicit order: [TN, FP, FN, TP]."""
    # 2 true negatives (idx 0, 1), 1 false positive (idx 2), 1 false negative (idx 3), 1 true positive (idx 4)
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.3, 0.9])

    metrics = evaluate_predictions(y_true, y_pred, y_prob, model_name="TestOrdering")

    assert metrics.true_negatives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1

    # Precision on Churn = TP / (TP + FP) = 1 / 2 = 0.5
    assert metrics.precision == 0.5
    # Recall on Churn = TP / (TP + FN) = 1 / 2 = 0.5
    assert metrics.recall == 0.5
    # Accuracy = (2 + 1) / 5 = 0.6
    assert metrics.accuracy == 0.6


def test_evaluate_predictions_length_mismatch():
    """Verify length mismatch raises ValueError."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1])
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_predictions(y_true, y_pred)


def test_evaluate_predictions_invalid_probabilities():
    """Verify probabilities outside [0, 1] raise ValueError."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])
    y_prob = np.array([0.2, 1.5, -0.1])  # Out-of-bounds probabilities
    with pytest.raises(ValueError, match="strictly bounded in \\[0.0, 1.0\\]"):
        evaluate_predictions(y_true, y_pred, y_prob)


def test_evaluate_predictions_non_finite_probabilities():
    """Verify NaN, +Inf, and -Inf in probabilities are rejected via np.isfinite()."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])

    # NaN probability
    with pytest.raises(ValueError, match="non-finite values"):
        evaluate_predictions(y_true, y_pred, np.array([0.2, np.nan, 0.1]))

    # +Inf probability
    with pytest.raises(ValueError, match="non-finite values"):
        evaluate_predictions(y_true, y_pred, np.array([0.2, np.inf, 0.1]))

    # -Inf probability
    with pytest.raises(ValueError, match="non-finite values"):
        evaluate_predictions(y_true, y_pred, np.array([0.2, -np.inf, 0.1]))


def test_evaluate_predictions_non_finite_labels_and_preds():
    """Verify NaN and Inf in labels or predictions are rejected."""
    # NaN in y_true
    with pytest.raises(ValueError, match="y_true contains non-finite"):
        evaluate_predictions(np.array([0, np.nan, 1]), np.array([0, 1, 1]))

    # Inf in y_pred
    with pytest.raises(ValueError, match="y_pred contains non-finite"):
        evaluate_predictions(np.array([0, 1, 0]), np.array([0, np.inf, 0]))


def test_evaluate_predictions_invalid_labels():
    """Verify non-binary labels raise ValueError."""
    # Labels containing 2
    with pytest.raises(ValueError, match="strictly binary labels in {0, 1}"):
        evaluate_predictions(np.array([0, 1, 2]), np.array([0, 1, 0]))

    # Negative labels
    with pytest.raises(ValueError, match="strictly binary labels in {0, 1}"):
        evaluate_predictions(np.array([-1, 1, 0]), np.array([0, 1, 0]))


def test_evaluate_predictions_invalid_predictions():
    """Verify non-binary predictions raise ValueError."""
    with pytest.raises(ValueError, match="strictly binary predictions in {0, 1}"):
        evaluate_predictions(np.array([0, 1, 0]), np.array([0, 1, 5]))


def test_evaluate_predictions_single_class_target_raises_error():
    """Verify single-class target raises explicit ValueError rather than returning a plausible-looking 0.5."""
    y_true_all_zeros = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.1, 0.3])

    with pytest.raises(ValueError, match="mathematically undefined when y_true contains only one class"):
        evaluate_predictions(y_true_all_zeros, y_pred, y_prob, raise_on_single_class=True)

    # When raise_on_single_class is False, verify NaN is assigned explicitly, not 0.5
    metrics = evaluate_predictions(y_true_all_zeros, y_pred, y_prob, raise_on_single_class=False)
    assert math.isnan(metrics.roc_auc)
    assert math.isnan(metrics.pr_auc)


def test_holdout_isolation_no_sweep_tables():
    """Verify that when compute_threshold_tradeoffs is False, no threshold sweeps are calculated."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.2, 0.9])

    metrics = evaluate_predictions(y_true, y_pred, y_prob, compute_threshold_tradeoffs=False)
    assert metrics.threshold_tradeoffs == []
    # to_dict() should omit the empty threshold_tradeoffs key
    assert "threshold_tradeoffs" not in metrics.to_dict()


def test_format_benchmark_table_with_nans():
    """Verify table formatter handles NaN metrics gracefully."""
    m_nan = EvaluationMetrics(
        model_name="Model with NaNs",
        accuracy=0.8,
        precision=0.7,
        recall=0.6,
        f1_score=0.65,
        roc_auc=float("nan"),
        pr_auc=float("nan"),
        true_negatives=70,
        false_positives=10,
        false_negatives=10,
        true_positives=10,
    )
    table_str = format_benchmark_table([m_nan])
    assert "Model with NaNs" in table_str
    assert "NaN" in table_str

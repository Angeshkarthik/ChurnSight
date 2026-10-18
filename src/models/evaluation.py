"""Model Evaluation and Diagnostics Module.

Computes comprehensive classification metrics tailored for imbalanced churn prediction:
- Accuracy, Precision, Recall, F1-score (for the positive Churn class)
- ROC-AUC (Discrimination ability across thresholds)
- PR-AUC (Average Precision, highly sensitive to minority class performance)
- Confusion Matrix (True Negatives, False Positives, False Negatives, True Positives)
- Probability Threshold Trade-Off Analysis

Business Context & Metric Framing:
- False Positive (FP): A customer predicted to churn who was actually happy.
  Business Impact: Wasted retention discount/offer, but customer is retained.
- False Negative (FN): A customer predicted to stay who cancels service.
  Business Impact: Catastrophic revenue loss. Re-acquiring a new customer costs 5x-25x more.
  Therefore, Recall is far more critical than Accuracy.
"""

from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Encapsulates all evaluation scores and confusion matrix counts for a model."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    threshold_tradeoffs: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics dataclass to dictionary."""
        return asdict(self)

    def summary_text(self) -> str:
        """Format metrics into a clear diagnostic report."""
        lines = [
            f"=== Performance Report: {self.model_name} ===",
            f"  Accuracy           : {self.accuracy:.4f}",
            f"  ROC-AUC            : {self.roc_auc:.4f}",
            f"  PR-AUC (Avg Prec)  : {self.pr_auc:.4f}",
            f"  Recall (Churners)  : {self.recall:.4f}  (Sensitivity: % of true churners caught)",
            f"  Precision (Churn)  : {self.precision:.4f}  (Reliability: % of predicted churners correct)",
            f"  F1-Score           : {self.f1_score:.4f}",
            "  Confusion Matrix   :",
            f"    [TN: {self.true_negatives:>4} | FP: {self.false_positives:>4}]",
            f"    [FN: {self.false_negatives:>4} | TP: {self.true_positives:>4}]",
        ]
        return "\n".join(lines)


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
    model_name: str = "Classifier",
) -> EvaluationMetrics:
    """Calculate comprehensive evaluation metrics on predictions and probabilities.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Ground-truth binary labels (0 or 1).
    y_pred : Union[pd.Series, np.ndarray]
        Discrete predicted binary labels (0 or 1).
    y_prob : Optional[Union[pd.Series, np.ndarray]], default None
        Predicted churn probabilities P(y=1|X). If None, ROC-AUC and PR-AUC are estimated from y_pred.
    model_name : str, default 'Classifier'
        Label identifying the evaluated model.

    Returns
    -------
    EvaluationMetrics
        Structured container with all computed metrics.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # Confusion matrix extraction: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    # Calculate ROC-AUC and PR-AUC using continuous probabilities if available
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)
        # If shape is (N, 2), extract positive class probabilities
        if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
            y_prob_arr = y_prob_arr[:, 1]
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except ValueError:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
        except ValueError:
            pr_auc = float(np.mean(y_true_arr))
        threshold_table = evaluate_threshold_tradeoffs(y_true_arr, y_prob_arr)
    else:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_pred_arr))
        except ValueError:
            roc_auc = 0.5
        pr_auc = float(average_precision_score(y_true_arr, y_pred_arr))
        threshold_table = []

    return EvaluationMetrics(
        model_name=model_name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        threshold_tradeoffs=threshold_table,
    )


def evaluate_threshold_tradeoffs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """Evaluate Precision, Recall, and F1 across multiple probability thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive churn class.
    thresholds : Optional[Sequence[float]], default None
        List of decision thresholds to evaluate. If None, uses 0.1 to 0.9 in steps of 0.1.

    Returns
    -------
    List[Dict[str, float]]
        Table of metrics for each threshold.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob)
    if y_prob_arr.ndim == 2:
        y_prob_arr = y_prob_arr[:, 1]

    results = []
    for th in thresholds:
        preds = (y_prob_arr >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_arr, preds, labels=[0, 1]).ravel()
        prec = precision_score(y_true_arr, preds, zero_division=0)
        rec = recall_score(y_true_arr, preds, zero_division=0)
        f1 = f1_score(y_true_arr, preds, zero_division=0)

        results.append({
            "threshold": round(float(th), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    return results


def format_benchmark_table(metrics_list: List[EvaluationMetrics]) -> str:
    """Format multiple model evaluation results into an ASCII comparison table."""
    header = (
        f"{'Model Name':<32} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Recall':<8} | {'Prec':<8} | {'F1':<8} | {'Acc':<8}"
    )
    separator = "-" * len(header)
    rows = [separator, header, separator]

    for m in metrics_list:
        row = (
            f"{m.model_name:<32} | "
            f"{m.roc_auc:<8.4f} | "
            f"{m.pr_auc:<8.4f} | "
            f"{m.recall:<8.4f} | "
            f"{m.precision:<8.4f} | "
            f"{m.f1_score:<8.4f} | "
            f"{m.accuracy:<8.4f}"
        )
        rows.append(row)

    rows.append(separator)
    return "\n".join(rows)

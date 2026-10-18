"""Model Evaluation and Diagnostics Module.

Computes comprehensive classification metrics tailored for imbalanced churn prediction:
- Accuracy, Precision, Recall, F1-score (for the positive Churn class)
- ROC-AUC (Discrimination ability across continuous probability thresholds)
- PR-AUC (Average Precision, highly sensitive to minority class performance)
- Confusion Matrix (Explicitly ordered: True Negatives, False Positives, False Negatives, True Positives)
- Probability Threshold Trade-Off Analysis

Business Context & Metric Framing:
- False Positive (FP): A loyal customer mistakenly predicted to churn.
  Business Impact: Wasted proactive discount or retention outreach cost, but the customer is retained.
- False Negative (FN): A customer predicted to stay who cancels service.
  Business Impact: Significant lost recurring revenue and Customer Lifetime Value (LTV).
  (Ref: Gallo, Amy. "The Value of Keeping the Right Customers", Harvard Business Review, 2014).
  Therefore, Recall on the churn class is typically prioritized over raw Accuracy.
"""

from dataclasses import asdict, dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
            f"  ROC-AUC            : {self.roc_auc:.4f}" if not math.isnan(self.roc_auc) else "  ROC-AUC            : NaN (Undefined)",
            f"  PR-AUC (Avg Prec)  : {self.pr_auc:.4f}" if not math.isnan(self.pr_auc) else "  PR-AUC (Avg Prec)  : NaN (Undefined)",
            f"  Recall (Churners)  : {self.recall:.4f}  (Sensitivity: % of true churners caught)",
            f"  Precision (Churn)  : {self.precision:.4f}  (Reliability: % of predicted churners correct)",
            f"  F1-Score           : {self.f1_score:.4f}",
            "  Confusion Matrix (labels=[0, 1]):",
            f"    [TN: {self.true_negatives:>4} | FP: {self.false_positives:>4}]",
            f"    [FN: {self.false_negatives:>4} | TP: {self.true_positives:>4}]",
        ]
        return "\n".join(lines)


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
    model_name: str = "Classifier",
    raise_on_single_class: bool = True,
) -> EvaluationMetrics:
    """Calculate comprehensive evaluation metrics on predictions and probabilities.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Ground-truth binary labels (0 or 1).
    y_pred : Union[pd.Series, np.ndarray]
        Discrete predicted binary labels (0 or 1).
    y_prob : Optional[Union[pd.Series, np.ndarray]], default None
        Predicted churn probabilities P(y=1|X).
    model_name : str, default 'Classifier'
        Label identifying the evaluated model.
    raise_on_single_class : bool, default True
        If True, raises ValueError when y_true contains fewer than 2 classes,
        preventing misleading fallback numbers.

    Returns
    -------
    EvaluationMetrics
        Structured container with all computed metrics.

    Raises
    ------
    ValueError
        If array lengths mismatch, values are out of bounds, or y_true is single-class.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true_arr)} != len(y_pred)={len(y_pred_arr)}"
        )

    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        msg = (
            f"ROC-AUC and binary evaluation metrics are mathematically undefined when y_true "
            f"contains only one class: {unique_classes.tolist()}. Both positive and negative classes required."
        )
        if raise_on_single_class:
            raise ValueError(msg)
        logger.warning(msg)

    # Core classification metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # Explicit confusion matrix unpacking: labels=[0, 1] guarantees [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Probability metrics
    threshold_table: List[Dict[str, float]] = []

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)

        # Validate probability array shape
        if y_prob_arr.ndim == 2:
            if y_prob_arr.shape[1] != 2:
                raise ValueError(
                    f"Expected 2-class probability array with shape (N, 2), got {y_prob_arr.shape}"
                )
            y_prob_arr = y_prob_arr[:, 1]
        elif y_prob_arr.ndim != 1:
            raise ValueError(
                f"Predicted probabilities must be 1D or 2D array, got shape {y_prob_arr.shape}"
            )

        if len(y_prob_arr) != len(y_true_arr):
            raise ValueError(
                f"Length mismatch between y_true ({len(y_true_arr)}) and y_prob ({len(y_prob_arr)})"
            )

        if (y_prob_arr < 0.0).any() or (y_prob_arr > 1.0).any():
            raise ValueError("Predicted probabilities must be strictly bounded in [0.0, 1.0].")

        if len(unique_classes) >= 2:
            roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
            pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
            threshold_table = evaluate_threshold_tradeoffs(y_true_arr, y_prob_arr)
        else:
            roc_auc = float("nan")
            pr_auc = float("nan")
    else:
        # Fallback to discrete predictions only if probabilities are not provided
        if len(unique_classes) >= 2:
            roc_auc = float(roc_auc_score(y_true_arr, y_pred_arr))
            pr_auc = float(average_precision_score(y_true_arr, y_pred_arr))
        else:
            roc_auc = float("nan")
            pr_auc = float("nan")

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
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """Evaluate Precision, Recall, and F1 across multiple probability decision thresholds.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Ground-truth binary labels.
    y_prob : Union[pd.Series, np.ndarray]
        Predicted probabilities for the positive churn class (class 1).
    thresholds : Optional[Sequence[float]], default None
        List of decision thresholds to evaluate. If None, uses 0.1 to 0.9 in steps of 0.05.

    Returns
    -------
    List[Dict[str, float]]
        Table of metrics for each threshold.
    """
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if y_prob_arr.ndim == 2:
        y_prob_arr = y_prob_arr[:, 1]

    results: List[Dict[str, float]] = []
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


def select_optimal_threshold(
    threshold_tradeoffs: List[Dict[str, float]],
    criterion: str = "max_f1",
) -> Tuple[float, Dict[str, float]]:
    """Select optimal operational threshold using an explicit, documented criterion.

    Parameters
    ----------
    threshold_tradeoffs : List[Dict[str, float]]
        Table of threshold evaluations generated from validation or OOF predictions.
    criterion : str, default 'max_f1'
        Selection rule: 'max_f1' chooses the threshold that maximizes F1 score.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        Chosen threshold value and its corresponding metrics dictionary.

    Raises
    ------
    ValueError
        If threshold_tradeoffs is empty or criterion is unrecognized.
    """
    if not threshold_tradeoffs:
        raise ValueError("Cannot select threshold from empty threshold_tradeoffs list.")

    if criterion == "max_f1":
        # Deterministic tie-breaking: highest F1, then closest to 0.5 threshold
        best_entry = max(
            threshold_tradeoffs,
            key=lambda row: (row["f1_score"], -abs(row["threshold"] - 0.5)),
        )
        return best_entry["threshold"], best_entry

    raise ValueError(f"Unknown threshold selection criterion: '{criterion}'. Supported: ['max_f1']")


def format_benchmark_table(metrics_list: List[EvaluationMetrics]) -> str:
    """Format multiple model evaluation results into an ASCII comparison table."""
    header = (
        f"{'Model Name':<32} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Recall':<8} | {'Prec':<8} | {'F1':<8} | {'Acc':<8}"
    )
    separator = "-" * len(header)
    rows = [separator, header, separator]

    for m in metrics_list:
        roc_str = f"{m.roc_auc:<8.4f}" if not math.isnan(m.roc_auc) else "NaN     "
        pr_str = f"{m.pr_auc:<8.4f}" if not math.isnan(m.pr_auc) else "NaN     "
        row = (
            f"{m.model_name:<32} | "
            f"{roc_str} | "
            f"{pr_str} | "
            f"{m.recall:<8.4f} | "
            f"{m.precision:<8.4f} | "
            f"{m.f1_score:<8.4f} | "
            f"{m.accuracy:<8.4f}"
        )
        rows.append(row)

    rows.append(separator)
    return "\n".join(rows)

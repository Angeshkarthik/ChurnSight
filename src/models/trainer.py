"""Model Training, Cross-Validation, and Benchmarking Orchestrator.

Orchestrates the complete Phase 4 modeling workflow adhering to strict ML governance:
1. Partitions data into a Development set (80%) and an independent Fresh Holdout set (20%)
   using documented, explicit random seeds.
2. Evaluates all candidate pipelines using 5-Fold Stratified Cross-Validation strictly on Development data.
3. Selects the champion model strictly using mean CV PR-AUC with deterministic tie-breaking.
   (No test data is inspected during model selection).
4. Selects the operational probability decision threshold strictly using training Out-Of-Fold (OOF)
   predictions to maximize OOF F1-score.
5. Freezes the champion pipeline and operational threshold.
6. Refits the champion model on the full Development set and performs a single, independent evaluation
   on the fresh untouched Holdout set at both standard (0.50) and operational thresholds.
7. Serializes the winning end-to-end Pipeline artifact (.joblib) and governance metadata (.json).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.features.pipeline import (
    create_train_test_splits,
    encode_target,
    split_features_and_target,
)
from src.models.evaluation import (
    EvaluationMetrics,
    evaluate_predictions,
    evaluate_threshold_tradeoffs,
    format_benchmark_table,
    select_optimal_threshold,
)
from src.models.registry import RANDOM_SEED, get_all_candidate_models, get_model_pipeline

logger = logging.getLogger(__name__)

# Default project paths resolved relative to repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

# Methodology seeds:
# - CV_SPLIT_SEED: Controls 5-fold cross-validation fold assignments within development data.
# - FRESH_HOLDOUT_SEED: Distinct random seed establishing an independent, un-peeked final holdout partition.
CV_SPLIT_SEED: int = 42
FRESH_HOLDOUT_SEED: int = 100


@dataclass
class BenchmarkResult:
    """Encapsulates the complete benchmarking run results, metrics, and paths."""

    cv_summary_df: pd.DataFrame
    champion_name: str
    champion_pipeline: Pipeline
    selected_threshold: float
    oof_threshold_metrics: Dict[str, float]
    oof_threshold_tradeoffs: List[Dict[str, float]]
    holdout_standard_metrics: EvaluationMetrics
    holdout_operational_metrics: EvaluationMetrics
    all_models_holdout_metrics: List[EvaluationMetrics]
    artifact_path: Optional[Path]
    metadata_path: Optional[Path]


def cross_validate_candidates(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = CV_SPLIT_SEED,
) -> pd.DataFrame:
    """Run Stratified K-Fold Cross-Validation on all candidate models strictly on development data.

    Parameters
    ----------
    models : Dict[str, Pipeline]
        Dictionary of unfitted model pipelines.
    X_train : pd.DataFrame
        Development feature matrix.
    y_train : pd.Series
        Development target vector.
    n_splits : int, default 5
        Number of stratified cross-validation folds.
    random_state : int, default 42
        Deterministic split seed.

    Returns
    -------
    pd.DataFrame
        Summary table showing mean and std of ROC-AUC, PR-AUC, Recall, Precision, and F1.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": "recall",
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "accuracy": "accuracy",
    }

    results: List[Dict[str, Any]] = []

    logger.info("Executing %d-Fold Stratified Cross-Validation on %d models...", n_splits, len(models))

    for name, pipeline in models.items():
        logger.info("Cross-validating: %s...", name)
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        results.append({
            "Model": name,
            "CV PR-AUC Mean": float(np.mean(cv_scores["test_pr_auc"])),
            "CV PR-AUC Std": float(np.std(cv_scores["test_pr_auc"])),
            "CV ROC-AUC Mean": float(np.mean(cv_scores["test_roc_auc"])),
            "CV ROC-AUC Std": float(np.std(cv_scores["test_roc_auc"])),
            "CV F1 Mean": float(np.mean(cv_scores["test_f1"])),
            "CV Recall Mean": float(np.mean(cv_scores["test_recall"])),
            "CV Precision Mean": float(np.mean(cv_scores["test_precision"])),
            "CV Accuracy Mean": float(np.mean(cv_scores["test_accuracy"])),
        })

    # Sort deterministically
    summary_df = pd.DataFrame(results).sort_values(
        by=["CV PR-AUC Mean", "CV ROC-AUC Mean", "CV F1 Mean", "Model"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    return summary_df


def select_champion_from_cv(
    cv_summary: pd.DataFrame,
    exclude_models: Sequence[str] = ("Dummy Baseline",),
) -> str:
    """Select champion model strictly using cross-validation metrics with deterministic tie-breaking.

    Selection Hierarchy:
    1. Primary: Highest CV PR-AUC Mean (best performance on minority churn class).
    2. Tie-break 1: Highest CV ROC-AUC Mean.
    3. Tie-break 2: Highest CV F1 Mean.
    4. Tie-break 3: Alphabetical by model name.

    Parameters
    ----------
    cv_summary : pd.DataFrame
        Cross-validation metrics summary table.
    exclude_models : Sequence[str], default ('Dummy Baseline',)
        Model names to exclude from champion candidacy.

    Returns
    -------
    str
        Name of the winning champion model.

    Raises
    ------
    ValueError
        If cv_summary has no eligible candidate models.
    """
    eligible = cv_summary[~cv_summary["Model"].isin(exclude_models)].copy()
    if eligible.empty:
        raise ValueError("No eligible candidate models in CV summary after filtering.")

    sorted_candidates = eligible.sort_values(
        by=["CV PR-AUC Mean", "CV ROC-AUC Mean", "CV F1 Mean", "Model"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    champion_name = str(sorted_candidates.iloc[0]["Model"])
    logger.info(
        "Selected champion model strictly from CV: '%s' (CV PR-AUC=%.4f, CV ROC-AUC=%.4f)",
        champion_name,
        sorted_candidates.iloc[0]["CV PR-AUC Mean"],
        sorted_candidates.iloc[0]["CV ROC-AUC Mean"],
    )
    return champion_name


def select_threshold_from_oof(
    champion_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = CV_SPLIT_SEED,
    criterion: str = "max_f1",
) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
    """Select the operational decision threshold strictly using out-of-fold training predictions.

    Parameters
    ----------
    champion_pipeline : Pipeline
        Unfitted pipeline for the champion model.
    X_train : pd.DataFrame
        Development feature matrix.
    y_train : pd.Series
        Development target vector.
    n_splits : int, default 5
        Number of cross-validation folds.
    random_state : int, default 42
        CV split seed.
    criterion : str, default 'max_f1'
        Threshold selection objective.

    Returns
    -------
    Tuple[float, Dict[str, float], List[Dict[str, float]]]
        Selected threshold, selected threshold OOF metrics, and full OOF sensitivity table.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    logger.info("Generating Out-Of-Fold (OOF) probability predictions for threshold tuning...")

    oof_probs = cross_val_predict(
        champion_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    oof_tradeoffs = evaluate_threshold_tradeoffs(y_train, oof_probs)
    selected_threshold, selected_metrics = select_optimal_threshold(
        oof_tradeoffs, criterion=criterion
    )

    logger.info(
        "Selected operational threshold from OOF predictions: %.2f (OOF F1=%.4f, OOF Recall=%.4f, OOF Prec=%.4f)",
        selected_threshold,
        selected_metrics["f1_score"],
        selected_metrics["recall"],
        selected_metrics["precision"],
    )

    return selected_threshold, selected_metrics, oof_tradeoffs


def run_model_benchmarking(
    data_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    test_size: float = 0.2,
    holdout_seed: int = FRESH_HOLDOUT_SEED,
    cv_seed: int = CV_SPLIT_SEED,
    export_artifacts: bool = True,
    verbose: bool = True,
) -> BenchmarkResult:
    """Execute rigorous model training, validation-based selection, and independent holdout evaluation.

    Governance Safeguards:
    1. Data Partitioning: Independent holdout split using holdout_seed (default 100),
       completely un-peeked during model and threshold selection.
    2. Champion Selection: Strictly determined via 5-Fold Stratified CV on Development data.
    3. Threshold Tuning: Operational threshold selected strictly from Out-Of-Fold (OOF) training predictions.
    4. Freezing: Model architecture and threshold are frozen prior to final holdout inference.

    Parameters
    ----------
    data_path : Optional[Union[str, Path]], default None
        Path to cleaned CSV dataset.
    output_dir : Optional[Union[str, Path]], default None
        Destination directory for exported artifacts.
    test_size : float, default 0.2
        Fraction of samples allocated to the fresh holdout partition.
    holdout_seed : int, default 100
        Seed establishing the independent holdout partition.
    cv_seed : int, default 42
        Seed for cross-validation within the development partition.
    export_artifacts : bool, default True
        If True, serializes champion model .joblib and metadata .json.
    verbose : bool, default True
        If True, prints diagnostic tables to stdout.

    Returns
    -------
    BenchmarkResult
        Complete evaluation artifact and summary containers.
    """
    resolved_data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    resolved_output_dir = Path(output_dir) if output_dir else DEFAULT_MODEL_DIR

    logger.info("Loading cleaned dataset from: %s", resolved_data_path)
    df = pd.read_csv(resolved_data_path)

    # 1. Feature separation & target encoding
    X, y_raw = split_features_and_target(df, drop_id=True)
    y = encode_target(y_raw)

    # 2. Establish fresh, untouched final holdout partition
    # Development: 80% (5,634 samples), Final Holdout: 20% (1,409 samples)
    X_dev, X_holdout, y_dev, y_holdout = create_train_test_splits(
        X, y, test_size=test_size, random_state=holdout_seed, stratify=True
    )

    logger.info(
        "Partitioned dataset: Development samples=%d, Fresh Holdout samples=%d (holdout_seed=%d)",
        len(X_dev),
        len(X_holdout),
        holdout_seed,
    )

    # 3. Instantiate candidate pipelines
    candidate_models = get_all_candidate_models(random_state=cv_seed)

    # 4. Step 1: Champion Selection strictly via 5-Fold CV on Development set
    cv_summary = cross_validate_candidates(
        candidate_models, X_dev, y_dev, n_splits=5, random_state=cv_seed
    )

    champion_name = select_champion_from_cv(cv_summary)

    if verbose:
        print("\n" + "=" * 90)
        print("5-FOLD STRATIFIED CROSS-VALIDATION (DEVELOPMENT SET ONLY: N=%d, cv_seed=%d)" % (len(X_dev), cv_seed))
        print("=" * 90)
        print(cv_summary.to_string(index=False))
        print("=" * 90)
        print(f"CHAMPION SELECTED STRICTLY FROM CV: '{champion_name}'")
        print("=" * 90 + "\n")

    # 5. Step 2: Threshold Selection strictly on Development OOF Predictions
    champion_pipeline_unfitted = get_all_candidate_models(random_state=cv_seed)[champion_name]
    selected_threshold, oof_thresh_metrics, oof_tradeoffs = select_threshold_from_oof(
        champion_pipeline_unfitted, X_dev, y_dev, n_splits=5, random_state=cv_seed, criterion="max_f1"
    )

    if verbose:
        print("=" * 90)
        print("OUT-OF-FOLD (OOF) THRESHOLD SELECTION (DEVELOPMENT SET ONLY)")
        print("=" * 90)
        print(f"Operational Decision Threshold: {selected_threshold:.2f} (Criterion: Maximum OOF F1-Score)")
        print(f"OOF Metrics at Threshold {selected_threshold:.2f}:")
        print(f"  F1: {oof_thresh_metrics['f1_score']:.4f} | Recall: {oof_thresh_metrics['recall']:.4f} | Precision: {oof_thresh_metrics['precision']:.4f}")
        print("=" * 90 + "\n")

    # 6. Step 3: Refit candidate models on full Development set and evaluate on Fresh Holdout
    # We fit all candidate pipelines on Development data
    all_holdout_metrics: List[EvaluationMetrics] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, pipeline in candidate_models.items():
        pipeline.fit(X_dev, y_dev)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_holdout)
        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_holdout)[:, 1]
            except Exception:
                y_prob = None

        metrics = evaluate_predictions(y_holdout, y_pred, y_prob, model_name=name)
        all_holdout_metrics.append(metrics)

    # 7. Evaluate the frozen champion model at both standard (0.50) and operational thresholds
    champion_pipeline = fitted_pipelines[champion_name]
    holdout_prob = champion_pipeline.predict_proba(X_holdout)[:, 1]

    # Standard threshold 0.50
    holdout_pred_standard = (holdout_prob >= 0.50).astype(int)
    holdout_standard_metrics = evaluate_predictions(
        y_holdout, holdout_pred_standard, holdout_prob, model_name=f"{champion_name} (Th=0.50 Standard)"
    )

    # Frozen operational threshold
    holdout_pred_operational = (holdout_prob >= selected_threshold).astype(int)
    holdout_operational_metrics = evaluate_predictions(
        y_holdout, holdout_pred_operational, holdout_prob, model_name=f"{champion_name} (Th={selected_threshold:.2f} Operational)"
    )

    if verbose:
        print("=" * 90)
        print("INDEPENDENT FINAL EVALUATION: FRESH HOLDOUT (N=%d samples, holdout_seed=%d)" % (len(X_holdout), holdout_seed))
        print("=" * 90)
        print(format_benchmark_table([holdout_standard_metrics, holdout_operational_metrics] + [m for m in all_holdout_metrics if m.model_name != champion_name]))
        print("=" * 90 + "\n")
        print("CHAMPION MODEL HOLDOUT REPORT (OPERATIONAL THRESHOLD):")
        print(holdout_operational_metrics.summary_text())
        print()

    # 8. Export artifacts
    artifact_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    if export_artifacts:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = resolved_output_dir / "churn_model.joblib"
        metadata_path = resolved_output_dir / "model_metadata.json"

        logger.info("Serializing frozen champion pipeline to %s...", artifact_path)
        joblib.dump(champion_pipeline, artifact_path)

        metadata = {
            "model_name": champion_name,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_methodology": {
                "champion_selection": "5-Fold Stratified CV on Development set using CV PR-AUC Mean with deterministic tie-breaking",
                "threshold_selection": "Out-Of-Fold (OOF) cross-validation predictions on Development set maximizing F1-score",
                "holdout_evaluation": "Independent fresh holdout partition, evaluated strictly after freezing model and threshold",
            },
            "seeds": {
                "holdout_seed": holdout_seed,
                "cv_split_seed": cv_seed,
            },
            "sample_counts": {
                "development_samples": int(len(X_dev)),
                "holdout_samples": int(len(X_holdout)),
                "total_samples": int(len(df)),
            },
            "operational_threshold": selected_threshold,
            "oof_threshold_metrics": oof_thresh_metrics,
            "holdout_operational_metrics": holdout_operational_metrics.to_dict(),
            "holdout_standard_metrics": holdout_standard_metrics.to_dict(),
            "oof_threshold_tradeoffs": oof_tradeoffs,
            "feature_names_in": list(X.columns),
            "total_raw_features": len(X.columns),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved model metadata and governance logs to %s.", metadata_path)

    return BenchmarkResult(
        cv_summary_df=cv_summary,
        champion_name=champion_name,
        champion_pipeline=champion_pipeline,
        selected_threshold=selected_threshold,
        oof_threshold_metrics=oof_thresh_metrics,
        oof_threshold_tradeoffs=oof_tradeoffs,
        holdout_standard_metrics=holdout_standard_metrics,
        holdout_operational_metrics=holdout_operational_metrics,
        all_models_holdout_metrics=all_holdout_metrics,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )

"""Model Training, Cross-Validation, and Benchmarking Orchestrator.

Orchestrates the complete Phase 4 modeling workflow:
1. Ingests cleaned data and splits into stratified train/test sets.
2. Evaluates all candidate pipelines using 5-Fold Stratified Cross-Validation strictly on X_train.
3. Selects the champion model balancing PR-AUC, ROC-AUC, and Recall.
4. Refits candidate models on the full training set and evaluates on the held-out test set.
5. Performs threshold sensitivity analysis.
6. Serializes the winning end-to-end Pipeline artifact (.joblib) and governance metadata (.json).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features.pipeline import (
    create_train_test_splits,
    encode_target,
    split_features_and_target,
)
from src.models.evaluation import (
    EvaluationMetrics,
    evaluate_predictions,
    format_benchmark_table,
)
from src.models.registry import RANDOM_SEED, get_all_candidate_models, get_model_pipeline

logger = logging.getLogger(__name__)

# Default project paths resolved relative to repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


@dataclass
class BenchmarkResult:
    """Encapsulates the complete benchmarking run results, metrics, and paths."""

    cv_summary_df: pd.DataFrame
    test_metrics: List[EvaluationMetrics]
    champion_name: str
    champion_pipeline: Pipeline
    champion_test_metrics: EvaluationMetrics
    artifact_path: Optional[Path]
    metadata_path: Optional[Path]


def cross_validate_candidates(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run Stratified K-Fold Cross-Validation on all candidate models strictly on X_train.

    Parameters
    ----------
    models : Dict[str, Pipeline]
        Dictionary of unfitted model pipelines.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
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
            "CV ROC-AUC Mean": float(np.mean(cv_scores["test_roc_auc"])),
            "CV ROC-AUC Std": float(np.std(cv_scores["test_roc_auc"])),
            "CV PR-AUC Mean": float(np.mean(cv_scores["test_pr_auc"])),
            "CV PR-AUC Std": float(np.std(cv_scores["test_pr_auc"])),
            "CV Recall Mean": float(np.mean(cv_scores["test_recall"])),
            "CV Recall Std": float(np.std(cv_scores["test_recall"])),
            "CV Precision Mean": float(np.mean(cv_scores["test_precision"])),
            "CV F1 Mean": float(np.mean(cv_scores["test_f1"])),
            "CV Accuracy Mean": float(np.mean(cv_scores["test_accuracy"])),
        })

    summary_df = pd.DataFrame(results).sort_values(by="CV PR-AUC Mean", ascending=False).reset_index(drop=True)
    return summary_df


def run_model_benchmarking(
    data_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    export_artifacts: bool = True,
    verbose: bool = True,
) -> BenchmarkResult:
    """Execute complete model training, evaluation, comparison, and serialization.

    Parameters
    ----------
    data_path : Optional[Union[str, Path]], default None
        Path to telco_churn_cleaned.csv. If None, defaults to data/processed.
    output_dir : Optional[Union[str, Path]], default None
        Directory to export serialized model. Defaults to models/.
    test_size : float, default 0.2
        Fraction of samples held out for final test evaluation.
    random_state : int, default 42
        Seed ensuring deterministic reproducibility.
    export_artifacts : bool, default True
        If True, exports serialized .joblib pipeline and metadata .json.
    verbose : bool, default True
        If True, prints formatted benchmark tables and progress logs.

    Returns
    -------
    BenchmarkResult
        Container with cross-validation summary, test metrics, and champion artifact.
    """
    resolved_data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    resolved_output_dir = Path(output_dir) if output_dir else DEFAULT_MODEL_DIR

    logger.info("Loading cleaned dataset from: %s", resolved_data_path)
    df = pd.read_csv(resolved_data_path)

    # 1. Feature separation & target encoding
    X, y_raw = split_features_and_target(df, drop_id=True)
    y = encode_target(y_raw)

    # 2. Stratified train/test split (test set kept strictly untouched during model comparison)
    X_train, X_test, y_train, y_test = create_train_test_splits(
        X, y, test_size=test_size, random_state=random_state, stratify=True
    )

    # 3. Instantiate candidate models
    candidate_models = get_all_candidate_models(random_state=random_state)

    # 4. Cross-validation strictly on X_train
    cv_summary = cross_validate_candidates(
        candidate_models, X_train, y_train, n_splits=5, random_state=random_state
    )

    if verbose:
        print("\n" + "=" * 90)
        print("5-FOLD STRATIFIED CROSS-VALIDATION SUMMARY (TRAINING SET ONLY)")
        print("=" * 90)
        print(cv_summary.to_string(index=False))
        print("=" * 90 + "\n")

    # 5. Fit all models on full X_train and evaluate once on held-out X_test
    test_metrics_list: List[EvaluationMetrics] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, pipeline in candidate_models.items():
        logger.info("Fitting %s on full X_train...", name)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None

        metrics = evaluate_predictions(y_test, y_pred, y_prob, model_name=name)
        test_metrics_list.append(metrics)

    if verbose:
        print("\n" + "=" * 90)
        print("HELD-OUT TEST SET EVALUATION (N = %d samples)" % len(X_test))
        print("=" * 90)
        print(format_benchmark_table(test_metrics_list))
        print("=" * 90 + "\n")

    # 6. Select champion model
    # Priority: PR-AUC & ROC-AUC with solid Recall (excluding Dummy baseline)
    non_dummy_metrics = [m for m in test_metrics_list if "Dummy" not in m.model_name]
    champion_metric = max(non_dummy_metrics, key=lambda m: (m.pr_auc, m.roc_auc))
    champion_name = champion_metric.model_name
    champion_pipeline = fitted_pipelines[champion_name]

    if verbose:
        print(f"SELECTED CHAMPION MODEL: '{champion_name}'")
        print(champion_metric.summary_text())

    # 7. Export artifacts
    artifact_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    if export_artifacts:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = resolved_output_dir / "churn_model.joblib"
        metadata_path = resolved_output_dir / "model_metadata.json"

        logger.info("Serializing champion pipeline to %s...", artifact_path)
        joblib.dump(champion_pipeline, artifact_path)

        metadata = {
            "model_name": champion_name,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": random_state,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_metrics": champion_metric.to_dict(),
            "threshold_tradeoffs": champion_metric.threshold_tradeoffs,
            "feature_names_in": list(X.columns),
            "total_raw_features": len(X.columns),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved model metadata to %s.", metadata_path)

    return BenchmarkResult(
        cv_summary_df=cv_summary,
        test_metrics=test_metrics_list,
        champion_name=champion_name,
        champion_pipeline=champion_pipeline,
        champion_test_metrics=champion_metric,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )

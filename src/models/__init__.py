"""Model training, evaluation, and benchmarking sub-package.

Exposes model factories, evaluation metrics, and benchmarking runners.
"""

from src.models.registry import (
    get_model_pipeline,
    get_all_candidate_models,
    RANDOM_SEED,
)
from src.models.evaluation import (
    EvaluationMetrics,
    evaluate_predictions,
    evaluate_threshold_tradeoffs,
    select_optimal_threshold,
    format_benchmark_table,
)
from src.models.trainer import (
    cross_validate_candidates,
    select_champion_from_cv,
    select_threshold_from_oof,
    run_model_benchmarking,
    BenchmarkResult,
    CV_SPLIT_SEED,
    FRESH_HOLDOUT_SEED,
)

__all__ = [
    "get_model_pipeline",
    "get_all_candidate_models",
    "RANDOM_SEED",
    "EvaluationMetrics",
    "evaluate_predictions",
    "evaluate_threshold_tradeoffs",
    "select_optimal_threshold",
    "format_benchmark_table",
    "cross_validate_candidates",
    "select_champion_from_cv",
    "select_threshold_from_oof",
    "run_model_benchmarking",
    "BenchmarkResult",
    "CV_SPLIT_SEED",
    "FRESH_HOLDOUT_SEED",
]

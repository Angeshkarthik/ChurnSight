"""Explainable AI (XAI) modules utilizing SHAP for global and local interpretability."""

from src.explainability.shap_explainer import (
    ChurnShapExplainer,
    CustomerExplanation,
    GlobalImportanceSummary,
    LocalFeatureContribution,
    explain_representative_customers,
    format_readable_feature,
    generate_contribution_reason,
    load_churn_model,
    load_representative_data,
    prepare_single_customer_df,
    run_global_explainability,
    validate_raw_feature_schema,
)

__all__ = [
    "ChurnShapExplainer",
    "CustomerExplanation",
    "GlobalImportanceSummary",
    "LocalFeatureContribution",
    "explain_representative_customers",
    "format_readable_feature",
    "generate_contribution_reason",
    "load_churn_model",
    "load_representative_data",
    "prepare_single_customer_df",
    "run_global_explainability",
    "validate_raw_feature_schema",
]

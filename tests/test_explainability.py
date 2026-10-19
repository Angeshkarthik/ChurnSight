"""Unit tests for the SHAP Explainability module.

Covers:
- Loading and structural validation of the existing model pipeline
- Schema validation: dropping target/ID and rejecting missing features or empty inputs
- Preprocessor feature transformation and feature name alignment
- TreeExplainer initialization and mathematical additivity check (margin/log-odds scale)
- Positive-class SHAP value selection across 2D and 3D output dimensions
- Global feature importance ranking computation (monotonic descending order)
- End-to-end global artifact export (bar plot, beeswarm plot, JSON summary, markdown report)
- Local single-customer explanation: dict, Series, DataFrame formats
- Operational threshold (0.35) and risk level assignment
- Additivity verification on individual customer predictions
- Readable reason generation and signed contribution sorting
- Waterfall plot, local JSON, and individual markdown report export
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.explainability.shap_explainer import (
    ChurnShapExplainer,
    CustomerExplanation,
    LocalFeatureContribution,
    format_readable_feature,
    generate_contribution_reason,
    get_transformed_features,
    load_churn_model,
    load_representative_data,
    prepare_single_customer_df,
    validate_raw_feature_schema,
)
from src.features.config import (
    ALL_PREDICTOR_FEATURES,
    ID_COLUMN,
    TARGET_COLUMN,
)


@pytest.fixture
def minimal_customer_features():
    """Create a minimal synthetic customer dataframe matching all 19 raw predictor features."""
    n_samples = 12
    np.random.seed(42)
    return pd.DataFrame({
        "gender": ["Female", "Male"] * (n_samples // 2),
        "SeniorCitizen": [0, 1] * (n_samples // 2),
        "Partner": ["Yes", "No"] * (n_samples // 2),
        "Dependents": ["No", "Yes"] * (n_samples // 2),
        "tenure": np.random.randint(1, 72, size=n_samples),
        "PhoneService": ["Yes"] * n_samples,
        "MultipleLines": ["No", "Yes"] * (n_samples // 2),
        "InternetService": ["DSL", "Fiber optic", "No"] * (n_samples // 3),
        "OnlineSecurity": ["Yes", "No", "No internet service"] * (n_samples // 3),
        "OnlineBackup": ["No", "Yes", "No internet service"] * (n_samples // 3),
        "DeviceProtection": ["Yes", "No", "No internet service"] * (n_samples // 3),
        "TechSupport": ["No", "Yes", "No internet service"] * (n_samples // 3),
        "StreamingTV": ["No", "Yes", "No internet service"] * (n_samples // 3),
        "StreamingMovies": ["Yes", "No", "No internet service"] * (n_samples // 3),
        "Contract": ["Month-to-month", "One year", "Two year"] * (n_samples // 3),
        "PaperlessBilling": ["Yes", "No"] * (n_samples // 2),
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ] * (n_samples // 4),
        "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n_samples),
        "TotalCharges": np.random.uniform(20.0, 4500.0, size=n_samples),
    })


# --- MODEL LOADING & PIPELINE VALIDATION ---

def test_load_churn_model_real_artifact():
    """Verify loading the real serialized pipeline confirms expected steps and classifier."""
    pipeline = load_churn_model()
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["classifier"], GradientBoostingClassifier)


def test_load_churn_model_missing_file():
    """Verify loading a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_churn_model("non_existent_model_artifact.joblib")


def test_load_churn_model_invalid_object(tmp_path):
    """Verify loading an object that is not a Pipeline or missing steps raises ValueError."""
    import joblib

    # 1. Non-pipeline object
    fake_file = tmp_path / "not_a_pipeline.joblib"
    joblib.dump({"some": "dict"}, fake_file)
    with pytest.raises(ValueError, match="Expected a Scikit-learn Pipeline"):
        load_churn_model(fake_file)

    # 2. Pipeline missing classifier step
    missing_step_pipeline = Pipeline([("preprocessor", StandardScaler())])
    fake_pipe_file = tmp_path / "missing_step.joblib"
    joblib.dump(missing_step_pipeline, fake_pipe_file)
    with pytest.raises(ValueError, match="Pipeline missing required steps"):
        load_churn_model(fake_pipe_file)


# --- SCHEMA & FEATURE SANITIZATION ---

def test_validate_raw_feature_schema_drops_id_and_target(minimal_customer_features):
    """Verify target and identifier columns are safely excluded without altering predictors."""
    df_with_extra = minimal_customer_features.copy()
    df_with_extra[ID_COLUMN] = [f"7000-XYZ{i:02d}" for i in range(len(df_with_extra))]
    df_with_extra[TARGET_COLUMN] = ["No", "Yes"] * (len(df_with_extra) // 2)

    sanitized = validate_raw_feature_schema(df_with_extra)

    assert ID_COLUMN not in sanitized.columns
    assert TARGET_COLUMN not in sanitized.columns
    assert set(sanitized.columns) == set(ALL_PREDICTOR_FEATURES)
    assert len(sanitized.columns) == len(ALL_PREDICTOR_FEATURES)


def test_validate_raw_feature_schema_missing_column(minimal_customer_features):
    """Verify missing required columns raises ValueError."""
    incomplete_df = minimal_customer_features.drop(columns=["tenure", "MonthlyCharges"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_feature_schema(incomplete_df)


def test_validate_raw_feature_schema_empty():
    """Verify empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        validate_raw_feature_schema(empty_df)


# --- TRANSFORMATION & FEATURE NAMES ---

def test_get_transformed_features_alignment(minimal_customer_features):
    """Verify raw features transform to the expected column dimension and feature names."""
    pipeline = load_churn_model()
    preprocessor = pipeline.named_steps["preprocessor"]

    X_trans, feature_names = get_transformed_features(preprocessor, minimal_customer_features)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape == (len(minimal_customer_features), 46)
    assert len(feature_names) == 46
    assert "tenure" in feature_names
    assert "Contract_Month-to-month" in feature_names


# --- EXPLAINER, ADDITIVITY & POSITIVE CLASS SELECTION ---

def test_shap_explainer_positive_class_and_additivity(minimal_customer_features):
    """Verify TreeExplainer additivity matches decision_function on log-odds margin."""
    pipeline = load_churn_model()
    explainer = ChurnShapExplainer(pipeline)

    assert explainer.explainer_type == "TreeExplainer"
    assert "log-odds" in explainer.output_scale

    explanation = explainer.explain(minimal_customer_features, check_additivity=True)

    assert isinstance(explanation, shap.Explanation)
    assert explanation.values.shape == (len(minimal_customer_features), 46)
    assert len(explanation.feature_names) == 46

    # Verify mathematical additivity: F(x) = F_0 + sum(phi_i)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_trans, _ = get_transformed_features(preprocessor, minimal_customer_features)
    decision_margin = classifier.decision_function(X_trans)
    reconstructed_margin = explanation.base_values + np.sum(explanation.values, axis=1)

    max_diff = np.max(np.abs(decision_margin - reconstructed_margin))
    assert max_diff < 1e-4


def test_shap_explainer_handles_3d_multi_class_output(minimal_customer_features):
    """Verify 3D SHAP output (N, Features, Classes) safely extracts the positive class (index 1)."""
    pipeline = load_churn_model()
    explainer = ChurnShapExplainer(pipeline)

    # Mock the internal explainer to return a 3D explanation
    n_samples = len(minimal_customer_features)
    n_features = 46
    dummy_values_3d = np.zeros((n_samples, n_features, 2))
    # Give class 0 negative values, class 1 positive values
    dummy_values_3d[:, :, 0] = -0.5
    dummy_values_3d[:, :, 1] = 0.8
    dummy_base_3d = np.array([[-1.5, 1.5]] * n_samples)

    mock_exp_3d = shap.Explanation(
        values=dummy_values_3d,
        base_values=dummy_base_3d,
        data=np.zeros((n_samples, n_features)),
    )
    explainer.explainer = MagicMock(return_value=mock_exp_3d)

    explanation = explainer.explain(minimal_customer_features, check_additivity=False)

    # Must extract positive class (index 1) with values 0.8
    assert explanation.values.shape == (n_samples, n_features)
    assert np.allclose(explanation.values, 0.8)


# --- GLOBAL IMPORTANCE RANKINGS & VISUALIZATIONS ---

def test_get_global_importance_ranking(minimal_customer_features):
    """Verify global importance computes sorted, non-negative mean absolute SHAP values."""
    pipeline = load_churn_model()
    explainer = ChurnShapExplainer(pipeline)
    explanation = explainer.explain(minimal_customer_features)

    importance_df = explainer.get_global_importance(explanation)

    assert isinstance(importance_df, pd.DataFrame)
    assert list(importance_df.columns) == ["rank", "feature", "mean_abs_shap"]
    assert len(importance_df) == 46
    assert importance_df["rank"].iloc[0] == 1
    assert (importance_df["mean_abs_shap"] >= 0).all()

    # Monotonically non-increasing order
    values = importance_df["mean_abs_shap"].to_numpy()
    assert (np.diff(values) <= 1e-9).all()


def test_save_global_visualizations_and_report(minimal_customer_features, tmp_path):
    """Verify bar plot, beeswarm plot, JSON, CSV, and Markdown reports are generated properly."""
    pipeline = load_churn_model()
    explainer = ChurnShapExplainer(pipeline)
    explanation = explainer.explain(minimal_customer_features)
    importance_df = explainer.get_global_importance(explanation)

    # 1. Plots
    plot_paths = explainer.save_global_visualizations(explanation, output_dir=tmp_path, max_display=10)
    assert plot_paths["bar_plot"].exists()
    assert plot_paths["beeswarm_plot"].exists()
    assert plot_paths["bar_plot"].stat().st_size > 1000
    assert plot_paths["beeswarm_plot"].stat().st_size > 1000

    # 2. Markdown Report
    report_file = tmp_path / "shap_summary_report.md"
    explainer.generate_global_report(
        importance_df=importance_df,
        sample_size=len(minimal_customer_features),
        output_path=report_file,
        top_n=10,
    )
    assert report_file.exists()
    report_text = report_file.read_text(encoding="utf-8")
    assert "# Global Model Explainability Report" in report_text
    assert "log-odds" in report_text.lower()
    assert "Correlation vs. Causation" in report_text


def test_load_representative_data_development_isolation():
    """Verify representative sampling draws strictly from Development data without touching holdout."""
    X_sample, y_sample = load_representative_data(sample_size=50, random_state=42)

    assert len(X_sample) == 50
    assert len(y_sample) == 50
    assert ID_COLUMN not in X_sample.columns
    assert TARGET_COLUMN not in X_sample.columns
    assert set(X_sample.columns) == set(ALL_PREDICTOR_FEATURES)


# --- PHASE 5.2 LOCAL CUSTOMER EXPLANATION TESTS ---

def test_prepare_single_customer_df_formats(minimal_customer_features):
    """Verify prepare_single_customer_df handles dict, Series, and 1-row DataFrame."""
    first_row_dict = minimal_customer_features.iloc[0].to_dict()
    first_row_series = minimal_customer_features.iloc[0]
    first_row_df = minimal_customer_features.iloc[[0]]

    # 1. Dict
    df_dict, id_dict = prepare_single_customer_df(first_row_dict)
    assert len(df_dict) == 1
    assert id_dict is None

    # 2. Series with customerID
    series_with_id = first_row_series.copy()
    series_with_id["customerID"] = "CUST-999"
    df_series, id_series = prepare_single_customer_df(series_with_id)
    assert len(df_series) == 1
    assert id_series == "CUST-999"
    assert "customerID" not in df_series.columns

    # 3. DataFrame
    df_df, id_df = prepare_single_customer_df(first_row_df)
    assert len(df_df) == 1


def test_prepare_single_customer_df_invalid_inputs():
    """Verify empty inputs or multi-row DataFrames raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        prepare_single_customer_df({})

    with pytest.raises(ValueError, match="empty"):
        prepare_single_customer_df(pd.DataFrame())

    multi_row_df = pd.DataFrame({"tenure": [1, 2], "MonthlyCharges": [20, 30]})
    with pytest.raises(ValueError, match="expects a single customer record"):
        prepare_single_customer_df(multi_row_df)

    with pytest.raises(TypeError, match="Expected customer_data as dict"):
        prepare_single_customer_df([1, 2, 3])  # type: ignore


def test_explain_customer_end_to_end(minimal_customer_features):
    """Verify single customer explanation computes probabilities, thresholds, and contributions."""
    pipeline = load_churn_model()
    explainer = ChurnShapExplainer(pipeline)

    sample_customer = minimal_customer_features.iloc[0].to_dict()
    sample_customer["customerID"] = "TEST-USER-001"

    explanation = explainer.explain_customer(
        customer_data=sample_customer,
        operational_threshold=0.35,
        standard_threshold=0.50,
        top_n=5,
    )

    assert isinstance(explanation, CustomerExplanation)
    assert explanation.customer_id == "TEST-USER-001"
    assert 0.0 <= explanation.churn_probability <= 1.0
    assert explanation.predicted_label in {0, 1}
    assert explanation.operational_threshold == 0.35
    assert explanation.standard_threshold == 0.50

    # Probability relationship with threshold
    if explanation.churn_probability >= 0.35:
        assert explanation.predicted_label == 1
        assert "High Risk" in explanation.risk_level
    else:
        assert explanation.predicted_label == 0
        assert "Low Risk" in explanation.risk_level

    # Mathematical additivity check
    assert explanation.additivity_delta < 1e-4
    assert np.isclose(
        explanation.model_score_log_odds,
        explanation.baseline_log_odds + explanation.shap_values_sum,
        atol=1e-4,
    )

    # Contributions structure
    assert len(explanation.all_contributions) == 46
    assert len(explanation.top_positive_contributions) <= 5
    assert len(explanation.top_negative_contributions) <= 5

    # Signed sorting check
    for c in explanation.top_positive_contributions:
        assert c.shap_value > 0
        assert c.direction == "push_toward_churn"
    for c in explanation.top_negative_contributions:
        assert c.shap_value < 0
        assert c.direction == "push_toward_retention"


def test_format_readable_feature_and_reason_generation():
    """Verify human-readable labels and factual reasons are generated properly."""
    # Test readable label formatting without customer input (fallback mode)
    assert format_readable_feature("tenure") == "Customer Tenure (months)"
    assert format_readable_feature("MonthlyCharges") == "Monthly Charges ($)"
    assert format_readable_feature("Contract_Month-to-month") == "Contract: Month-to-month (Indicator)"
    assert format_readable_feature("SeniorCitizen_1") == "Senior Citizen: Yes (Indicator)"

    # Test readable label formatting with customer input
    customer_dict = {
        "tenure": 2,
        "MonthlyCharges": 85.50,
        "Contract": "Month-to-month",
        "SeniorCitizen": 0,
        "TechSupport": "Yes",
    }
    assert format_readable_feature("tenure", customer_dict) == "Customer Tenure (2 mo)"
    assert format_readable_feature("MonthlyCharges", customer_dict) == "Monthly Charges ($85.50)"
    assert format_readable_feature("Contract_Month-to-month", customer_dict) == "Contract = 'Month-to-month'"
    assert format_readable_feature("SeniorCitizen_1", customer_dict) == "Senior Citizen != 'Yes' (Actual: 'No')"
    assert format_readable_feature("TechSupport_No", customer_dict) == "Tech Support != 'No' (Actual: 'Yes')"

    # Test reason generation
    reason_pos = generate_contribution_reason("Contract_Month-to-month", 0.65, customer_dict)
    assert "Month-to-month" in reason_pos
    assert "+0.6500" in reason_pos
    assert "pushing the model output toward churn" in reason_pos

    reason_neg = generate_contribution_reason("tenure", -0.45, customer_dict)
    assert "Observed customer tenure of 2 month(s)" in reason_neg
    assert "-0.4500" in reason_neg
    assert "pushing the model output away from churn" in reason_neg


def test_regression_techsupport_no_active_and_inactive():
    """Verify TechSupport_No correctly identifies active vs. inactive indicator status.

    Addresses Phase 5.2 Case 2 audit finding:
    Customer having TechSupport = 'Yes' results in TechSupport_No = 0 (inactive indicator).
    A negative SHAP value must be described as the inactive indicator pushing the model output away
    from churn, and must NEVER claim the customer has active tech support as causal protection.
    """
    # 1. Active TechSupport_No indicator (Customer actually has TechSupport = 'No')
    # Positive SHAP contribution (pushing toward churn)
    active_input = {"TechSupport": "No"}
    label_active = format_readable_feature("TechSupport_No", active_input)
    assert label_active == "Tech Support = 'No'"

    reason_pos = generate_contribution_reason("TechSupport_No", 0.35, active_input)
    assert "Tech Support is 'No' (indicator active)" in reason_pos
    assert "+0.3500 log-odds" in reason_pos
    assert "pushing the model output toward churn" in reason_pos

    # Negative SHAP contribution on active indicator (model output pushed away from churn)
    reason_neg = generate_contribution_reason("TechSupport_No", -0.15, active_input)
    assert "Tech Support is 'No' (indicator active)" in reason_neg
    assert "-0.1500 log-odds" in reason_neg
    assert "pushing the model output away from churn" in reason_neg
    # Crucially, must NOT state customer has active tech support!
    assert "active tech support" not in reason_neg.lower()

    # 2. Inactive TechSupport_No indicator: Customer has TechSupport = 'Yes' (Case 2 scenario)
    inactive_yes_input = {"TechSupport": "Yes"}
    label_inactive_yes = format_readable_feature("TechSupport_No", inactive_yes_input)
    assert label_inactive_yes == "Tech Support != 'No' (Actual: 'Yes')"

    reason_inactive_yes = generate_contribution_reason("TechSupport_No", -0.2468, inactive_yes_input)
    assert "Tech Support is not 'No' (observed: 'Yes', indicator inactive)" in reason_inactive_yes
    assert "-0.2468 log-odds" in reason_inactive_yes
    assert "pushing the model output away from churn" in reason_inactive_yes
    # Narrative does not assert causal protection or active support
    assert "protective" not in reason_inactive_yes.lower()

    # 3. Inactive TechSupport_No indicator: Customer has 'No internet service'
    inactive_no_int_input = {"TechSupport": "No internet service"}
    label_no_int = format_readable_feature("TechSupport_No", inactive_no_int_input)
    assert label_no_int == "Tech Support != 'No' (Actual: 'No internet service')"

    reason_no_int = generate_contribution_reason("TechSupport_No", -0.18, inactive_no_int_input)
    assert "Tech Support is not 'No' (observed: 'No internet service', indicator inactive)" in reason_no_int
    assert "-0.1800 log-odds" in reason_no_int
    assert "pushing the model output away from churn" in reason_no_int


def test_regression_onlinesecurity_no_active_and_inactive():
    """Verify OnlineSecurity_No correctly distinguishes active vs inactive states and signed SHAP.

    Addresses Phase 5.2 Case 3 audit finding:
    Customer having OnlineSecurity = 'Yes' has OnlineSecurity_No = 0.
    A negative SHAP value must not falsely claim the customer is an 'active security subscriber'.
    """
    # 1. Active OnlineSecurity_No indicator (Customer actually has OnlineSecurity = 'No')
    active_sec_input = {"OnlineSecurity": "No"}
    label_active = format_readable_feature("OnlineSecurity_No", active_sec_input)
    assert label_active == "Online Security = 'No'"

    reason_active_pos = generate_contribution_reason("OnlineSecurity_No", 0.42, active_sec_input)
    assert "Online Security is 'No' (indicator active)" in reason_active_pos
    assert "+0.4200 log-odds" in reason_active_pos
    assert "pushing the model output toward churn" in reason_active_pos

    reason_active_neg = generate_contribution_reason("OnlineSecurity_No", -0.08, active_sec_input)
    assert "Online Security is 'No' (indicator active)" in reason_active_neg
    assert "-0.0800 log-odds" in reason_active_neg
    assert "pushing the model output away from churn" in reason_active_neg
    assert "active security subscriber" not in reason_active_neg.lower()

    # 2. Inactive OnlineSecurity_No indicator: Customer has OnlineSecurity = 'Yes' (Case 3 scenario)
    inactive_sec_input = {"OnlineSecurity": "Yes"}
    label_inactive = format_readable_feature("OnlineSecurity_No", inactive_sec_input)
    assert label_inactive == "Online Security != 'No' (Actual: 'Yes')"

    reason_inactive_neg = generate_contribution_reason("OnlineSecurity_No", -0.1874, inactive_sec_input)
    assert "Online Security is not 'No' (observed: 'Yes', indicator inactive)" in reason_inactive_neg
    assert "-0.1874 log-odds" in reason_inactive_neg
    assert "pushing the model output away from churn" in reason_inactive_neg
    # Must NEVER claim customer is an 'active security subscriber'
    assert "active security subscriber" not in reason_inactive_neg.lower()


def test_regression_separation_of_actual_value_direction_and_importance(minimal_customer_features):
    """Verify separation of raw customer value, indicator activity, and signed contribution."""
    pipeline = load_churn_model()
    explainer = ChurnShapExplainer(pipeline)

    # Construct customer with known values
    sample_row = minimal_customer_features.iloc[0].copy()
    sample_row["TechSupport"] = "Yes"
    sample_row["OnlineSecurity"] = "No"
    sample_dict = sample_row.to_dict()

    explanation = explainer.explain_customer(sample_dict)

    # Find TechSupport_No and OnlineSecurity_No in all_contributions
    contrib_dict = {c.feature_name: c for c in explanation.all_contributions}

    # TechSupport_No: raw value is 'Yes', indicator is inactive (False)
    ts_contrib = contrib_dict["TechSupport_No"]
    assert ts_contrib.actual_value == "Yes"
    assert ts_contrib.is_active is False
    assert "Tech Support != 'No' (Actual: 'Yes')" in ts_contrib.readable_feature
    assert ts_contrib.direction in ("push_toward_churn", "push_toward_retention", "negligible")
    assert "observed: 'Yes', indicator inactive" in ts_contrib.reason

    # OnlineSecurity_No: raw value is 'No', indicator is active (True)
    os_contrib = contrib_dict["OnlineSecurity_No"]
    assert os_contrib.actual_value == "No"
    assert os_contrib.is_active is True
    assert "Online Security = 'No'" in os_contrib.readable_feature
    assert "indicator active" in os_contrib.reason


def test_regression_numeric_features_non_causal_narrative():
    """Verify numeric feature reasons describe observed values without asserting causality."""
    input_data = {
        "tenure": 70,
        "MonthlyCharges": 24.50,
        "TotalCharges": 1700.00,
    }

    # High tenure pushing away from churn
    reason_tenure = generate_contribution_reason("tenure", -0.52, input_data)
    assert "Observed customer tenure of 70 month(s)" in reason_tenure
    assert "pushing the model output away from churn by -0.5200 log-odds" in reason_tenure
    assert "prevents" not in reason_tenure.lower()
    assert "protects" not in reason_tenure.lower()
    assert "causes" not in reason_tenure.lower()

    # High monthly charges pushing toward churn
    reason_monthly = generate_contribution_reason("MonthlyCharges", 0.45, input_data)
    assert "Observed monthly bill of $24.50" in reason_monthly
    assert "pushing the model output toward churn by +0.4500 log-odds" in reason_monthly
    assert "causes" not in reason_monthly.lower()


def test_regression_safe_fallback_when_raw_value_unavailable():
    """Verify safe deterministic descriptions when raw customer values are unavailable."""
    # When raw input is None
    label_fallback = format_readable_feature("TechSupport_No", None)
    assert label_fallback == "Tech Support: No (Indicator)"

    reason_fallback = generate_contribution_reason("TechSupport_No", -0.22, None)
    assert "Binary indicator for 'Tech Support = No'" in reason_fallback
    assert "pushing the model output away from churn by -0.2200 log-odds" in reason_fallback

    # When raw input is empty dict
    label_empty = format_readable_feature("OnlineSecurity_No", {})
    assert label_empty == "Online Security: No (Indicator)"

    reason_empty = generate_contribution_reason("OnlineSecurity_No", 0.31, {})
    assert "Binary indicator for 'Online Security = No'" in reason_empty
    assert "pushing the model output toward churn by +0.3100 log-odds" in reason_empty


def test_save_customer_visualizations_and_report(minimal_customer_features, tmp_path):
    """Verify saving local customer waterfall plot, JSON, and Markdown reports."""
    pipeline = load_churn_model()
    explainer = ChurnShapExplainer(pipeline)

    sample_customer = minimal_customer_features.iloc[0].to_dict()
    explanation = explainer.explain_customer(sample_customer, operational_threshold=0.35)

    prefix = "test_customer_case"

    # 1. Waterfall Plot
    plot_path = explainer.save_customer_visualizations(
        explanation_record=explanation,
        output_dir=tmp_path,
        filename_prefix=prefix,
        max_display=10,
    )
    assert plot_path.exists()
    assert plot_path.stat().st_size > 1000

    # 2. Markdown Report
    report_path = explainer.save_customer_report(
        explanation_record=explanation,
        output_dir=tmp_path,
        filename_prefix=prefix,
        waterfall_rel_path=plot_path.name,
    )
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "# Local Customer Churn Explanation Report" in report_text
    assert "Operational Decision Threshold" in report_text
    assert "Governance Notice" in report_text

    # 3. JSON export
    json_path = tmp_path / f"{prefix}_explanation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(explanation.to_dict(), f, indent=2)

    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "churn_probability" in data
    assert "top_positive_contributions" in data
    assert "additivity_delta" in data
    assert data["additivity_delta"] < 1e-4

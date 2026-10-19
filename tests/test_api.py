"""Integration and Unit Tests for FastAPI Customer Intelligence Backend.

Covers all Phase 5.3 verification invariants:
1. Health endpoint when model is ready.
2. Health endpoint when model loading fails.
3. Valid prediction request.
4. Valid explanation request.
5. Probability bounds and threshold classification (0.35).
6. Missing required feature validation.
7. Unexpected feature rejection (extra fields / leakage prevention).
8. Invalid categorical value rejection.
9. Invalid numeric value rejection (out-of-range bounds).
10. Non-finite numeric input rejection (NaN / Infinity).
11. Model-loading failure behavior (503 Service Unavailable).
12. Probability consistency with underlying Scikit-learn Pipeline.
13. Probability and label consistency between /explain and /predict.
14. SHAP additivity, scale guarantees, and margin reconstruction.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import ModelService, get_model_service
from src.explainability.shap_explainer import load_churn_model


@pytest.fixture
def valid_high_risk_customer() -> Dict[str, Any]:
    """Synthetic customer record exhibiting high churn risk characteristics."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 95.50,
    }


@pytest.fixture
def valid_low_risk_customer() -> Dict[str, Any]:
    """Synthetic customer record exhibiting low churn risk characteristics."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 55.20,
        "TotalCharges": 3588.0,
    }


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient with default model service."""
    # Ensure any previous dependency overrides are cleared
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# =============================================================================
# 1 & 2. HEALTH ENDPOINT TESTS
# =============================================================================

def test_health_when_model_ready(client: TestClient) -> None:
    """Verify /health reports healthy status and model readiness when loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_ready"] is True
    assert data["model_name"] == "Gradient Boosting"
    assert data["operational_threshold"] == 0.35
    assert data["total_features_expected"] == 19


def test_health_when_model_loading_fails(client: TestClient) -> None:
    """Verify /health reports degraded status and model_ready=False when unready."""
    unready_service = ModelService()
    unready_service.is_ready = False

    app.dependency_overrides[get_model_service] = lambda: unready_service
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_ready"] is False
    finally:
        app.dependency_overrides.clear()


# =============================================================================
# 3 & 4. VALID PREDICTION AND EXPLANATION ENDPOINTS
# =============================================================================

def test_valid_prediction_request(client: TestClient, valid_high_risk_customer: Dict[str, Any]) -> None:
    """Verify /predict accepts valid 19 features and returns complete schema."""
    response = client.post("/predict", json=valid_high_risk_customer)
    assert response.status_code == 200
    data = response.json()

    assert "churn_probability" in data
    assert "retention_probability" in data
    assert "predicted_class" in data
    assert "threshold_used" in data
    assert "risk_level" in data
    assert "model_name" in data

    assert isinstance(data["churn_probability"], float)
    assert isinstance(data["retention_probability"], float)
    assert data["predicted_class"] in (0, 1)
    assert data["threshold_used"] == 0.35
    assert data["model_name"] == "Gradient Boosting"


def test_valid_explanation_request(client: TestClient, valid_high_risk_customer: Dict[str, Any]) -> None:
    """Verify /explain accepts valid customer and returns full SHAP attribution."""
    response = client.post("/explain", json=valid_high_risk_customer)
    assert response.status_code == 200
    data = response.json()

    assert "churn_probability" in data
    assert "retention_probability" in data
    assert "predicted_class" in data
    assert "threshold_used" in data
    assert "baseline_model_output" in data
    assert "baseline_probability" in data
    assert "model_score_log_odds" in data
    assert "shap_values_sum" in data
    assert "additivity_delta" in data
    assert "output_scale" in data
    assert "scale_explanation" in data
    assert "contributions" in data
    assert "top_positive_contributions" in data
    assert "top_negative_contributions" in data
    assert "summary_reasons" in data
    assert "governance_notice" in data

    assert len(data["contributions"]) > 0
    first_contrib = data["contributions"][0]
    assert "feature_name" in first_contrib
    assert "readable_feature" in first_contrib
    assert "shap_value" in first_contrib
    assert "direction" in first_contrib
    assert "magnitude" in first_contrib
    assert "reason" in first_contrib


# =============================================================================
# 5. PROBABILITY BOUNDS AND THRESHOLD CLASSIFICATION
# =============================================================================

def test_probability_bounds_and_threshold_classification(
    client: TestClient,
    valid_high_risk_customer: Dict[str, Any],
    valid_low_risk_customer: Dict[str, Any],
) -> None:
    """Verify probability mathematical bounds and discrete thresholding at 0.35."""
    # Test High-Risk Sample
    resp_high = client.post("/predict", json=valid_high_risk_customer)
    assert resp_high.status_code == 200
    data_high = resp_high.json()

    assert 0.0 <= data_high["churn_probability"] <= 1.0
    assert 0.0 <= data_high["retention_probability"] <= 1.0
    assert data_high["churn_probability"] + data_high["retention_probability"] == pytest.approx(1.0, abs=1e-5)
    assert data_high["churn_probability"] >= 0.35
    assert data_high["predicted_class"] == 1
    assert data_high["risk_level"] == "High Risk (Operational Churn Alert)"

    # Test Low-Risk Sample
    resp_low = client.post("/predict", json=valid_low_risk_customer)
    assert resp_low.status_code == 200
    data_low = resp_low.json()

    assert 0.0 <= data_low["churn_probability"] <= 1.0
    assert 0.0 <= data_low["retention_probability"] <= 1.0
    assert data_low["churn_probability"] + data_low["retention_probability"] == pytest.approx(1.0, abs=1e-5)
    assert data_low["churn_probability"] < 0.35
    assert data_low["predicted_class"] == 0
    assert data_low["risk_level"] == "Low Risk (Retained)"


# =============================================================================
# 6, 7, 8, 9, 10. REQUEST VALIDATION AND ERROR HANDLING
# =============================================================================

def test_missing_required_feature(client: TestClient, valid_high_risk_customer: Dict[str, Any]) -> None:
    """Verify omitting any required feature triggers 422 Unprocessable Entity."""
    bad_payload = valid_high_risk_customer.copy()
    del bad_payload["MonthlyCharges"]

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_unexpected_feature_rejected(client: TestClient, valid_high_risk_customer: Dict[str, Any]) -> None:
    """Verify passing unexpected columns (e.g., customerID, Churn, or arbitrary fields) triggers 422."""
    # Test 1: customerID injection (leakage prevention)
    bad_payload_id = valid_high_risk_customer.copy()
    bad_payload_id["customerID"] = "9999-XYZAB"
    resp_id = client.post("/predict", json=bad_payload_id)
    assert resp_id.status_code == 422

    # Test 2: Target Churn injection
    bad_payload_target = valid_high_risk_customer.copy()
    bad_payload_target["Churn"] = "Yes"
    resp_target = client.post("/predict", json=bad_payload_target)
    assert resp_target.status_code == 422

    # Test 3: Arbitrary extra field
    bad_payload_extra = valid_high_risk_customer.copy()
    bad_payload_extra["credit_score"] = 720
    resp_extra = client.post("/predict", json=bad_payload_extra)
    assert resp_extra.status_code == 422


def test_invalid_categorical_value(client: TestClient, valid_high_risk_customer: Dict[str, Any]) -> None:
    """Verify unapproved category strings trigger 422 validation failure."""
    bad_payload = valid_high_risk_customer.copy()
    bad_payload["Contract"] = "3-Year Custom Enterprise"

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_invalid_numeric_value_out_of_range(client: TestClient, valid_high_risk_customer: Dict[str, Any]) -> None:
    """Verify negative tenure or negative charges trigger 422 validation failure."""
    # Negative tenure
    bad_tenure = valid_high_risk_customer.copy()
    bad_tenure["tenure"] = -3
    assert client.post("/predict", json=bad_tenure).status_code == 422

    # Negative MonthlyCharges
    bad_charges = valid_high_risk_customer.copy()
    bad_charges["MonthlyCharges"] = -10.0
    assert client.post("/predict", json=bad_charges).status_code == 422


def test_non_finite_numeric_input(client: TestClient, valid_high_risk_customer: Dict[str, Any]) -> None:
    """Verify NaN, +Inf, and -Inf inputs are rejected with 422 Unprocessable Content."""
    import json

    # 1. Raw NaN serialized with allow_nan=True
    bad_nan = valid_high_risk_customer.copy()
    bad_nan["MonthlyCharges"] = float("nan")
    raw_nan = json.dumps(bad_nan, allow_nan=True)
    resp_nan = client.post("/predict", content=raw_nan, headers={"Content-Type": "application/json"})
    assert resp_nan.status_code == 422

    # 2. Raw +Infinity serialized with allow_nan=True
    bad_inf = valid_high_risk_customer.copy()
    bad_inf["TotalCharges"] = float("inf")
    raw_inf = json.dumps(bad_inf, allow_nan=True)
    resp_inf = client.post("/predict", content=raw_inf, headers={"Content-Type": "application/json"})
    assert resp_inf.status_code == 422

    # 3. Raw -Infinity serialized with allow_nan=True
    bad_neginf = valid_high_risk_customer.copy()
    bad_neginf["TotalCharges"] = float("-inf")
    raw_neginf = json.dumps(bad_neginf, allow_nan=True)
    resp_neginf = client.post("/predict", content=raw_neginf, headers={"Content-Type": "application/json"})
    assert resp_neginf.status_code == 422

    # 4. String representations like "nan" or "Infinity"
    bad_str = valid_high_risk_customer.copy()
    bad_str["MonthlyCharges"] = "nan"
    resp_str = client.post("/predict", json=bad_str)
    assert resp_str.status_code == 422


# =============================================================================
# 11. MODEL LOADING FAILURE BEHAVIOR
# =============================================================================

def test_model_loading_failure_behavior(client: TestClient, valid_high_risk_customer: Dict[str, Any]) -> None:
    """Verify /predict and /explain return 503 when model is not ready, without leaking internal details."""
    unready_service = ModelService()
    unready_service.is_ready = False

    app.dependency_overrides[get_model_service] = lambda: unready_service
    try:
        resp_predict = client.post("/predict", json=valid_high_risk_customer)
        assert resp_predict.status_code == 503
        data_p = resp_predict.json()
        assert "unavailable" in data_p["detail"].lower()
        # Verify no traceback or path leakage
        assert "traceback" not in data_p["detail"].lower()
        assert "C:\\" not in data_p["detail"]
        assert "/home" not in data_p["detail"]

        resp_explain = client.post("/explain", json=valid_high_risk_customer)
        assert resp_explain.status_code == 503
        data_e = resp_explain.json()
        assert "unavailable" in data_e["detail"].lower()
    finally:
        app.dependency_overrides.clear()


# =============================================================================
# 12. CONSISTENCY WITH EXISTING MODEL PIPELINE
# =============================================================================

def test_consistency_with_underlying_model(client: TestClient, valid_high_risk_customer: Dict[str, Any]) -> None:
    """Verify API prediction strictly matches direct scikit-learn pipeline calculation."""
    # Direct model pipeline inference
    pipeline = load_churn_model()
    raw_df = pd.DataFrame([valid_high_risk_customer])
    expected_prob = float(pipeline.predict_proba(raw_df)[0, 1])
    expected_class = 1 if expected_prob >= 0.35 else 0

    # API inference
    response = client.post("/predict", json=valid_high_risk_customer)
    assert response.status_code == 200
    data = response.json()

    assert data["churn_probability"] == pytest.approx(expected_prob, abs=1e-6)
    assert data["predicted_class"] == expected_class


# =============================================================================
# 13. CONSISTENCY BETWEEN PREDICTION AND EXPLANATION ENDPOINTS
# =============================================================================

def test_consistency_between_explain_and_predict(
    client: TestClient, valid_high_risk_customer: Dict[str, Any]
) -> None:
    """Verify /explain and /predict yield identical probability, class, and threshold for identical input."""
    resp_predict = client.post("/predict", json=valid_high_risk_customer)
    resp_explain = client.post("/explain", json=valid_high_risk_customer)

    assert resp_predict.status_code == 200
    assert resp_explain.status_code == 200

    pred_data = resp_predict.json()
    exp_data = resp_explain.json()

    assert exp_data["churn_probability"] == pytest.approx(pred_data["churn_probability"], abs=1e-6)
    assert exp_data["predicted_class"] == pred_data["predicted_class"]
    assert exp_data["threshold_used"] == pred_data["threshold_used"]
    assert exp_data["risk_level"] == pred_data["risk_level"]


# =============================================================================
# 14. SHAP CONTRIBUTION SCALE AND ADDITIVITY
# =============================================================================

def test_shap_contribution_scale_and_additivity(
    client: TestClient, valid_high_risk_customer: Dict[str, Any]
) -> None:
    """Verify SHAP log-odds additivity F(x) = E[F(X)] + sum(phi) within machine tolerance."""
    response = client.post("/explain", json=valid_high_risk_customer)
    assert response.status_code == 200
    data = response.json()

    assert data["output_scale"] == "log-odds (raw margin)"
    assert "log-odds" in data["scale_explanation"].lower()

    base_val = data["baseline_model_output"]
    all_contribs = data["contributions"]
    shap_sum = sum(c["shap_value"] for c in all_contribs)
    margin = data["model_score_log_odds"]
    additivity_delta = data["additivity_delta"]

    # Additivity guarantee: base_val + sum(phi) == model_score_log_odds
    reconstructed_margin = base_val + shap_sum
    assert abs(margin - reconstructed_margin) == pytest.approx(additivity_delta, abs=1e-7)
    assert additivity_delta < 1e-4

    # Sigmoid consistency: P(Churn) = 1 / (1 + exp(-margin))
    prob_from_margin = 1.0 / (1.0 + np.exp(-margin))
    assert prob_from_margin == pytest.approx(data["churn_probability"], abs=1e-4)

    # Baseline probability consistency: P0 = 1 / (1 + exp(-base_val))
    expected_base_prob = 1.0 / (1.0 + np.exp(-base_val))
    assert expected_base_prob == pytest.approx(data["baseline_probability"], abs=1e-4)

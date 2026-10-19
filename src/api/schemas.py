"""Pydantic Request and Response Schemas for Customer Intelligence API.

Enforces strict input validation, data typing, categorical domain constraints,
non-finite numeric checks, and structured, audit-ready prediction and explanation responses.
"""

import math
from typing import Any, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class CustomerFeaturesInput(BaseModel):
    """Raw 19 predictor features for an individual customer.

    Strictly forbids unexpected fields (including customerID and Churn) to prevent
    data leakage and schema drift. All categorical attributes must strictly conform
    to categories present in the trained model preprocessor.
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    # Demographic Features
    gender: Literal["Female", "Male"] = Field(
        ..., description="Customer gender: 'Female' or 'Male'"
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Whether the customer is a senior citizen: 1 (Yes) or 0 (No)"
    )
    Partner: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer has a partner: 'Yes' or 'No'"
    )
    Dependents: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer has dependents: 'Yes' or 'No'"
    )

    # Account and Service Tenure
    tenure: int = Field(
        ..., ge=0, le=120, description="Number of months customer has stayed with the company (0-120)"
    )

    # Phone Services
    PhoneService: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer has a phone service: 'Yes' or 'No'"
    )
    MultipleLines: Literal["No", "No phone service", "Yes"] = Field(
        ..., description="Whether customer has multiple lines: 'No', 'Yes', or 'No phone service'"
    )

    # Internet & Add-on Services
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer's internet service provider: 'DSL', 'Fiber optic', or 'No'"
    )
    OnlineSecurity: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether customer has online security add-on"
    )
    OnlineBackup: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether customer has online backup add-on"
    )
    DeviceProtection: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether customer has device protection add-on"
    )
    TechSupport: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether customer has tech support add-on"
    )
    StreamingTV: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether customer has streaming TV add-on"
    )
    StreamingMovies: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether customer has streaming movies add-on"
    )

    # Contract & Billing
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="The contract term: 'Month-to-month', 'One year', or 'Two year'"
    )
    PaperlessBilling: Literal["No", "Yes"] = Field(
        ..., description="Whether customer has paperless billing: 'Yes' or 'No'"
    )
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(..., description="Customer payment method")

    # Financial Charges
    MonthlyCharges: float = Field(
        ..., ge=0.0, le=1000.0, allow_inf_nan=False, description="Amount charged to customer monthly in USD"
    )
    TotalCharges: float = Field(
        ..., ge=0.0, le=50000.0, allow_inf_nan=False, description="Total cumulative amount charged in USD"
    )

    @field_validator("MonthlyCharges", "TotalCharges")
    @classmethod
    def validate_finite_numbers(cls, v: float, info) -> float:
        """Reject non-finite numbers (NaN, +Inf, -Inf) explicitly."""
        if not math.isfinite(v):
            raise ValueError(f"Field '{info.field_name}' must be a finite numeric value (NaN and Infinity are not permitted).")
        return v

    def to_feature_dict(self) -> dict:
        """Export validated attributes as a clean dictionary ready for model inference."""
        return self.model_dump()


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

class HealthResponse(BaseModel):
    """Service health and model readiness status."""
    status: str = Field(..., description="Application health status: 'healthy' or 'degraded'")
    model_ready: bool = Field(..., description="Whether machine learning pipeline is loaded and ready")
    model_name: Optional[str] = Field(None, description="Champion model family identification")
    operational_threshold: Optional[float] = Field(None, description="Preselected operational decision threshold")
    total_features_expected: Optional[int] = Field(None, description="Total number of predictor features required")


class PredictionResponse(BaseModel):
    """Churn prediction inference response distinguishing probability from threshold classification."""
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of churn P(Churn=1|X) between 0.0 and 1.0"
    )
    retention_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Derived probability of retention P(Churn=0|X) = 1.0 - churn_probability"
    )
    predicted_class: int = Field(
        ..., description="Discrete binary class: 1 (Churn Risk) if churn_probability >= threshold_used else 0 (Retained)"
    )
    threshold_used: float = Field(
        ..., description="Operational probability decision threshold applied for classification"
    )
    risk_level: str = Field(
        ..., description="Human-readable operational risk tier: 'High Risk (Operational Churn Alert)' or 'Low Risk (Retained)'"
    )
    model_name: str = Field(
        ..., description="Model architecture/family used for scoring"
    )


class LocalFeatureContributionSchema(BaseModel):
    """Local SHAP contribution for a single feature on the log-odds margin scale."""
    feature_name: str = Field(..., description="Transformed pipeline feature column name")
    readable_feature: str = Field(..., description="Human-interpretable feature and value description")
    observed_value: Optional[Any] = Field(None, description="Raw customer input value for this feature")
    transformed_value: Optional[Any] = Field(None, description="Transformed numerical or one-hot indicator value")
    is_active: Optional[bool] = Field(None, description="Whether one-hot indicator is active for this customer")
    shap_value: float = Field(..., description="Signed contribution value on log-odds margin scale")
    direction: str = Field(..., description="'push_toward_churn', 'push_toward_retention', or 'negligible'")
    magnitude: float = Field(..., ge=0.0, description="Absolute magnitude |shap_value|")
    reason: str = Field(..., description="Factual natural language description of contribution")


class ExplanationResponse(BaseModel):
    """Comprehensive SHAP local explanation with mathematical additivity guarantees."""
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of churn P(Churn=1|X)"
    )
    retention_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Derived probability of retention 1.0 - churn_probability"
    )
    predicted_class: int = Field(
        ..., description="Operational binary classification label (0 or 1)"
    )
    threshold_used: float = Field(
        ..., description="Operational decision threshold used for classification"
    )
    risk_level: str = Field(
        ..., description="Human-readable operational risk tier"
    )
    baseline_model_output: float = Field(
        ..., description="Expected model log-odds base value E[F(X)] across training distribution"
    )
    baseline_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Baseline probability sigmoid(baseline_model_output)"
    )
    model_score_log_odds: float = Field(
        ..., description="Actual model raw decision margin F(x) = classifier.decision_function(x)"
    )
    shap_values_sum: float = Field(
        ..., description="Sum of all individual feature SHAP contributions sum(phi_i)"
    )
    additivity_delta: float = Field(
        ..., ge=0.0, description="Discrepancy |F(x) - (baseline + sum(phi))|, verified < 1e-4"
    )
    output_scale: str = Field(
        ..., description="Output scale: 'log-odds (raw margin)'"
    )
    scale_explanation: str = Field(
        ..., description="Detailed explanation clarifying that SHAP values are additive in log-odds space, not direct probability increments"
    )
    contributions: List[LocalFeatureContributionSchema] = Field(
        ..., description="All individual feature contributions ordered by magnitude"
    )
    top_positive_contributions: List[LocalFeatureContributionSchema] = Field(
        ..., description="Top features driving the prediction toward churn"
    )
    top_negative_contributions: List[LocalFeatureContributionSchema] = Field(
        ..., description="Top protective features driving the prediction toward retention"
    )
    summary_reasons: List[str] = Field(
        ..., description="Executive bullet-point summary of key risk and retention factors"
    )
    governance_notice: str = Field(
        ..., description="ML governance and non-causal interpretation disclaimer"
    )


class SafeErrorResponse(BaseModel):
    """Standardized safe error response avoiding leakage of internal implementation details."""
    detail: Union[str, List[dict]] = Field(..., description="Safe human-readable error description")

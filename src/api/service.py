"""Model Inference and Explainability Service.

Encapsulates singleton model loading, cached pipeline execution, and local SHAP explanations.
Ensures the machine learning pipeline is loaded once at startup and reused safely across requests.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from src.api.config import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OPERATIONAL_THRESHOLD,
    METADATA_PATH,
    MODEL_PATH,
)
from src.api.schemas import (
    ExplanationResponse,
    LocalFeatureContributionSchema,
    PredictionResponse,
)
from src.explainability.shap_explainer import (
    ChurnShapExplainer,
    CustomerExplanation,
    load_churn_model,
)

logger = logging.getLogger(__name__)


class ModelNotReadyError(Exception):
    """Raised when an inference request is received but the model is not ready or failed to load."""
    pass


class ModelService:
    """Production service managing the trained model pipeline and SHAP explainer."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        operational_threshold: float = DEFAULT_OPERATIONAL_THRESHOLD,
    ) -> None:
        self.model_path = model_path or MODEL_PATH
        self.metadata_path = metadata_path or METADATA_PATH
        self.operational_threshold = operational_threshold

        self.pipeline: Optional[Pipeline] = None
        self.explainer: Optional[ChurnShapExplainer] = None
        self.metadata: Dict[str, Any] = {}
        self.model_name: str = "Gradient Boosting"
        self.total_raw_features: int = 19
        self.is_ready: bool = False
        self._load_error: Optional[str] = None

    def load_artifacts(self) -> None:
        """Load the trained Scikit-learn pipeline and metadata once."""
        try:
            logger.info("Initializing ModelService from %s...", self.model_path)
            self.pipeline = load_churn_model(self.model_path)

            # Load model governance metadata if available
            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    self.model_name = self.metadata.get("model_name", self.model_name)
                    self.operational_threshold = float(
                        self.metadata.get("operational_threshold", self.operational_threshold)
                    )
                    self.total_raw_features = int(
                        self.metadata.get("total_raw_features", self.total_raw_features)
                    )
            else:
                logger.warning(
                    "Model metadata file not found at %s. Proceeding with defaults.",
                    self.metadata_path,
                )

            # Initialize SHAP TreeExplainer
            logger.info("Initializing SHAP TreeExplainer for %s...", self.model_name)
            self.explainer = ChurnShapExplainer(self.pipeline)

            self.is_ready = True
            self._load_error = None
            logger.info(
                "ModelService successfully initialized. Champion: '%s', Threshold: %.2f.",
                self.model_name,
                self.operational_threshold,
            )
        except Exception as exc:
            self.is_ready = False
            self.pipeline = None
            self.explainer = None
            self._load_error = str(exc)
            logger.error("Failed to initialize model service: %s", exc, exc_info=True)

    def predict(self, features_dict: Dict[str, Any]) -> PredictionResponse:
        """Run single-record inference using the loaded pipeline."""
        if not self.is_ready or self.pipeline is None:
            raise ModelNotReadyError("Model service is not ready. Inference is temporarily unavailable.")

        # Construct single-row DataFrame matching the 19 predictor schema
        df = pd.DataFrame([features_dict])

        # Positive class probability P(Churn=1|X)
        prob_churn = float(self.pipeline.predict_proba(df)[0, 1])
        prob_retention = float(1.0 - prob_churn)

        # Operational decision threshold classification
        predicted_class = 1 if prob_churn >= self.operational_threshold else 0
        risk_level = (
            "High Risk (Operational Churn Alert)"
            if predicted_class == 1
            else "Low Risk (Retained)"
        )

        return PredictionResponse(
            churn_probability=prob_churn,
            retention_probability=prob_retention,
            predicted_class=predicted_class,
            threshold_used=self.operational_threshold,
            risk_level=risk_level,
            model_name=self.model_name,
        )

    def explain(self, features_dict: Dict[str, Any]) -> ExplanationResponse:
        """Compute single-customer prediction and local SHAP explanation in a single pass."""
        if not self.is_ready or self.explainer is None or self.pipeline is None:
            raise ModelNotReadyError("Model service is not ready. Explainability is temporarily unavailable.")

        # Compute explanation using established ChurnShapExplainer
        explanation: CustomerExplanation = self.explainer.explain_customer(
            customer_data=features_dict,
            operational_threshold=self.operational_threshold,
        )

        # Map local feature contributions to API schema
        all_contributions = [
            LocalFeatureContributionSchema(
                feature_name=c.feature_name,
                readable_feature=c.readable_feature,
                observed_value=c.actual_value,
                transformed_value=c.feature_value,
                is_active=c.is_active,
                shap_value=c.shap_value,
                direction=c.direction,
                magnitude=c.magnitude,
                reason=c.reason,
            )
            for c in explanation.all_contributions
        ]

        top_pos = [
            LocalFeatureContributionSchema(
                feature_name=c.feature_name,
                readable_feature=c.readable_feature,
                observed_value=c.actual_value,
                transformed_value=c.feature_value,
                is_active=c.is_active,
                shap_value=c.shap_value,
                direction=c.direction,
                magnitude=c.magnitude,
                reason=c.reason,
            )
            for c in explanation.top_positive_contributions
        ]

        top_neg = [
            LocalFeatureContributionSchema(
                feature_name=c.feature_name,
                readable_feature=c.readable_feature,
                observed_value=c.actual_value,
                transformed_value=c.feature_value,
                is_active=c.is_active,
                shap_value=c.shap_value,
                direction=c.direction,
                magnitude=c.magnitude,
                reason=c.reason,
            )
            for c in explanation.top_negative_contributions
        ]

        scale_explanation = (
            "SHAP values quantify each feature's additive contribution to the raw decision margin (log-odds scale), "
            "where model_score_log_odds = baseline_model_output + sum(shap_values). Positive values push the model "
            "toward churn, while negative values push toward retention. Contributions are additive strictly in log-odds "
            "space and must not be added directly to probabilities."
        )

        return ExplanationResponse(
            churn_probability=explanation.churn_probability,
            retention_probability=float(1.0 - explanation.churn_probability),
            predicted_class=explanation.predicted_label,
            threshold_used=explanation.operational_threshold,
            risk_level=explanation.risk_level,
            baseline_model_output=explanation.baseline_log_odds,
            baseline_probability=explanation.baseline_probability,
            model_score_log_odds=explanation.model_score_log_odds,
            shap_values_sum=explanation.shap_values_sum,
            additivity_delta=explanation.additivity_delta,
            output_scale=explanation.output_scale,
            scale_explanation=scale_explanation,
            contributions=all_contributions,
            top_positive_contributions=top_pos,
            top_negative_contributions=top_neg,
            summary_reasons=explanation.summary_reasons,
            governance_notice=explanation.governance_notice,
        )


# Global singleton instance for the application
_default_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Dependency provider returning the active ModelService instance."""
    global _default_service
    if _default_service is None:
        _default_service = ModelService()
        _default_service.load_artifacts()
    return _default_service


def set_model_service(service: Optional[ModelService]) -> None:
    """Setter for testing or custom initialization."""
    global _default_service
    _default_service = service

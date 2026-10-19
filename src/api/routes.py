"""API Endpoint Routes for Churn Prediction and Explainability.

Exposes:
- GET /health: Readiness and liveness diagnostics
- POST /predict: Single-customer churn risk scoring
- POST /explain: Local SHAP attribution and mathematical additivity analysis
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import (
    CustomerFeaturesInput,
    ExplanationResponse,
    HealthResponse,
    PredictionResponse,
    SafeErrorResponse,
)
from src.api.service import ModelNotReadyError, ModelService, get_model_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Model Readiness Check",
    description="Reports application liveness and whether the trained churn pipeline is loaded and ready.",
    responses={
        200: {"description": "Service is running; reports whether model is ready or degraded."},
    },
)
def get_health(
    model_service: ModelService = Depends(get_model_service),
) -> HealthResponse:
    """Check service health and model readiness status."""
    if model_service.is_ready:
        return HealthResponse(
            status="healthy",
            model_ready=True,
            model_name=model_service.model_name,
            operational_threshold=model_service.operational_threshold,
            total_features_expected=model_service.total_raw_features,
        )
    return HealthResponse(
        status="degraded",
        model_ready=False,
        model_name=None,
        operational_threshold=None,
        total_features_expected=model_service.total_raw_features,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Customer Churn Risk",
    description=(
        "Scores an individual customer record against the trained Gradient Boosting champion pipeline. "
        "Returns continuous churn probability, retention probability, and threshold classification at 0.35."
    ),
    responses={
        200: {"model": PredictionResponse, "description": "Successful churn prediction."},
        422: {"model": SafeErrorResponse, "description": "Validation error: invalid types, missing/unexpected fields, or out-of-range values."},
        503: {"model": SafeErrorResponse, "description": "Model is not loaded or temporarily unavailable."},
    },
)
def predict_churn(
    customer: CustomerFeaturesInput,
    model_service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Predict churn probability and operational classification for a single customer."""
    try:
        return model_service.predict(customer.to_feature_dict())
    except ModelNotReadyError as err:
        logger.warning("Predict request rejected: model is not ready.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable or failed to initialize.",
        ) from err
    except Exception as exc:
        logger.error("Unexpected error during prediction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the prediction request.",
        ) from exc


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Generate Local SHAP Explanation",
    description=(
        "Computes local feature contributions for an individual customer using TreeExplainer on the "
        "raw margin (log-odds) scale. Preserves additivity F(x) = E[F(X)] + sum(phi_i)."
    ),
    responses={
        200: {"model": ExplanationResponse, "description": "Successful local SHAP explanation."},
        422: {"model": SafeErrorResponse, "description": "Validation error: malformed input schema."},
        503: {"model": SafeErrorResponse, "description": "Model or explainer is not available."},
    },
)
def explain_churn(
    customer: CustomerFeaturesInput,
    model_service: ModelService = Depends(get_model_service),
) -> ExplanationResponse:
    """Generate detailed local SHAP attribution and feature reasons for a single customer."""
    try:
        return model_service.explain(customer.to_feature_dict())
    except ModelNotReadyError as err:
        logger.warning("Explain request rejected: model is not ready.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model explainer service is currently unavailable or failed to initialize.",
        ) from err
    except Exception as exc:
        logger.error("Unexpected error during explainability: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while calculating the model explanation.",
        ) from exc

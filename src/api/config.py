"""API Configuration and Environment Settings.

Provides centralized configuration for the FastAPI service, including model paths,
governance thresholds, CORS origins, and server metadata.
"""

import os
from pathlib import Path
from typing import List

# Resolve repository root directory dynamically (2 levels up from src/api)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Model and Metadata Artifact Paths
DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "models" / "churn_model.joblib"
DEFAULT_METADATA_PATH: Path = PROJECT_ROOT / "models" / "model_metadata.json"

MODEL_PATH: Path = Path(os.getenv("CHURN_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
METADATA_PATH: Path = Path(os.getenv("CHURN_METADATA_PATH", str(DEFAULT_METADATA_PATH)))

# Default Governance Operational Threshold (from Phase 4 Out-Of-Fold optimization)
DEFAULT_OPERATIONAL_THRESHOLD: float = 0.35

# CORS Allowed Origins (controlled explicit list for local development, no wildcards)
DEFAULT_CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

def get_cors_origins() -> List[str]:
    """Retrieve allowed CORS origins from environment or default to local dev origins."""
    env_origins = os.getenv("CORS_ORIGINS")
    if env_origins:
        return [origin.strip() for origin in env_origins.split(",") if origin.strip()]
    return DEFAULT_CORS_ORIGINS


API_TITLE: str = "Customer Intelligence & Churn Prevention API"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = (
    "Production-grade RESTful API serving trained Scikit-learn churn prediction pipelines "
    "and TreeExplainer SHAP local model interpretability."
)

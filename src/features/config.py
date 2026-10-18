"""Feature Engineering Configuration and Schema Taxonomy.

Defines the explicit roles of all 21 columns in the Telco Customer Churn dataset:
- Identifier columns (excluded from predictive models)
- Target column (isolated binary label)
- Numerical features (scaled with StandardScaler)
- Categorical features (encoded with OneHotEncoder)
"""

from typing import List

# Primary record identifier (must NOT be used as a predictive feature)
ID_COLUMN: str = "customerID"

# Binary prediction target: 'Yes' (churned = 1) vs 'No' (retained = 0)
TARGET_COLUMN: str = "Churn"

# Continuous and discrete numerical features requiring scaling
NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Categorical features requiring one-hot encoding
# Note: SeniorCitizen is included here because although it is coded as 0/1 in the raw data,
# it is conceptually a binary demographic categorical variable.
CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Combined list of all valid predictor feature names (excluding customerID and Churn)
ALL_PREDICTOR_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

/**
 * Core API and Model Types for Customer Intelligence & Churn Prevention Platform.
 * Derived directly from backend Pydantic schemas in src/api/schemas.py.
 */

export interface CustomerFeaturesInput {
  gender: 'Female' | 'Male';
  SeniorCitizen: 0 | 1;
  Partner: 'No' | 'Yes';
  Dependents: 'No' | 'Yes';
  tenure: number;
  PhoneService: 'No' | 'Yes';
  MultipleLines: 'No' | 'No phone service' | 'Yes';
  InternetService: 'DSL' | 'Fiber optic' | 'No';
  OnlineSecurity: 'No' | 'No internet service' | 'Yes';
  OnlineBackup: 'No' | 'No internet service' | 'Yes';
  DeviceProtection: 'No' | 'No internet service' | 'Yes';
  TechSupport: 'No' | 'No internet service' | 'Yes';
  StreamingTV: 'No' | 'No internet service' | 'Yes';
  StreamingMovies: 'No' | 'No internet service' | 'Yes';
  Contract: 'Month-to-month' | 'One year' | 'Two year';
  PaperlessBilling: 'No' | 'Yes';
  PaymentMethod:
    | 'Bank transfer (automatic)'
    | 'Credit card (automatic)'
    | 'Electronic check'
    | 'Mailed check';
  MonthlyCharges: number;
  TotalCharges: number;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded';
  model_ready: boolean;
  model_name: string | null;
  operational_threshold: number | null;
  total_features_expected: number | null;
}

export interface PredictionResponse {
  churn_probability: number;
  retention_probability: number;
  predicted_class: 0 | 1;
  threshold_used: number;
  risk_level: string;
  model_name: string;
}

export interface LocalFeatureContribution {
  feature_name: string;
  readable_feature: string;
  observed_value: any;
  transformed_value: any;
  is_active: boolean | null;
  shap_value: number;
  direction: 'push_toward_churn' | 'push_toward_retention' | 'negligible';
  magnitude: number;
  reason: string;
}

export interface ExplanationResponse {
  churn_probability: number;
  retention_probability: number;
  predicted_class: number;
  threshold_used: number;
  risk_level: string;
  baseline_model_output: number;
  baseline_probability: number;
  model_score_log_odds: number;
  shap_values_sum: number;
  additivity_delta: number;
  output_scale: string;
  scale_explanation: string;
  contributions: LocalFeatureContribution[];
  top_positive_contributions: LocalFeatureContribution[];
  top_negative_contributions: LocalFeatureContribution[];
  summary_reasons: string[];
  governance_notice: string;
}

export interface ApiErrorDetail {
  field?: string;
  message: string;
  type?: string;
}

export interface ApiErrorResponse {
  detail: string | ApiErrorDetail[];
  message?: string;
}

export interface CustomerAnalysisRecord {
  id: string;
  timestamp: string;
  customerName?: string;
  input: CustomerFeaturesInput;
  prediction: PredictionResponse;
  explanation: ExplanationResponse | null;
}

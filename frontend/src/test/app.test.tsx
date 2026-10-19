import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { App } from '../App';
import { CustomerForm } from '../components/analysis/CustomerForm';
import { PredictionCard } from '../components/analysis/PredictionCard';
import { ShapWaterfall } from '../components/analysis/ShapWaterfall';
import { SAMPLE_PROFILES } from '../data/sampleCustomers';
import { MODEL_METADATA } from '../data/modelArtifacts';
import type { ExplanationResponse, PredictionResponse } from '../types/api';

describe('Frontend Component & Integration Tests', () => {
  it('renders application shell, title, and dashboard navigation', () => {
    render(<App />);
    expect(screen.getAllByText(/Customer Intelligence & Churn Prevention Platform/i).length).toBeGreaterThan(0);
    expect(screen.getByText('ChurnSight Dashboard')).toBeInTheDocument();
    expect(screen.getAllByText(/See risk\. Understand why\./i).length).toBeGreaterThan(0);
    expect(screen.getByText('Churn')).toBeInTheDocument();
    expect(screen.getByText('Sight')).toBeInTheDocument();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Customer Analysis')).toBeInTheDocument();
    expect(screen.getByText('Churn Predictions')).toBeInTheDocument();
    expect(screen.getByText('Model Insights')).toBeInTheDocument();
    expect(screen.getByText('Reports')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('renders KPI metrics and empty state when no customer has been scored', () => {
    render(<App />);
    expect(screen.getAllByText(/Operational Cutoff/i).length).toBeGreaterThan(0);
    expect(screen.getByText('Latest Churn Probability')).toBeInTheDocument();
    expect(screen.getByText('No Customer Analyzed In This Session')).toBeInTheDocument();
  });


  it('navigates between tabs upon clicking sidebar links', () => {
    render(<App />);
    const analysisTab = screen.getByText('Customer Analysis');
    fireEvent.click(analysisTab);
    expect(screen.getByText(/Single Customer Churn Risk Analysis/i)).toBeInTheDocument();

    const insightsTab = screen.getByText('Model Insights');
    fireEvent.click(insightsTab);
    expect(screen.getByText(/Machine Learning Model Insights & Benchmarks/i)).toBeInTheDocument();

    const settingsTab = screen.getByText('Settings');
    fireEvent.click(settingsTab);
    expect(screen.getByText(/Platform Connectivity & Settings/i)).toBeInTheDocument();
  });

  it('renders customer scoring form with all 19 raw predictor features', () => {
    const handleSubmit = vi.fn();
    render(<CustomerForm onSubmit={handleSubmit} isLoading={false} />);

    // Profile fields
    expect(screen.getByLabelText(/Gender/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Senior Citizen/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Has Partner/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Has Dependents/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Customer Tenure/i)).toBeInTheDocument();

    // Services fields
    expect(screen.getByLabelText(/Phone Service/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Multiple Lines/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Internet Service Tier/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Online Security/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Online Backup/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Device Protection/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Tech Support/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Streaming TV/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Streaming Movies/i)).toBeInTheDocument();

    // Billing fields
    expect(screen.getByLabelText(/Contract Commitment/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Paperless Billing/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Payment Method/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Monthly Charges/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Total Cumulative Charges/i)).toBeInTheDocument();
  });

  it('populates form fields when clicking a synthetic sample customer button', () => {
    const handleSubmit = vi.fn();
    render(<CustomerForm onSubmit={handleSubmit} isLoading={false} />);

    const sampleBtn = screen.getByText(/High Risk Profile/i);
    fireEvent.click(sampleBtn);

    const tenureInput = screen.getByLabelText(/Customer Tenure/i) as HTMLInputElement;
    expect(tenureInput.value).toBe(String(SAMPLE_PROFILES[0].data.tenure));

    const chargesInput = screen.getByLabelText(/Monthly Charges/i) as HTMLInputElement;
    expect(chargesInput.value).toBe(String(SAMPLE_PROFILES[0].data.MonthlyCharges));
  });

  it('validates numeric range bounds and prevents submission of invalid values', () => {
    const handleSubmit = vi.fn();
    render(<CustomerForm onSubmit={handleSubmit} isLoading={false} />);

    const tenureInput = screen.getByLabelText(/Customer Tenure/i);
    fireEvent.change(tenureInput, { target: { value: '-5' } });

    const submitBtn = screen.getByText(/Analyze Churn Risk/i);
    fireEvent.click(submitBtn);

    expect(screen.getByText(/Tenure must be between 0 and 120/i)).toBeInTheDocument();
    expect(handleSubmit).not.toHaveBeenCalled();
  });

  it('renders PredictionCard with probability, retention, threshold, and classification badge', () => {
    const mockPrediction: PredictionResponse = {
      churn_probability: 0.6318,
      retention_probability: 0.3682,
      predicted_class: 1,
      threshold_used: 0.35,
      risk_level: 'High Risk (Operational Churn Alert)',
      model_name: 'Gradient Boosting',
    };

    render(<PredictionCard prediction={mockPrediction} customerName="Test Account" />);
    expect(screen.getByText('63.2%')).toBeInTheDocument();
    expect(screen.getByText('36.8%')).toBeInTheDocument();
    expect(screen.getByText(/High Risk \(Churn Alert\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Evaluated with Champion Model: Gradient Boosting/i)).toBeInTheDocument();
  });

  it('renders ShapWaterfall with additivity check, log-odds scale notice, and signed contributions', () => {
    const mockExplanation: ExplanationResponse = {
      churn_probability: 0.6318,
      retention_probability: 0.3682,
      predicted_class: 1,
      threshold_used: 0.35,
      risk_level: 'High Risk (Operational Churn Alert)',
      baseline_model_output: -1.5893,
      baseline_probability: 0.1695,
      model_score_log_odds: 0.5401,
      shap_values_sum: 2.1294,
      additivity_delta: 0.0000000000000002,
      output_scale: 'log-odds (raw margin)',
      scale_explanation:
        'SHAP values quantify each feature additive contribution to the raw decision margin (log-odds scale).',
      contributions: [
        {
          feature_name: 'Contract_Month-to-month',
          readable_feature: "Contract = 'Month-to-month'",
          observed_value: 'Month-to-month',
          transformed_value: 1.0,
          is_active: true,
          shap_value: 0.5781,
          direction: 'push_toward_churn',
          magnitude: 0.5781,
          reason: "Contract is 'Month-to-month' contributing +0.5781 log-odds toward churn.",
        },
        {
          feature_name: 'tenure',
          readable_feature: 'Customer Tenure (1 mo)',
          observed_value: 1,
          transformed_value: 1.0,
          is_active: null,
          shap_value: -0.2049,
          direction: 'push_toward_retention',
          magnitude: 0.2049,
          reason: 'Observed tenure of 1 mo contributes -0.2049 log-odds away from churn.',
        },
      ],
      top_positive_contributions: [
        {
          feature_name: 'Contract_Month-to-month',
          readable_feature: "Contract = 'Month-to-month'",
          observed_value: 'Month-to-month',
          transformed_value: 1.0,
          is_active: true,
          shap_value: 0.5781,
          direction: 'push_toward_churn',
          magnitude: 0.5781,
          reason: "Contract is 'Month-to-month' contributing +0.5781 log-odds toward churn.",
        },
      ],
      top_negative_contributions: [
        {
          feature_name: 'tenure',
          readable_feature: 'Customer Tenure (1 mo)',
          observed_value: 1,
          transformed_value: 1.0,
          is_active: null,
          shap_value: -0.2049,
          direction: 'push_toward_retention',
          magnitude: 0.2049,
          reason: 'Observed tenure of 1 mo contributes -0.2049 log-odds away from churn.',
        },
      ],
      summary_reasons: ["Top Factor: Contract = 'Month-to-month' (+0.58)"],
      governance_notice: 'Governance Notice: Observational predictive attribution, not causal proof.',
    };

    render(<ShapWaterfall explanation={mockExplanation} />);
    expect(screen.getByText(/Local SHAP Model Interpretability/i)).toBeInTheDocument();
    expect(screen.getByText(/Additivity \u0394 < 1e-4 verified/i)).toBeInTheDocument();
    expect(screen.getByText(/log-odds margin scale/i)).toBeInTheDocument();
    expect(screen.getByText('+0.5781')).toBeInTheDocument();
    expect(screen.getByText('-0.2049')).toBeInTheDocument();
  });

  it('renders verified offline holdout evaluation metrics in Model Insights', () => {
    expect(MODEL_METADATA.holdout_operational_metrics.recall).toBeCloseTo(0.7139, 3);
    expect(MODEL_METADATA.holdout_operational_metrics.precision).toBeCloseTo(0.5597, 3);
    expect(MODEL_METADATA.holdout_operational_metrics.f1_score).toBeCloseTo(0.6275, 3);
    expect(MODEL_METADATA.operational_threshold).toBe(0.35);
  });
});

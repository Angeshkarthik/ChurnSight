import React, { useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  BookmarkCheck,
  RotateCcw,
  Sparkles,
} from 'lucide-react';
import { SAMPLE_PROFILES } from '../../data/sampleCustomers';
import type { SampleProfile } from '../../data/sampleCustomers';
import type { CustomerFeaturesInput } from '../../types/api';
import { Button } from '../common/Button';


interface CustomerFormProps {
  onSubmit: (data: CustomerFeaturesInput, customerName?: string) => void;
  isLoading: boolean;
  error?: string | null;
}

const DEFAULT_FORM_STATE: CustomerFeaturesInput = {
  gender: 'Female',
  SeniorCitizen: 0,
  Partner: 'No',
  Dependents: 'No',
  tenure: 12,
  PhoneService: 'Yes',
  MultipleLines: 'No',
  InternetService: 'Fiber optic',
  OnlineSecurity: 'No',
  OnlineBackup: 'No',
  DeviceProtection: 'No',
  TechSupport: 'No',
  StreamingTV: 'No',
  StreamingMovies: 'No',
  Contract: 'Month-to-month',
  PaperlessBilling: 'Yes',
  PaymentMethod: 'Electronic check',
  MonthlyCharges: 70.0,
  TotalCharges: 840.0,
};

export const CustomerForm: React.FC<CustomerFormProps> = ({
  onSubmit,
  isLoading,
  error,
}) => {
  const [formData, setFormData] = useState<CustomerFeaturesInput>(DEFAULT_FORM_STATE);
  const [customerName, setCustomerName] = useState<string>('');
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [activeProfileId, setActiveProfileId] = useState<string | null>(null);

  const handleChange = (
    field: keyof CustomerFeaturesInput,
    value: any
  ) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
    // Clear validation error when user edits
    if (validationErrors[field]) {
      setValidationErrors((prev) => {
        const next = { ...prev };
        delete next[field];
        return next;
      });
    }
  };

  const handleNumericChange = (
    field: 'tenure' | 'MonthlyCharges' | 'TotalCharges',
    value: string
  ) => {
    const num = parseFloat(value);
    handleChange(field, isNaN(num) ? '' : num);
  };

  const handleLoadSample = (profile: SampleProfile) => {
    setFormData({ ...profile.data });
    setCustomerName(profile.name.replace('Synthetic Example: ', ''));
    setActiveProfileId(profile.id);
    setValidationErrors({});
  };

  const handleReset = () => {
    setFormData(DEFAULT_FORM_STATE);
    setCustomerName('');
    setActiveProfileId(null);
    setValidationErrors({});
  };

  const validate = (): boolean => {
    const errors: Record<string, string> = {};

    if (formData.tenure === undefined || formData.tenure === null || isNaN(formData.tenure)) {
      errors.tenure = 'Tenure is required';
    } else if (formData.tenure < 0 || formData.tenure > 120) {
      errors.tenure = 'Tenure must be between 0 and 120 months';
    }

    if (formData.MonthlyCharges === undefined || formData.MonthlyCharges === null || isNaN(formData.MonthlyCharges)) {
      errors.MonthlyCharges = 'Monthly charges is required';
    } else if (formData.MonthlyCharges < 0 || formData.MonthlyCharges > 1000) {
      errors.MonthlyCharges = 'Monthly charges must be between $0 and $1,000';
    }

    if (formData.TotalCharges === undefined || formData.TotalCharges === null || isNaN(formData.TotalCharges)) {
      errors.TotalCharges = 'Total charges is required';
    } else if (formData.TotalCharges < 0 || formData.TotalCharges > 50000) {
      errors.TotalCharges = 'Total charges must be between $0 and $50,000';
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;
    onSubmit(formData, customerName || undefined);
  };

  return (
    <form onSubmit={handleSubmit} className="saas-form-wrapper" noValidate>
      {/* Synthetic Profiles Quick Selector */}
      <div className="saas-sample-selector-card">
        <div className="selector-header">
          <div className="selector-title">
            <Sparkles size={16} className="text-emerald" />
            <span>Load Pre-configured Example Customer Profile</span>
          </div>
          <span className="selector-caption">
            Populate all 19 model features with verified synthetic development cases
          </span>
        </div>
        <div className="selector-options">
          {SAMPLE_PROFILES.map((profile) => {
            const isSelected = activeProfileId === profile.id;
            return (
              <button
                type="button"
                key={profile.id}
                className={`saas-sample-pill ${isSelected ? 'saas-sample-pill-active' : ''}`}
                onClick={() => handleLoadSample(profile)}
              >
                <div className="pill-top">
                  <span className={`pill-tier pill-tier-${profile.expectedTier}`}>
                    {profile.expectedTier.toUpperCase()} RISK
                  </span>
                  {isSelected && <BookmarkCheck size={14} className="text-emerald" />}
                </div>
                <span className="pill-name">{profile.name.replace('Synthetic Example: ', '')}</span>
                <span className="pill-tagline">{profile.tagline}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Global Error Banner if API failed */}
      {error && (
        <div className="saas-alert-banner saas-alert-error" role="alert">
          <AlertCircle size={18} className="alert-icon" />
          <div className="alert-body">
            <strong>Inference Request Failed:</strong> {error}
          </div>
        </div>
      )}

      {/* Section 1: Customer Profile */}
      <div className="saas-form-section">
        <div className="form-section-header">
          <h4 className="form-section-title">1. Customer Demographics & Account Profile</h4>
          <p className="form-section-desc">Primary demographic attributes and relationship length.</p>
        </div>

        <div className="saas-form-grid">
          {/* Optional Name Identifier for UI Reference */}
          <div className="saas-form-group col-span-2">
            <label className="saas-label" htmlFor="customerName">
              Customer Reference / Identifier (Optional, UI only)
            </label>
            <input
              id="customerName"
              type="text"
              className="saas-input"
              placeholder="e.g. Acme Corp Account #42"
              value={customerName}
              onChange={(e) => setCustomerName(e.target.value)}
            />
            <span className="saas-helper-text">
              Note: customerID is never submitted to the ML model to prevent data leakage.
            </span>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="gender">
              Gender <span className="text-required">*</span>
            </label>
            <select
              id="gender"
              className="saas-select"
              value={formData.gender}
              onChange={(e) => handleChange('gender', e.target.value as 'Female' | 'Male')}
            >
              <option value="Female">Female</option>
              <option value="Male">Male</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="SeniorCitizen">
              Senior Citizen Status <span className="text-required">*</span>
            </label>
            <select
              id="SeniorCitizen"
              className="saas-select"
              value={formData.SeniorCitizen}
              onChange={(e) => handleChange('SeniorCitizen', parseInt(e.target.value, 10) as 0 | 1)}
            >
              <option value={0}>No (0)</option>
              <option value={1}>Yes (1)</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="Partner">
              Has Partner <span className="text-required">*</span>
            </label>
            <select
              id="Partner"
              className="saas-select"
              value={formData.Partner}
              onChange={(e) => handleChange('Partner', e.target.value as 'No' | 'Yes')}
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="Dependents">
              Has Dependents <span className="text-required">*</span>
            </label>
            <select
              id="Dependents"
              className="saas-select"
              value={formData.Dependents}
              onChange={(e) => handleChange('Dependents', e.target.value as 'No' | 'Yes')}
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="tenure">
              Customer Tenure (Months) <span className="text-required">*</span>
            </label>
            <input
              id="tenure"
              type="number"
              min={0}
              max={120}
              className={`saas-input ${validationErrors.tenure ? 'saas-input-error' : ''}`}
              value={formData.tenure}
              onChange={(e) => handleNumericChange('tenure', e.target.value)}
            />
            {validationErrors.tenure ? (
              <span className="saas-error-text">{validationErrors.tenure}</span>
            ) : (
              <span className="saas-helper-text">Total months subscribed (0 - 120).</span>
            )}
          </div>
        </div>
      </div>

      {/* Section 2: Subscribed Services */}
      <div className="saas-form-section">
        <div className="form-section-header">
          <h4 className="form-section-title">2. Subscribed Telco & Digital Services</h4>
          <p className="form-section-desc">Connectivity tier and security/streaming add-ons.</p>
        </div>

        <div className="saas-form-grid">
          <div className="saas-form-group">
            <label className="saas-label" htmlFor="PhoneService">
              Phone Service <span className="text-required">*</span>
            </label>
            <select
              id="PhoneService"
              className="saas-select"
              value={formData.PhoneService}
              onChange={(e) => handleChange('PhoneService', e.target.value as 'No' | 'Yes')}
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="MultipleLines">
              Multiple Lines <span className="text-required">*</span>
            </label>
            <select
              id="MultipleLines"
              className="saas-select"
              value={formData.MultipleLines}
              onChange={(e) =>
                handleChange(
                  'MultipleLines',
                  e.target.value as 'No' | 'No phone service' | 'Yes'
                )
              }
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No phone service">No phone service</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="InternetService">
              Internet Service Tier <span className="text-required">*</span>
            </label>
            <select
              id="InternetService"
              className="saas-select"
              value={formData.InternetService}
              onChange={(e) =>
                handleChange('InternetService', e.target.value as 'DSL' | 'Fiber optic' | 'No')
              }
            >
              <option value="Fiber optic">Fiber optic</option>
              <option value="DSL">DSL</option>
              <option value="No">No (No internet)</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="OnlineSecurity">
              Online Security <span className="text-required">*</span>
            </label>
            <select
              id="OnlineSecurity"
              className="saas-select"
              value={formData.OnlineSecurity}
              onChange={(e) =>
                handleChange(
                  'OnlineSecurity',
                  e.target.value as 'No' | 'No internet service' | 'Yes'
                )
              }
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="OnlineBackup">
              Online Backup <span className="text-required">*</span>
            </label>
            <select
              id="OnlineBackup"
              className="saas-select"
              value={formData.OnlineBackup}
              onChange={(e) =>
                handleChange(
                  'OnlineBackup',
                  e.target.value as 'No' | 'No internet service' | 'Yes'
                )
              }
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="DeviceProtection">
              Device Protection Plan <span className="text-required">*</span>
            </label>
            <select
              id="DeviceProtection"
              className="saas-select"
              value={formData.DeviceProtection}
              onChange={(e) =>
                handleChange(
                  'DeviceProtection',
                  e.target.value as 'No' | 'No internet service' | 'Yes'
                )
              }
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="TechSupport">
              Tech Support Plan <span className="text-required">*</span>
            </label>
            <select
              id="TechSupport"
              className="saas-select"
              value={formData.TechSupport}
              onChange={(e) =>
                handleChange(
                  'TechSupport',
                  e.target.value as 'No' | 'No internet service' | 'Yes'
                )
              }
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="StreamingTV">
              Streaming TV <span className="text-required">*</span>
            </label>
            <select
              id="StreamingTV"
              className="saas-select"
              value={formData.StreamingTV}
              onChange={(e) =>
                handleChange(
                  'StreamingTV',
                  e.target.value as 'No' | 'No internet service' | 'Yes'
                )
              }
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="StreamingMovies">
              Streaming Movies <span className="text-required">*</span>
            </label>
            <select
              id="StreamingMovies"
              className="saas-select"
              value={formData.StreamingMovies}
              onChange={(e) =>
                handleChange(
                  'StreamingMovies',
                  e.target.value as 'No' | 'No internet service' | 'Yes'
                )
              }
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
        </div>
      </div>

      {/* Section 3: Billing & Contract */}
      <div className="saas-form-section">
        <div className="form-section-header">
          <h4 className="form-section-title">3. Contract Term & Billing Financials</h4>
          <p className="form-section-desc">Commitment period, invoicing, and cumulative spend.</p>
        </div>

        <div className="saas-form-grid">
          <div className="saas-form-group">
            <label className="saas-label" htmlFor="Contract">
              Contract Commitment Term <span className="text-required">*</span>
            </label>
            <select
              id="Contract"
              className="saas-select"
              value={formData.Contract}
              onChange={(e) =>
                handleChange(
                  'Contract',
                  e.target.value as 'Month-to-month' | 'One year' | 'Two year'
                )
              }
            >
              <option value="Month-to-month">Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="PaperlessBilling">
              Paperless Billing <span className="text-required">*</span>
            </label>
            <select
              id="PaperlessBilling"
              className="saas-select"
              value={formData.PaperlessBilling}
              onChange={(e) =>
                handleChange('PaperlessBilling', e.target.value as 'No' | 'Yes')
              }
            >
              <option value="Yes">Yes (Paperless)</option>
              <option value="No">No (Paper mail)</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="PaymentMethod">
              Payment Method <span className="text-required">*</span>
            </label>
            <select
              id="PaymentMethod"
              className="saas-select"
              value={formData.PaymentMethod}
              onChange={(e) => handleChange('PaymentMethod', e.target.value as any)}
            >
              <option value="Electronic check">Electronic check</option>
              <option value="Mailed check">Mailed check</option>
              <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
              <option value="Credit card (automatic)">Credit card (automatic)</option>
            </select>
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="MonthlyCharges">
              Monthly Charges ($ USD) <span className="text-required">*</span>
            </label>
            <input
              id="MonthlyCharges"
              type="number"
              step="0.05"
              min={0}
              max={1000}
              className={`saas-input ${validationErrors.MonthlyCharges ? 'saas-input-error' : ''}`}
              value={formData.MonthlyCharges}
              onChange={(e) => handleNumericChange('MonthlyCharges', e.target.value)}
            />
            {validationErrors.MonthlyCharges ? (
              <span className="saas-error-text">{validationErrors.MonthlyCharges}</span>
            ) : (
              <span className="saas-helper-text">Current recurring monthly fee.</span>
            )}
          </div>

          <div className="saas-form-group">
            <label className="saas-label" htmlFor="TotalCharges">
              Total Cumulative Charges ($ USD) <span className="text-required">*</span>
            </label>
            <input
              id="TotalCharges"
              type="number"
              step="0.1"
              min={0}
              max={50000}
              className={`saas-input ${validationErrors.TotalCharges ? 'saas-input-error' : ''}`}
              value={formData.TotalCharges}
              onChange={(e) => handleNumericChange('TotalCharges', e.target.value)}
            />
            {validationErrors.TotalCharges ? (
              <span className="saas-error-text">{validationErrors.TotalCharges}</span>
            ) : (
              <span className="saas-helper-text">Historical cumulative customer spend.</span>
            )}
          </div>
        </div>
      </div>

      {/* Form Action Controls */}
      <div className="saas-form-actions">
        <Button
          type="button"
          variant="outline"
          onClick={handleReset}
          disabled={isLoading}
          icon={<RotateCcw size={16} />}
        >
          Reset Form
        </Button>
        <Button
          type="submit"
          variant="primary"
          isLoading={isLoading}
          icon={<ArrowRight size={16} />}
          iconPosition="right"
        >
          Analyze Churn Risk & Explain
        </Button>
      </div>
    </form>
  );
};

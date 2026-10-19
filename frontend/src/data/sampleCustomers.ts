import type { CustomerFeaturesInput } from '../types/api';


export interface SampleProfile {
  id: string;
  name: string;
  tagline: string;
  expectedTier: 'high' | 'borderline' | 'low';
  data: CustomerFeaturesInput;
}

export const SAMPLE_PROFILES: SampleProfile[] = [
  {
    id: 'sample-high-risk',
    name: 'Synthetic Example: High Risk Profile',
    tagline: 'New customer, month-to-month contract, fiber optic internet, electronic check payment.',
    expectedTier: 'high',
    data: {
      gender: 'Female',
      SeniorCitizen: 0,
      Partner: 'No',
      Dependents: 'No',
      tenure: 1,
      PhoneService: 'Yes',
      MultipleLines: 'No',
      InternetService: 'Fiber optic',
      OnlineSecurity: 'No',
      OnlineBackup: 'No',
      DeviceProtection: 'No',
      TechSupport: 'No',
      StreamingTV: 'Yes',
      StreamingMovies: 'Yes',
      Contract: 'Month-to-month',
      PaperlessBilling: 'Yes',
      PaymentMethod: 'Electronic check',
      MonthlyCharges: 95.5,
      TotalCharges: 95.5,
    },
  },
  {
    id: 'sample-borderline-risk',
    name: 'Synthetic Example: Borderline Profile',
    tagline: 'Moderate tenure (14 mo), 1-year contract, bundled services, paperless billing.',
    expectedTier: 'borderline',
    data: {
      gender: 'Male',
      SeniorCitizen: 0,
      Partner: 'Yes',
      Dependents: 'No',
      tenure: 14,
      PhoneService: 'Yes',
      MultipleLines: 'Yes',
      InternetService: 'Fiber optic',
      OnlineSecurity: 'No',
      OnlineBackup: 'Yes',
      DeviceProtection: 'No',
      TechSupport: 'No',
      StreamingTV: 'Yes',
      StreamingMovies: 'No',
      Contract: 'One year',
      PaperlessBilling: 'Yes',
      PaymentMethod: 'Bank transfer (automatic)',
      MonthlyCharges: 85.0,
      TotalCharges: 1190.0,
    },
  },
  {
    id: 'sample-low-risk',
    name: 'Synthetic Example: Low Risk Profile',
    tagline: 'Long tenure (65 mo), 2-year contract, DSL internet, automatic credit card billing.',
    expectedTier: 'low',
    data: {
      gender: 'Male',
      SeniorCitizen: 0,
      Partner: 'Yes',
      Dependents: 'Yes',
      tenure: 65,
      PhoneService: 'Yes',
      MultipleLines: 'Yes',
      InternetService: 'DSL',
      OnlineSecurity: 'Yes',
      OnlineBackup: 'Yes',
      DeviceProtection: 'Yes',
      TechSupport: 'Yes',
      StreamingTV: 'No',
      StreamingMovies: 'No',
      Contract: 'Two year',
      PaperlessBilling: 'No',
      PaymentMethod: 'Credit card (automatic)',
      MonthlyCharges: 55.2,
      TotalCharges: 3588.0,
    },
  },
];

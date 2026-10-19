/**
 * API Client for Customer Intelligence & Churn Prevention Backend.
 * Connects to FastAPI backend running on http://127.0.0.1:8000 by default.
 */

import type {
  CustomerFeaturesInput,
  ExplanationResponse,
  HealthResponse,
  PredictionResponse,
} from '../types/api';


const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export class ApiError extends Error {
  status: number;
  details?: any;

  constructor(message: string, status: number, details?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    let details: any = null;
    try {
      const data = await response.json();
      details = data;
      if (typeof data.detail === 'string') {
        errorDetail = data.detail;
      } else if (Array.isArray(data.detail)) {
        errorDetail = data.detail
          .map((d: any) => `${d.field ? `${d.field}: ` : ''}${d.message}`)
          .join('; ');
      } else if (data.message) {
        errorDetail = data.message;
      }
    } catch {
      // Non-JSON error body
    }
    throw new ApiError(errorDetail, response.status, details);
  }
  return response.json() as Promise<T>;
}

export const apiClient = {
  getBaseUrl(): string {
    return API_BASE_URL;
  },

  async getHealth(): Promise<HealthResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
        },
      });
      return handleResponse<HealthResponse>(response);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        `Unable to reach backend at ${API_BASE_URL}. Ensure FastAPI is running on port 8000.`,
        0,
        err
      );
    }
  },

  async predictChurn(input: CustomerFeaturesInput): Promise<PredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(input),
    });
    return handleResponse<PredictionResponse>(response);
  },

  async explainChurn(input: CustomerFeaturesInput): Promise<ExplanationResponse> {
    const response = await fetch(`${API_BASE_URL}/explain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(input),
    });
    return handleResponse<ExplanationResponse>(response);
  },
};

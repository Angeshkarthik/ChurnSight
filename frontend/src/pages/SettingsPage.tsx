import React from 'react';
import {
  ExternalLink,
  Info,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react';

import { apiClient } from '../api/client';
import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import { Card } from '../components/common/Card';
import { useAnalysis } from '../context/AnalysisContext';

export const SettingsPage: React.FC = () => {
  const { health, isHealthLoading, healthError, refreshHealth } = useAnalysis();

  const isModelReady = health?.model_ready === true;
  const baseUrl = apiClient.getBaseUrl();

  return (
    <div className="saas-page-container">
      <div className="page-header-block">
        <h2 className="page-main-title">Platform Connectivity & Settings</h2>
        <p className="page-main-desc">
          Inspect backend service configuration, API endpoint endpoints, and governance thresholds.
        </p>
      </div>

      <div className="settings-grid">
        {/* Backend Connectivity Status */}
        <Card
          title="FastAPI Service Connectivity"
          subtitle="Real-time status check with the Customer Intelligence backend service"
          action={
            <Button
              variant="outline"
              size="sm"
              icon={<RefreshCw size={14} className={isHealthLoading ? 'animate-spin' : ''} />}
              onClick={() => refreshHealth()}
              disabled={isHealthLoading}
            >
              Test Connection
            </Button>
          }
        >
          <div className="settings-field-list">
            <div className="settings-field-row">
              <span className="field-name">Service Health:</span>
              <div>
                {isHealthLoading ? (
                  <Badge variant="neutral" size="sm" dot>
                    Probing...
                  </Badge>
                ) : isModelReady ? (
                  <Badge variant="healthy" size="sm" dot>
                    Online & Ready
                  </Badge>
                ) : (
                  <Badge variant="degraded" size="sm" dot>
                    Degraded
                  </Badge>
                )}
              </div>
            </div>

            <div className="settings-field-row">
              <span className="field-name">Active Backend Base URL:</span>
              <span className="field-val font-mono bg-muted p-1 px-2 rounded">
                {baseUrl}
              </span>
            </div>

            <div className="settings-field-row">
              <span className="field-name">Champion Estimator:</span>
              <span className="field-val font-medium text-charcoal">
                {health?.model_name || 'Gradient Boosting (Loading...)'}
              </span>
            </div>

            <div className="settings-field-row">
              <span className="field-name">Required Predictor Features:</span>
              <span className="field-val">{health?.total_features_expected || 19} Raw Attributes</span>
            </div>

            {healthError && (
              <div className="settings-error-banner">
                <ShieldAlert size={16} className="text-risk-high shrink-0 mt-0.5" />
                <span>{healthError}</span>
              </div>
            )}
          </div>
        </Card>

        {/* Operational Threshold Governance (Read-Only) */}
        <Card
          title="Operational Decision Threshold"
          subtitle="Strictly managed under Phase 4 ML governance"
        >
          <div className="threshold-settings-body">
            <div className="threshold-val-box">
              <span className="number font-mono text-emerald">
                {health?.operational_threshold ?? 0.35}
              </span>
              <span className="caption">Fixed Operational Cutoff (&theta;)</span>
            </div>
            <p className="threshold-desc">
              To guarantee reproducible compliance with offline cross-validation results, the operational
              decision threshold is locked to <strong>0.35</strong>.
              It cannot be arbitrarily changed in the client UI to prevent uncalibrated decision-making.
            </p>
            <div className="threshold-notice">
              <Info size={14} className="text-secondary shrink-0 mt-0.5" />
              <span>
                To evaluate alternative thresholds, inspect the cross-validation trade-off table on the{' '}
                <strong>Model Insights</strong> page.
              </span>
            </div>
          </div>
        </Card>

        {/* API Documentation Quick Links */}
        <Card
          title="Developer & API Resources"
          subtitle="Direct links to interactive API documentation and health endpoints"
        >
          <div className="api-links-grid">
            <a
              href={`${baseUrl}/docs`}
              target="_blank"
              rel="noreferrer"
              className="api-link-card"
            >
              <div className="link-title-row">
                <span className="font-semibold text-charcoal">Swagger UI Documentation</span>
                <ExternalLink size={14} className="text-secondary" />
              </div>
              <p className="link-desc">
                Interactive OpenAPI explorer at <code>/docs</code> for inspecting schemas and executing test requests.
              </p>
            </a>

            <a
              href={`${baseUrl}/redoc`}
              target="_blank"
              rel="noreferrer"
              className="api-link-card"
            >
              <div className="link-title-row">
                <span className="font-semibold text-charcoal">ReDoc Specification</span>
                <ExternalLink size={14} className="text-secondary" />
              </div>
              <p className="link-desc">
                Clean, readable API endpoint reference documentation at <code>/redoc</code>.
              </p>
            </a>

            <a
              href={`${baseUrl}/health`}
              target="_blank"
              rel="noreferrer"
              className="api-link-card"
            >
              <div className="link-title-row">
                <span className="font-semibold text-charcoal">Raw /health JSON Endpoint</span>
                <ExternalLink size={14} className="text-secondary" />
              </div>
              <p className="link-desc">
                Standard health probe endpoint returning model readiness status and metadata.
              </p>
            </a>
          </div>
        </Card>

        {/* System & Runtime Environment */}
        <Card title="Environment & Build Details">
          <div className="settings-field-list">
            <div className="settings-field-row">
              <span className="field-name">Frontend Framework:</span>
              <span className="field-val">React 19 + TypeScript (Vite 8)</span>
            </div>
            <div className="settings-field-row">
              <span className="field-name">Backend Framework:</span>
              <span className="field-val">FastAPI + Uvicorn + Pydantic v2</span>
            </div>
            <div className="settings-field-row">
              <span className="field-name">Explainability Library:</span>
              <span className="field-val">SHAP 0.49 (TreeExplainer)</span>
            </div>
            <div className="settings-field-row">
              <span className="field-name">UI Design System:</span>
              <span className="field-val">Enterprise Light SaaS (Charcoal #111827 + Emerald #10B981)</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

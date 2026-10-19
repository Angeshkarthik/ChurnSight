import React from 'react';
import {
  CheckCircle2,
  ShieldAlert,
} from 'lucide-react';
import type { PredictionResponse } from '../../types/api';
import { Badge } from '../common/Badge';
import { Card } from '../common/Card';


interface PredictionCardProps {
  prediction: PredictionResponse;
  customerName?: string;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  prediction,
  customerName,
}) => {
  const churnPct = (prediction.churn_probability * 100).toFixed(1);
  const retentionPct = (prediction.retention_probability * 100).toFixed(1);
  const isHighRisk = prediction.predicted_class === 1;
  const thresholdPct = (prediction.threshold_used * 100).toFixed(0);

  return (
    <Card
      title={
        <div className="prediction-title-row">
          <span>Inference Result</span>
          {customerName && <span className="customer-tag">{customerName}</span>}
        </div>
      }
      subtitle={`Evaluated with Champion Model: ${prediction.model_name}`}
      action={
        <Badge
          variant={isHighRisk ? 'high-risk' : 'low-risk'}
          size="md"
          dot
        >
          {isHighRisk ? 'High Risk (Churn Alert)' : 'Low Risk (Retained)'}
        </Badge>
      }
      className="prediction-card-highlight"
    >
      <div className="prediction-metrics-grid">
        {/* Metric 1: Churn Probability */}
        <div className="prediction-stat-box">
          <div className="stat-label-row">
            <span className="stat-label">Churn Probability</span>
            <span className="stat-formula">P(Churn=1|X)</span>
          </div>
          <div className={`stat-value ${isHighRisk ? 'text-risk-high' : 'text-risk-low'}`}>
            {churnPct}%
          </div>
          <div className="stat-caption">Continuous model estimate</div>
        </div>

        {/* Metric 2: Retention Probability */}
        <div className="prediction-stat-box">
          <div className="stat-label-row">
            <span className="stat-label">Retention Probability</span>
            <span className="stat-formula">1 - P(Churn)</span>
          </div>
          <div className="stat-value text-charcoal">{retentionPct}%</div>
          <div className="stat-caption">Complementary likelihood</div>
        </div>

        {/* Metric 3: Decision Cutoff */}
        <div className="prediction-stat-box">
          <div className="stat-label-row">
            <span className="stat-label">Operational Cutoff</span>
            <span className="stat-formula">&theta; Decision Rule</span>
          </div>
          <div className="stat-value text-emerald">{prediction.threshold_used.toFixed(2)}</div>
          <div className="stat-caption">OOF F1-score optimized</div>
        </div>
      </div>

      {/* Visual Risk Meter / Gauge Bar */}
      <div className="saas-risk-meter-container">
        <div className="meter-header">
          <span className="meter-label">Model Risk Spectrum</span>
          <span className="meter-status-text">
            {churnPct}% probability is{' '}
            <strong>{isHighRisk ? 'above' : 'below'}</strong> operational threshold ({thresholdPct}%)
          </span>
        </div>

        <div className="meter-track" role="progressbar" aria-valuenow={parseFloat(churnPct)} aria-valuemin={0} aria-valuemax={100}>
          {/* Threshold Marker Indicator */}
          <div
            className="meter-threshold-marker"
            style={{ left: `${thresholdPct}%` }}
            title={`Operational Decision Cutoff: ${thresholdPct}%`}
          >
            <div className="threshold-line" />
            <span className="threshold-tag">Threshold: {thresholdPct}%</span>
          </div>

          {/* Probability Fill Bar */}
          <div
            className={`meter-fill ${isHighRisk ? 'meter-fill-high' : 'meter-fill-low'}`}
            style={{ width: `${Math.min(parseFloat(churnPct), 100)}%` }}
          />
        </div>

        <div className="meter-footer-ticks">
          <span>0% (Safe)</span>
          <span className="tick-mid">50% (Standard)</span>
          <span>100% (Certain Churn)</span>
        </div>
      </div>

      {/* Operational Threshold Governance Explainer */}
      <div className="saas-threshold-callout">
        <div className="callout-icon">
          {isHighRisk ? (
            <ShieldAlert size={18} className="text-risk-high" />
          ) : (
            <CheckCircle2 size={18} className="text-risk-low" />
          )}
        </div>
        <div className="callout-body">
          <div className="callout-title">
            Classification Rule: Probability &ge; {prediction.threshold_used.toFixed(2)} = Predicted Churn
          </div>
          <p className="callout-text">
            Because telco churn is inherently imbalanced (~26.5% base rate), standard 0.50 cutoffs miss too many at-risk customers.
            The operational threshold of <strong>{prediction.threshold_used.toFixed(2)}</strong> catches <strong>71.4%</strong> of churners in holdout evaluation.
            <em>This threshold is a classification decision rule, not a guarantee of churn certainty.</em>
          </p>
        </div>
      </div>
    </Card>
  );
};

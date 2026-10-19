import React from 'react';
import {
  Activity,
  BarChart2,
  CheckCircle2,
  Clock,
  Info,
  Layers,
  Scale,
  ShieldAlert,
  Sparkles,
  UserCheck,
} from 'lucide-react';

import { PredictionCard } from '../components/analysis/PredictionCard';
import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import { Card } from '../components/common/Card';
import { useAnalysis } from '../context/AnalysisContext';
import { MODEL_METADATA } from '../data/modelArtifacts';

export const DashboardPage: React.FC = () => {
  const {
    setActiveTab,
    health,
    latestAnalysis,
    sessionHistory,
    setSelectedRecordForModal,
  } = useAnalysis();

  const isModelReady = health?.model_ready === true;
  const holdout = MODEL_METADATA.holdout_operational_metrics;

  return (
    <div className="saas-page-container">
      {/* 5.1 Welcome Banner */}
      <div className="saas-welcome-banner">
        <div className="welcome-banner-content">
          <div className="welcome-banner-header">
            <span className="welcome-tag">ChurnSight Platform</span>
            {isModelReady ? (
              <Badge variant="healthy" size="sm" dot>
                Live Backend Connected
              </Badge>
            ) : (
              <Badge variant="degraded" size="sm" dot>
                Backend Degraded / Reconnecting
              </Badge>
            )}
          </div>
          <h2 className="welcome-title">ChurnSight — Customer Intelligence & Churn Prevention Platform</h2>
          <p className="welcome-tagline-text">See risk. Understand why.</p>
          <p className="welcome-desc">
            Analyze customer churn risk, understand individual predictions, and review the factors
            influencing model outputs using verified Scikit-learn champion pipelines and SHAP local interpretability.
          </p>
          <div className="welcome-actions">
            <Button
              variant="primary"
              size="sm"
              icon={<UserCheck size={15} />}
              onClick={() => setActiveTab('analysis')}
            >
              Analyze Customer
            </Button>
            <Button
              variant="outline"
              size="sm"
              icon={<BarChart2 size={15} />}
              onClick={() => setActiveTab('insights')}
            >
              View Model Benchmarks
            </Button>
          </div>
        </div>
      </div>

      {/* 5.2 KPI Metrics Row */}
      <div className="saas-kpi-grid">
        {/* KPI 1: Operational Cutoff */}
        <div className="saas-kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">Operational Cutoff</span>
            <span className="kpi-icon-wrap text-emerald">
              <Scale size={18} />
            </span>
          </div>
          <div className="kpi-value">
            {health?.operational_threshold?.toFixed(2) ?? MODEL_METADATA.operational_threshold.toFixed(2)}
          </div>
          <div className="kpi-subtext">
            <span>Probability threshold for churn alerts</span>
          </div>
        </div>

        {/* KPI 2: Latest Churn Probability */}
        <div className="saas-kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">Latest Churn Probability</span>
            <span className="kpi-icon-wrap text-secondary">
              <Activity size={18} />
            </span>
          </div>
          <div className="kpi-value">
            {latestAnalysis ? (
              <span
                className={
                  latestAnalysis.prediction.predicted_class === 1
                    ? 'text-risk-high'
                    : 'text-risk-low'
                }
              >
                {(latestAnalysis.prediction.churn_probability * 100).toFixed(1)}%
              </span>
            ) : (
              <span className="text-muted">--</span>
            )}
          </div>
          <div className="kpi-subtext">
            <span>
              {latestAnalysis
                ? `Latest: ${latestAnalysis.customerName || latestAnalysis.id}`
                : 'No customers analyzed yet'}
            </span>
          </div>
        </div>

        {/* KPI 3: Latest Predicted Class */}
        <div className="saas-kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">Latest Risk Classification</span>
            <span className="kpi-icon-wrap text-secondary">
              <ShieldAlert size={18} />
            </span>
          </div>
          <div className="kpi-value text-base font-semibold">
            {latestAnalysis ? (
              <Badge
                variant={
                  latestAnalysis.prediction.predicted_class === 1
                    ? 'high-risk'
                    : 'low-risk'
                }
                size="md"
              >
                {latestAnalysis.prediction.predicted_class === 1
                  ? 'High Risk (Alert)'
                  : 'Low Risk (Retained)'}
              </Badge>
            ) : (
              <span className="text-muted">--</span>
            )}
          </div>
          <div className="kpi-subtext">
            <span>
              {latestAnalysis
                ? `Based on >= 0.35 threshold rule`
                : 'Ready for customer scoring'}
            </span>
          </div>
        </div>

        {/* KPI 4: Backend Model Engine */}
        <div className="saas-kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">Champion Estimator</span>
            <span className="kpi-icon-wrap text-emerald">
              <Layers size={18} />
            </span>
          </div>
          <div className="kpi-value text-base font-semibold text-charcoal">
            {health?.model_name ?? MODEL_METADATA.model_name}
          </div>
          <div className="kpi-subtext">
            <span className="text-emerald font-medium">
              {isModelReady ? 'Pipeline & TreeExplainer Ready' : 'Service Connecting...'}
            </span>
          </div>
        </div>
      </div>

      {/* 5.3 Main Dashboard Content (Balanced Two-Column Layout) */}
      <div className="saas-dashboard-grid">
        {/* Left / Main Column */}
        <div className="dashboard-main-col">
          {latestAnalysis ? (
            <div className="latest-result-wrapper">
              <div className="section-label-row">
                <h3 className="section-title">Latest Customer Scoring Result</h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setActiveTab('analysis')}
                >
                  Analyze Another Customer
                </Button>
              </div>
              <PredictionCard
                prediction={latestAnalysis.prediction}
                customerName={latestAnalysis.customerName || latestAnalysis.id}
              />
            </div>
          ) : (
            /* 5.4 Polished Empty State */
            <Card className="empty-analysis-card">
              <div className="saas-empty-state">
                <div className="empty-icon-wrap">
                  <UserCheck size={22} className="text-secondary" />
                </div>
                <h4 className="empty-title">No Customer Analyzed In This Session</h4>
                <p className="empty-desc">
                  Enter a customer's contract, service, and billing profile to calculate real-time
                  churn risk and inspect local SHAP explainability factors.
                </p>
                <div className="empty-actions">
                  <Button
                    variant="primary"
                    size="sm"
                    icon={<Sparkles size={15} />}
                    onClick={() => setActiveTab('analysis')}
                  >
                    Open Customer Analysis
                  </Button>
                </div>
              </div>
            </Card>
          )}

          {/* Session History Table Card */}
          <Card
            title="Current Session Activity"
            subtitle="Customer records analyzed in the current browser session"
            action={
              sessionHistory.length > 0 ? (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setActiveTab('predictions')}
                >
                  View All ({sessionHistory.length})
                </Button>
              ) : null
            }
          >
            {sessionHistory.length === 0 ? (
              <div className="empty-table-state">
                <Clock size={18} className="text-muted" />
                <span>No analyses performed yet this session.</span>
              </div>
            ) : (
              <div className="saas-table-wrapper">
                <table className="saas-table">
                  <thead>
                    <tr>
                      <th>Customer / Ref</th>
                      <th>Contract</th>
                      <th>Tenure</th>
                      <th>Monthly</th>
                      <th>Churn Prob</th>
                      <th>Classification</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sessionHistory.slice(0, 5).map((rec) => {
                      const isHigh = rec.prediction.predicted_class === 1;
                      return (
                        <tr key={rec.id}>
                          <td className="font-medium">
                            {rec.customerName || rec.id}
                          </td>
                          <td>{rec.input.Contract}</td>
                          <td>{rec.input.tenure} mo</td>
                          <td>${rec.input.MonthlyCharges.toFixed(2)}</td>
                          <td className={`font-mono font-semibold ${isHigh ? 'text-risk-high' : 'text-risk-low'}`}>
                            {(rec.prediction.churn_probability * 100).toFixed(1)}%
                          </td>
                          <td>
                            <Badge variant={isHigh ? 'high-risk' : 'low-risk'} size="sm">
                              {isHigh ? 'High Risk' : 'Low Risk'}
                            </Badge>
                          </td>
                          <td>
                            <button
                              className="table-action-link"
                              onClick={() => {
                                setSelectedRecordForModal(rec);
                              }}
                            >
                              Details
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>

        {/* Right / Secondary Column */}
        <div className="dashboard-side-col">
          {/* Card 1: Operational Cutoff Rule */}
          <Card
            title="Operational Threshold Rule"
            subtitle="Governance policy for retention intervention"
          >
            <div className="threshold-explainer-body">
              <div className="cutoff-metric-display">
                <span className="cutoff-number">&ge; 0.35</span>
                <span className="cutoff-tagline">Trigger Churn Intervention</span>
              </div>
              <p className="cutoff-explanation">
                Standard classification models default to 0.50 cutoffs. In customer retention,
                failing to identify a churning customer (false negative) is significantly more expensive
                than reaching out to a customer who wasn't planning to leave (false positive).
              </p>
              <div className="cutoff-bullet-list">
                <div className="cutoff-bullet">
                  <CheckCircle2 size={16} className="text-emerald shrink-0 mt-0.5" />
                  <span><strong>71.4% Recall:</strong> Identifies almost 3 out of 4 true churners in holdout testing.</span>
                </div>
                <div className="cutoff-bullet">
                  <CheckCircle2 size={16} className="text-emerald shrink-0 mt-0.5" />
                  <span><strong>Optimized on OOF:</strong> Tuned strictly on cross-validation out-of-fold data to maximize F1-score.</span>
                </div>
              </div>
            </div>
          </Card>

          {/* Card 2: Verified Holdout Performance Summary */}
          <Card
            title="Champion Model Holdout Performance"
            subtitle="Verified offline evaluation on unseen holdout partition (N=1,409)"
            action={
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setActiveTab('insights')}
              >
                Full Metrics
              </Button>
            }
          >
            <div className="holdout-metrics-grid">
              <div className="holdout-metric-cell">
                <span className="cell-label">Recall (Sensitivity)</span>
                <span className="cell-val text-emerald">{(holdout.recall * 100).toFixed(1)}%</span>
                <span className="cell-desc">% of true churners detected</span>
              </div>
              <div className="holdout-metric-cell">
                <span className="cell-label">Precision</span>
                <span className="cell-val">{(holdout.precision * 100).toFixed(1)}%</span>
                <span className="cell-desc">Reliability of alerts</span>
              </div>
              <div className="holdout-metric-cell">
                <span className="cell-label">F1-Score</span>
                <span className="cell-val">{(holdout.f1_score * 100).toFixed(1)}%</span>
                <span className="cell-desc">Harmonic balance</span>
              </div>
              <div className="holdout-metric-cell">
                <span className="cell-label">PR-AUC (Avg Prec)</span>
                <span className="cell-val">{(holdout.pr_auc * 100).toFixed(1)}%</span>
                <span className="cell-desc">Imbalanced benchmark</span>
              </div>
            </div>
            <div className="model-offline-disclaimer">
              <Info size={14} className="text-muted shrink-0 mt-0.5" />
              <span>
                These figures represent verified offline evaluation on the independent holdout partition,
                not live customer population estimates.
              </span>
            </div>
          </Card>

          {/* Card 3: Quick Guide to SHAP Interpretation */}
          <Card
            title="How SHAP Values Are Interpreted"
            subtitle="Guidance for customer success & retention teams"
          >
            <div className="guide-content">
              <p className="guide-p">
                <strong>Log-Odds Scale:</strong> Individual feature contributions quantify mathematical pushes
                on the model's decision margin, not linear percentage changes.
              </p>
              <div className="guide-legend">
                <div className="legend-item">
                  <span className="legend-color legend-red" />
                  <span><strong>+ Positive Contribution:</strong> Pushes prediction toward churn.</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color legend-green" />
                  <span><strong>- Negative Contribution:</strong> Pushes prediction away from churn (retention).</span>
                </div>
              </div>
              <p className="guide-disclaimer">
                Attribution reflects patterns in historical data, not confirmed causal levers. Always combine
                model predictions with business domain context.
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

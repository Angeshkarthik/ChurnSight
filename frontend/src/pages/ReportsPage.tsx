import React from 'react';
import {
  BarChart3,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  FileText,
  ShieldAlert,
  Sparkles,
  Users,
} from 'lucide-react';
import { Button } from '../components/common/Button';
import { Card } from '../components/common/Card';

import { useAnalysis } from '../context/AnalysisContext';

export const ReportsPage: React.FC = () => {
  const { sessionHistory, setActiveTab } = useAnalysis();

  const totalCount = sessionHistory.length;
  const highRiskCount = sessionHistory.filter(
    (r) => r.prediction.predicted_class === 1
  ).length;
  const lowRiskCount = totalCount - highRiskCount;
  const avgProb =
    totalCount > 0
      ? (
          sessionHistory.reduce(
            (acc, r) => acc + r.prediction.churn_probability,
            0
          ) / totalCount
        ) * 100
      : 0;

  const handleExportJson = () => {
    if (sessionHistory.length === 0) return;
    const blob = new Blob([JSON.stringify(sessionHistory, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `churn-session-report-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportCsv = () => {
    if (sessionHistory.length === 0) return;

    // Build CSV headers
    const headers = [
      'Record_ID',
      'Timestamp',
      'Customer_Name',
      'Churn_Probability',
      'Predicted_Class',
      'Risk_Level',
      'Threshold_Used',
      'Contract',
      'Tenure_Months',
      'Internet_Service',
      'Monthly_Charges',
      'Total_Charges',
    ];

    const rows = sessionHistory.map((rec) => [
      `"${rec.id}"`,
      `"${rec.timestamp}"`,
      `"${rec.customerName || ''}"`,
      (rec.prediction.churn_probability * 100).toFixed(2),
      rec.prediction.predicted_class,
      `"${rec.prediction.risk_level}"`,
      rec.prediction.threshold_used,
      `"${rec.input.Contract}"`,
      rec.input.tenure,
      `"${rec.input.InternetService}"`,
      rec.input.MonthlyCharges.toFixed(2),
      rec.input.TotalCharges.toFixed(2),
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join(
      '\n'
    );
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `churn-session-report-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="saas-page-container">
      <div className="page-header-block">
        <h2 className="page-main-title">Session Analytical Reports</h2>
        <p className="page-main-desc">
          Review risk distributions across customer records scored during the current session.
          Export verified session scoring data for offline reporting and team review.
        </p>
      </div>

      {/* Summary KPI Cards */}
      <div className="saas-kpi-grid">
        <div className="saas-kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">Session Customers Scored</span>
            <span className="kpi-icon-wrap text-secondary">
              <Users size={18} />
            </span>
          </div>
          <div className="kpi-value">{totalCount}</div>
          <div className="kpi-subtext">Active session records</div>
        </div>

        <div className="saas-kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">High Churn Risk (&ge;0.35)</span>
            <span className="kpi-icon-wrap text-risk-high">
              <ShieldAlert size={18} />
            </span>
          </div>
          <div className="kpi-value text-risk-high">{highRiskCount}</div>
          <div className="kpi-subtext">
            {totalCount > 0
              ? `${((highRiskCount / totalCount) * 100).toFixed(1)}% of session total`
              : 'No analyses yet'}
          </div>
        </div>

        <div className="saas-kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">Low Churn Risk (&lt;0.35)</span>
            <span className="kpi-icon-wrap text-emerald">
              <CheckCircle2 size={18} />
            </span>
          </div>
          <div className="kpi-value text-emerald">{lowRiskCount}</div>
          <div className="kpi-subtext">
            {totalCount > 0
              ? `${((lowRiskCount / totalCount) * 100).toFixed(1)}% of session total`
              : 'No analyses yet'}
          </div>
        </div>

        <div className="saas-kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">Mean Churn Probability</span>
            <span className="kpi-icon-wrap text-secondary">
              <BarChart3 size={18} />
            </span>
          </div>
          <div className="kpi-value font-mono">
            {totalCount > 0 ? `${avgProb.toFixed(1)}%` : '--'}
          </div>
          <div className="kpi-subtext">Average across scored batch</div>
        </div>
      </div>

      {/* Export Action Card */}
      <Card
        title="Session Data Export"
        subtitle="Download raw scoring records and predictions currently held in browser session state"
      >
        {totalCount === 0 ? (
          <div className="saas-empty-state">
            <div className="empty-icon-wrap">
              <FileSpreadsheet size={32} className="text-secondary" />
            </div>
            <h4 className="empty-title">No Session Records to Export</h4>
            <p className="empty-desc">
              Once you analyze customers, their inputs and predictions will be ready for export here
              in CSV and JSON formats.
            </p>
            <Button
              variant="primary"
              size="md"
              icon={<Sparkles size={16} />}
              onClick={() => setActiveTab('analysis')}
            >
              Analyze a Customer
            </Button>
          </div>
        ) : (
          <div className="export-controls-row">
            <div className="export-info">
              <p className="text-secondary mb-2">
                Ready to export <strong>{totalCount}</strong> customer scoring record(s).
                Exports contain all 19 raw predictor features, calculated churn probabilities, and
                operational threshold classifications.
              </p>
            </div>
            <div className="export-btn-group">
              <Button
                variant="outline"
                size="md"
                icon={<Download size={16} />}
                onClick={handleExportCsv}
              >
                Export CSV (.csv)
              </Button>
              <Button
                variant="primary"
                size="md"
                icon={<FileText size={16} />}
                onClick={handleExportJson}
              >
                Export JSON (.json)
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

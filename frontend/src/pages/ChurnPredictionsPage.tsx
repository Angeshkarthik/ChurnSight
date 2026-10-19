import React, { useState } from 'react';
import {
  Clock,
  Eye,
  Sparkles,
  Trash2,
} from 'lucide-react';

import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import { Card } from '../components/common/Card';
import { useAnalysis } from '../context/AnalysisContext';

export const ChurnPredictionsPage: React.FC = () => {
  const {
    sessionHistory,
    clearSessionHistory,
    setActiveTab,
    setSelectedRecordForModal,
  } = useAnalysis();

  const [riskFilter, setRiskFilter] = useState<'all' | 'high' | 'low'>('all');

  const filteredRecords = sessionHistory.filter((rec) => {
    if (riskFilter === 'high') return rec.prediction.predicted_class === 1;
    if (riskFilter === 'low') return rec.prediction.predicted_class === 0;
    return true;
  });

  return (
    <div className="saas-page-container">
      <div className="page-header-block">
        <h2 className="page-main-title">Session Churn Predictions</h2>
        <p className="page-main-desc">
          Audit and review all customer scoring analyses executed during the current browser session.
          Detailed SHAP attribution factors can be inspected for each prediction.
        </p>
      </div>

      <Card
        title={
          <div className="history-card-header">
            <span>Session Scoring Records ({filteredRecords.length})</span>
            <div className="history-filter-group">
              <button
                type="button"
                className={`filter-btn ${riskFilter === 'all' ? 'filter-btn-active' : ''}`}
                onClick={() => setRiskFilter('all')}
              >
                All ({sessionHistory.length})
              </button>
              <button
                type="button"
                className={`filter-btn ${riskFilter === 'high' ? 'filter-btn-active' : ''}`}
                onClick={() => setRiskFilter('high')}
              >
                High Risk ({sessionHistory.filter((r) => r.prediction.predicted_class === 1).length})
              </button>
              <button
                type="button"
                className={`filter-btn ${riskFilter === 'low' ? 'filter-btn-active' : ''}`}
                onClick={() => setRiskFilter('low')}
              >
                Low Risk ({sessionHistory.filter((r) => r.prediction.predicted_class === 0).length})
              </button>
            </div>
          </div>
        }
        action={
          sessionHistory.length > 0 ? (
            <Button
              variant="outline"
              size="sm"
              icon={<Trash2 size={14} />}
              onClick={clearSessionHistory}
            >
              Clear Session
            </Button>
          ) : null
        }
      >
        {sessionHistory.length === 0 ? (
          <div className="saas-empty-state">
            <div className="empty-icon-wrap">
              <Clock size={32} className="text-secondary" />
            </div>
            <h4 className="empty-title">No Predictions Logged Yet</h4>
            <p className="empty-desc">
              Customer predictions made during this session will be recorded here for quick comparison
              and SHAP explanation inspection.
            </p>
            <Button
              variant="primary"
              size="md"
              icon={<Sparkles size={16} />}
              onClick={() => setActiveTab('analysis')}
            >
              Score a Customer
            </Button>
          </div>
        ) : filteredRecords.length === 0 ? (
          <div className="empty-table-state">
            <p>No records match the current filter selection.</p>
          </div>
        ) : (
          <div className="saas-table-wrapper">
            <table className="saas-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Customer Reference</th>
                  <th>Contract Term</th>
                  <th>Tenure</th>
                  <th>Monthly Spend</th>
                  <th>Churn Probability</th>
                  <th>Classification (&theta;&ge;0.35)</th>
                  <th>SHAP Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredRecords.map((rec) => {
                  const isHigh = rec.prediction.predicted_class === 1;
                  const timeFormatted = new Date(rec.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                  });

                  return (
                    <tr key={rec.id}>
                      <td className="text-secondary font-mono text-sm">{timeFormatted}</td>
                      <td className="font-medium text-charcoal">{rec.customerName || rec.id}</td>
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
                        {rec.explanation ? (
                          <span className="text-emerald text-sm font-medium">Available</span>
                        ) : (
                          <span className="text-muted text-sm">Unavailable</span>
                        )}
                      </td>
                      <td>
                        <Button
                          variant="outline"
                          size="sm"
                          icon={<Eye size={13} />}
                          onClick={() => setSelectedRecordForModal(rec)}
                        >
                          Details
                        </Button>
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
  );
};

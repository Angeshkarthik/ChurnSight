import React from 'react';
import { ExternalLink, RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';
import { useAnalysis } from '../../context/AnalysisContext';
import { Badge } from '../common/Badge';

const TAB_TITLES: Record<string, { title: string; subtitle: string }> = {
  dashboard: {
    title: 'ChurnSight Dashboard',
    subtitle: 'Customer Intelligence & Churn Prevention Platform • See risk. Understand why.',
  },
  analysis: {
    title: 'Customer Analysis',
    subtitle: 'Score customer profiles against trained Gradient Boosting champion model',
  },
  predictions: {
    title: 'Churn Predictions',
    subtitle: 'Session scoring history and retention risk classifications',
  },
  insights: {
    title: 'Model Insights',
    subtitle: 'Offline holdout benchmark evaluation and governance metadata',
  },
  reports: {
    title: 'Analytics Reports',
    subtitle: 'Session risk distribution and analytical data export',
  },
  settings: {
    title: 'Platform Settings',
    subtitle: 'Backend connectivity, operational thresholds, and API documentation',
  },
};

export const TopNav: React.FC = () => {
  const { activeTab, health, isHealthLoading, refreshHealth } = useAnalysis();
  const current = TAB_TITLES[activeTab] || {
    title: 'ChurnSight',
    subtitle: 'Customer Intelligence & Churn Prevention Platform • See risk. Understand why.',
  };

  const isModelReady = health?.model_ready === true;

  return (
    <header className="saas-topnav">
      <div className="saas-topnav-left">
        <div className="saas-page-header">
          <h1 className="saas-page-title">{current.title}</h1>
          <span className="saas-page-subtitle">{current.subtitle}</span>
        </div>
      </div>

      <div className="saas-topnav-right">
        {/* Backend & Model Readiness Status */}
        <div className="saas-health-widget">
          <div className="health-widget-info">
            <span className="health-widget-label">Backend Status</span>
            <div className="health-widget-status">
              {isHealthLoading ? (
                <Badge variant="neutral" size="sm" dot>
                  Checking...
                </Badge>
              ) : isModelReady ? (
                <Badge variant="healthy" size="sm" dot>
                  Model Ready (Th={health?.operational_threshold ?? 0.35})
                </Badge>
              ) : (
                <Badge variant="degraded" size="sm" dot>
                  Service Degraded
                </Badge>
              )}
            </div>
          </div>
          <button
            className="saas-icon-btn"
            onClick={() => refreshHealth()}
            title="Refresh API Health"
            disabled={isHealthLoading}
          >
            <RefreshCw size={14} className={isHealthLoading ? 'animate-spin' : ''} />
          </button>
        </div>

        {/* API Docs Quick Link */}
        <a
          href={`${apiClient.getBaseUrl()}/docs`}
          target="_blank"
          rel="noreferrer"
          className="saas-link-btn"
          title="Open FastAPI Swagger Documentation"
        >
          <span>API Docs</span>
          <ExternalLink size={12} />
        </a>
      </div>
    </header>
  );
};

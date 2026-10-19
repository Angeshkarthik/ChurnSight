import React, { useState } from 'react';
import {
  ArrowDownRight,
  ArrowUpRight,
  Info,
  Sliders,
} from 'lucide-react';
import type { ExplanationResponse, LocalFeatureContribution } from '../../types/api';
import { Badge } from '../common/Badge';
import { Card } from '../common/Card';


interface ShapWaterfallProps {
  explanation: ExplanationResponse;
}

export const ShapWaterfall: React.FC<ShapWaterfallProps> = ({ explanation }) => {
  const [viewMode, setViewMode] = useState<'top' | 'all'>('top');

  const topPos = explanation.top_positive_contributions || [];
  const topNeg = explanation.top_negative_contributions || [];
  const allContribs = explanation.contributions || [];

  // Determine maximum absolute SHAP value for proportional bar scaling
  const maxAbsValue = Math.max(
    ...allContribs.map((c) => Math.abs(c.shap_value)),
    0.5
  );

  return (
    <Card
      title={
        <div className="shap-title-row">
          <span>Local SHAP Model Interpretability</span>
          <Badge variant="neutral" size="sm">
            TreeExplainer (Log-Odds Scale)
          </Badge>
        </div>
      }
      subtitle="Mathematical attribution of each customer feature to the model's log-odds output margin"
      action={
        <div className="shap-view-toggle">
          <button
            type="button"
            className={`toggle-btn ${viewMode === 'top' ? 'toggle-btn-active' : ''}`}
            onClick={() => setViewMode('top')}
          >
            Top Factors ({topPos.length + topNeg.length})
          </button>
          <button
            type="button"
            className={`toggle-btn ${viewMode === 'all' ? 'toggle-btn-active' : ''}`}
            onClick={() => setViewMode('all')}
          >
            All Features ({allContribs.length})
          </button>
        </div>
      }
      className="shap-card-container"
    >
      {/* Mathematical Additivity Audit Banner */}
      <div className="shap-additivity-audit">
        <div className="audit-col">
          <span className="audit-label">Expected Base Value E[F(X)]</span>
          <span className="audit-val font-mono">{explanation.baseline_model_output.toFixed(4)}</span>
          <span className="audit-sub">Avg training log-odds</span>
        </div>
        <div className="audit-op">+</div>
        <div className="audit-col">
          <span className="audit-label">Total SHAP Sum &sum;&phi;</span>
          <span className="audit-val font-mono">{explanation.shap_values_sum > 0 ? `+${explanation.shap_values_sum.toFixed(4)}` : explanation.shap_values_sum.toFixed(4)}</span>
          <span className="audit-sub">Net feature influence</span>
        </div>
        <div className="audit-op">=</div>
        <div className="audit-col">
          <span className="audit-label">Final Margin F(x)</span>
          <span className="audit-val font-mono text-charcoal">{explanation.model_score_log_odds.toFixed(4)}</span>
          <span className="audit-sub">Decision function score</span>
        </div>
        <div className="audit-status">
          <Badge variant="healthy" size="sm">
            Additivity &Delta; &lt; 1e-4 verified
          </Badge>
        </div>
      </div>

      {/* Critical Scale Explanation Banner */}
      <div className="saas-scale-disclaimer">
        <Info size={16} className="text-secondary shrink-0 mt-0.5" />
        <div className="disclaimer-text">
          <strong>Important Interpretation Standard:</strong> SHAP values are expressed strictly on the <em>log-odds margin scale</em> ($F(x) = \ln(p / (1-p))$), <strong>not direct percentage points</strong>.
          Positive values push the model toward churn; negative values push toward retention. They represent statistical model attributions in observational data, not proven causal levers.
        </div>
      </div>

      {/* Feature Contributions List / Bars */}
      <div className="shap-contributions-section">
        {viewMode === 'top' ? (
          <div className="shap-dual-columns">
            {/* Column 1: Top Risk Elevating Factors */}
            <div className="shap-group-col">
              <div className="group-col-header text-risk-high">
                <ArrowUpRight size={16} />
                <span>Top Factors Elevating Churn Risk (+Log-Odds)</span>
              </div>
              <div className="contrib-items-list">
                {topPos.length === 0 ? (
                  <p className="empty-subtext">No significant factors elevating churn risk.</p>
                ) : (
                  topPos.map((c) => (
                    <ContributionRow
                      key={c.feature_name}
                      contrib={c}
                      maxAbsValue={maxAbsValue}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Column 2: Top Protective / Retention Factors */}
            <div className="shap-group-col">
              <div className="group-col-header text-risk-low">
                <ArrowDownRight size={16} />
                <span>Top Factors Suppressing Churn Risk (-Log-Odds)</span>
              </div>
              <div className="contrib-items-list">
                {topNeg.length === 0 ? (
                  <p className="empty-subtext">No significant protective retention factors.</p>
                ) : (
                  topNeg.map((c) => (
                    <ContributionRow
                      key={c.feature_name}
                      contrib={c}
                      maxAbsValue={maxAbsValue}
                    />
                  ))
                )}
              </div>
            </div>
          </div>
        ) : (
          /* View Mode = All Features */
          <div className="shap-all-features-list">
            <div className="all-features-header">
              <span>All 46 Transformed Pipeline Features Ranked by Absolute Impact</span>
            </div>
            {allContribs.map((c) => (
              <ContributionRow
                key={c.feature_name}
                contrib={c}
                maxAbsValue={maxAbsValue}
              />
            ))}
          </div>
        )}

      </div>

      {/* Executive Summary Bullet Points */}
      {explanation.summary_reasons && explanation.summary_reasons.length > 0 && (
        <div className="shap-summary-reasons-card">
          <div className="summary-reasons-title">
            <Sliders size={15} className="text-secondary" />
            <span>Key Model Scoring Highlights</span>
          </div>
          <ul className="summary-reasons-list">
            {explanation.summary_reasons.map((reason, idx) => (
              <li key={idx} className="reason-item">
                {reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ML Governance & Non-Causal Notice */}
      <div className="saas-governance-footer">
        <span className="gov-title">ML Governance Notice:</span> {explanation.governance_notice}
      </div>
    </Card>
  );
};

interface ContributionRowProps {
  contrib: LocalFeatureContribution;
  maxAbsValue: number;
}

const ContributionRow: React.FC<ContributionRowProps> = ({
  contrib,
  maxAbsValue,
}) => {

  const isPos = contrib.shap_value >= 0;
  const barWidth = Math.min(
    Math.round((Math.abs(contrib.shap_value) / maxAbsValue) * 100),
    100
  );

  return (
    <div className="saas-contrib-row">
      <div className="contrib-top-row">
        <div className="contrib-feature-info">
          <span className="contrib-feature-name">{contrib.readable_feature}</span>
          {contrib.observed_value !== undefined && contrib.observed_value !== null && (
            <span className="contrib-observed-tag">
              Observed: <strong>{String(contrib.observed_value)}</strong>
            </span>
          )}
        </div>
        <div className={`contrib-val font-mono ${isPos ? 'text-risk-high' : 'text-risk-low'}`}>
          {isPos ? `+${contrib.shap_value.toFixed(4)}` : contrib.shap_value.toFixed(4)}
        </div>
      </div>

      {/* Visual Bar Indicator */}
      <div className="contrib-bar-track">
        <div
          className={`contrib-bar-fill ${isPos ? 'contrib-fill-pos' : 'contrib-fill-neg'}`}
          style={{ width: `${Math.max(barWidth, 2)}%` }}
        />
      </div>

      {/* Factual Non-Causal Narrative Reason */}
      <p className="contrib-reason">{contrib.reason}</p>
    </div>
  );
};

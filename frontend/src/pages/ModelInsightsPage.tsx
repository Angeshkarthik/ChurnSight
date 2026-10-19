import { Badge } from '../components/common/Badge';
import { Card } from '../components/common/Card';
import { MODEL_METADATA } from '../data/modelArtifacts';


export const ModelInsightsPage: React.FC = () => {
  const meta = MODEL_METADATA;
  const opMetrics = meta.holdout_operational_metrics;
  const stdMetrics = meta.holdout_standard_metrics;

  return (
    <div className="saas-page-container">
      <div className="page-header-block">
        <h2 className="page-main-title">Machine Learning Model Insights & Benchmarks</h2>
        <p className="page-main-desc">
          Verified offline evaluation metrics, Out-Of-Fold (OOF) threshold trade-offs, and governance
          specifications for the deployed champion Gradient Boosting pipeline.
        </p>
      </div>

      {/* Model Specifications Top Cards */}
      <div className="insights-overview-grid">
        <Card title="Champion Pipeline Architecture" className="h-full">
          <div className="spec-item-list">
            <div className="spec-item">
              <span className="spec-label">Estimator Family:</span>
              <span className="spec-val font-semibold text-charcoal">{meta.model_name} Classifier</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Feature Pipeline:</span>
              <span className="spec-val">ColumnTransformer (StandardScaler + OneHotEncoder)</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Input Predictor Features:</span>
              <span className="spec-val">19 Raw Features (3 numeric, 16 categorical)</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Transformed Dimension:</span>
              <span className="spec-val">46 Encoded Columns (unknown category resilient)</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Explainability Engine:</span>
              <span className="spec-val">SHAP TreeExplainer (Exact polynomial tree paths)</span>
            </div>
          </div>
        </Card>

        <Card title="Governance & Data Partitions" className="h-full">
          <div className="spec-item-list">
            <div className="spec-item">
              <span className="spec-label">Development Partition (80%):</span>
              <span className="spec-val">{meta.sample_counts.development_samples.toLocaleString()} samples</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Fresh Holdout Partition (20%):</span>
              <span className="spec-val">{meta.sample_counts.holdout_samples.toLocaleString()} samples (seed=100)</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Model Selection Methodology:</span>
              <span className="spec-val">5-Fold Stratified CV on Dev set (PR-AUC Mean)</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Threshold Tuning Protocol:</span>
              <span className="spec-val">OOF cross-validation F1-score maximization</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Holdout Integrity Guarantee:</span>
              <span className="spec-val text-emerald">Evaluated strictly after freezing model & cutoff</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Benchmark Comparison: Operational (0.35) vs Standard (0.50) */}
      <Card
        title="Offline Holdout Benchmark Comparison"
        subtitle="Independent evaluation on unseen fresh holdout partition (N=1,409)"
        action={
          <Badge variant="emerald" size="sm">
            Holdout Seed = 100
          </Badge>
        }
      >
        <div className="saas-table-wrapper">
          <table className="saas-table">
            <thead>
              <tr>
                <th>Evaluation Metric</th>
                <th>Standard Cutoff (&theta; = 0.50)</th>
                <th>Operational Cutoff (&theta; = 0.35)</th>
                <th>Delta / Business Impact</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="font-medium">
                  Recall (Sensitivity)
                  <span className="cell-sublabel">% of true churners caught</span>
                </td>
                <td className="font-mono">{(stdMetrics.recall * 100).toFixed(1)}%</td>
                <td className="font-mono text-emerald font-semibold">{(opMetrics.recall * 100).toFixed(1)}%</td>
                <td className="text-emerald font-medium">
                  +17.1% more churners detected (267 vs 203)
                </td>
              </tr>
              <tr>
                <td className="font-medium">
                  F1-Score
                  <span className="cell-sublabel">Harmonic balance</span>
                </td>
                <td className="font-mono">{(stdMetrics.f1_score * 100).toFixed(1)}%</td>
                <td className="font-mono text-emerald font-semibold">{(opMetrics.f1_score * 100).toFixed(1)}%</td>
                <td className="text-emerald font-medium">+3.8% improved balance</td>
              </tr>
              <tr>
                <td className="font-medium">
                  Precision (Reliability)
                  <span className="cell-sublabel">% of predicted churners correct</span>
                </td>
                <td className="font-mono">{(stdMetrics.precision * 100).toFixed(1)}%</td>
                <td className="font-mono">{(opMetrics.precision * 100).toFixed(1)}%</td>
                <td className="text-secondary">-8.5% expected trade-off for higher recall</td>
              </tr>
              <tr>
                <td className="font-medium">
                  Accuracy
                  <span className="cell-sublabel">Overall classification rate</span>
                </td>
                <td className="font-mono">{(stdMetrics.accuracy * 100).toFixed(1)}%</td>
                <td className="font-mono">{(opMetrics.accuracy * 100).toFixed(1)}%</td>
                <td className="text-secondary">-2.4% due to proactive sensitivity</td>
              </tr>
              <tr>
                <td className="font-medium">
                  PR-AUC (Average Precision)
                  <span className="cell-sublabel">Area under Precision-Recall curve</span>
                </td>
                <td className="font-mono" colSpan={2}>
                  {(opMetrics.pr_auc * 100).toFixed(1)}% (Threshold-Independent)
                </td>
                <td className="text-secondary">Champion selection benchmark</td>
              </tr>
              <tr>
                <td className="font-medium">
                  ROC-AUC
                  <span className="cell-sublabel">Discrimination ability</span>
                </td>
                <td className="font-mono" colSpan={2}>
                  {(opMetrics.roc_auc * 100).toFixed(1)}% (Threshold-Independent)
                </td>
                <td className="text-secondary">Strong overall ranking capacity</td>
              </tr>
              <tr>
                <td className="font-medium">
                  Confusion Matrix Breakdown
                  <span className="cell-sublabel">[TN | FP | FN | TP]</span>
                </td>
                <td className="font-mono text-sm">
                  TN: {stdMetrics.true_negatives} | FP: {stdMetrics.false_positives}
                  <br />
                  FN: {stdMetrics.false_negatives} | TP: {stdMetrics.true_positives}
                </td>
                <td className="font-mono text-sm text-emerald">
                  TN: {opMetrics.true_negatives} | FP: {opMetrics.false_positives}
                  <br />
                  FN: {opMetrics.false_negatives} | TP: {opMetrics.true_positives}
                </td>
                <td className="text-sm">
                  <strong>64 fewer false negatives</strong> (undetected churners)
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      {/* Out-of-fold Threshold Trade-off Table */}
      <Card
        title="Out-Of-Fold (OOF) Decision Threshold Sweep"
        subtitle="Cross-validation trade-off analysis on Development partition establishing 0.35 cutoff"
      >
        <div className="saas-table-wrapper">
          <table className="saas-table">
            <thead>
              <tr>
                <th>Decision Cutoff (&theta;)</th>
                <th>Recall (Sensitivity)</th>
                <th>Precision</th>
                <th>OOF F1-Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {meta.oof_threshold_tradeoffs.map((row) => {
                const isSelected = row.is_optimal;
                return (
                  <tr
                    key={row.threshold}
                    className={isSelected ? 'table-row-selected' : ''}
                  >
                    <td className="font-mono font-semibold">
                      {row.threshold.toFixed(2)}
                    </td>
                    <td className="font-mono">{(row.recall * 100).toFixed(1)}%</td>
                    <td className="font-mono">{(row.precision * 100).toFixed(1)}%</td>
                    <td className="font-mono font-semibold">
                      {(row.f1_score * 100).toFixed(1)}%
                    </td>
                    <td>
                      {isSelected ? (
                        <Badge variant="emerald" size="sm">
                          Selected Operational Cutoff
                        </Badge>
                      ) : (
                        <span className="text-muted text-sm">Evaluated</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Model Governance and Scope Distinction Card */}
      <div className="insights-footer-grid">
        <Card title="Critical Scope Distinctions">
          <div className="scope-bullet-list">
            <div className="scope-bullet">
              <span className="bullet-num">1</span>
              <div>
                <strong>Global Model Behavior:</strong> Refers to population-wide feature importances across the entire dataset. It identifies general trends but cannot predict one customer's outcome alone.
              </div>
            </div>
            <div className="scope-bullet">
              <span className="bullet-num">2</span>
              <div>
                <strong>Local SHAP Attribution:</strong> Deconstructs one customer's prediction into additive log-odds contributions ($F(x) = F_0 + \sum \phi_i$). It explains why the model scored that individual, not population averages.
              </div>
            </div>
            <div className="scope-bullet">
              <span className="bullet-num">3</span>
              <div>
                <strong>Offline Holdout Evaluation:</strong> Computed on a locked 20% test partition (N=1,409) that was completely isolated during training. It serves as an un-peeked audit of model generalization.
              </div>
            </div>
          </div>
        </Card>

        <Card title="Model Limitations & Governance Guidelines">
          <div className="scope-bullet-list">
            <div className="scope-bullet">
              <span className="bullet-num">&bull;</span>
              <div>
                <strong>Non-Causal Interpretation:</strong> Features with high positive SHAP values are statistical risk correlates in observational data, not proven causal drivers. Changing an attribute will not guarantee retention.
              </div>
            </div>
            <div className="scope-bullet">
              <span className="bullet-num">&bull;</span>
              <div>
                <strong>Probabilistic Output:</strong> Predictions represent calibrated likelihoods P(Churn=1 | X) between 0.0 and 1.0, not binary certainties.

              </div>
            </div>
            <div className="scope-bullet">
              <span className="bullet-num">&bull;</span>
              <div>
                <strong>Domain Context Required:</strong> Frontline customer success representatives must combine model predictions with direct customer feedback and relationship history.
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

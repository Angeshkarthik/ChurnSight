/**
 * Verified Model Governance and Offline Evaluation Metadata.
 * Sourced directly from models/model_metadata.json generated during Phase 4 benchmarking.
 */

export const MODEL_METADATA = {
  model_name: 'Gradient Boosting',
  trained_at_utc: '2026-10-18T10:09:50.306068+00:00',
  selection_methodology: {
    champion_selection:
      '5-Fold Stratified CV on Development set using CV PR-AUC Mean with deterministic tie-breaking',
    threshold_selection:
      'Out-Of-Fold (OOF) cross-validation predictions on Development set maximizing F1-score',
    holdout_evaluation:
      'Independent fresh holdout partition, evaluated strictly on frozen champion model',
  },
  seeds: {
    holdout_seed: 100,
    cv_split_seed: 42,
  },
  sample_counts: {
    development_samples: 5634,
    holdout_samples: 1409,
    total_samples: 7043,
  },
  operational_threshold: 0.35,
  standard_threshold: 0.5,

  // Offline Evaluation on Fresh Holdout Partition (N=1,409) at Operational Threshold 0.35
  holdout_operational_metrics: {
    accuracy: 0.775,
    recall: 0.7139,
    precision: 0.5597,
    f1_score: 0.6275,
    roc_auc: 0.8413,
    pr_auc: 0.6447,
    true_negatives: 825,
    false_positives: 210,
    false_negatives: 107,
    true_positives: 267,
  },

  // Comparison: Offline Evaluation on Fresh Holdout Partition at Standard Threshold 0.50
  holdout_standard_metrics: {
    accuracy: 0.7991,
    recall: 0.5428,
    precision: 0.6444,
    f1_score: 0.5893,
    roc_auc: 0.8413,
    pr_auc: 0.6447,
    true_negatives: 923,
    false_positives: 112,
    false_negatives: 171,
    true_positives: 203,
  },

  // Key Out-Of-Fold Threshold Sweep Points on Development Set
  oof_threshold_tradeoffs: [
    { threshold: 0.2, precision: 0.4886, recall: 0.8441, f1_score: 0.6189 },
    { threshold: 0.25, precision: 0.5161, recall: 0.7926, f1_score: 0.6252 },
    { threshold: 0.3, precision: 0.5452, recall: 0.7418, f1_score: 0.6285 },
    { threshold: 0.35, precision: 0.5756, recall: 0.7003, f1_score: 0.6319, is_optimal: true },
    { threshold: 0.4, precision: 0.6106, recall: 0.6462, f1_score: 0.6279 },
    { threshold: 0.45, precision: 0.6352, recall: 0.5846, f1_score: 0.6088 },
    { threshold: 0.5, precision: 0.6619, recall: 0.5211, f1_score: 0.5831 },
    { threshold: 0.6, precision: 0.7321, recall: 0.404, f1_score: 0.5207 },
  ],
};

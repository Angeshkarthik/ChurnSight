import React, { useRef } from 'react';
import { CustomerForm } from '../components/analysis/CustomerForm';
import { PredictionCard } from '../components/analysis/PredictionCard';
import { ShapWaterfall } from '../components/analysis/ShapWaterfall';
import { Card } from '../components/common/Card';
import { useAnalysis } from '../context/AnalysisContext';
import type { CustomerFeaturesInput } from '../types/api';


export const CustomerAnalysisPage: React.FC = () => {
  const {
    runAnalysis,
    latestAnalysis,
    isAnalyzing,
    analysisError,
  } = useAnalysis();

  const resultsRef = useRef<HTMLDivElement>(null);

  const handleFormSubmit = async (
    data: CustomerFeaturesInput,
    customerName?: string
  ) => {
    try {
      await runAnalysis(data, customerName);
      // Smoothly scroll down to results
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch {
      // Handled via context analysisError
    }
  };

  return (
    <div className="saas-page-container">
      <div className="page-header-block">
        <h2 className="page-main-title">Single Customer Churn Risk Analysis</h2>
        <p className="page-main-desc">
          Enter an individual customer record to evaluate churn probability against the trained
          Gradient Boosting champion pipeline and generate local SHAP attribution explanations.
        </p>
      </div>

      {/* Main Scoring Form */}
      <Card
        title="Customer Input Attributes (19 Raw Predictor Features)"
        subtitle="Values are validated strictly against the model schema. Required fields are marked with an asterisk (*)."
      >
        <CustomerForm
          onSubmit={handleFormSubmit}
          isLoading={isAnalyzing}
          error={analysisError}
        />
      </Card>

      {/* Results Section (Appears after submission) */}
      <div ref={resultsRef}>
        {latestAnalysis && (
          <div className="analysis-results-section">
            <div className="results-divider">
              <span>Customer Prediction & SHAP Analysis</span>
            </div>

            <div className="results-grid">
              {/* Primary Prediction Result Card */}
              <PredictionCard
                prediction={latestAnalysis.prediction}
                customerName={latestAnalysis.customerName || latestAnalysis.id}
              />

              {/* Local SHAP Explanations */}
              {latestAnalysis.explanation ? (
                <ShapWaterfall explanation={latestAnalysis.explanation} />
              ) : (
                <Card title="Explainability Status">
                  <p className="text-secondary">
                    Prediction succeeded, but detailed SHAP explanation was unavailable or timed out.
                  </p>
                </Card>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

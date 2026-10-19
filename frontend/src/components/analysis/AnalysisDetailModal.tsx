import React from 'react';
import type { CustomerAnalysisRecord } from '../../types/api';

import { Modal } from '../common/Modal';
import { PredictionCard } from './PredictionCard';
import { ShapWaterfall } from './ShapWaterfall';

interface AnalysisDetailModalProps {
  record: CustomerAnalysisRecord | null;
  onClose: () => void;
}

export const AnalysisDetailModal: React.FC<AnalysisDetailModalProps> = ({
  record,
  onClose,
}) => {
  if (!record) return null;

  return (
    <Modal
      isOpen={Boolean(record)}
      onClose={onClose}
      title={record.customerName || `Customer Analysis: ${record.id}`}
      subtitle={`Recorded at ${new Date(record.timestamp).toLocaleString()}`}
      maxWidth="xl"
    >
      <div className="modal-analysis-body">
        {/* Prediction Results Card */}
        <PredictionCard
          prediction={record.prediction}
          customerName={record.customerName || record.id}
        />

        {/* Local SHAP Waterfall */}
        {record.explanation && (
          <div className="mt-4">
            <ShapWaterfall explanation={record.explanation} />
          </div>
        )}

        {/* Input Attributes Table */}
        <div className="modal-inputs-section">
          <h4 className="modal-section-title">Submitted Customer Input Features</h4>
          <div className="inputs-grid">
            {Object.entries(record.input).map(([key, val]) => (
              <div key={key} className="input-field-badge">
                <span className="input-key">{key}:</span>
                <span className="input-val font-semibold">{String(val)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Modal>
  );
};

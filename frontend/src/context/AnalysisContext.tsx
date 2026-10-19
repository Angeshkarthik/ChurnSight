import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react';
import { apiClient, ApiError } from '../api/client';
import type {
  CustomerAnalysisRecord,
  CustomerFeaturesInput,
  ExplanationResponse,
  HealthResponse,
  PredictionResponse,
} from '../types/api';


export type NavigationTab =
  | 'dashboard'
  | 'analysis'
  | 'predictions'
  | 'insights'
  | 'reports'
  | 'settings';

interface AnalysisContextType {
  activeTab: NavigationTab;
  setActiveTab: (tab: NavigationTab) => void;
  health: HealthResponse | null;
  isHealthLoading: boolean;
  healthError: string | null;
  refreshHealth: () => Promise<void>;
  latestAnalysis: CustomerAnalysisRecord | null;
  sessionHistory: CustomerAnalysisRecord[];
  isAnalyzing: boolean;
  analysisError: string | null;
  runAnalysis: (
    input: CustomerFeaturesInput,
    customerName?: string
  ) => Promise<CustomerAnalysisRecord>;
  clearSessionHistory: () => void;
  selectedRecordForModal: CustomerAnalysisRecord | null;
  setSelectedRecordForModal: (record: CustomerAnalysisRecord | null) => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(
  undefined
);

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [activeTab, setActiveTab] = useState<NavigationTab>('dashboard');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isHealthLoading, setIsHealthLoading] = useState<boolean>(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [latestAnalysis, setLatestAnalysis] =
    useState<CustomerAnalysisRecord | null>(null);
  const [sessionHistory, setSessionHistory] = useState<
    CustomerAnalysisRecord[]
  >([]);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const [selectedRecordForModal, setSelectedRecordForModal] =
    useState<CustomerAnalysisRecord | null>(null);

  const refreshHealth = useCallback(async () => {
    setIsHealthLoading(true);
    try {
      const data = await apiClient.getHealth();
      setHealth(data);
      setHealthError(null);
    } catch (err: any) {
      setHealth(null);
      setHealthError(err.message || 'Failed to connect to backend.');
    } finally {
      setIsHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    // Periodic health check every 30 seconds
    const interval = setInterval(refreshHealth, 30000);
    return () => clearInterval(interval);
  }, [refreshHealth]);

  const runAnalysis = async (
    input: CustomerFeaturesInput,
    customerName?: string
  ): Promise<CustomerAnalysisRecord> => {
    setIsAnalyzing(true);
    setAnalysisError(null);

    let prediction: PredictionResponse;
    let explanation: ExplanationResponse | null = null;

    try {
      // 1. Prediction request
      prediction = await apiClient.predictChurn(input);

      // 2. Explanation request (handled resiliently)
      try {
        explanation = await apiClient.explainChurn(input);
      } catch (explainErr: any) {
        console.warn('Explain request failed, continuing with prediction:', explainErr);
      }

      const record: CustomerAnalysisRecord = {
        id: `REC-${Date.now().toString(36).toUpperCase()}`,
        timestamp: new Date().toISOString(),
        customerName: customerName || undefined,
        input,
        prediction,
        explanation,
      };

      setLatestAnalysis(record);
      setSessionHistory((prev) => [record, ...prev]);
      return record;
    } catch (err: any) {
      const msg =
        err instanceof ApiError
          ? err.message
          : err.message || 'An unexpected error occurred during prediction.';
      setAnalysisError(msg);
      throw err;
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearSessionHistory = () => {
    setSessionHistory([]);
    setLatestAnalysis(null);
    setSelectedRecordForModal(null);
  };

  return (
    <AnalysisContext.Provider
      value={{
        activeTab,
        setActiveTab,
        health,
        isHealthLoading,
        healthError,
        refreshHealth,
        latestAnalysis,
        sessionHistory,
        isAnalyzing,
        analysisError,
        runAnalysis,
        clearSessionHistory,
        selectedRecordForModal,
        setSelectedRecordForModal,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = (): AnalysisContextType => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};

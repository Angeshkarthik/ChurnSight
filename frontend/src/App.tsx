import React from 'react';
import { AnalysisDetailModal } from './components/analysis/AnalysisDetailModal';
import { AppShell } from './components/layout/AppShell';
import { AnalysisProvider, useAnalysis } from './context/AnalysisContext';
import { ChurnPredictionsPage } from './pages/ChurnPredictionsPage';
import { CustomerAnalysisPage } from './pages/CustomerAnalysisPage';
import { DashboardPage } from './pages/DashboardPage';
import { ModelInsightsPage } from './pages/ModelInsightsPage';
import { ReportsPage } from './pages/ReportsPage';
import { SettingsPage } from './pages/SettingsPage';

const AppContent: React.FC = () => {
  const { activeTab, selectedRecordForModal, setSelectedRecordForModal } =
    useAnalysis();

  const renderActivePage = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardPage />;
      case 'analysis':
        return <CustomerAnalysisPage />;
      case 'predictions':
        return <ChurnPredictionsPage />;
      case 'insights':
        return <ModelInsightsPage />;
      case 'reports':
        return <ReportsPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <DashboardPage />;
    }
  };

  return (
    <AppShell>
      {renderActivePage()}
      <AnalysisDetailModal
        record={selectedRecordForModal}
        onClose={() => setSelectedRecordForModal(null)}
      />
    </AppShell>
  );
};

export const App: React.FC = () => {
  return (
    <AnalysisProvider>
      <AppContent />
    </AnalysisProvider>
  );
};

export default App;

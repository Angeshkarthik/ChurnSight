import React from 'react';
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  FileText,
  History,
  LayoutDashboard,
  LineChart,
  Settings,
  UserCheck,
} from 'lucide-react';
import type { NavigationTab } from '../../context/AnalysisContext';
import { useAnalysis } from '../../context/AnalysisContext';
import { BrandWordmark } from '../common/BrandWordmark';


interface SidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

interface NavItem {
  key: NavigationTab;
  label: string;
  icon: React.ReactNode;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isCollapsed,
  onToggleCollapse,
}) => {
  const { activeTab, setActiveTab } = useAnalysis();

  const navItems: NavItem[] = [
    {
      key: 'dashboard',
      label: 'Dashboard',
      icon: <LayoutDashboard size={18} />,
    },
    {
      key: 'analysis',
      label: 'Customer Analysis',
      icon: <UserCheck size={18} />,
    },
    {
      key: 'predictions',
      label: 'Churn Predictions',
      icon: <History size={18} />,
    },
    {
      key: 'insights',
      label: 'Model Insights',
      icon: <LineChart size={18} />,
    },
    {
      key: 'reports',
      label: 'Reports',
      icon: <FileText size={18} />,
    },
    {
      key: 'settings',
      label: 'Settings',
      icon: <Settings size={18} />,
    },
  ];

  return (
    <aside className={`saas-sidebar ${isCollapsed ? 'saas-sidebar-collapsed' : ''}`}>
      <div className="saas-sidebar-header">
        <div
          className="saas-brand"
          onClick={isCollapsed ? onToggleCollapse : undefined}
          style={isCollapsed ? { cursor: 'pointer' } : undefined}
          title={isCollapsed ? 'Click to expand sidebar (ChurnSight)' : 'ChurnSight'}
        >
          <BrandWordmark
            collapsed={isCollapsed}
            size="md"
            showTagline={!isCollapsed}
          />
        </div>
        <button
          id="sidebar-toggle-btn"
          className="saas-sidebar-toggle"
          onClick={onToggleCollapse}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      <nav className="saas-sidebar-nav" aria-label="Main Navigation">
        <div className="saas-nav-section-title">
          {!isCollapsed && <span>ANALYTICS & SCORING</span>}
        </div>
        <ul className="saas-nav-list">
          {navItems.map((item) => {
            const isActive = activeTab === item.key;
            return (
              <li key={item.key}>
                <button
                  className={`saas-nav-item ${isActive ? 'saas-nav-item-active' : ''}`}
                  onClick={() => setActiveTab(item.key)}
                  title={isCollapsed ? item.label : undefined}
                >
                  <span className="saas-nav-item-icon">{item.icon}</span>
                  {!isCollapsed && (
                    <span className="saas-nav-item-label">{item.label}</span>
                  )}
                  {isActive && <span className="saas-nav-indicator" />}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="saas-sidebar-footer">
        {!isCollapsed && (
          <div className="saas-sidebar-footer-card">
            <div className="footer-card-header">
              <BarChart3 size={14} className="text-emerald" />
              <span>Operational Cutoff</span>
            </div>
            <p className="footer-card-body">
              Threshold <strong>&ge; 0.35</strong> triggers high-risk churn intervention.
            </p>
          </div>
        )}
      </div>
    </aside>
  );
};

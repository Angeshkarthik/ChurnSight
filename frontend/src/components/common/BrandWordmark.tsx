import React from 'react';

interface BrandWordmarkProps {
  collapsed?: boolean;
  size?: 'sm' | 'md' | 'lg';
  showTagline?: boolean;
  className?: string;
}

/**
 * Custom typographic wordmark for ChurnSight.
 * - 'Churn' rendered in Primary Charcoal (#111827)
 * - 'Sight' rendered in Emerald Accent (#10B981)
 * - Collapsed state: Compact typographic mark 'CS' with matching colors
 */
export const BrandWordmark: React.FC<BrandWordmarkProps> = ({
  collapsed = false,
  size = 'md',
  showTagline = false,
  className = '',
}) => {
  if (collapsed) {
    return (
      <div
        className={`brand-wordmark-collapsed brand-size-${size} ${className}`}
        aria-label="ChurnSight"
      >
        <span className="brand-word-churn font-mono-brand">C</span>
        <span className="brand-word-sight font-mono-brand">S</span>
      </div>
    );
  }

  return (
    <div className={`brand-wordmark-container brand-size-${size} ${className}`}>
      <div className="brand-wordmark-main" aria-label="ChurnSight">
        <span className="brand-word-churn">Churn</span>
        <span className="brand-word-sight">Sight</span>
      </div>
      {showTagline && (
        <span className="brand-wordmark-tagline">See risk. Understand why.</span>
      )}
    </div>
  );
};

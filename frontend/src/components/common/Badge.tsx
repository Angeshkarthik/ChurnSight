import React from 'react';

export type BadgeVariant =
  | 'high-risk'
  | 'medium-risk'
  | 'low-risk'
  | 'healthy'
  | 'degraded'
  | 'neutral'
  | 'emerald';

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  dot?: boolean;
  className?: string;
  size?: 'sm' | 'md';
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'neutral',
  children,
  dot = false,
  className = '',
  size = 'md',
}) => {
  return (
    <span className={`saas-badge saas-badge-${variant} saas-badge-${size} ${className}`}>
      {dot && <span className="saas-badge-dot" />}
      {children}
    </span>
  );
};

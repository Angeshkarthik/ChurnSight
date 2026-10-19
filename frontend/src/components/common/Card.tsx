import React from 'react';

interface CardProps {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  action,
  children,
  footer,
  className = '',
  padding = 'md',
}) => {
  return (
    <div className={`saas-card saas-card-padding-${padding} ${className}`}>
      {(title || subtitle || action) && (
        <div className="saas-card-header">
          <div>
            {title && <h3 className="saas-card-title">{title}</h3>}
            {subtitle && <p className="saas-card-subtitle">{subtitle}</p>}
          </div>
          {action && <div className="saas-card-action">{action}</div>}
        </div>
      )}
      <div className="saas-card-body">{children}</div>
      {footer && <div className="saas-card-footer">{footer}</div>}
    </div>
  );
};

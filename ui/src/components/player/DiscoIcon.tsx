// DiscoIcon.tsx — Custom disco ball icon in Lucide style
// 24×24 viewBox, 2px stroke, round linecaps — matches existing Lucide icons.
// Depicts a faceted mirror ball with horizontal and vertical tile lines.

import React from 'react';

interface DiscoIconProps {
  size?: number;
  className?: string;
}

export const DiscoIcon: React.FC<DiscoIconProps> = ({ size = 24, className }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    {/* Hanging string */}
    <line x1="12" y1="1" x2="12" y2="3" />
    {/* Main sphere */}
    <circle cx="12" cy="12" r="9" />
    {/* Horizontal facet arcs — mirror tile rows */}
    <path d="M4.2 8.5 C7 7, 17 7, 19.8 8.5" />
    <path d="M3 12 C7 10.5, 17 10.5, 21 12" />
    <path d="M4.2 15.5 C7 17, 17 17, 19.8 15.5" />
    {/* Vertical centre line */}
    <path d="M12 3 L12 21" />
    {/* Vertical facet arcs — mirror tile columns */}
    <path d="M8.5 3.5 C6.5 8, 6.5 16, 8.5 20.5" />
    <path d="M15.5 3.5 C17.5 8, 17.5 16, 15.5 20.5" />
  </svg>
);

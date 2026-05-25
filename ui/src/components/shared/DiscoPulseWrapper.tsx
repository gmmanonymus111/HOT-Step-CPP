// DiscoPulseWrapper.tsx — Beat-reactive panel wrapper for disco mode.
//
// Wraps a UI panel to apply scale transform + multi-coloured INSET box-shadow
// glow that pulses with the kick drum frequency. Uses inset shadows because
// regular box-shadow is clipped by overflow:hidden on parent containers.
//
// Uses direct DOM style manipulation (via ref) to avoid React re-renders at
// 60fps — the disco store notifies at ~60fps, and we write straight to the
// DOM element.

import React, { useRef, useEffect } from 'react';
import { usePulseIntensity, useDiscoMode } from '../../stores/discoStore';

interface DiscoPulseWrapperProps {
  /** Neon glow colour for this panel (hex, e.g. '#ff1493') */
  glowColor: string;
  /** Stagger delay in ms — creates a left-to-right ripple via CSS transition-delay */
  staggerMs?: number;
  /** Extra CSS classes to pass through */
  className?: string;
  /** Inline styles to pass through */
  style?: React.CSSProperties;
  children: React.ReactNode;
}

export const DiscoPulseWrapper: React.FC<DiscoPulseWrapperProps> = ({
  glowColor,
  staggerMs = 0,
  className = '',
  style,
  children,
}) => {
  const discoMode = useDiscoMode();
  const pulseIntensity = usePulseIntensity();
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Apply visual effects directly to DOM to avoid React re-render churn.
  // This runs on every pulseIntensity change (~60fps when active).
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;

    if (!discoMode || pulseIntensity < 0.01) {
      el.style.transform = '';
      el.style.boxShadow = '';
      el.style.filter = '';
      return;
    }

    applyPulse(el, pulseIntensity, glowColor);
  }, [pulseIntensity, discoMode, glowColor]);

  // Clean up styles when disco mode is toggled off
  useEffect(() => {
    if (!discoMode) {
      const el = wrapperRef.current;
      if (el) {
        el.style.transform = '';
        el.style.boxShadow = '';
        el.style.filter = '';
      }
    }
  }, [discoMode]);

  return (
    <div
      ref={wrapperRef}
      className={`disco-pulse-wrapper ${className}`}
      style={{
        ...style,
        // Apply stagger as CSS transition-delay — clean, no JS timer spam
        transitionDelay: staggerMs > 0 ? `${staggerMs}ms` : undefined,
      }}
      data-disco-active={discoMode || undefined}
    >
      {children}
    </div>
  );
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function applyPulse(el: HTMLElement, intensity: number, color: string): void {
  // Scale: 1.0 → 1.025 at full intensity (2.5% — clearly visible)
  const scale = 1 + intensity * 0.025;
  el.style.transform = `scale(${scale})`;

  // INSET multi-layer glow — not clipped by overflow:hidden on parents!
  // Three concentric inset layers: tight inner + mid spread + wide ambient
  const tight = Math.round(6 * intensity);
  const mid = Math.round(16 * intensity);
  const wide = Math.round(30 * intensity);
  el.style.boxShadow = [
    `inset 0 0 ${tight}px 0 ${color}`,
    `inset 0 0 ${mid}px 0 ${hexToRgba(color, 0.5)}`,
    `inset 0 0 ${wide}px 0 ${hexToRgba(color, 0.2)}`,
    // Also add a small outer glow for panels NOT inside overflow:hidden
    `0 0 ${Math.round(8 * intensity)}px 0 ${hexToRgba(color, 0.4)}`,
  ].join(', ');

  // Combined filter: brightness pulse + slow hue rotation for colour drift
  const brightness = 1 + intensity * 0.08;
  // Slow sinusoidal hue drift: ±15° over ~8 seconds
  const hueShift = Math.sin(performance.now() / 4000) * 15;
  el.style.filter = `brightness(${brightness}) hue-rotate(${hueShift.toFixed(1)}deg)`;
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

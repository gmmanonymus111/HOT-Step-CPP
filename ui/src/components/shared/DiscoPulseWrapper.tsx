// DiscoPulseWrapper.tsx — iOS 18-style animated rainbow border for disco mode.
//
// Uses a spinning conic-gradient masked to a border ring around the panel.
// CSS mask-composite: exclude cuts out the interior, so the gradient is
// ONLY visible as a glowing border — no layout changes or margin tricks needed.
//
// Structure:
//   <wrapper>          ← position: relative, isolation: isolate
//     <glow>           ← spinning gradient, masked to border ring
//       <glow-bg>      ← oversized spinning conic-gradient
//     </glow>
//     <content>        ← children (unchanged layout)
//   </wrapper>

import React, { useRef, useEffect } from 'react';
import { usePulseIntensity, useDiscoMode } from '../../stores/discoStore';

interface DiscoPulseWrapperProps {
  /** Base hue for this panel's glow (0-360). Each panel gets a different hue. */
  hue?: number;
  /** Extra CSS classes to pass through to the content wrapper */
  className?: string;
  /** Inline styles for the content wrapper */
  style?: React.CSSProperties;
  children: React.ReactNode;
}

export const DiscoPulseWrapper: React.FC<DiscoPulseWrapperProps> = ({
  hue = 145,
  className = '',
  style,
  children,
}) => {
  const discoMode = useDiscoMode();
  const pulseIntensity = usePulseIntensity();
  const glowRef = useRef<HTMLDivElement>(null);

  // Modulate glow intensity via direct DOM writes (avoids React re-renders at 60fps)
  useEffect(() => {
    const glow = glowRef.current;
    if (!glow) return;

    if (!discoMode) {
      glow.style.opacity = '0';
      return;
    }

    // Power curve: low-intensity moments are subtle, hits POP
    const intensity = Math.pow(pulseIntensity, 0.6);
    glow.style.opacity = String(Math.min(1, intensity * 1.3));
  }, [pulseIntensity, discoMode]);

  // Clean up on disco mode toggle off
  useEffect(() => {
    if (!discoMode && glowRef.current) {
      glowRef.current.style.opacity = '0';
    }
  }, [discoMode]);

  return (
    <div
      className={`disco-wrapper ${discoMode ? 'disco-active' : ''} ${className}`}
      style={style}
    >
      {/* Spinning gradient border ring */}
      <div
        ref={glowRef}
        className="disco-glow"
        style={{ opacity: 0 }}
      >
        <div
          className="disco-glow-bg"
          style={{
            '--disco-hue': hue,
          } as React.CSSProperties}
        />
      </div>

      {/* Actual panel content — no wrapper div needed, children render directly */}
      {children}
    </div>
  );
};

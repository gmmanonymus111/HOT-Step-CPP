// SnareFlashOverlay.tsx — Full-viewport white flash on snare hits.
// "Strobe at a gig" effect: jumps to ~6% opacity on snare, decays in ~80ms.
// Uses direct DOM writes for 60fps performance.

import React, { useRef, useEffect } from 'react';
import { useDiscoMode } from '../../stores/discoStore';
import { useSnarePulse } from '../../stores/discoStore';

export const SnareFlashOverlay: React.FC = () => {
  const discoMode = useDiscoMode();
  const snarePulse = useSnarePulse();
  const flashRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = flashRef.current;
    if (!el) return;

    if (!discoMode) {
      el.style.opacity = '0';
      return;
    }

    // Sharp power curve — only visible on strong hits
    const intensity = Math.pow(snarePulse, 0.8);
    el.style.opacity = String(Math.min(0.15, intensity * 0.15));
  }, [snarePulse, discoMode]);

  if (!discoMode) return null;

  return (
    <div
      ref={flashRef}
      className="disco-snare-flash"
      style={{ opacity: 0 }}
    />
  );
};

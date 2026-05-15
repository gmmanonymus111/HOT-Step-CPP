/**
 * Disguise Mode — Press X to swap all visible artist/album names and images
 * with convincing fakes. Completely invisible to observers.
 *
 * Display-only: sorting, filtering, API calls, and stored data are never affected.
 */
import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from 'react';

// ── Fake Band Names ───────────────────────────────────────────────────────────
const FAKE_ARTISTS: string[] = [
  'Velvet Parallax', 'Chromatic Drift', 'Neon Cartography', 'The Phosphor Collective',
  'Hollow Meridian', 'Glass Antenna', 'Burnt Ochre', 'Solar Fugue',
  'The Wren Experiment', 'Moth & Lantern', 'Cubic Haze', 'Pale Voltage',
  'Iron Pastoral', 'The Quiet Engines', 'Sable Frequency', 'Lumen Archive',
  'Tidal Construct', 'Brass Meridian', 'The Zinc Parade', 'Ember Cartograph',
  'Phantom Lattice', 'The Gilt Fraction', 'Verdigris', 'Null Pasture',
  'Wax Cathedral', 'The Oscillate', 'Fern & Static', 'Heliograph',
  'Charcoal Prism', 'The Amber Drift', 'Slate Province', 'Mercury Garden',
  'The Analog Saints', 'Ochre Window', 'Rust Dialect', 'The Pollen Index',
  'Nova Scaffold', 'Dusk Protocol', 'The Marble Signal', 'Aether Loom',
  'Concrete Lyric', 'The Copper Phase', 'Graphite Sun', 'Paper Meridian',
  'The Indigo Shift', 'Fossil Circuit', 'Lichen Radio', 'The Voltage Choir',
  'Tin Cathedral', 'Quartz Assembly', 'The Basalt Method', 'Saffron Engine',
  'Porcelain Static', 'The Driftwood Codex', 'Oxide Narrative', 'Celadon Wire',
  'The Flicker Atlas', 'Tungsten Bloom', 'Cobalt Pastoral', 'The Filament',
  'Magnetic Vesper', 'The Chalk Meridian', 'Radium Sketch', 'Silica Hymn',
  'The Parallax Garden', 'Mineral Choir', 'Bronze Aperture', 'The Lacquer Dispatch',
  'Ivory Tremor', 'The Signal Meadow', 'Cypress Voltage', 'Antimony Dusk',
  'The Glass Almanac', 'Sulfur Arcade', 'Limestone Echo', 'The Carbon Vespers',
  'Feldspar Ensemble', 'Jasper Wireframe', 'The Zinc Hymnal', 'Pewter Solstice',
];

// ── Fake Album Names ──────────────────────────────────────────────────────────
const FAKE_ALBUMS: string[] = [
  'Midnight Architecture', 'Signal & Noise', 'The Quiet Voltage', 'Borrowed Light',
  'Concrete Hymns', 'Atlas of Edges', 'The Still Frequency', 'Paper Meridians',
  'Slow Geometry', 'Through the Lattice', 'Post-Meridian', 'Warm Static',
  'The Vanishing Index', 'Coastal Engines', 'All the Small Fires', 'Tidal Grammar',
  'Negative Space', 'The Architecture of Rain', 'Blank Cartography', 'Ember & Ash',
  'Luminous Debris', 'The Weight of Glass', 'Seasonal Drift', 'Carbon Copy',
  'The View from Here', 'Analog Sundays', 'Minor Key Weather', 'The Fold',
  'Distance Calls', 'Everything in Waves', 'Soft Machines', 'The Usual Silence',
  'Ghost Frequency', 'We Were Voltage', 'Surface Tension', 'The Long Equation',
  'Afterimage', 'Map of Echoes', 'The Rust Sessions', 'Quiet Turbulence',
];

// ── Number of fake band images in ui/public/disguise/ ─────────────────────────
const FAKE_IMAGE_COUNT = 20;

// ── Deterministic hash ────────────────────────────────────────────────────────
function stableHash(s: string): number {
  let hash = 5381;
  for (let i = 0; i < s.length; i++) {
    hash = ((hash << 5) + hash + s.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

// ── Context type ──────────────────────────────────────────────────────────────
interface DisguiseModeContextValue {
  isDisguised: boolean;
  /** Map a real artist name to a fake one (passthrough when disguise is off) */
  disguiseArtist: (name: string) => string;
  /** Map a real album name to a fake one (passthrough when disguise is off) */
  disguiseAlbum: (name: string) => string;
  /** Swap an image URL for a fake band image (passthrough when disguise is off) */
  disguiseImageUrl: (realUrl: string | undefined, name: string) => string | undefined;
}

const DisguiseModeContext = createContext<DisguiseModeContextValue>({
  isDisguised: false,
  disguiseArtist: (n) => n,
  disguiseAlbum: (n) => n,
  disguiseImageUrl: (u) => u,
});

// ── Provider ──────────────────────────────────────────────────────────────────
export const DisguiseModeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isDisguised, setIsDisguised] = useState(false);

  // Global keyboard listener — toggle on X when not in an input
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Ignore if any modifier is held
      if (e.ctrlKey || e.altKey || e.metaKey || e.shiftKey) return;
      if (e.key !== 'x') return;

      // Ignore if typing in an input/textarea/select/contenteditable
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
      if ((e.target as HTMLElement)?.isContentEditable) return;

      e.preventDefault();
      setIsDisguised(prev => !prev);
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const disguiseArtist = useCallback((name: string): string => {
    if (!isDisguised || !name) return name;
    return FAKE_ARTISTS[stableHash(name) % FAKE_ARTISTS.length];
  }, [isDisguised]);

  const disguiseAlbum = useCallback((name: string): string => {
    if (!isDisguised || !name) return name;
    return FAKE_ALBUMS[stableHash(name) % FAKE_ALBUMS.length];
  }, [isDisguised]);

  const disguiseImageUrl = useCallback((realUrl: string | undefined, name: string): string | undefined => {
    if (!isDisguised) return realUrl;
    if (!name) return realUrl;
    const idx = (stableHash(name) % FAKE_IMAGE_COUNT) + 1;
    return `/disguise/band_${String(idx).padStart(2, '0')}.webp`;
  }, [isDisguised]);

  const value = useMemo(() => ({
    isDisguised,
    disguiseArtist,
    disguiseAlbum,
    disguiseImageUrl,
  }), [isDisguised, disguiseArtist, disguiseAlbum, disguiseImageUrl]);

  return (
    <DisguiseModeContext.Provider value={value}>
      {children}
    </DisguiseModeContext.Provider>
  );
};

// ── Hook ──────────────────────────────────────────────────────────────────────
export function useDisguiseMode(): DisguiseModeContextValue {
  return useContext(DisguiseModeContext);
}

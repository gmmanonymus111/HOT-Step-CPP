// TrackSelector.tsx — Checkbox grid for selecting stem extraction tracks
import React from 'react';
import { Lock } from 'lucide-react';
import { EXTRACT_TRACKS, TRACK_LABELS, TRACK_CATEGORIES } from '../../services/stemStudioApi';

interface TrackSelectorProps {
  selectedTracks: string[];
  onTracksChange: (tracks: string[]) => void;
  mode: 'extract' | 'supersep';
  onModeChange: (mode: 'extract' | 'supersep') => void;
  onExtract: () => void;
  isExtracting: boolean;
  canExtract: boolean; // false if no source audio selected
}

const CATEGORY_COLORS: Record<string, string> = {
  vocals: '#e879f9',
  instruments: '#60a5fa',
  drums: '#f97316',
  other: '#a3a3a3',
};

export const TrackSelector: React.FC<TrackSelectorProps> = ({
  selectedTracks, onTracksChange, mode, onModeChange,
  onExtract, isExtracting, canExtract,
}) => {
  const toggleTrack = (track: string) => {
    if (selectedTracks.includes(track)) {
      onTracksChange(selectedTracks.filter(t => t !== track));
    } else {
      onTracksChange([...selectedTracks, track]);
    }
  };

  const selectAll = () => onTracksChange([...EXTRACT_TRACKS]);
  const clearAll = () => onTracksChange([]);

  // Group tracks by category for display
  const grouped = EXTRACT_TRACKS.reduce<Record<string, string[]>>((acc, t) => {
    const cat = TRACK_CATEGORIES[t] || 'other';
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(t);
    return acc;
  }, {});

  return (
    <div style={styles.container}>
      {/* Mode selector */}
      <div style={styles.modeRow}>
        <button
          onClick={() => onModeChange('extract')}
          style={{
            ...styles.modeBtn,
            background: mode === 'extract' ? 'rgba(167,139,250,0.15)' : 'transparent',
            color: mode === 'extract' ? '#a78bfa' : '#888',
            borderColor: mode === 'extract' ? 'rgba(167,139,250,0.3)' : 'rgba(255,255,255,0.08)',
          }}
        >
          🎵 Extract (DiT)
        </button>
        <button
          onClick={() => {}}
          disabled
          style={{
            ...styles.modeBtn,
            opacity: 0.4,
            cursor: 'not-allowed',
            color: '#666',
          }}
          title="Coming soon — Phase 2"
        >
          <Lock size={12} /> SuperSep
        </button>
      </div>

      <h3 style={styles.sectionTitle}>Select Tracks</h3>

      {/* Quick actions */}
      <div style={styles.quickActions}>
        <button onClick={selectAll} style={styles.quickBtn}>Select All</button>
        <button onClick={clearAll} style={styles.quickBtn}>Clear</button>
        <span style={styles.selectedCount}>{selectedTracks.length} selected</span>
      </div>

      {/* Track grid grouped by category */}
      <div style={styles.trackGrid}>
        {(['vocals', 'drums', 'instruments', 'other'] as const).map(cat => {
          const tracks = grouped[cat];
          if (!tracks) return null;
          return (
            <div key={cat} style={styles.categoryGroup}>
              <div style={{ ...styles.categoryLabel, color: CATEGORY_COLORS[cat] }}>
                <span>●</span> {cat.charAt(0).toUpperCase() + cat.slice(1)}
              </div>
              {tracks.map(track => (
                <label key={track} style={styles.trackItem}>
                  <input
                    type="checkbox"
                    checked={selectedTracks.includes(track)}
                    onChange={() => toggleTrack(track)}
                    style={{ accentColor: CATEGORY_COLORS[cat], cursor: 'pointer' }}
                  />
                  <span style={{
                    ...styles.trackLabel,
                    color: selectedTracks.includes(track) ? '#d4d4d4' : '#888',
                  }}>
                    {TRACK_LABELS[track] || track}
                  </span>
                </label>
              ))}
            </div>
          );
        })}
      </div>

      {/* Extract button */}
      <button
        onClick={onExtract}
        disabled={!canExtract || isExtracting || selectedTracks.length === 0}
        style={{
          ...styles.extractBtn,
          opacity: (!canExtract || isExtracting || selectedTracks.length === 0) ? 0.5 : 1,
          cursor: (!canExtract || isExtracting || selectedTracks.length === 0) ? 'not-allowed' : 'pointer',
        }}
      >
        {isExtracting ? '⏳ Extracting...' : `▶ Extract ${selectedTracks.length} Track${selectedTracks.length !== 1 ? 's' : ''}`}
      </button>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  modeRow: {
    display: 'flex',
    gap: 6,
  },
  modeBtn: {
    flex: 1,
    padding: '8px 12px',
    borderRadius: 8,
    border: '1px solid',
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    transition: 'all 0.15s ease',
  },
  sectionTitle: {
    margin: 0,
    fontSize: 14,
    fontWeight: 600,
    color: '#d4d4d4',
    letterSpacing: '0.02em',
  },
  quickActions: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  quickBtn: {
    padding: '4px 10px',
    borderRadius: 6,
    border: '1px solid rgba(255,255,255,0.08)',
    background: 'rgba(255,255,255,0.04)',
    color: '#aaa',
    fontSize: 11,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.1s ease',
  },
  selectedCount: {
    fontSize: 11,
    color: '#666',
    marginLeft: 'auto',
  },
  trackGrid: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  categoryGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
  categoryLabel: {
    fontSize: 10,
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    padding: '2px 0',
    display: 'flex',
    alignItems: 'center',
    gap: 4,
  },
  trackItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '4px 8px',
    borderRadius: 6,
    cursor: 'pointer',
    transition: 'background 0.1s ease',
  },
  trackLabel: {
    fontSize: 13,
    fontWeight: 500,
    transition: 'color 0.1s ease',
  },
  extractBtn: {
    padding: '10px 16px',
    borderRadius: 10,
    border: 'none',
    background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
    color: '#fff',
    fontSize: 14,
    fontWeight: 600,
    transition: 'all 0.15s ease',
    marginTop: 4,
  },
};

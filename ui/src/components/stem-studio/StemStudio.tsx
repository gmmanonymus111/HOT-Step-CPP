// StemStudio.tsx — Main Stem Studio orchestrator
//
// Composes: SourceSelector, TrackSelector, StemMixer, RecentExtractions
// Manages extraction state, polls progress, loads results into mixer.

import React, { useState, useCallback } from 'react';
import { AlertTriangle } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import {
  submitExtraction, waitForExtraction, getExtractResult,
  getStemUrl, getDownloadAllUrl, TRACK_CATEGORIES,
  type ExtractProgress, type ExtractJobResult,
} from '../../services/stemStudioApi';
import { StemMixer, type StemControl, type MixerStemInfo } from '../shared/StemMixer';
import { SourceSelector } from './SourceSelector';
import { TrackSelector } from './TrackSelector';
import { RecentExtractions } from './RecentExtractions';

export const StemStudio: React.FC = () => {
  const gp = useGlobalParams();

  // Source audio
  const [sourceAudioUrl, setSourceAudioUrl] = useState('');
  const [sourceFileName, setSourceFileName] = useState('');

  // Track selection
  const [selectedTracks, setSelectedTracks] = useState<string[]>(['vocals', 'drums', 'bass', 'guitar']);
  const [mode, setMode] = useState<'extract' | 'supersep'>('extract');

  // Optional enhancement
  const [style, setStyle] = useState('');
  const [lyrics, setLyrics] = useState('');

  // Extraction state
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractProgress, setExtractProgress] = useState<ExtractProgress | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  // Results
  const [mixerStems, setMixerStems] = useState<MixerStemInfo[] | null>(null);
  const [stemControls, setStemControls] = useState<StemControl[]>([]);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Turbo warning
  const turboWarning = gp.ditModel?.toLowerCase().includes('turbo');

  const handleSourceChange = useCallback((url: string, fileName: string) => {
    setSourceAudioUrl(url);
    setSourceFileName(fileName);
  }, []);

  const handleExtract = useCallback(async () => {
    if (!sourceAudioUrl || selectedTracks.length === 0) return;

    setIsExtracting(true);
    setExtractProgress(null);
    setMixerStems(null);

    try {
      // Collect DiT settings from global params
      const ditSettings = {
        ditModel: gp.ditModel,
        inferenceSteps: gp.inferenceSteps,
        inferMethod: gp.inferMethod,
        scheduler: gp.scheduler,
        guidanceMode: gp.guidanceMode,
        guidanceScale: gp.guidanceScale,
        shift: gp.shift,
        loraPath: gp.adapter,
        loraScale: gp.adapterScale,
        seed: gp.randomSeed ? -1 : gp.seed,
      };

      const jobId = await submitExtraction({
        sourceAudioUrl,
        sourceFileName,
        tracks: selectedTracks,
        style: style || undefined,
        lyrics: lyrics || undefined,
        ditSettings,
      });

      setActiveJobId(jobId);

      // Poll for progress
      const result = await waitForExtraction(jobId, (progress) => {
        setExtractProgress(progress);
      });

      loadResultIntoMixer(jobId, result);
      setRefreshTrigger(prev => prev + 1);

    } catch (err: any) {
      console.error('Extraction failed:', err);
      setExtractProgress({
        status: 'failed',
        progress: 0,
        currentTrack: '',
        completedStems: [],
        totalTracks: selectedTracks.length,
        error: err.message,
      });
    } finally {
      setIsExtracting(false);
    }
  }, [sourceAudioUrl, sourceFileName, selectedTracks, style, lyrics, gp]);

  const loadResultIntoMixer = useCallback((jobId: string, result: ExtractJobResult) => {
    const stems: MixerStemInfo[] = result.stems.map((s, idx) => ({
      name: s.trackName,
      category: TRACK_CATEGORIES[s.trackName] || 'other',
      audioUrl: getStemUrl(jobId, s.trackName),
      index: idx,
    }));
    setMixerStems(stems);
    setStemControls(stems.map(s => ({ index: s.index, volume: 1.0, muted: false })));
  }, []);

  const handleSelectPastJob = useCallback(async (jobId: string) => {
    try {
      setActiveJobId(jobId);
      const result = await getExtractResult(jobId);
      loadResultIntoMixer(jobId, result);
    } catch (err) {
      console.error('Failed to load past job:', err);
    }
  }, [loadResultIntoMixer]);

  const handleDownloadStem = useCallback((stem: MixerStemInfo) => {
    if (!activeJobId) return;
    const a = document.createElement('a');
    a.href = getStemUrl(activeJobId, stem.name);
    a.download = `${stem.name}.wav`;
    a.click();
  }, [activeJobId]);

  const handleDownloadAll = useCallback(() => {
    if (!activeJobId) return;
    const a = document.createElement('a');
    a.href = getDownloadAllUrl(activeJobId);
    a.download = 'stems.zip';
    a.click();
  }, [activeJobId]);

  return (
    <div style={styles.outerContainer}>
      <div style={styles.layout}>
        {/* Left column — Source Audio */}
        <div style={styles.leftCol}>
          <SourceSelector
            sourceAudioUrl={sourceAudioUrl}
            sourceFileName={sourceFileName}
            onSourceChange={handleSourceChange}
          />

          {/* Optional: style hint + lyrics */}
          <div style={styles.optionalSection}>
            <details>
              <summary style={styles.optionalSummary}>Optional: Style & Lyrics (improves extraction)</summary>
              <div style={styles.optionalFields}>
                <label style={styles.fieldLabel}>Style Hint</label>
                <input
                  type="text"
                  value={style}
                  onChange={e => setStyle(e.target.value)}
                  placeholder="e.g. indie rock, distorted guitar, raw vocals"
                  style={styles.textInput}
                />
                <label style={styles.fieldLabel}>Lyrics</label>
                <textarea
                  value={lyrics}
                  onChange={e => setLyrics(e.target.value)}
                  placeholder="Paste lyrics here to improve vocal extraction..."
                  style={styles.textarea}
                  rows={4}
                />
              </div>
            </details>
          </div>
        </div>

        {/* Center column — Track Selection + Mixer */}
        <div style={styles.centerCol}>
          {/* Turbo warning */}
          {turboWarning && (
            <div style={styles.turboWarning}>
              <AlertTriangle size={14} />
              <span>Active DiT model is turbo. Extract requires a base/SFT model for coherent output.</span>
            </div>
          )}

          <TrackSelector
            selectedTracks={selectedTracks}
            onTracksChange={setSelectedTracks}
            mode={mode}
            onModeChange={setMode}
            onExtract={handleExtract}
            isExtracting={isExtracting}
            canExtract={!!sourceAudioUrl}
          />

          {/* Progress */}
          {extractProgress && extractProgress.status === 'extracting' && (
            <div style={styles.progressSection}>
              <div style={styles.progressLabel}>
                Extracting {extractProgress.currentTrack} ({extractProgress.completedStems.length + 1}/{extractProgress.totalTracks})
              </div>
              <div style={styles.progressBar}>
                <div style={{ ...styles.progressFill, width: `${extractProgress.progress}%` }} />
              </div>
              <span style={styles.progressPercent}>{extractProgress.progress}%</span>
            </div>
          )}

          {/* Error */}
          {extractProgress?.status === 'failed' && (
            <div style={styles.errorMsg}>
              ❌ Extraction failed: {extractProgress.error}
            </div>
          )}

          {/* Mixer */}
          {mixerStems && activeJobId && (
            <div style={styles.mixerSection}>
              <StemMixer
                jobId={activeJobId}
                stems={mixerStems}
                controls={stemControls}
                onControlsChange={setStemControls}
                onDownloadStem={handleDownloadStem}
                onDownloadAll={handleDownloadAll}
              />
            </div>
          )}
        </div>

        {/* Right column — Recent Extractions */}
        <div style={styles.rightCol}>
          <RecentExtractions
            onSelectJob={handleSelectPastJob}
            activeJobId={activeJobId || undefined}
            refreshTrigger={refreshTrigger}
          />
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  outerContainer: {
    height: '100%',
    padding: 16,
    overflowY: 'auto',
  },
  layout: {
    display: 'grid',
    gridTemplateColumns: '280px 1fr 240px',
    gap: 16,
    maxWidth: 1400,
    margin: '0 auto',
    height: '100%',
  },
  leftCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
    padding: 16,
    borderRadius: 12,
    background: 'rgba(255,255,255,0.02)',
    border: '1px solid rgba(255,255,255,0.06)',
    overflowY: 'auto',
  },
  centerCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
    padding: 16,
    borderRadius: 12,
    background: 'rgba(255,255,255,0.02)',
    border: '1px solid rgba(255,255,255,0.06)',
    overflowY: 'auto',
  },
  rightCol: {
    display: 'flex',
    flexDirection: 'column',
    padding: 12,
    borderRadius: 12,
    background: 'rgba(255,255,255,0.02)',
    border: '1px solid rgba(255,255,255,0.06)',
    overflowY: 'auto',
  },
  turboWarning: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '8px 12px',
    borderRadius: 8,
    background: 'rgba(234,179,8,0.1)',
    border: '1px solid rgba(234,179,8,0.25)',
    color: '#eab308',
    fontSize: 12,
    fontWeight: 500,
  },
  progressSection: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '10px 0',
  },
  progressLabel: {
    fontSize: 12,
    color: '#a78bfa',
    fontWeight: 600,
    whiteSpace: 'nowrap',
  },
  progressBar: {
    flex: 1,
    height: 6,
    borderRadius: 3,
    background: 'rgba(255,255,255,0.06)',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
    background: 'linear-gradient(90deg, #8b5cf6, #a78bfa)',
    transition: 'width 0.3s ease',
  },
  progressPercent: {
    fontSize: 11,
    color: '#888',
    fontFamily: 'monospace',
    width: 32,
    textAlign: 'right',
  },
  errorMsg: {
    padding: '10px 14px',
    borderRadius: 8,
    background: 'rgba(239,68,68,0.1)',
    border: '1px solid rgba(239,68,68,0.2)',
    color: '#ef4444',
    fontSize: 12,
  },
  mixerSection: {
    marginTop: 8,
  },
  optionalSection: {
    borderTop: '1px solid rgba(255,255,255,0.05)',
    paddingTop: 12,
  },
  optionalSummary: {
    fontSize: 12,
    color: '#888',
    cursor: 'pointer',
    fontWeight: 500,
  },
  optionalFields: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    marginTop: 10,
  },
  fieldLabel: {
    fontSize: 11,
    fontWeight: 600,
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  textInput: {
    padding: '8px 10px',
    borderRadius: 6,
    border: '1px solid rgba(255,255,255,0.08)',
    background: 'rgba(255,255,255,0.04)',
    color: '#d4d4d4',
    fontSize: 12,
    outline: 'none',
  },
  textarea: {
    padding: '8px 10px',
    borderRadius: 6,
    border: '1px solid rgba(255,255,255,0.08)',
    background: 'rgba(255,255,255,0.04)',
    color: '#d4d4d4',
    fontSize: 12,
    outline: 'none',
    resize: 'vertical',
    fontFamily: 'inherit',
  },
};

export default StemStudio;

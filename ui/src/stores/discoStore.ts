// discoStore.ts — Disco mode state + beat detection engine.
//
// Creates a dedicated AnalyserNode (smoothingTimeConstant=0.15) and uses
// adaptive normalisation to extract maximum dynamic range from kick band
// energy. The raw energy varies ~0.15 between kick hits and gaps — we
// stretch that to fill 0–1 using a rolling min/max tracker.

import { useSyncExternalStore, useRef, useCallback } from 'react';

// ── Types ────────────────────────────────────────────────────────────────────

interface DiscoState {
  discoMode: boolean;
  pulseIntensity: number;  // 0.0–1.0
}

// ── localStorage ─────────────────────────────────────────────────────────────

const DISCO_PREFS_KEY = 'disco-prefs';

function loadDiscoPrefs(): { discoMode: boolean } {
  try {
    const raw = localStorage.getItem(DISCO_PREFS_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return { discoMode: false };
}

function saveDiscoPrefs(discoMode: boolean): void {
  localStorage.setItem(DISCO_PREFS_KEY, JSON.stringify({ discoMode }));
}

// ── Audio Graph — Dedicated Beat Detection AnalyserNode ──────────────────────

let _audioMotion: any = null;
let _beatAnalyser: AnalyserNode | null = null;
let _freqData: Uint8Array | null = null;

// Kick drum FFT bin range
let _kickBinStart = 0;
let _kickBinEnd = 0;

export function registerAudioMotion(instance: any): void {
  _audioMotion = instance;

  try {
    const ctx: AudioContext = instance.audioCtx;

    _beatAnalyser = ctx.createAnalyser();
    _beatAnalyser.fftSize = 2048;
    _beatAnalyser.smoothingTimeConstant = 0.15; // Very low — we want raw transients

    const sources = instance.connectedSources;
    if (sources && sources.length > 0) {
      sources[0].connect(_beatAnalyser);
      console.log('[Disco] Beat analyser connected');
    }

    _freqData = new Uint8Array(_beatAnalyser.frequencyBinCount);

    const binHz = ctx.sampleRate / _beatAnalyser.fftSize;
    _kickBinStart = Math.floor(50 / binHz);
    _kickBinEnd = Math.ceil(180 / binHz);
    console.log(`[Disco] Kick bins: ${_kickBinStart}-${_kickBinEnd} (binHz=${binHz.toFixed(1)})`);
  } catch (err) {
    console.error('[Disco] Failed to create beat analyser:', err);
  }
}

export function unregisterAudioMotion(): void {
  _audioMotion = null;
  _beatAnalyser = null;
  _freqData = null;
}

/** Read kick energy from dedicated low-smoothing AnalyserNode */
function readKickEnergy(): number {
  if (!_beatAnalyser || !_freqData) return 0;
  _beatAnalyser.getByteFrequencyData(_freqData);

  let sum = 0;
  let count = 0;
  for (let i = _kickBinStart; i <= _kickBinEnd && i < _freqData.length; i++) {
    sum += _freqData[i];
    count++;
  }
  return count > 0 ? (sum / count) / 255 : 0;
}

// ── State ────────────────────────────────────────────────────────────────────

const prefs = loadDiscoPrefs();

let _state: DiscoState = {
  discoMode: prefs.discoMode,
  pulseIntensity: 0,
};

// ── Reactivity (useSyncExternalStore) ────────────────────────────────────────

const _listeners = new Set<() => void>();

function notify(): void {
  _listeners.forEach(cb => cb());
}

function setState(updates: Partial<DiscoState>): void {
  _state = { ..._state, ...updates };
  notify();
}

function subscribe(cb: () => void): () => void {
  _listeners.add(cb);
  return () => _listeners.delete(cb);
}

function getSnapshot(): DiscoState {
  return _state;
}

// ── Beat Detection Loop (Adaptive Normalisation) ─────────────────────────────

let _rafId: number | null = null;
let _lastFrameTime = 0;
let _debugFrameCount = 0;

// Adaptive normalisation — tracks rolling min/max to stretch available range
let _rollingMin = 1.0;       // Tracks recent minimum energy (floor)
let _rollingMax = 0.0;       // Tracks recent maximum energy (ceiling)
let _pulse = 0;              // Current output pulse (0–1)

// Tuning knobs
const MIN_DECAY = 0.002;     // How fast the floor rises per frame (adapts upward)
const MAX_DECAY = 0.005;     // How fast the ceiling drops per frame (adapts downward)
const MIN_RANGE = 0.03;      // Minimum gap between min and max to avoid noise amplification
const ATTACK_SPEED = 0.6;    // How fast pulse rises toward target (0–1, higher = snappier)
const DECAY_SPEED = 0.85;    // Per-frame pulse decay multiplier (lower = faster drop)

function beatDetectionLoop(timestamp: number): void {
  if (!_state.discoMode) {
    _rafId = null;
    _pulse = 0;
    setState({ pulseIntensity: 0 });
    return;
  }

  _lastFrameTime = timestamp;

  const rawEnergy = readKickEnergy();

  // Update rolling min/max with slow adaptation
  // Min creeps upward, max creeps downward — they converge toward current energy
  // When energy spikes above max or drops below min, they snap to the new value
  if (rawEnergy < _rollingMin) {
    _rollingMin = rawEnergy;  // Snap to new low
  } else {
    _rollingMin += MIN_DECAY; // Slowly rise
  }

  if (rawEnergy > _rollingMax) {
    _rollingMax = rawEnergy;  // Snap to new high
  } else {
    _rollingMax -= MAX_DECAY; // Slowly drop
  }

  // Ensure min < max with minimum range
  if (_rollingMax - _rollingMin < MIN_RANGE) {
    const mid = (_rollingMax + _rollingMin) / 2;
    _rollingMin = mid - MIN_RANGE / 2;
    _rollingMax = mid + MIN_RANGE / 2;
  }

  // Normalise raw energy into 0–1 using the adaptive range
  const normalised = Math.max(0, Math.min(1,
    (rawEnergy - _rollingMin) / (_rollingMax - _rollingMin)
  ));

  // Apply dynamics: fast attack, fast decay
  if (normalised > _pulse) {
    // Attack — lerp quickly toward the peak
    _pulse += (normalised - _pulse) * ATTACK_SPEED;
  } else {
    // Decay — multiplicative drop
    _pulse *= DECAY_SPEED;
  }

  if (_pulse < 0.01) _pulse = 0;

  // Debug: log every 30 frames (~2x per second) to see more detail
  _debugFrameCount++;
  if (_debugFrameCount % 30 === 0) {
    console.log(`[Disco] raw=${rawEnergy.toFixed(3)} min=${_rollingMin.toFixed(3)} max=${_rollingMax.toFixed(3)} norm=${normalised.toFixed(3)} pulse=${_pulse.toFixed(3)}`);
  }

  // Notify subscribers
  if (Math.abs(_pulse - _state.pulseIntensity) > 0.003) {
    _state = { ..._state, pulseIntensity: _pulse };
    notify();
  }

  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function startLoop(): void {
  if (_rafId !== null) return;
  _lastFrameTime = 0;
  _rollingMin = 1.0;
  _rollingMax = 0.0;
  _pulse = 0;
  _debugFrameCount = 0;
  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function stopLoop(): void {
  if (_rafId !== null) {
    cancelAnimationFrame(_rafId);
    _rafId = null;
  }
  _pulse = 0;
  setState({ pulseIntensity: 0 });
}

// ── Public API ───────────────────────────────────────────────────────────────

export function setDiscoMode(on: boolean): void {
  setState({ discoMode: on });
  saveDiscoPrefs(on);
  if (on) startLoop(); else stopLoop();
}

export function toggleDiscoMode(): void {
  setDiscoMode(!_state.discoMode);
}

export function setDiscoPlaying(isPlaying: boolean): void {
  if (_state.discoMode && isPlaying) startLoop();
  else if (!isPlaying) stopLoop();
}

// ── React Hooks ──────────────────────────────────────────────────────────────

export function useDiscoSelector<T>(selector: (state: DiscoState) => T): T {
  const selectorRef = useRef(selector);
  selectorRef.current = selector;

  const selectedRef = useRef<T>(selector(_state));

  const getSelectedSnapshot = useCallback(() => {
    const next = selectorRef.current(_state);
    if (Object.is(selectedRef.current, next)) return selectedRef.current;
    selectedRef.current = next;
    return next;
  }, []);

  return useSyncExternalStore(subscribe, getSelectedSnapshot);
}

export function usePulseIntensity(): number {
  return useDiscoSelector(s => s.pulseIntensity);
}

export function useDiscoMode(): boolean {
  return useDiscoSelector(s => s.discoMode);
}

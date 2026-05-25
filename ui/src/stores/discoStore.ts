// discoStore.ts — Disco mode state + beat detection engine.
//
// Creates a dedicated AnalyserNode (smoothingTimeConstant=0.2) connected to
// the same audio graph as audioMotion. This gives us raw transient data for
// kick detection, while audioMotion keeps its smooth display (smoothing=0.7).

import { useSyncExternalStore, useRef, useCallback } from 'react';

// ── Types ────────────────────────────────────────────────────────────────────

interface DiscoState {
  discoMode: boolean;
  pulseIntensity: number;  // 0.0–1.0, onset-detected pulse
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
let _sampleRate = 44100;

// Kick drum FFT bin range (computed from sample rate + fftSize)
let _kickBinStart = 0;
let _kickBinEnd = 0;

export function registerAudioMotion(instance: any): void {
  _audioMotion = instance;

  // Create our own AnalyserNode with LOW smoothing for beat detection
  try {
    const ctx: AudioContext = instance.audioCtx;
    _sampleRate = ctx.sampleRate;

    _beatAnalyser = ctx.createAnalyser();
    _beatAnalyser.fftSize = 2048;
    _beatAnalyser.smoothingTimeConstant = 0.2;  // Much lower than audioMotion's 0.7

    // Connect the audio source to our analyser
    const sources = instance.connectedSources;
    if (sources && sources.length > 0) {
      sources[0].connect(_beatAnalyser);
      console.log('[Disco] Beat analyser connected to audio source');
    } else {
      console.warn('[Disco] No connected sources found on audioMotion');
    }

    // Allocate frequency data buffer
    _freqData = new Uint8Array(_beatAnalyser.frequencyBinCount);

    // Compute FFT bin indices for kick drum range (50–180Hz)
    const binHz = _sampleRate / _beatAnalyser.fftSize;
    _kickBinStart = Math.floor(50 / binHz);
    _kickBinEnd = Math.ceil(180 / binHz);
    console.log(`[Disco] Kick bins: ${_kickBinStart}-${_kickBinEnd} (${(_kickBinStart * binHz).toFixed(0)}-${(_kickBinEnd * binHz).toFixed(0)}Hz, binHz=${binHz.toFixed(1)})`);
  } catch (err) {
    console.error('[Disco] Failed to create beat analyser:', err);
  }
}

export function unregisterAudioMotion(): void {
  _audioMotion = null;
  _beatAnalyser = null;
  _freqData = null;
}

/** Read kick drum energy from our dedicated low-smoothing AnalyserNode */
function readKickEnergy(): number {
  if (!_beatAnalyser || !_freqData) return 0;

  _beatAnalyser.getByteFrequencyData(_freqData);

  // Average the kick drum bins (50–180Hz)
  let sum = 0;
  let count = 0;
  for (let i = _kickBinStart; i <= _kickBinEnd && i < _freqData.length; i++) {
    sum += _freqData[i];
    count++;
  }
  if (count === 0) return 0;

  // Normalise from 0-255 to 0-1
  return (sum / count) / 255;
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

// ── Beat Detection Loop (Onset Detection) ────────────────────────────────────

let _rafId: number | null = null;
let _lastFrameTime = 0;
let _debugFrameCount = 0;

// Onset detection state
let _runningAvg = 0;         // Slow-moving average of kick energy
let _pulse = 0;              // Current pulse value (0–1), decays per frame
let _lastHitTime = 0;        // Prevent double-triggers

// Tuning knobs — calibrated for low-smoothing AnalyserNode
const AVG_SMOOTHING = 0.96;  // Slow-moving baseline
const ONSET_THRESHOLD = 0.06; // Absolute delta above running avg to trigger
const REFRACTORY_MS = 100;   // Min ms between hits
const DECAY_RATE = 0.92;     // Per-frame decay (~200ms to near-zero at 60fps)
const HIT_SCALE = 3.0;       // How much to amplify the hit intensity

function beatDetectionLoop(timestamp: number): void {
  if (!_state.discoMode) {
    _rafId = null;
    _pulse = 0;
    setState({ pulseIntensity: 0 });
    return;
  }

  const dt = _lastFrameTime ? timestamp - _lastFrameTime : 16;
  _lastFrameTime = timestamp;

  // Read from our dedicated low-smoothing AnalyserNode
  const rawEnergy = readKickEnergy();

  // Update running average (slow-moving baseline)
  _runningAvg = _runningAvg * AVG_SMOOTHING + rawEnergy * (1 - AVG_SMOOTHING);

  // Onset detection: sudden increase above baseline
  const delta = rawEnergy - _runningAvg;
  const timeSinceHit = timestamp - _lastHitTime;

  if (delta > ONSET_THRESHOLD && timeSinceHit > REFRACTORY_MS) {
    // HIT! Spike proportional to delta strength
    const hitIntensity = Math.min(1.0, delta * HIT_SCALE);
    _pulse = Math.max(_pulse, hitIntensity);
    _lastHitTime = timestamp;
  }

  // Decay the pulse each frame
  _pulse *= DECAY_RATE;
  if (_pulse < 0.01) _pulse = 0;

  // Debug: log once per second
  _debugFrameCount++;
  if (_debugFrameCount % 60 === 0) {
    console.log(`[Disco] raw=${rawEnergy.toFixed(3)} avg=${_runningAvg.toFixed(3)} delta=${delta.toFixed(3)} pulse=${_pulse.toFixed(3)} analyser=${!!_beatAnalyser}`);
  }

  // Only notify if meaningfully changed
  if (Math.abs(_pulse - _state.pulseIntensity) > 0.003) {
    _state = { ..._state, pulseIntensity: _pulse };
    notify();
  }

  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function startLoop(): void {
  if (_rafId !== null) return;
  _lastFrameTime = 0;
  _runningAvg = 0;
  _pulse = 0;
  _lastHitTime = 0;
  _debugFrameCount = 0;
  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function stopLoop(): void {
  if (_rafId !== null) {
    cancelAnimationFrame(_rafId);
    _rafId = null;
  }
  _pulse = 0;
  _runningAvg = 0;
  setState({ pulseIntensity: 0 });
}

// ── Public API ───────────────────────────────────────────────────────────────

export function setDiscoMode(on: boolean): void {
  setState({ discoMode: on });
  saveDiscoPrefs(on);
  if (on) {
    startLoop();
  } else {
    stopLoop();
  }
}

export function toggleDiscoMode(): void {
  setDiscoMode(!_state.discoMode);
}

export function setDiscoPlaying(isPlaying: boolean): void {
  if (_state.discoMode && isPlaying) {
    startLoop();
  } else if (!isPlaying) {
    stopLoop();
  }
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

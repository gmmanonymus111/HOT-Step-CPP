// discoStore.ts — Disco mode state + beat detection engine.
//
// Uses the same useSyncExternalStore pattern as playbackStore.
// Reads kick drum energy from the registered audioMotion-analyzer instance
// via getEnergy(60, 150). No new AudioContext or AnalyserNode needed.

import { useSyncExternalStore, useRef, useCallback } from 'react';

// ── Types ────────────────────────────────────────────────────────────────────

interface DiscoState {
  discoMode: boolean;
  pulseIntensity: number;  // 0.0–1.0, smoothed kick energy
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

// ── audioMotion Registration ─────────────────────────────────────────────────

// Holds a reference to the audioMotion-analyzer instance from SpectrumAnalyzer.
// We call getEnergy() on it — no new AudioContext or AnalyserNode needed.
let _audioMotion: any = null;

export function registerAudioMotion(instance: any): void {
  _audioMotion = instance;
}

export function unregisterAudioMotion(): void {
  _audioMotion = null;
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

// ── Beat Detection Loop ──────────────────────────────────────────────────────

let _rafId: number | null = null;
let _smoothedEnergy = 0;
let _lastFrameTime = 0;
let _debugFrameCount = 0;

// Envelope follower parameters
const ATTACK_MS = 30;    // Fast attack — snappy response to beat hits
const DECAY_MS = 180;    // Slow decay — organic release

function beatDetectionLoop(timestamp: number): void {
  if (!_state.discoMode) {
    _rafId = null;
    setState({ pulseIntensity: 0 });
    return;
  }

  const dt = _lastFrameTime ? timestamp - _lastFrameTime : 16;
  _lastFrameTime = timestamp;

  let rawEnergy = 0;
  if (_audioMotion) {
    try {
      // getEnergy() with two numeric args returns average energy in that Hz range
      rawEnergy = _audioMotion.getEnergy(60, 150) ?? 0;
    } catch {
      rawEnergy = 0;
    }
  }

  // Debug: log once per second so we can verify values are coming through
  _debugFrameCount++;
  if (_debugFrameCount % 60 === 0) {
    console.log(`[Disco] raw=${rawEnergy.toFixed(3)} smoothed=${_smoothedEnergy.toFixed(3)} audioMotion=${!!_audioMotion}`);
  }

  // Envelope follower: fast attack, slow decay
  if (rawEnergy > _smoothedEnergy) {
    const attackFactor = 1 - Math.exp(-dt / ATTACK_MS);
    _smoothedEnergy += (rawEnergy - _smoothedEnergy) * attackFactor;
  } else {
    const decayFactor = 1 - Math.exp(-dt / DECAY_MS);
    _smoothedEnergy += (rawEnergy - _smoothedEnergy) * decayFactor;
  }

  // Clamp to 0–1
  const intensity = Math.max(0, Math.min(1, _smoothedEnergy));

  // Only notify if meaningfully changed (avoid excess re-renders)
  if (Math.abs(intensity - _state.pulseIntensity) > 0.005) {
    _state = { ..._state, pulseIntensity: intensity };
    notify();
  }

  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function startLoop(): void {
  if (_rafId !== null) return;
  _lastFrameTime = 0;
  _smoothedEnergy = 0;
  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function stopLoop(): void {
  if (_rafId !== null) {
    cancelAnimationFrame(_rafId);
    _rafId = null;
  }
  _smoothedEnergy = 0;
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

/**
 * Called by App.tsx when isPlaying changes — starts/stops the rAF loop
 * so we don't waste CPU when nothing is playing.
 */
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

/** Convenience: get just the pulse intensity (updates ~60fps when active) */
export function usePulseIntensity(): number {
  return useDiscoSelector(s => s.pulseIntensity);
}

/** Convenience: get disco mode on/off */
export function useDiscoMode(): boolean {
  return useDiscoSelector(s => s.discoMode);
}

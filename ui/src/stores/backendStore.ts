// backendStore.ts — Zustand store for the multi-backend registry
//
// Backends are the registered generation engines behind /api/backends — ACE-Step
// 1.5 today, future engines (e.g. MiniMax-Music3) later. See
// docs/plans/multi-backend-architecture.md §4.5. Almost every install only ever
// has ONE backend registered ('ace'); BackendToggle stays hidden until a second
// one exists, so this store is inert (one entry, no UI) for the common case.
//
// Fetch-on-demand, not polled: callers refetch capabilities explicitly (on
// mount / on backend switch) rather than the store running an interval loop.

import { create } from 'zustand';

// ── Types (mirror server/src/services/backends/types.ts — §4.2) ──

export interface BackendInfo {
  id: string;
  displayName: string;
  resourcePool: 'gpu' | 'remote';
  active: boolean;
}

export interface BackendCoreCapabilities {
  duration: { max: number };
  bpm: boolean;
  keyscale: boolean;
  negativePrompt: boolean;
  batch: { max: number };
  seed: boolean;
}

export interface BackendFeatureCapabilities {
  lm: boolean;
  plugins: boolean;
  adapters: boolean;
  cover: boolean;
  repaint: boolean;
  lego: boolean;
  extract: boolean;
  streaming: boolean;
  training: boolean;
  midi: boolean;
  stems: boolean;
  understand: boolean;
  conceptSteering: boolean;
}

/** Backend-specific knob schema — rendered generically by PluginControls,
 *  same shape as the Lua plugin param schema (types/pluginTypes.ts). */
export interface BackendExtensionParam {
  key: string;
  type: 'slider' | 'select' | 'toggle' | 'text';
  label: string;
  hint?: string;
  transform?: string;
  default?: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  visible_when?: { key: string; equals: string };
}

export interface BackendCapabilities {
  backend: string;
  up: boolean;
  core: BackendCoreCapabilities;
  features: BackendFeatureCapabilities;
  extensions: BackendExtensionParam[];
}

interface BackendState {
  backends: BackendInfo[];
  activeBackendId: string;
  /** Per-backend capability manifest, keyed by backend id. */
  capabilities: Record<string, BackendCapabilities>;
  loading: boolean;
  error: string | null;

  fetchBackends: () => Promise<void>;
  fetchCapabilities: (id?: string) => Promise<void>;
  switchBackend: (id: string) => Promise<void>;
}

export const useBackendStore = create<BackendState>((set, get) => ({
  backends: [],
  // Default 'ace' — matches the server's default and every pre-multi-backend
  // install; overwritten by fetchBackends() once the registry responds.
  activeBackendId: 'ace',
  capabilities: {},
  loading: false,
  error: null,

  fetchBackends: async () => {
    set({ loading: true, error: null });
    try {
      const res = await fetch('/api/backends');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: { backends: BackendInfo[]; activeId: string } = await res.json();
      set({
        backends: Array.isArray(data.backends) ? data.backends : [],
        activeBackendId: data.activeId || 'ace',
        loading: false,
      });
    } catch (err) {
      // Advisory — a failed fetch must not break generation. Every consumer
      // (getGlobalParams, BackendToggle) treats an empty/single-entry list as
      // "no multi-backend UI", which is exactly the pre-feature behaviour.
      console.warn('[Backends] fetch failed:', err instanceof Error ? err.message : String(err));
      set({ loading: false, error: err instanceof Error ? err.message : String(err) });
    }
  },

  fetchCapabilities: async (id) => {
    const backendId = id || get().activeBackendId;
    if (!backendId) return;
    try {
      const res = await fetch(`/api/capabilities?backend=${encodeURIComponent(backendId)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: BackendCapabilities = await res.json();
      set({ capabilities: { ...get().capabilities, [backendId]: data } });
    } catch (err) {
      console.warn(`[Backends] capabilities fetch failed for "${backendId}":`, err instanceof Error ? err.message : String(err));
    }
  },

  switchBackend: async (id) => {
    set({ error: null });
    try {
      const res = await fetch('/api/backends/active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: { activeId: string } = await res.json();
      set({
        activeBackendId: data.activeId,
        backends: get().backends.map(b => ({ ...b, active: b.id === data.activeId })),
      });
      // Refetch on success — the new active backend's manifest may not be
      // cached yet (or may be stale from a previous session).
      await get().fetchCapabilities(data.activeId);
    } catch (err) {
      console.warn('[Backends] switch failed:', err instanceof Error ? err.message : String(err));
      set({ error: err instanceof Error ? err.message : String(err) });
    }
  },
}));

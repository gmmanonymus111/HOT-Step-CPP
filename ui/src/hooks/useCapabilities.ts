// useCapabilities.ts — the ACTIVE backend's capability manifest
//
// PHASE 3 INTENT (docs/plans/multi-backend-architecture.md §4.2/§4.5): this is
// the hook studios, panels, and knob groups will gate on —
// `useCapabilities().capabilities?.features.*` — the same pattern
// `trainingStore.capabilities` + `CapabilityBanner` already establish for the
// Training Studio. Nothing consumes it yet; this file only fetches and exposes
// the manifest so Phase 3 gating work has a ready-made hook to import.
//
// Fetches once per backend id (cached in backendStore.capabilities) and
// refetches automatically when the active backend switches.

import { useEffect } from 'react';
import { useBackendStore, type BackendCapabilities } from '../stores/backendStore';

export interface UseCapabilitiesResult {
  /** null until the first fetch resolves (or the backend has never reported). */
  capabilities: BackendCapabilities | null;
  loading: boolean;
}

export function useCapabilities(): UseCapabilitiesResult {
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const capabilities = useBackendStore(s => s.capabilities[s.activeBackendId] ?? null);
  const loading = useBackendStore(s => s.loading);
  const fetchCapabilities = useBackendStore(s => s.fetchCapabilities);

  useEffect(() => {
    if (!activeBackendId) return;
    // Already cached for this backend — don't refetch on every mount, only on
    // an actual backend switch (fetchCapabilities is called explicitly there
    // too, from switchBackend, so this covers first-use and page reload).
    if (useBackendStore.getState().capabilities[activeBackendId]) return;
    void fetchCapabilities(activeBackendId);
  }, [activeBackendId, fetchCapabilities]);

  return { capabilities, loading };
}

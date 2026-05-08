// usePluginRegistry.ts — Fetch and cache the Lua plugin registry from the server
//
// Fetches once on mount, re-fetches when invalidated. Falls back to
// empty arrays if the engine is offline — the UI gracefully degrades
// to showing "no plugins available" messages.

import { useState, useEffect, useCallback } from 'react';
import type { PluginRegistry, PluginInfo } from '../types/pluginTypes';

const EMPTY_REGISTRY: PluginRegistry = { solvers: [], schedulers: [], guidance: [] };

let globalCache: PluginRegistry | null = null;

export function usePluginRegistry() {
  const [registry, setRegistry] = useState<PluginRegistry>(globalCache || EMPTY_REGISTRY);
  const [loading, setLoading] = useState(!globalCache);
  const [error, setError] = useState<string | null>(null);

  const fetchRegistry = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch('/api/plugins');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: PluginRegistry = await res.json();
      globalCache = data;
      setRegistry(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load plugins');
      // Keep existing data if available
      if (!globalCache) setRegistry(EMPTY_REGISTRY);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!globalCache) {
      fetchRegistry();
    }
  }, [fetchRegistry]);

  const reload = useCallback(async () => {
    globalCache = null;
    await fetchRegistry();
  }, [fetchRegistry]);

  // Lookup helpers
  const findSolver = useCallback((name: string): PluginInfo | undefined => {
    return registry.solvers.find(s => s.name === name);
  }, [registry]);

  const findScheduler = useCallback((name: string): PluginInfo | undefined => {
    return registry.schedulers.find(s => s.name === name);
  }, [registry]);

  const findGuidance = useCallback((name: string): PluginInfo | undefined => {
    return registry.guidance.find(g => g.name === name);
  }, [registry]);

  return {
    registry,
    loading,
    error,
    reload,
    findSolver,
    findScheduler,
    findGuidance,
  };
}

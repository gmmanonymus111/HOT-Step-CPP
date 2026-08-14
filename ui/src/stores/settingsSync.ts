// settingsSync.ts — Background sync: debounced save of global params to user's server settings
import { useGlobalParamsStore } from './globalParamsStore';
import { authApi } from '../services/api';

let syncTimer: ReturnType<typeof setTimeout> | null = null;
const SYNC_DELAY = 2000; // 2 seconds debounce

/** Get a snapshot of all non-function store values */
function getStoreSnapshot(): Record<string, unknown> {
  const store = useGlobalParamsStore.getState();
  const snapshot: Record<string, unknown> = {};
  for (const key of Object.keys(store)) {
    if (typeof store[key] !== 'function') {
      snapshot[key] = store[key];
    }
  }
  return snapshot;
}

/** Schedule a debounced sync to the server */
function scheduleSync(token: string) {
  if (syncTimer) clearTimeout(syncTimer);
  syncTimer = setTimeout(async () => {
    try {
      const snapshot = getStoreSnapshot();
      await authApi.updateSettings(snapshot, token);
    } catch {
      // Silently fail — network issues, etc.
    }
  }, SYNC_DELAY);
}

/** Subscribe to store changes and sync on modify */
export function startSettingsSync(token: string | null) {
  // Clear any existing subscription
  stopSettingsSync();

  if (!token) return;

  // Subscribe to all store changes
  useGlobalParamsStore.subscribe(() => {
    scheduleSync(token);
  });
}

/** Stop the sync subscription */
export function stopSettingsSync() {
  if (syncTimer) {
    clearTimeout(syncTimer);
    syncTimer = null;
  }
}
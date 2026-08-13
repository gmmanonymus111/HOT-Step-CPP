// backends/registry.ts — id → EngineBackend map, active-backend selection
//
// Shape copied from services/lireek/llm/registry.ts (the provider registry
// blueprint named in the plan, §3.6). Active id persists through the same
// generic settings key-value table training/activeModels.ts already uses
// (server/src/db/lireekDb.ts `settings` table) — no new persistence
// mechanism, no schema change.

import { getSetting, setSetting } from '../../db/lireekDb.js';
import type { EngineBackend } from './types.js';
import { aceBackend } from './ace/index.js';
import { minimaxBackend } from './minimax/index.js';

const SETTING_KEY = 'active_backend_id';
const DEFAULT_BACKEND_ID = 'ace';

// Registration order is the UI's toggle order. ACE stays first and stays the
// default — an existing install must behave exactly as before until the user
// deliberately switches.
const backends: Record<string, EngineBackend> = {
  ace: aceBackend,
  'minimax-m3': minimaxBackend,
};

/** Look up a backend by id, or undefined if unregistered. */
export function getBackend(id: string): EngineBackend | undefined {
  return backends[id];
}

/** All registered backends, in registration order. */
export function listBackends(): EngineBackend[] {
  return Object.values(backends);
}

/** The persisted active backend id. Falls back to the default if the stored
 *  value doesn't name a registered backend (e.g. a backend was removed). */
export function getActiveBackendId(): string {
  const id = getSetting(SETTING_KEY, DEFAULT_BACKEND_ID);
  return backends[id] ? id : DEFAULT_BACKEND_ID;
}

/** Set the active backend id. Returns false (no-op) if `id` isn't registered. */
export function setActiveBackendId(id: string): boolean {
  if (!backends[id]) return false;
  setSetting(SETTING_KEY, id);
  return true;
}

/** The active EngineBackend instance. */
export function getActiveBackend(): EngineBackend {
  return backends[getActiveBackendId()] ?? backends[DEFAULT_BACKEND_ID];
}

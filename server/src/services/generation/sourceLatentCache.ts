// generation/sourceLatentCache.ts — LRU cache for VAE-encoded source audio latents
//
// Keyed by (sourceAudioUrl, tempoScale, pitchShift) so each unique audio
// processing configuration gets its own cached source latent. Eliminates
// ~45s of redundant VAE encoding when re-running cover tasks with the
// same source audio and tempo/pitch settings.

const MAX_ENTRIES = 20;  // ~20 songs × ~3MB each = ~60MB worst case

interface SourceLatentEntry {
  rawLatent: Buffer;
  timestamp: number;
}

const cache = new Map<string, SourceLatentEntry>();

/** Build a cache key from source audio URL and processing params */
export function sourceLatentCacheKey(
  sourceAudioUrl: string,
  tempoScale: number = 1.0,
  pitchShift: number = 0,
): string {
  // Normalise tempo/pitch to avoid floating-point key drift
  const t = Math.round(tempoScale * 1000) / 1000;
  const p = Math.round(pitchShift * 10) / 10;
  return `${sourceAudioUrl}|t=${t}|p=${p}`;
}

/** Look up a cached source latent. Returns raw f32 bytes or undefined. */
export function getSourceLatentCache(key: string): Buffer | undefined {
  const entry = cache.get(key);
  if (!entry) return undefined;
  entry.timestamp = Date.now();  // refresh LRU
  return entry.rawLatent;
}

/** Store a source latent in the cache. Evicts oldest if full. */
export function setSourceLatentCache(key: string, rawLatent: Buffer): void {
  if (rawLatent.length === 0) return;

  // Evict oldest if at capacity
  while (cache.size >= MAX_ENTRIES) {
    let oldestKey = '';
    let oldestTs = Infinity;
    for (const [k, v] of cache) {
      if (v.timestamp < oldestTs) {
        oldestTs = v.timestamp;
        oldestKey = k;
      }
    }
    if (oldestKey) cache.delete(oldestKey);
  }

  cache.set(key, { rawLatent, timestamp: Date.now() });
}

/** Current number of cached entries */
export function getSourceLatentCacheSize(): number {
  return cache.size;
}

/** Invalidate a specific cache entry */
export function invalidateSourceLatentCache(key: string): boolean {
  return cache.delete(key);
}

/** Clear the entire cache */
export function clearSourceLatentCache(): void {
  cache.clear();
}

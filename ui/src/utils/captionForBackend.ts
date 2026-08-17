/**
 * captionForBackend.ts — pick the caption the ACTIVE backend was trained on.
 *
 * A Lyric Studio generation carries two captions, and they are NOT two renderings
 * of the same text:
 *
 *   caption      ACE-Step 1.5 — 2-4 sentences of flowing description.
 *   caption_mm3  MiniMax-Music3 — a three-heading Structured Caption with thirteen
 *                fixed labels (Global Metadata / Vocal Details / Arrangement).
 *
 * Handing one model the other's caption is a measured quality loss, not a
 * cosmetic mismatch: in a controlled A/B (2026-08-14, one track, 5 seeds per arm,
 * no adapter) a rich ACE-style caption fed to MM3 produced the right genre in 1
 * take of 5, while the same content in MM3's format was on-genre throughout.
 * See server/src/services/lireek/prompts.ts for the full write-up.
 *
 * Fallback is deliberate and one-way: an MM3 run with no MM3 caption gets the
 * ACE caption, because degraded conditioning beats no conditioning — every
 * generation written before this field existed is in that state. The reverse
 * never happens; an ACE run is never handed a Structured Caption.
 */

/** The registered id of the MiniMax-Music3 backend (server/src/services/backends/registry.ts). */
export const MM3_BACKEND_ID = 'minimax-m3';

export function captionForBackend(
  gen: { caption?: string | null; caption_mm3?: string | null },
  backendId: string | undefined,
): string {
  if (backendId === MM3_BACKEND_ID && gen.caption_mm3?.trim()) return gen.caption_mm3;
  return gen.caption || '';
}

// wildcardUtils.ts — Seeded recursive wildcard expander for STORM Streaming
// MDMAchine / A&E Concepts 2026
// GPL v3 — safe for public MD_Nodes repo
//
// Resolves nested {A|B|C} syntax deterministically from a seed.
// Zero dependencies — runs entirely client-side in the browser.
//
// Design notes:
//   - Mulberry32 PRNG: fast, well-distributed, seedable with a uint32.
//   - Iterative (not recursive) to avoid call-stack blowup on deep nesting.
//   - Innermost braces resolved first (same behaviour as Python RecursiveWildcardProcessor).
//   - Slot-scoped seeding: pass (streamSeed ^ playingSlot) so two DJ decks
//     running the same template expand to independent variants.
//   - hasWildcards() lets the UI show/hide the expand button cheaply.
//   - expandInPlace() wraps the textarea selection so you can expand just
//     the selected text if the user highlights a region.

// ─────────────────────────────────────────────────────────────────────────────
// PRNG — mulberry32
// Reference: https://gist.github.com/tommyettinger/46a874533244883189143505d203312c
// ─────────────────────────────────────────────────────────────────────────────

function mulberry32(seed: number): () => number {
  // Seed must be a uint32. JS bitwise ops already truncate to 32-bit.
  let s = seed >>> 0;
  return function () {
    s += 0x6d2b79f5;
    let z = s;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
}

// Seed mixer: XOR fold a JS number (up to 53-bit safe integer) into uint32.
// streamSeed ^ playingSlot can exceed 32 bits if streamSeed is large, so we
// fold the high bits down before handing to mulberry32.
function mixSeed(seed: number, slot = 0): number {
  const combined = seed ^ slot;
  // XOR high 21 bits into low 32
  return ((combined ^ (combined / 0x100000000)) >>> 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Core expansion
// ─────────────────────────────────────────────────────────────────────────────

const INNER_BRACES = /\{([^{}]+)\}/g;
const MAX_PASSES   = 64; // safety ceiling for deeply nested templates

/**
 * Expand all {A|B|C} wildcards in `text` using the given seed.
 * Nested wildcards are resolved innermost-first.
 * delimiter defaults to '|' — matches ComfyUI behaviour.
 */
export function expandWildcards(
  text: string,
  seed: number,
  slot     = 0,
  delimiter = '|',
): string {
  if (!text || !hasWildcards(text)) return text;

  const rng = mulberry32(mixSeed(seed, slot));

  let current = text;
  for (let pass = 0; pass < MAX_PASSES; pass++) {
    // Check if there's anything left to expand
    if (!hasWildcards(current)) break;

    // Replace all innermost {…} groups in one pass
    let matched = false;
    current = current.replace(INNER_BRACES, (_full, content: string) => {
      matched = true;
      const options = content.split(delimiter).map(o => o.trim()).filter(o => o.length > 0);
      if (options.length === 0) return '';
      if (options.length === 1) return options[0];
      const idx = Math.floor(rng() * options.length);
      return options[idx];
    });

    // Reset regex lastIndex (global flag state)
    INNER_BRACES.lastIndex = 0;

    if (!matched) break;
  }

  return current;
}

/**
 * Cheap check — does the string contain any {…} groups?
 * Used to decide whether to show the expand button in the UI.
 */
export function hasWildcards(text: string): boolean {
  return /\{[^{}]*\}/.test(text);
}

/**
 * Expand wildcards inside a textarea's current selection only.
 * If nothing is selected, expands the entire value.
 * Returns { value, selectionStart, selectionEnd } — apply to the textarea element.
 */
export function expandInPlace(
  el:    HTMLTextAreaElement,
  seed:  number,
  slot   = 0,
  delim  = '|',
): { value: string; selectionStart: number; selectionEnd: number } {
  const { value, selectionStart: ss, selectionEnd: se } = el;

  if (ss === se || ss === null || se === null) {
    // No selection — expand everything
    const expanded = expandWildcards(value, seed, slot, delim);
    return { value: expanded, selectionStart: 0, selectionEnd: expanded.length };
  }

  // Selection only
  const before   = value.slice(0, ss);
  const selected = value.slice(ss, se);
  const after    = value.slice(se);
  const expanded = expandWildcards(selected, seed, slot, delim);
  const newValue = before + expanded + after;
  return {
    value:          newValue,
    selectionStart: ss,
    selectionEnd:   ss + expanded.length,
  };
}

/**
 * Generate a random seed suitable for wildcard expansion.
 * Uses crypto.getRandomValues when available (browser), falls back to Math.random.
 * Matches the JS_MAX_SAFE_INTEGER ceiling used by MD_Nodes SeedSaver.
 */
export function randomWildcardSeed(): number {
  const JS_MAX_SAFE = 9007199254740991;
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    const buf = new Uint32Array(2);
    crypto.getRandomValues(buf);
    // Combine two uint32s into a safe integer
    const hi = buf[0] & 0x1fffff; // top 21 bits
    const lo = buf[1] >>> 0;      // bottom 32 bits
    return Math.min(hi * 0x100000000 + lo, JS_MAX_SAFE);
  }
  return Math.floor(Math.random() * JS_MAX_SAFE);
}

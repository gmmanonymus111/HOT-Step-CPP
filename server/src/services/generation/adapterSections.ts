// generation/adapterSections.ts — Per-section adapter masking (regional LoRA)
//
// Parses inline per-section adapter-influence directives from the lyrics, e.g.
//
//   [Intro]{adapter_1=1; adapter_2=0}
//   [Verse 1]{adapter_1=0.5; adapter_2=0.5}
//   ...lines...
//   [Chorus]{adapter_1=0; adapter_2=1}
//
// and turns them into a per-section weight table indexed to the loaded adapter
// stack, plus the lyrics with the {…} directives stripped (the model must never
// see them). Keyed by trigger word (adapter filename stem), or positional #N.
//
// Sum/Blend (issue #72) is reused per section; directive-less sections fall back
// to the stack's normal effective scales ("uniform blend of the stack").
// See docs/plans/per-section-adapter-masking.md.

export interface AdapterSection {
  weights: number[]; // effective per-adapter scale, indexed to the stack
  size: number;      // relative frame-allocation hint (section char count)
}

export interface ParsedAdapterSections {
  lyrics: string;                 // directives stripped
  sections?: AdapterSection[];    // undefined when the feature is inactive
}

/** filename stem (trigger word) for an adapter path */
function triggerOf(p: string): string {
  return (p.split(/[\\/]/).pop() || p).replace(/\.safetensors$/i, '');
}

/** Resolve a directive key ("adapter_1", trigger word, or "#2") to a stack index, or -1. */
function resolveKey(key: string, triggers: string[]): number {
  const k = key.trim().toLowerCase();
  const byTrigger = triggers.findIndex(t => t.toLowerCase() === k);
  if (byTrigger >= 0) return byTrigger;
  const m = k.match(/^#?(\d+)$/);
  if (m) {
    const idx = parseInt(m[1], 10) - 1; // 1-based
    if (idx >= 0 && idx < triggers.length) return idx;
  }
  return -1;
}

/** Parse a `key=val; key=val` directive body into raw per-adapter weights (unmentioned → 0). */
function parseDirective(body: string, triggers: string[]): number[] {
  const raw = new Array(triggers.length).fill(0);
  for (const part of body.split(/[;,]/)) {
    const eq = part.indexOf('=');
    if (eq < 0) continue;
    const key = part.slice(0, eq).trim();
    const val = parseFloat(part.slice(eq + 1).trim());
    if (!key || !Number.isFinite(val)) continue;
    const idx = resolveKey(key, triggers);
    if (idx >= 0) raw[idx] = val;
  }
  return raw;
}

/** Apply the #72 Sum/Blend transform to a section's raw weights. */
function applyMode(raw: number[], mode: string, budget: number): number[] {
  if (mode === 'blend') {
    const sum = raw.reduce((a, b) => a + (b || 0), 0);
    if (sum > 0) return raw.map(w => +(budget * (w || 0) / sum).toFixed(4));
    return raw.map(() => 0); // explicit all-zero directive → base only
  }
  return raw.map(w => w || 0); // sum: raw as-is
}

/**
 * Parse per-section adapter directives from lyrics.
 * @param lyrics    raw lyrics (may contain `[Section]{…}` directives)
 * @param stack     loaded adapter stack (effective scales), order matches the engine
 * @param mode      'sum' | 'blend'
 * @param budget    combined-strength budget (blend)
 */
export function parseAdapterSections(
  lyrics: string,
  stack: { path: string; scale: number }[],
  mode: string,
  budget: number,
): ParsedAdapterSections {
  if (!lyrics || !Array.isArray(stack) || stack.length < 2) return { lyrics };
  // Fast bail-out: no directive syntax at all.
  if (!/\]\s*\{[^}]*\}/.test(lyrics) && !/^\s*\{[^}]*\}/.test(lyrics)) return { lyrics };

  const triggers = stack.map(s => triggerOf(s.path));
  const defaultWeights = stack.map(s => s.scale); // uniform blend of the stack

  // Split into sections at [Header] lines, capturing an optional {…} directive
  // that follows the header. Content before the first header is an implicit
  // directive-less section.
  const headerRe = /\[[^\]\n]+\]/g;
  const sections: AdapterSection[] = [];
  let cleaned = '';
  let lastIndex = 0;
  let pendingDirective: number[] | null = null; // for the preamble (none)

  // Helper to push a section given its body text and directive (raw weights or null).
  const pushSection = (body: string, raw: number[] | null) => {
    const size = Math.max(1, body.replace(/\s+/g, ' ').trim().length);
    const weights = raw ? applyMode(raw, mode, budget) : defaultWeights.slice();
    sections.push({ weights, size });
  };

  const matches = [...lyrics.matchAll(headerRe)];
  if (matches.length === 0) return { lyrics };

  // Preamble before the first header (rare) → implicit default section.
  const firstStart = matches[0].index ?? 0;
  if (firstStart > 0 && lyrics.slice(0, firstStart).trim().length > 0) {
    pushSection(lyrics.slice(0, firstStart), null);
  }
  cleaned += lyrics.slice(0, firstStart);
  lastIndex = firstStart;

  for (let mi = 0; mi < matches.length; mi++) {
    const h = matches[mi];
    const hStart = h.index ?? 0;
    const header = h[0];
    let cursor = hStart + header.length;

    // Optional directive immediately after the header (allowing whitespace).
    let raw: number[] | null = null;
    const after = lyrics.slice(cursor);
    const dm = after.match(/^[ \t]*\{([^}]*)\}/);
    let headerOut = header;
    if (dm) {
      raw = parseDirective(dm[1], triggers);
      cursor += dm[0].length; // skip the directive in the output
    } else {
      raw = pendingDirective; // (unused; kept for symmetry)
    }

    // Body runs until the next header (or end).
    const bodyEnd = (mi + 1 < matches.length) ? (matches[mi + 1].index ?? lyrics.length) : lyrics.length;
    const body = lyrics.slice(cursor, bodyEnd);

    pushSection(body, raw);
    cleaned += headerOut + body;
    lastIndex = bodyEnd;
  }
  cleaned += lyrics.slice(lastIndex);

  if (sections.length === 0) return { lyrics };
  return { lyrics: cleaned, sections };
}

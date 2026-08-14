/**
 * Trigger-word composition — one implementation, two call sites.
 *
 * A trigger word is the token an adapter was trained to respond to. It has to
 * appear in the caption at the SAME position the training captions carried it
 * (`preprocess-run.h:192-204` prepends or appends `custom_tag`), which is why a
 * trigger travels as a `{word, placement}` pair rather than a bare string.
 *
 * Two places inject: `translateParams` (before the request leaves for the
 * engine) and `routes/generate.ts` (again after the planner LM's chain-of-
 * thought rewrites the caption and drops whatever was there). Both use this
 * module so the two can never disagree.
 *
 * docs/plans/2026-07-28-adapter-trigger-embedding.md T7
 */

export type TriggerPlacement = 'prepend' | 'append' | 'replace';

/** Where a resolved trigger came from — surfaced on the Create panel's chip. */
export type TriggerSource = 'embedded' | 'filename' | 'override' | 'none';

export interface TriggerSpec {
  word: string;
  placement: TriggerPlacement;
  source?: TriggerSource;
  /** Adapter this trigger belongs to, when the caller knows it. */
  path?: string;
}

/** The subset of GenerationParams this module reads. */
interface TriggerParamsLike {
  triggerSpecs?: Array<{ word?: string; placement?: string; source?: string; path?: string }>;
  triggerWords?: string[];
  triggerWord?: string;
  triggerPlacement?: string;
}

function asPlacement(v: unknown, dflt: TriggerPlacement = 'prepend'): TriggerPlacement {
  return v === 'append' || v === 'replace' || v === 'prepend' ? v : dflt;
}

/**
 * Normalise whatever the caller sent into a spec list.
 *
 * `triggerSpecs` is the current shape (one placement per adapter, because a
 * stack can mix adapters trained at different positions). The older flat
 * `triggerWords` + single `triggerPlacement` form is still accepted — saved
 * presets and the queue path both persist it — and maps to one placement for
 * every word.
 */
export function resolveTriggerSpecs(params: TriggerParamsLike): TriggerSpec[] {
  if (Array.isArray(params.triggerSpecs) && params.triggerSpecs.length) {
    return params.triggerSpecs
      .map(s => ({
        word: (s?.word || '').trim(),
        placement: asPlacement(s?.placement),
        source: (s?.source as TriggerSource) || undefined,
        path: s?.path || undefined,
      }))
      .filter(s => s.word.length > 0);
  }
  const words = (Array.isArray(params.triggerWords) && params.triggerWords.length)
    ? params.triggerWords
    : (params.triggerWord ? [params.triggerWord] : []);
  if (!words.length || !params.triggerPlacement) return [];
  const placement = asPlacement(params.triggerPlacement);
  return words.map(w => (w || '').trim()).filter(Boolean).map(word => ({ word, placement }));
}

/**
 * Decide the trigger for every loaded adapter.
 *
 * Precedence, per adapter (T6):
 *   1. a manual override the user typed for that adapter
 *   2. the trigger embedded in the adapter's own safetensors metadata — what it
 *      was actually TRAINED with, so it beats any guess
 *   3. whatever the client derived from the filename (only sent when the
 *      "use filename" setting is on)
 *
 * When no adapter path resolves anything, the caller's flat spec list is used
 * verbatim — that is the legacy path saved presets and older clients take.
 */
export function resolveAdapterTriggers(
  adapterPaths: string[],
  clientSpecs: TriggerSpec[],
  readEmbedded: (adapterPath: string) => { trigger: string; position: 'prepend' | 'append' | 'replace' | '' },
): TriggerSpec[] {
  const out: TriggerSpec[] = [];
  const matched = new Set<TriggerSpec>();

  for (const p of adapterPaths) {
    if (!p) continue;

    const override = clientSpecs.find(s => s.path === p && s.source === 'override');
    if (override) { out.push(override); matched.add(override); continue; }

    let embedded = { trigger: '', position: '' as 'prepend' | 'append' | 'replace' | '' };
    try { embedded = readEmbedded(p); } catch { /* an unreadable adapter simply has no trigger */ }
    if (embedded.trigger) {
      // All three positions come through verbatim. 'replace' in particular MUST
      // NOT be downgraded to 'prepend': an adapter trained with
      // tag_position=replace only ever saw the bare trigger as its caption, so
      // prepending it to prose is exactly the off-distribution case the
      // embedded position exists to prevent.
      out.push({
        word: embedded.trigger,
        placement: embedded.position === 'append' ? 'append'
          : embedded.position === 'replace' ? 'replace'
            : 'prepend',
        source: 'embedded',
        path: p,
      });
      continue;
    }

    const fromClient = clientSpecs.find(s => s.path === p && !matched.has(s));
    if (fromClient) { out.push(fromClient); matched.add(fromClient); }
  }

  // Specs the client sent without a path (the flat triggerWords form) only
  // apply when per-adapter resolution produced nothing at all — otherwise a
  // stale preset would duplicate a trigger we already resolved properly.
  if (!out.length) return clientSpecs;

  // Path-less client specs alongside resolved ones: keep them only if their word
  // is not already present, so an override for one adapter in a stack does not
  // silently drop the others.
  const words = new Set(out.map(s => s.word));
  for (const s of clientSpecs) {
    if (!s.path && !words.has(s.word)) { out.push(s); words.add(s.word); }
  }
  return out;
}

export interface ApplyTriggersResult {
  caption: string;
  /** Words actually written into the caption this call. */
  applied: string[];
  /** Words skipped because the caption already contained them. */
  skipped: string[];
}

/**
 * Compose `specs` into `caption`.
 *
 * A `replace` spec wins outright — it is only reachable from the global setting
 * or a manual override (training never produces one), and it means "the caption
 * IS the trigger". Otherwise prepend-group words go in front and append-group
 * words go behind, each in the order the adapters were stacked.
 *
 * `skipPresent` drops words the caption already contains, which is what the
 * post-CoT re-injection wants: the LM often keeps the trigger it was trained on,
 * and re-adding it would duplicate the token.
 */
export function applyTriggers(
  caption: string,
  specs: TriggerSpec[],
  opts: { skipPresent?: boolean } = {},
): ApplyTriggersResult {
  const base = caption || '';
  if (!specs.length) return { caption: base, applied: [], skipped: [] };

  const replace = specs.filter(s => s.placement === 'replace');
  if (replace.length) {
    const words = dedupe(replace.map(s => s.word));
    return { caption: words.join(', '), applied: words, skipped: [] };
  }

  const skipped: string[] = [];
  const keep = (s: TriggerSpec) => {
    if (opts.skipPresent && base.includes(s.word)) { skipped.push(s.word); return false; }
    return true;
  };
  const pre = dedupe(specs.filter(s => s.placement === 'prepend').filter(keep).map(s => s.word));
  const post = dedupe(specs.filter(s => s.placement === 'append').filter(keep).map(s => s.word));

  let out = base;
  if (pre.length) out = out ? `${pre.join(', ')}, ${out}` : pre.join(', ');
  if (post.length) out = out ? `${out}, ${post.join(', ')}` : post.join(', ');
  return { caption: out, applied: [...pre, ...post], skipped };
}

function dedupe(words: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const w of words) {
    if (w && !seen.has(w)) { seen.add(w); out.push(w); }
  }
  return out;
}

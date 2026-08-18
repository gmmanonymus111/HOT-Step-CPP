// training/mossCaption.ts — the first caption provider that actually LISTENS
//
// Every other provider in the registry rewrites text. MOSS-Music-8B runs locally
// through `ace-caption` (engine/tools/ace-caption.cpp) and writes captions from
// the audio itself, which is a different kind of output, not a cheaper one:
//
//   Rob's blind-ish A/B, 2026-08-15, same track / lyrics / 5 seeds, Basic
//   Attributes forced identical in both arms so prose was the only variable:
//     restructured-from-Gemini -> "more plain rock, not particularly punk/emo"
//     MOSS-from-audio          -> "sounds like alkaline trio already, that's
//                                  crazy ... MOSS is WAY better"
//   Two of five seeds hit the target artist with NO ADAPTER LOADED.
//
// So this is the preferred path when the binary and weights are present, and the
// cloud rewrite in enhanceService.ts is the fallback, not the other way round.
//
// ── What this module does NOT trust MOSS with ────────────────────────────────
//
// Tempo and key. MOSS is a documented blind spot on both — measured on
// `03-alkaline_trio-burn.flac` it said ~102 BPM / C# minor where Essentia says
// 90 / E major, and Essentia is right. Those two numbers are already known
// exactly from local analysis, so `applyFactSubstitution` overwrites MOSS's
// `Basic Attributes:` line with the sidecar's values and keeps every other
// section MOSS wrote. This supersedes engine/tools/mm3-caption-hybrid.py, which
// did the same job as a post-pass over dataset folders.

import fs from 'fs';
import os from 'os';
import path from 'path';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { config, getFFmpegPath } from '../../config.js';
import { CAPTION_INSTRUCTIONS } from './captionPrompt.js';

/** Modes `ace-caption` can decode off a single encode. */
export type MossMode = 'prose' | 'mm3' | 'lyrics';

export interface MossPaths {
  exe: string;
  modelsDir: string;
  lm: string;
  audio: string;
}

const BIN = process.platform === 'win32' ? '.exe' : '';

/**
 * Resolve the binary + weights, or explain what is missing.
 *
 * The LM is looked up q8_0-first to match `ace-caption`'s own probe order: q8_0
 * is half the download and measurably no worse for captioning, so a user who has
 * both should get the small one, and the resolution order here must not disagree
 * with the engine's or the UI would advertise a model the engine won't load.
 */
export function resolveMossPaths(): { paths: MossPaths } | { missing: string } {
  const exe = path.join(path.dirname(config.aceServer.exe), `ace-caption${BIN}`);
  if (!fs.existsSync(exe)) return { missing: `ace-caption${BIN} not built` };

  const modelsDir = path.join(config.aceServer.models, 'moss');
  const lm = ['moss-lm-q8_0.gguf', 'moss-lm-f16.gguf']
    .map(n => path.join(modelsDir, n))
    .find(p => fs.existsSync(p));
  const audio = path.join(modelsDir, 'moss-aud-f16.gguf');

  if (!lm) return { missing: `no moss-lm-*.gguf in ${modelsDir}` };
  if (!fs.existsSync(audio)) return { missing: `moss-aud-f16.gguf missing from ${modelsDir}` };
  return { paths: { exe, modelsDir, lm, audio } };
}

export function mossCaptionAvailable(): boolean {
  return 'paths' in resolveMossPaths();
}

// ── Running the model ────────────────────────────────────────────────────

export interface MossRunOptions {
  modes: MossMode[];
  /** Per-mode prompt overrides. `prose` normally carries CAPTION_INSTRUCTIONS. */
  prompts?: Partial<Record<MossMode, string>>;
  maxTokens?: number;
  /** Hard audio cap in seconds. Cost is roughly linear past the encode. */
  maxSeconds?: number;
  temperature?: number;
  repPenalty?: number;
  freqPenalty?: number;
  signal?: AbortSignal;
  log?: (level: 'info' | 'warn', message: string) => void;
}

/**
 * Sampling defaults, measured rather than guessed.
 *
 * rep 1.05 + freq 0.3 was the optimum of a sweep over the repetition failure
 * mode that made MOSS's early lyric output look far worse than it is: bare
 * `repetition_penalty` fires on PRESENCE, so at the strength needed to break a
 * loop it also suppresses words a lyric legitimately repeats — a chorus. Adding
 * `frequency_penalty`, which scales with COUNT, breaks loops without flattening
 * choruses. Do not raise rep to fix a loop; raise freq.
 */
export const MOSS_TEMPERATURE = 0;      // greedy — captions are facts, not prose style
// 1.0 = off, and that is parity, not a guess: the "rep 1.05 + freq 0.3" sweep
// ran through SGLang, which parses repetition_penalty and NEVER applies it (no
// repetition_penalty.py in its penaltylib, nothing in the sampling path reads
// it). So the sweep's optimum was frequency_penalty alone, and applying rep
// here is a divergence from the runtime the settings were tuned on. Verified
// 2026-08-18: rep 1.0 vs 1.05 changes none of the 9 americanfootball captions.
export const MOSS_REP_PENALTY = 1.0;
export const MOSS_FREQ_PENALTY = 0.3;

/**
 * MOSS runs strictly one at a time, enforced HERE rather than at the call sites.
 *
 * The caption path is governed by `captionLimiter`, whose concurrency is
 * user-configurable (`config.labeling.captionConcurrency`) and which several
 * people will reasonably raise — for a cloud provider, N parallel API calls is
 * exactly the right thing to do. For MOSS it is fatal: each run is a separate
 * process loading ~8.1 GB of LM plus 1.6 GB of audio tower, so concurrency 2
 * exceeds a 5090 before either run reaches the first decode. Serialising inside
 * the module means raising the limiter stays safe for the cloud providers it was
 * meant for, and no future caller has to know this.
 */
let mossQueue: Promise<unknown> = Promise.resolve();

function serialised<T>(body: () => Promise<T>): Promise<T> {
  const run = mossQueue.then(body, body);
  // Keep the chain alive after a rejection, and never leak the rejection into
  // the next waiter — it must see a settled predecessor, not a failed one.
  mossQueue = run.then(() => undefined, () => undefined);
  return run;
}

/**
 * One encode, N decodes. Returns the text per requested mode.
 *
 * The encoder (mel + 32 Whisper layers) is prompt-independent, so asking for
 * prose+mm3 together costs one audio pass and two short decodes rather than two
 * full runs. That is the whole reason `--mode` takes a list.
 */
export function runMossCaption(
  audioPath: string,
  opts: MossRunOptions,
): Promise<Partial<Record<MossMode, string>>> {
  return serialised(() => runMossCaptionInner(audioPath, opts));
}

async function runMossCaptionInner(
  audioPath: string,
  opts: MossRunOptions,
): Promise<Partial<Record<MossMode, string>>> {
  const resolved = resolveMossPaths();
  if ('missing' in resolved) throw new Error(`MOSS captioning unavailable: ${resolved.missing}`);
  const { exe, modelsDir } = resolved.paths;

  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'hs_moss_'));
  const outBase = path.join(work, 'out.txt');
  const args: string[] = [
    '--models', modelsDir,
    '--src-audio', audioPath,
    '--mode', opts.modes.join(','),
    '-o', outBase,
    '--temperature', String(opts.temperature ?? MOSS_TEMPERATURE),
    '--rep-penalty', String(opts.repPenalty ?? MOSS_REP_PENALTY),
    '--freq-penalty', String(opts.freqPenalty ?? MOSS_FREQ_PENALTY),
  ];
  if (opts.maxTokens) args.push('--max-tokens', String(opts.maxTokens));
  if (opts.maxSeconds) args.push('--max-seconds', String(opts.maxSeconds));

  // ffmpeg is only consulted for non-WAV/MP3 input, but passing our resolved
  // path means a portable release doesn't depend on ffmpeg being on PATH.
  const ffmpeg = getFFmpegPath();
  if (ffmpeg) args.push('--ffmpeg', ffmpeg);

  // Per-mode prompt files. The `<mode>=` prefix is what keeps the AS1.5 block and
  // the MM3 Structured Caption on ONE encode — a bare --prompt-file would force
  // both modes onto the same prompt and defeat the point.
  for (const [mode, text] of Object.entries(opts.prompts ?? {})) {
    if (!text?.trim()) continue;
    const pf = path.join(work, `prompt.${mode}.txt`);
    fs.writeFileSync(pf, text, 'utf8');
    args.push('--prompt-file', `${mode}=${pf}`);
  }

  try {
    const stderr = await new Promise<string>((resolve, reject) => {
      execFile(exe, args,
        { timeout: 30 * 60_000, maxBuffer: 32 * 1024 * 1024, signal: opts.signal },
        (error, _stdout, errText) => {
          if (error) { reject(new Error(`ace-caption failed: ${String(errText || error).slice(0, 400)}`)); return; }
          resolve(errText || '');
        });
    });
    const timing = /encode[^\n]*|\b\d+\.\d+\s*s\b/.exec(stderr);
    if (timing) opts.log?.('info', `MOSS: ${timing[0].trim()}`);

    const out: Partial<Record<MossMode, string>> = {};
    for (const mode of opts.modes) {
      // `-o out.txt --mode a,b` writes out.a.txt / out.b.txt — but with a SINGLE
      // mode ace-caption writes the bare `-o` path with no infix. Reading only
      // the suffixed name silently yields no caption for single-mode calls: the
      // run costs a full encode, exits 0, and returns nothing. Try both.
      const text = [`out.${mode}.txt`, 'out.txt']
        .map(n => path.join(work, n))
        .filter(p => fs.existsSync(p))
        .map(p => fs.readFileSync(p, 'utf8').trim())
        .find(t => t.length > 0);
      if (text) out[mode] = text;
    }
    if (Object.keys(out).length === 0) {
      // Never fail silently. A zero-output run means the CLI contract moved, and
      // the caller must not mistake it for "the model had nothing to say".
      opts.log?.('warn', `MOSS produced no output for modes: ${opts.modes.join(',')}`);
    }
    return out;
  } finally {
    try { fs.rmSync(work, { recursive: true, force: true }); } catch { /* tmp */ }
  }
}

// ── Fact substitution (supersedes mm3-caption-hybrid.py) ─────────────────

/**
 * Genre labels, MOST SPECIFIC FIRST — the ordering is load-bearing.
 *
 * `pickGenre` scans this list against every source at once and returns the first
 * hit, so a broad label placed too early swallows the specific one. Barry's
 * first version searched MOSS's own Basic Attributes line first and returned on
 * the first match there, which let a generic "rock" beat "Emo" in MOSS's body
 * and "punk rock" in the Gemini caption — collapsing every track to `Rock`.
 * That is not cosmetic: plain "Rock" is precisely the failure mode Rob
 * identified by ear in the losing arm of the A/B ("more plain rock, not
 * particularly punk/emo"). Genre is the single biggest lever on MM3 adherence.
 *
 * The rock family below is Barry's validated list, order preserved verbatim.
 * The other families follow the same most-specific-first rule but have not been
 * ear-tested; add to them freely, but never insert an umbrella term above a
 * subgenre it contains.
 *
 * ── The second trap: specific-sounding words that are WEAK evidence ──────────
 *
 * Ordering by specificity is necessary but not sufficient, because this scans
 * MOSS's whole body — and that body is production prose, where several genre
 * words appear as ordinary adjectives. A real capture of Daft Punk's "One More
 * Time" contained "blends vocal into ambient space", and a bare /\bambient\b/
 * placed above /\bhouse\b/ (correct by specificity!) labelled the track
 * `Ambient`. The negative lookaheads below exist for exactly that: "ambient
 * space", "classical guitar" on a folk track, "blues scale" on a rock one. A new
 * single common word needs the same treatment — check it against a production
 * description before adding it bare.
 */
export const GENRE_HINTS: ReadonlyArray<readonly [RegExp, string]> = [
  // — rock family (validated ordering, do not reorder) —
  [/pop[-\s]?punk/i, 'Pop-Punk'],
  [/post[-\s]?hardcore/i, 'Post-Hardcore'],
  [/punk rock/i, 'Punk Rock'],
  [/alternative rock/i, 'Alternative Rock'],
  [/indie rock/i, 'Indie Rock'],
  [/\bemo\b/i, 'Emo'],
  [/\bhardcore\b/i, 'Hardcore'],
  [/\bmetal\b/i, 'Metal'],
  [/\brock\b/i, 'Rock'],
  // — electronic —
  [/drum\s*(?:and|'?n'?|&)\s*bass|\bdnb\b/i, 'Drum and Bass'],
  [/\bbass house\b/i, 'Bass House'],
  [/\belectro house\b/i, 'Electro House'],
  [/\bprogressive house\b/i, 'Progressive House'],
  [/\bdeep house\b/i, 'Deep House'],
  [/\btech house\b/i, 'Tech House'],
  [/\bdubstep\b/i, 'Dubstep'],
  [/\bhardstyle\b/i, 'Hardstyle'],
  [/\beurodance\b/i, 'Eurodance'],
  [/\bsynth[-\s]?pop\b/i, 'Synth-Pop'],
  [/\bsynthwave\b/i, 'Synthwave'],
  [/\bfrench house\b/i, 'French House'],
  [/\btrance\b/i, 'Trance'],
  [/\btechno\b/i, 'Techno'],
  [/\bhouse\b/i, 'House'],
  // "ambient <production noun>" is mix vocabulary, not a genre claim.
  [/\bambient\b(?!\s+(?:space|texture|pad|tail|wash|noise|bed|layer|reverb|atmosphere)s?\b)/i, 'Ambient'],
  // — hip-hop / r&b —
  [/\bboom bap\b/i, 'Boom Bap'],
  [/\btrap\b/i, 'Trap'],
  [/\bhip[-\s]?hop\b|\brap\b/i, 'Hip-Hop'],
  [/\br\s*&\s*b\b|\brnb\b/i, 'R&B'],
  [/\bsoul\b/i, 'Soul'],
  [/\bfunk\b/i, 'Funk'],
  // — acoustic / traditional —
  [/\bbluegrass\b/i, 'Bluegrass'],
  [/\bcountry\b/i, 'Country'],
  [/\bfolk\b/i, 'Folk'],
  [/\bjazz\b/i, 'Jazz'],
  [/\breggae\b/i, 'Reggae'],
  // "blues scale"/"blues note" is theory vocabulary; "classical guitar" and
  // "orchestral pad" are instrument names. All three appear constantly in
  // descriptions of tracks that are not remotely those genres.
  [/\bblues\b(?!\s+(?:scale|note|progression|lick|inflection)s?\b)/i, 'Blues'],
  [/\bclassical\b(?!\s+(?:guitar|piano|influence|training)s?\b)|\borchestral\b(?!\s+(?:pad|swell|hit|stab|sample|texture)s?\b)/i, 'Classical'],
  [/\bpop\b/i, 'Pop'],
];

/**
 * The most specific genre named by any source that actually OBSERVED the track.
 *
 * MOSS heard the audio; an existing audio-written caption (e.g. a Gemini
 * sidecar) also did. The dataset's `genre` field is the weak fallback because it
 * is usually a hand-typed umbrella like "Rock".
 */
export function pickGenre(
  mossText: string | undefined,
  observedCaption: string | undefined,
  fallback: string | undefined,
): string {
  const hay = `${mossText ?? ''}\n${observedCaption ?? ''}`;
  for (const [re, label] of GENRE_HINTS) {
    if (re.test(hay)) return label;
  }
  return (fallback ?? '').trim() || 'Alternative Rock';
}

/** `C# minor` / `F major` -> parts. Anything else -> nulls, and the caller omits the clause. */
export function parseKeyScale(keyscale: string | undefined | null): { key: string | null; scale: string | null } {
  const m = /^\s*([A-G][#b]?)\s+(major|minor)\s*$/i.exec(String(keyscale ?? '').trim());
  return m ? { key: m[1], scale: m[2].toLowerCase() } : { key: null, scale: null };
}

export interface LocalFacts {
  bpm?: number | null;
  keyscale?: string | null;
  signature?: string | null;
  /** Dataset genre field — weak fallback only. */
  genre?: string | null;
  /** A caption previously written FROM AUDIO, if one exists. */
  observedCaption?: string | null;
}

/** MM3's `Basic Attributes:` line, built from facts we know exactly. */
export function buildBasicAttributes(facts: LocalFacts, mossText?: string): string {
  const bpm = Math.round(Number(facts.bpm ?? 0));
  const { key, scale } = parseKeyScale(facts.keyscale);
  const sig = String(facts.signature ?? '').trim() || '4/4';
  const genre = pickGenre(mossText, facts.observedCaption ?? undefined, facts.genre ?? undefined);

  const parts: string[] = [];
  if (bpm > 0) parts.push(`bpm is ${bpm}.`);
  if (key) parts.push(`key is ${key}, and scale is ${scale}.`);
  parts.push(`${genre}, in ${sig}.`);
  return `Basic Attributes: ${parts.join(' ')}`;
}

/**
 * Replace ONLY the `Basic Attributes:` line of an MM3 Structured Caption with
 * one built from local analysis. Everything MOSS heard — Global Emotional
 * Progression, Application Scenarios & Imagery, Sonics & Production Profile,
 * Vocal Details, Arrangement — is kept verbatim.
 *
 * If MOSS omitted the line entirely, it is inserted directly under
 * `Global Metadata` rather than dropping known-good facts on the floor.
 */
export function applyFactSubstitution(mm3: string, facts: LocalFacts): string {
  const lines = mm3.split('\n');
  const bi = lines.findIndex(l => l.startsWith('Basic Attributes:'));
  const built = buildBasicAttributes(facts, mm3);
  if (bi >= 0) {
    lines[bi] = built;
  } else {
    const gi = lines.findIndex(l => l.trim() === 'Global Metadata');
    lines.splice(gi + 1, 0, built);
  }
  return lines.join('\n');
}

/**
 * Correct tempo/key claims that MOSS baked into CAPTION PROSE.
 *
 * `applyFactSubstitution` fixes the MM3 side by replacing a whole labelled line.
 * The AS1.5 caption has no such line — the numbers are welded into the sentence
 * ("A driving Electro House track in B minor at 120 BPM, this piece pulses…") —
 * so a line replacement cannot reach them, and a caption that asserts the wrong
 * tempo is worse than one that omits it: it is training data that disagrees with
 * the `bpm` field sitting next to it in the same sidecar.
 *
 * Measured on the Daft Punk capture: caption prose said "120 BPM", the MM3
 * section of the SAME encode said "approximately 128", truth is 123.
 *
 * Values are REWRITTEN rather than deleted. Deleting leaves "…track  , this
 * piece…" and needs fragile comma surgery; rewriting keeps MOSS's sentence
 * intact and makes it true. Only substitutes what local analysis actually knows,
 * and only where the value differs — so a caption that happens to be right, or a
 * dataset with no Essentia pass, is left completely alone.
 *
 * Chord spellings are untouched: the key pattern requires a space and a literal
 * "major"/"minor", so "Bm7", "Gmaj7" and "A6" never match.
 */
export function correctFactsInProse(caption: string, facts: LocalFacts): string {
  let out = caption;

  const bpm = Math.round(Number(facts.bpm ?? 0));
  if (bpm > 0) {
    out = out.replace(/\b(\d{2,3})(\s*)(BPM|bpm)\b/g, (m, n, sp, unit) =>
      Number(n) === bpm ? m : `${bpm}${sp}${unit}`);
  }

  const { key, scale } = parseKeyScale(facts.keyscale);
  if (key && scale) {
    const truth = `${key} ${scale}`;
    // Replace only the FIRST key statement. Later ones are usually modulations
    // ("shifts to D minor"), and rewriting those would produce "from F# minor to
    // F# minor" — nonsense that reads as a transcription error in the dataset.
    let done = false;
    out = out.replace(/\b([A-G][#b]?)\s+(major|minor)\b/g, (m) => {
      if (done) return m;
      done = true;
      return m.toLowerCase() === truth.toLowerCase() ? m : truth;
    });
  }

  return out;
}

// ── The Training Studio entry point ──────────────────────────────────────

// ── Batch prefetch ───────────────────────────────────────────────────────

/**
 * Results from a `--src-list` run, keyed by audio path, consumed once.
 *
 * The alternative was restructuring `labelingQueue` to caption in bulk, which
 * would have meant duplicating its per-sample merge policy, error marking,
 * progress and cancellation. Prefetching instead keeps every one of those
 * per-sample behaviours exactly as they are: the queue still walks samples one
 * at a time, but the expensive part is already done.
 */
const batchCache = new Map<string, Partial<Record<MossMode, string>>>();

/** Drop anything left over. Call when a job ends, however it ends. */
export function clearMossBatch(): void {
  batchCache.clear();
}

/**
 * Caption every track in one `ace-caption` process, holding the model across
 * them. Failure is non-fatal: an empty cache just means each sample falls back
 * to its own run, which is slower but produces identical output.
 */
export async function prepareMossBatch(
  audioPaths: string[],
  opts: {
    wantMm3?: boolean; wantLyrics?: boolean; maxSeconds?: number;
    signal?: AbortSignal; log?: (level: 'info' | 'warn', message: string) => void;
  } = {},
): Promise<void> {
  clearMossBatch();
  if (audioPaths.length < 2) return;   // one track cannot amortise anything

  const modes: MossMode[] = ['prose'];
  if (opts.wantMm3) modes.push('mm3');
  if (opts.wantLyrics) modes.push('lyrics');

  try {
    const results = await runMossBatch(audioPaths, {
      modes, prompts: { prose: CAPTION_INSTRUCTIONS },
      maxSeconds: opts.maxSeconds, signal: opts.signal, log: opts.log,
    });
    for (const [audioPath, modeText] of results) batchCache.set(audioPath, modeText);
    opts.log?.('info', `MOSS: captioned ${results.size}/${audioPaths.length} tracks in one load`);
  } catch (err: any) {
    if ((err as NodeJS.ErrnoException)?.name === 'AbortError') throw err;
    opts.log?.('warn', `MOSS batch failed (${String(err?.message || err).slice(0, 160)}) — falling back to per-track runs`);
  }
}

/**
 * One process, N tracks. Returns a map of audio path -> per-mode text.
 *
 * Output bases are numbered (`t0`, `t1`, …) rather than derived from filenames:
 * dataset filenames collide across albums, and two tracks resolving to the same
 * base would silently overwrite one caption with another's.
 */
async function runMossBatch(
  audioPaths: string[],
  opts: MossRunOptions,
): Promise<Map<string, Partial<Record<MossMode, string>>>> {
  return serialised(async () => {
    const resolved = resolveMossPaths();
    if ('missing' in resolved) throw new Error(`MOSS captioning unavailable: ${resolved.missing}`);
    const { exe, modelsDir } = resolved.paths;

    const work = fs.mkdtempSync(path.join(os.tmpdir(), 'hs_mossb_'));
    try {
      const bases = audioPaths.map((_, i) => path.join(work, `t${i}`));
      fs.writeFileSync(
        path.join(work, 'list.txt'),
        audioPaths.map((p, i) => `${p}\t${bases[i]}`).join('\n'),
        'utf8',
      );

      const args: string[] = [
        '--models', modelsDir,
        '--src-list', path.join(work, 'list.txt'),
        '--mode', opts.modes.join(','),
        '--temperature', String(opts.temperature ?? MOSS_TEMPERATURE),
        '--rep-penalty', String(opts.repPenalty ?? MOSS_REP_PENALTY),
        '--freq-penalty', String(opts.freqPenalty ?? MOSS_FREQ_PENALTY),
      ];
      if (opts.maxTokens) args.push('--max-tokens', String(opts.maxTokens));
      if (opts.maxSeconds) args.push('--max-seconds', String(opts.maxSeconds));
      const ffmpeg = getFFmpegPath();
      if (ffmpeg) args.push('--ffmpeg', ffmpeg);
      for (const [mode, text] of Object.entries(opts.prompts ?? {})) {
        if (!text?.trim()) continue;
        const pf = path.join(work, `prompt.${mode}.txt`);
        fs.writeFileSync(pf, text, 'utf8');
        args.push('--prompt-file', `${mode}=${pf}`);
      }

      await new Promise<void>((resolve, reject) => {
        // A non-zero exit means SOME file failed, not that all did — the CLI
        // captions every readable track first. So resolve either way and let the
        // per-file read below decide what actually landed.
        execFile(exe, args,
          { timeout: 6 * 60 * 60_000, maxBuffer: 64 * 1024 * 1024, signal: opts.signal },
          (error) => {
            if (error && (error as NodeJS.ErrnoException).name === 'AbortError') { reject(error); return; }
            resolve();
          });
      });

      const out = new Map<string, Partial<Record<MossMode, string>>>();
      audioPaths.forEach((audioPath, i) => {
        const modeText: Partial<Record<MossMode, string>> = {};
        for (const mode of opts.modes) {
          const text = [`${bases[i]}.${mode}.txt`, `${bases[i]}.txt`]
            .filter(p => fs.existsSync(p))
            .map(p => fs.readFileSync(p, 'utf8').trim())
            .find(t => t.length > 0);
          if (text) modeText[mode] = text;
        }
        if (Object.keys(modeText).length > 0) out.set(audioPath, modeText);
      });
      return out;
    } finally {
      try { fs.rmSync(work, { recursive: true, force: true }); } catch { /* tmp */ }
    }
  });
}

export interface MossCaptionResult {
  /** Side-Step's 5-line block, unparsed — the caller runs parseStructuredResponse. */
  prose?: string;
  /** MM3 Structured Caption, already fact-substituted. */
  mm3?: string;
  lyrics?: string;
}

/**
 * Caption one track locally, in every format the app consumes.
 *
 * `wantMm3` and `wantLyrics` are separate because each adds a decode (cheap)
 * but lyrics in particular adds a LOT of tokens on a long song. The AS1.5 block
 * is always produced — it is what the sidecar needs.
 */
export async function captionWithMoss(
  audioPath: string,
  facts: LocalFacts,
  opts: {
    wantMm3?: boolean;
    wantLyrics?: boolean;
    maxSeconds?: number;
    signal?: AbortSignal;
    log?: (level: 'info' | 'warn', message: string) => void;
  } = {},
): Promise<MossCaptionResult> {
  const modes: MossMode[] = ['prose'];
  if (opts.wantMm3) modes.push('mm3');
  if (opts.wantLyrics) modes.push('lyrics');

  // Prefetched by prepareMossBatch, if the caller ran one. Deleted on read so a
  // retry of the same sample re-runs the model rather than replaying a stale
  // result — and so the map cannot grow across a long job.
  const prefetched = batchCache.get(audioPath);
  if (prefetched) batchCache.delete(audioPath);

  const raw = prefetched ?? await runMossCaption(audioPath, {
    modes,
    // The AS1.5 prompt is passed by file rather than restated in C++ so the two
    // can't drift; it is the same string the cloud providers get.
    prompts: { prose: CAPTION_INSTRUCTIONS },
    maxSeconds: opts.maxSeconds,
    signal: opts.signal,
    log: opts.log,
  });

  const out: MossCaptionResult = { prose: raw.prose, lyrics: raw.lyrics };
  if (raw.mm3) out.mm3 = applyFactSubstitution(raw.mm3, facts);
  return out;
}

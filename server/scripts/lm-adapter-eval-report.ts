// lm-adapter-eval-report.ts — analysis + HTML report for the planner-LM
// adapter eval (lm-adapter-eval.ts). No CLI of its own; the eval script calls
// analyzeRuns() + renderReport().
//
// What it measures, and why each metric is there:
//   The training claim is "the adapter moves the LM's audio-code distribution
//   toward the artist's". Every metric here compares three token populations —
//   the dataset's ground-truth codes (lm_codes.jsonl), base-LM generations and
//   adapter generations on the SAME captions/lyrics/seeds — so the adapter is
//   the only variable, mirroring the audition A/B discipline.
//
//   - Jensen-Shannon divergence (base-2, normalized 0..1) between token
//     distributions. JS needs no smoothing (its mixture term is nonzero
//     wherever either side is), unlike KL.
//   - Codes are FSQ indices over levels [8,8,8,5,5,5] (fsq-quant.h): the flat
//     64000-code histogram is sparse at dataset scale, so we also decompose
//     every code into its 6 FSQ digits and compare the dense per-dimension
//     marginals and per-dimension transition matrices.
//   - The "floor" is JS between two halves of the ground truth itself (odd vs
//     even songs): the artist's own within-dataset variability. An adapter at
//     or near the floor is statistically inside the artist's own spread. The
//     halves have ~half the tokens of the full set, which biases the floor UP
//     — i.e. it is a conservative bar, never a flattering one.
//   - Memorization: longest common contiguous run (LCS substring) between each
//     generation and any full-length training sequence. A "style" adapter
//     should score low; a replaying adapter scores in the tens of seconds and
//     would make every other metric look good for the wrong reason.
//
// Chart styling follows the dataviz reference palette (3 categorical slots,
// light+dark selected, floors as dashed reference lines, tables as the
// accessibility fallback).

export const FSQ_LEVELS = [8, 8, 8, 5, 5, 5];
export const FSQ_STRIDES = [1, 8, 64, 512, 2560, 12800];
export const FSQ_VOCAB = 64000;
const NDIMS = 6;

// ── Run-file shapes (written by lm-adapter-eval.ts) ───────────────────────

export interface EvalRow {
  id: string;
  file: string;
  caption: string;
  lyrics: string;
  bpm: number;
  duration: number;      // full song seconds (lm_codes.jsonl)
  durUsed: number;       // seconds actually generated for this row
  gtCodes: number[];     // FULL ground-truth code sequence
}

export type EvalSide = 'base' | 'adapter';

export interface EvalGen {
  rowId: string;
  side: EvalSide;
  seed: number;
  codes: number[];
  lmMs: number;
  bpm: number;
  keyscale: string;
}

export interface EvalRuns {
  version: number;
  createdAt: string;
  dataset: string;
  variant: string;
  adapterPath: string;
  adapterScale: number;
  lmModel: string;
  params: {
    temperature: number; topP: number; cfgScale: number; repPenalty: number;
    maxDuration: number; seeds: number; seedBase: number;
  };
  rows: EvalRow[];
  gens: EvalGen[];
}

// ── Analysis result shapes ────────────────────────────────────────────────

export interface PairJs { base: number; adapter: number; floor: number; baseVsAdapter: number }

interface Triple { gt: number; base: number; adapter: number }
interface TripleSd { gt: number; gtSd: number; base: number; baseSd: number; adapter: number; adapterSd: number }

export interface MemHit {
  side: EvalSide; rowId: string; rowFile: string; seed: number;
  matchFile: string; tokens: number; seconds: number;
}

export interface PerSongRow {
  rowId: string; file: string; durUsed: number;
  jsBase: number; jsAdapter: number;       // mean per-dim marginal JS, song GT vs gens
  divergeSec: number;                       // median same-seed divergence point (s)
}

export interface EvalResults {
  tokenCounts: Triple;
  distinctCodes: Triple;
  unigram: PairJs;
  marginalMean: PairJs;
  transitionMean: PairJs;
  perDimMarginalJs: { base: number[]; adapter: number[]; floor: number[] };
  perDimTransitionJs: { base: number[]; adapter: number[]; floor: number[] };
  /** [dim][level] probabilities per side. */
  marginals: { gt: number[][]; base: number[][]; adapter: number[][] };
  changeRate: TripleSd;
  repetition: TripleSd;
  entropyFull: Triple;
  oovMass: { base: number; adapter: number };
  memorization: { hits: MemHit[]; maxSec: { base: number; adapter: number }; meanSec: { base: number; adapter: number } };
  divergence: { medianSec: number; minSec: number; maxSec: number; count: number };
  perSong: PerSongRow[];
  verdict: {
    status: 'toward' | 'away' | 'inconclusive';
    unigramDelta: number;
    dimsWon: number;
    transWon: number;
    withinFloor: boolean;
    memFlag: 'ok' | 'warn' | 'critical';
    summary: string;
  };
}

// ── Small math helpers ────────────────────────────────────────────────────

function normalize(counts: Float64Array): Float64Array {
  let s = 0;
  for (let i = 0; i < counts.length; i++) s += counts[i];
  const out = new Float64Array(counts.length);
  if (s > 0) for (let i = 0; i < counts.length; i++) out[i] = counts[i] / s;
  return out;
}

/** Jensen-Shannon divergence, base-2 (0 = identical, 1 = disjoint). */
function jsDiv(pCounts: Float64Array, qCounts: Float64Array): number {
  const p = normalize(pCounts);
  const q = normalize(qCounts);
  let js = 0;
  for (let i = 0; i < p.length; i++) {
    const pi = p[i], qi = q[i];
    if (pi === 0 && qi === 0) continue;
    const m = (pi + qi) / 2;
    if (pi > 0) js += 0.5 * pi * Math.log2(pi / m);
    if (qi > 0) js += 0.5 * qi * Math.log2(qi / m);
  }
  return js;
}

function entropyBits(counts: Float64Array): number {
  const p = normalize(counts);
  let h = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log2(p[i]);
  return h;
}

function meanSd(values: number[]): { mean: number; sd: number } {
  if (!values.length) return { mean: NaN, sd: NaN };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = values.length > 1
    ? Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1))
    : 0;
  return { mean, sd };
}

function digitOf(code: number, dim: number): number {
  return Math.floor(code / FSQ_STRIDES[dim]) % FSQ_LEVELS[dim];
}

// ── Population accumulators ───────────────────────────────────────────────

class Population {
  unigram = new Float64Array(FSQ_VOCAB);
  marginals: Float64Array[] = FSQ_LEVELS.map(L => new Float64Array(L));
  transitions: Float64Array[] = FSQ_LEVELS.map(L => new Float64Array(L * L));
  tokens = 0;
  changeRates: number[] = [];   // one per sequence
  repRates: number[] = [];      // one per sequence

  addSequence(codes: number[]): void {
    let changes = 0;
    const grams = new Set<string>();
    let reps = 0, repWindows = 0;
    for (let t = 0; t < codes.length; t++) {
      const c = codes[t];
      this.unigram[c]++;
      this.tokens++;
      for (let d = 0; d < NDIMS; d++) {
        const dig = digitOf(c, d);
        this.marginals[d][dig]++;
        if (t > 0) this.transitions[d][digitOf(codes[t - 1], d) * FSQ_LEVELS[d] + dig]++;
      }
      if (t > 0 && c !== codes[t - 1]) changes++;
      if (t >= 7) {
        const key = codes.slice(t - 7, t + 1).join(',');
        repWindows++;
        if (grams.has(key)) reps++;
        grams.add(key);
      }
    }
    if (codes.length > 1) this.changeRates.push(changes / (codes.length - 1));
    if (repWindows > 0) this.repRates.push(reps / repWindows);
  }

  distinct(): number {
    let n = 0;
    for (let i = 0; i < this.unigram.length; i++) if (this.unigram[i] > 0) n++;
    return n;
  }
}

/** Longest common contiguous substring length between two int sequences. */
function lcsLen(a: number[], b: number[]): number {
  if (!a.length || !b.length) return 0;
  let prev = new Int32Array(b.length + 1);
  let curr = new Int32Array(b.length + 1);
  let best = 0;
  for (let i = 1; i <= a.length; i++) {
    const ai = a[i - 1];
    for (let j = 1; j <= b.length; j++) {
      const v = ai === b[j - 1] ? prev[j - 1] + 1 : 0;
      curr[j] = v;
      if (v > best) best = v;
    }
    const t = prev; prev = curr; curr = t;
    curr.fill(0);
  }
  return best;
}

// ── The analysis ──────────────────────────────────────────────────────────

export function analyzeRuns(runs: EvalRuns): EvalResults {
  const rowById = new Map(runs.rows.map(r => [r.id, r]));
  const gens = runs.gens.filter(g => g.codes.length > 0 && rowById.has(g.rowId));
  const baseGens = gens.filter(g => g.side === 'base');
  const adapterGens = gens.filter(g => g.side === 'adapter');
  if (!baseGens.length || !adapterGens.length) {
    throw new Error(`analysis needs both sides: ${baseGens.length} base / ${adapterGens.length} adapter generations`);
  }

  // Ground truth, truncated per row to the generated duration so intro-heavy
  // truncation biases GT and generations identically.
  const gtPop = new Population();
  const gtHalfA = new Population();
  const gtHalfB = new Population();
  runs.rows.forEach((row, i) => {
    const cut = row.gtCodes.slice(0, Math.max(1, Math.round(row.durUsed * 5)));
    gtPop.addSequence(cut);
    (i % 2 === 0 ? gtHalfA : gtHalfB).addSequence(cut);
  });

  const basePop = new Population();
  for (const g of baseGens) basePop.addSequence(g.codes);
  const adapterPop = new Population();
  for (const g of adapterGens) adapterPop.addSequence(g.codes);

  const haveFloor = runs.rows.length >= 2;
  const floorOf = (a: Float64Array, b: Float64Array) => haveFloor ? jsDiv(a, b) : NaN;

  const unigram: PairJs = {
    base: jsDiv(gtPop.unigram, basePop.unigram),
    adapter: jsDiv(gtPop.unigram, adapterPop.unigram),
    floor: floorOf(gtHalfA.unigram, gtHalfB.unigram),
    baseVsAdapter: jsDiv(basePop.unigram, adapterPop.unigram),
  };

  const perDimMarginalJs = {
    base: FSQ_LEVELS.map((_, d) => jsDiv(gtPop.marginals[d], basePop.marginals[d])),
    adapter: FSQ_LEVELS.map((_, d) => jsDiv(gtPop.marginals[d], adapterPop.marginals[d])),
    floor: FSQ_LEVELS.map((_, d) => floorOf(gtHalfA.marginals[d], gtHalfB.marginals[d])),
  };
  const perDimTransitionJs = {
    base: FSQ_LEVELS.map((_, d) => jsDiv(gtPop.transitions[d], basePop.transitions[d])),
    adapter: FSQ_LEVELS.map((_, d) => jsDiv(gtPop.transitions[d], adapterPop.transitions[d])),
    floor: FSQ_LEVELS.map((_, d) => floorOf(gtHalfA.transitions[d], gtHalfB.transitions[d])),
  };

  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const marginalMean: PairJs = {
    base: mean(perDimMarginalJs.base),
    adapter: mean(perDimMarginalJs.adapter),
    floor: haveFloor ? mean(perDimMarginalJs.floor) : NaN,
    baseVsAdapter: mean(FSQ_LEVELS.map((_, d) => jsDiv(basePop.marginals[d], adapterPop.marginals[d]))),
  };
  const transitionMean: PairJs = {
    base: mean(perDimTransitionJs.base),
    adapter: mean(perDimTransitionJs.adapter),
    floor: haveFloor ? mean(perDimTransitionJs.floor) : NaN,
    baseVsAdapter: mean(FSQ_LEVELS.map((_, d) => jsDiv(basePop.transitions[d], adapterPop.transitions[d]))),
  };

  // OOV mass: fraction of a side's tokens on codes the (truncated) GT never used.
  const oovOf = (pop: Population): number => {
    let oov = 0;
    for (let c = 0; c < FSQ_VOCAB; c++) if (gtPop.unigram[c] === 0) oov += pop.unigram[c];
    return pop.tokens > 0 ? oov / pop.tokens : NaN;
  };

  // Memorization vs FULL GT sequences.
  const hits: MemHit[] = [];
  for (const g of gens) {
    let bestTokens = 0;
    let bestFile = '';
    for (const row of runs.rows) {
      const l = lcsLen(g.codes, row.gtCodes);
      if (l > bestTokens) { bestTokens = l; bestFile = row.file; }
    }
    hits.push({
      side: g.side, rowId: g.rowId, rowFile: rowById.get(g.rowId)!.file, seed: g.seed,
      matchFile: bestFile, tokens: bestTokens, seconds: bestTokens / 5,
    });
  }
  hits.sort((a, b) => b.tokens - a.tokens);
  const memSide = (side: EvalSide) => hits.filter(h => h.side === side).map(h => h.seconds);
  const memorization = {
    hits,
    maxSec: { base: Math.max(0, ...memSide('base')), adapter: Math.max(0, ...memSide('adapter')) },
    meanSec: { base: meanSd(memSide('base')).mean, adapter: meanSd(memSide('adapter')).mean },
  };

  // Same-seed divergence point: first token where base and adapter differ.
  const divergeSecs: number[] = [];
  const divergeByRow = new Map<string, number[]>();
  for (const b of baseGens) {
    const a = adapterGens.find(g => g.rowId === b.rowId && g.seed === b.seed);
    if (!a) continue;
    const n = Math.min(a.codes.length, b.codes.length);
    let idx = n;
    for (let i = 0; i < n; i++) if (a.codes[i] !== b.codes[i]) { idx = i; break; }
    const sec = idx / 5;
    divergeSecs.push(sec);
    const list = divergeByRow.get(b.rowId) ?? [];
    list.push(sec);
    divergeByRow.set(b.rowId, list);
  }
  divergeSecs.sort((x, y) => x - y);
  const divergence = {
    medianSec: divergeSecs.length ? divergeSecs[Math.floor(divergeSecs.length / 2)] : NaN,
    minSec: divergeSecs.length ? divergeSecs[0] : NaN,
    maxSec: divergeSecs.length ? divergeSecs[divergeSecs.length - 1] : NaN,
    count: divergeSecs.length,
  };

  // Per-song: mean per-dim marginal JS of the song's own GT vs its generations.
  const perSong: PerSongRow[] = runs.rows.map(row => {
    const cut = row.gtCodes.slice(0, Math.max(1, Math.round(row.durUsed * 5)));
    const songGt = new Population();
    songGt.addSequence(cut);
    const sideJs = (side: EvalSide): number => {
      const pop = new Population();
      const mine = gens.filter(g => g.rowId === row.id && g.side === side);
      if (!mine.length) return NaN;
      for (const g of mine) pop.addSequence(g.codes);
      return mean(FSQ_LEVELS.map((_, d) => jsDiv(songGt.marginals[d], pop.marginals[d])));
    };
    const dv = (divergeByRow.get(row.id) ?? []).sort((x, y) => x - y);
    return {
      rowId: row.id, file: row.file, durUsed: row.durUsed,
      jsBase: sideJs('base'), jsAdapter: sideJs('adapter'),
      divergeSec: dv.length ? dv[Math.floor(dv.length / 2)] : NaN,
    };
  });

  // Verdict.
  const unigramDelta = unigram.base - unigram.adapter;
  const dimsWon = perDimMarginalJs.adapter.filter((v, d) => v < perDimMarginalJs.base[d]).length;
  const transWon = perDimTransitionJs.adapter.filter((v, d) => v < perDimTransitionJs.base[d]).length;
  const withinFloor = haveFloor && unigram.adapter <= unigram.floor * 1.25;
  const memFlag: 'ok' | 'warn' | 'critical' =
    memorization.maxSec.adapter > 15 ? 'critical' : memorization.maxSec.adapter > 5 ? 'warn' : 'ok';
  let status: 'toward' | 'away' | 'inconclusive' = 'inconclusive';
  if (unigramDelta > 0.005 && dimsWon >= 4) status = 'toward';
  else if (unigramDelta < -0.005 && dimsWon <= 2) status = 'away';
  const pct = unigram.base > 0 ? Math.abs(unigramDelta) / unigram.base * 100 : 0;
  const summary =
    status === 'toward'
      ? `The adapter moves generations TOWARD the artist: unigram JS to ground truth drops ${unigramDelta.toFixed(3)} `
        + `(${pct.toFixed(0)}% closer than base), winning ${dimsWon}/6 FSQ dimensions and ${transWon}/6 transition matrices.`
        + (withinFloor ? ' Adapter output sits within 1.25x of the artist’s own within-dataset variability.' : '')
        + (memFlag !== 'ok' ? ` CAUTION: longest replayed training run is ${memorization.maxSec.adapter.toFixed(1)}s — check the memorization table.` : '')
      : status === 'away'
        ? `The adapter moves generations AWAY from the artist (unigram JS rises ${(-unigramDelta).toFixed(3)}; only ${dimsWon}/6 dimensions improve). Something is off — check adapter scale, trigger tag, and base-model match.`
        : `No decisive shift: unigram JS delta ${unigramDelta.toFixed(3)}, ${dimsWon}/6 dimensions improved. `
          + `More seeds/samples may resolve it, or the adapter's effect is below this test's sensitivity.`;

  return {
    tokenCounts: { gt: gtPop.tokens, base: basePop.tokens, adapter: adapterPop.tokens },
    distinctCodes: { gt: gtPop.distinct(), base: basePop.distinct(), adapter: adapterPop.distinct() },
    unigram, marginalMean, transitionMean, perDimMarginalJs, perDimTransitionJs,
    marginals: {
      gt: gtPop.marginals.map(m => Array.from(normalize(m))),
      base: basePop.marginals.map(m => Array.from(normalize(m))),
      adapter: adapterPop.marginals.map(m => Array.from(normalize(m))),
    },
    changeRate: {
      gt: meanSd(gtPop.changeRates).mean, gtSd: meanSd(gtPop.changeRates).sd,
      base: meanSd(basePop.changeRates).mean, baseSd: meanSd(basePop.changeRates).sd,
      adapter: meanSd(adapterPop.changeRates).mean, adapterSd: meanSd(adapterPop.changeRates).sd,
    },
    repetition: {
      gt: meanSd(gtPop.repRates).mean, gtSd: meanSd(gtPop.repRates).sd,
      base: meanSd(basePop.repRates).mean, baseSd: meanSd(basePop.repRates).sd,
      adapter: meanSd(adapterPop.repRates).mean, adapterSd: meanSd(adapterPop.repRates).sd,
    },
    entropyFull: {
      gt: entropyBits(gtPop.unigram), base: entropyBits(basePop.unigram), adapter: entropyBits(adapterPop.unigram),
    },
    oovMass: { base: oovOf(basePop), adapter: oovOf(adapterPop) },
    memorization, divergence, perSong,
    verdict: { status, unigramDelta, dimsWon, transWon, withinFloor, memFlag, summary },
  };
}

// ── HTML report ───────────────────────────────────────────────────────────

const SERIES = {
  gt:      { label: 'Ground truth', varName: '--series-gt' },
  base:    { label: 'Base LM',      varName: '--series-base' },
  adapter: { label: 'Adapter',      varName: '--series-adapter' },
} as const;

function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function fmt(n: number, digits = 3): string {
  return Number.isFinite(n) ? n.toFixed(digits) : '—';
}

function niceMax(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 1;
  const exp = Math.floor(Math.log10(v));
  const f = v / 10 ** exp;
  const nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nf * 10 ** exp;
}

function roundedTopRect(x: number, y: number, w: number, h: number, r: number): string {
  const rr = Math.min(r, w / 2, h);
  return `M${x},${y + h} L${x},${y + rr} Q${x},${y} ${x + rr},${y} L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr} L${x + w},${y + h} Z`;
}

interface Bar { series: keyof typeof SERIES; value: number; tip?: string }
interface Group { label: string; bars: Bar[]; floor?: number }

interface BarChartOpts {
  groups: Group[];
  height?: number;
  barW?: number;
  valueLabels?: boolean;
  valueFmt?: (n: number) => string;
  yMax?: number;
}

/** Grouped bar chart as inline SVG. Baseline-anchored rounded data-ends, 2px
 *  bar gaps, hairline grid, muted ink labels, optional dashed floor marker. */
function barChart(opts: BarChartOpts): string {
  const H = opts.height ?? 220;
  const barW = opts.barW ?? 26;
  const gap = 2;
  const groupPad = 18;
  const mLeft = 46, mRight = 10, mTop = 14, mBottom = 26;
  const vFmt = opts.valueFmt ?? ((n: number) => fmt(n, 3));

  const allVals = opts.groups.flatMap(g => g.bars.map(b => b.value).concat(Number.isFinite(g.floor ?? NaN) ? [g.floor as number] : []));
  const yMax = opts.yMax ?? niceMax(Math.max(...allVals.filter(Number.isFinite), 1e-9) * 1.15);
  const plotH = H - mTop - mBottom;
  const y = (v: number) => mTop + plotH * (1 - Math.max(0, Math.min(1, v / yMax)));

  const groupW = (g: Group) => g.bars.length * barW + (g.bars.length - 1) * gap;
  const W = mLeft + mRight + opts.groups.reduce((a, g) => a + groupW(g) + groupPad, 0) - groupPad + 8;

  const parts: string[] = [];
  parts.push(`<svg viewBox="0 0 ${W} ${H}" style="max-width:${W}px" role="img">`);

  // grid + y labels (4 ticks)
  for (let i = 0; i <= 4; i++) {
    const v = yMax * i / 4;
    const yy = y(v);
    parts.push(`<line x1="${mLeft}" x2="${W - mRight}" y1="${yy}" y2="${yy}" class="grid"/>`);
    parts.push(`<text x="${mLeft - 6}" y="${yy + 3}" class="tick" text-anchor="end">${vFmt(v)}</text>`);
  }

  let x = mLeft + 4;
  for (const g of opts.groups) {
    const gw = groupW(g);
    let bx = x;
    for (const b of g.bars) {
      const by = y(b.value);
      const bh = mTop + plotH - by;
      const tip = b.tip ?? `${SERIES[b.series].label} — ${g.label}: ${vFmt(b.value)}`;
      if (bh > 0.5) {
        parts.push(`<path d="${roundedTopRect(bx, by, barW, bh, 4)}" fill="var(${SERIES[b.series].varName})" data-tip="${esc(tip)}"/>`);
      } else {
        parts.push(`<rect x="${bx}" y="${mTop + plotH - 1}" width="${barW}" height="1" fill="var(${SERIES[b.series].varName})" data-tip="${esc(tip)}"/>`);
      }
      if (opts.valueLabels) {
        parts.push(`<text x="${bx + barW / 2}" y="${Math.max(by - 4, 10)}" class="val" text-anchor="middle">${vFmt(b.value)}</text>`);
      }
      bx += barW + gap;
    }
    if (Number.isFinite(g.floor ?? NaN)) {
      const fy = y(g.floor as number);
      parts.push(`<line x1="${x - 4}" x2="${x + gw + 4}" y1="${fy}" y2="${fy}" class="floor" data-tip="${esc(`GT self-similarity floor — ${g.label}: ${vFmt(g.floor as number)}`)}"/>`);
    }
    parts.push(`<text x="${x + gw / 2}" y="${H - 8}" class="tick" text-anchor="middle">${esc(g.label)}</text>`);
    x += gw + groupPad;
  }
  parts.push(`<line x1="${mLeft}" x2="${W - mRight}" y1="${mTop + plotH}" y2="${mTop + plotH}" class="axis"/>`);
  parts.push('</svg>');
  return parts.join('');
}

function legend(sides: Array<keyof typeof SERIES>, withFloor = false): string {
  const chips = sides.map(s =>
    `<span class="chip"><span class="swatch" style="background:var(${SERIES[s].varName})"></span>${SERIES[s].label}</span>`);
  if (withFloor) chips.push('<span class="chip"><span class="swatch floor-swatch"></span>GT self-similarity floor</span>');
  return `<div class="legend">${chips.join('')}</div>`;
}

export function renderReport(runs: EvalRuns, r: EvalResults): string {
  const v = r.verdict;
  const verdictClass = v.status === 'toward' ? 'good' : v.status === 'away' ? 'bad' : 'mid';
  const verdictTitle = v.status === 'toward' ? 'Adapter shifts output toward the artist'
    : v.status === 'away' ? 'Adapter shifts output AWAY from the artist' : 'Inconclusive';

  const distanceChart = barChart({
    groups: [
      { label: 'Unigram (64k codes)', bars: [
        { series: 'base', value: r.unigram.base }, { series: 'adapter', value: r.unigram.adapter }], floor: r.unigram.floor },
      { label: 'FSQ marginals (mean)', bars: [
        { series: 'base', value: r.marginalMean.base }, { series: 'adapter', value: r.marginalMean.adapter }], floor: r.marginalMean.floor },
      { label: 'Transitions (mean)', bars: [
        { series: 'base', value: r.transitionMean.base }, { series: 'adapter', value: r.transitionMean.adapter }], floor: r.transitionMean.floor },
    ],
    valueLabels: true, height: 240, barW: 40,
  });

  const dimChart = (js: { base: number[]; adapter: number[]; floor: number[] }) => barChart({
    groups: FSQ_LEVELS.map((L, d) => ({
      label: `dim ${d} (L${L})`,
      bars: [{ series: 'base' as const, value: js.base[d] }, { series: 'adapter' as const, value: js.adapter[d] }],
      floor: js.floor[d],
    })),
    valueLabels: false, height: 200, barW: 22,
  });

  const marginalPanels = FSQ_LEVELS.map((L, d) => {
    const chart = barChart({
      groups: Array.from({ length: L }, (_, lvl) => ({
        label: String(lvl),
        bars: (['gt', 'base', 'adapter'] as const).map(s => ({
          series: s, value: r.marginals[s][d][lvl],
          tip: `${SERIES[s].label} — dim ${d}, level ${lvl}: ${(r.marginals[s][d][lvl] * 100).toFixed(1)}%`,
        })),
      })),
      height: 150, barW: 9, valueFmt: n => (n * 100).toFixed(0) + '%',
    });
    return `<div class="panel"><h4>FSQ dim ${d} <span class="muted">(${L} levels)</span></h4>${chart}</div>`;
  }).join('');

  const dynChart = barChart({
    groups: [
      { label: 'Code change rate', bars: (['gt', 'base', 'adapter'] as const).map(s => ({
        series: s, value: r.changeRate[s],
        tip: `${SERIES[s].label}: ${fmt(r.changeRate[s])} ± ${fmt(s === 'gt' ? r.changeRate.gtSd : s === 'base' ? r.changeRate.baseSd : r.changeRate.adapterSd)}` })) },
      { label: '8-gram repetition', bars: (['gt', 'base', 'adapter'] as const).map(s => ({
        series: s, value: r.repetition[s],
        tip: `${SERIES[s].label}: ${fmt(r.repetition[s])} ± ${fmt(s === 'gt' ? r.repetition.gtSd : s === 'base' ? r.repetition.baseSd : r.repetition.adapterSd)}` })) },
    ],
    valueLabels: true, valueFmt: n => fmt(n, 2), height: 220, barW: 34,
  });

  const memRows = r.memorization.hits.slice(0, 12).map(h => `
    <tr><td>${esc(SERIES[h.side].label)}</td><td class="file">${esc(h.rowFile)}</td><td>${h.seed}</td>
    <td class="file">${esc(h.matchFile)}</td><td class="num">${h.tokens}</td><td class="num">${h.seconds.toFixed(1)}s</td></tr>`).join('');

  const songRows = r.perSong.map(s => {
    const delta = s.jsBase - s.jsAdapter;
    const cls = !Number.isFinite(delta) ? '' : delta > 0 ? 'delta-good' : delta < 0 ? 'delta-bad' : '';
    return `<tr><td class="file">${esc(s.file)}</td><td class="num">${s.durUsed}s</td>
      <td class="num">${fmt(s.jsBase)}</td><td class="num">${fmt(s.jsAdapter)}</td>
      <td class="num ${cls}">${Number.isFinite(delta) ? (delta > 0 ? '−' : '+') + Math.abs(delta).toFixed(3) : '—'}</td>
      <td class="num">${fmt(s.divergeSec, 1)}s</td></tr>`;
  }).join('');

  const p = runs.params;
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LM adapter eval — ${esc(runs.dataset)}</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
    --series-gt: #2a78d6; --series-base: #eb6834; --series-adapter: #1baf7a;
    --good-text: #006300; --bad-text: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
      --series-gt: #3987e5; --series-base: #d95926; --series-adapter: #199e70;
      --good-text: #0ca30c; --bad-text: #e66767;
    }
  }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--page); color: var(--ink); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 24px; }
  main { max-width: 1060px; margin: 0 auto; display: grid; gap: 16px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; overflow-x: auto; }
  h1 { font-size: 20px; } h2 { font-size: 16px; margin-bottom: 8px; } h4 { font-size: 12px; font-weight: 600; margin-bottom: 4px; }
  .muted { color: var(--muted); font-weight: 400; }
  .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 4px 24px; margin-top: 8px; font-size: 13px; color: var(--ink-2); }
  .meta b { color: var(--ink); font-weight: 600; }
  .verdict { border-left: 4px solid var(--axis); }
  .verdict.good { border-left-color: #0ca30c; } .verdict.bad { border-left-color: #d03b3b; } .verdict.mid { border-left-color: #fab219; }
  .verdict p { margin-top: 6px; color: var(--ink-2); max-width: 72ch; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
  .tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .tile .k { font-size: 12px; color: var(--muted); } .tile .v { font-size: 22px; font-weight: 650; margin-top: 2px; }
  .tile .s { font-size: 12px; color: var(--ink-2); margin-top: 2px; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 6px 0 10px; font-size: 12px; color: var(--ink-2); }
  .chip { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
  .floor-swatch { background: none; border-top: 2px dashed var(--muted); height: 0; width: 14px; border-radius: 0; }
  svg { display: block; width: 100%; height: auto; }
  svg .grid { stroke: var(--grid); stroke-width: 1; } svg .axis { stroke: var(--axis); stroke-width: 1; }
  svg .tick { fill: var(--muted); font-size: 10px; font-family: system-ui, sans-serif; }
  svg .val { fill: var(--ink-2); font-size: 10px; font-family: system-ui, sans-serif; font-variant-numeric: tabular-nums; }
  svg .floor { stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 5 3; }
  .panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--grid); }
  th { color: var(--muted); font-weight: 600; font-size: 12px; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.file { max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .delta-good { color: var(--good-text); font-weight: 600; } .delta-bad { color: var(--bad-text); font-weight: 600; }
  .note { font-size: 12.5px; color: var(--ink-2); max-width: 78ch; }
  .note li { margin: 3px 0 3px 16px; }
  #tip { position: absolute; pointer-events: none; opacity: 0; background: var(--ink); color: var(--page);
         padding: 4px 8px; border-radius: 6px; font-size: 12px; max-width: 320px; z-index: 10; transition: opacity .08s; }
</style></head><body><main>

<div class="card">
  <h1>Planner-LM adapter eval <span class="muted">— does it move generations toward the artist?</span></h1>
  <div class="meta">
    <span><b>Dataset</b> ${esc(runs.dataset)} <span class="muted">(${esc(runs.variant)})</span></span>
    <span><b>Adapter</b> ${esc(runs.adapterPath)} <span class="muted">×${runs.adapterScale}</span></span>
    <span><b>Base LM</b> ${esc(runs.lmModel)}</span>
    <span><b>Songs</b> ${runs.rows.length} · <b>Seeds</b> ${p.seeds} · <b>Gens/side</b> ${runs.gens.filter(g => g.side === 'base').length}/${runs.gens.filter(g => g.side === 'adapter').length}</span>
    <span><b>Sampling</b> T ${p.temperature} · top-p ${p.topP} · cfg ${p.cfgScale} · rep ${p.repPenalty}</span>
    <span><b>Generated</b> ${esc(runs.createdAt)}</span>
  </div>
</div>

<div class="card verdict ${verdictClass}">
  <h2>${verdictTitle}</h2>
  <p>${esc(v.summary)}</p>
</div>

<div class="tiles">
  <div class="tile"><div class="k">Unigram JS → ground truth</div>
    <div class="v">${fmt(r.unigram.adapter)}</div>
    <div class="s">adapter vs ${fmt(r.unigram.base)} base <span class="${v.unigramDelta > 0 ? 'delta-good' : v.unigramDelta < 0 ? 'delta-bad' : ''}">(${v.unigramDelta > 0 ? '−' : '+'}${Math.abs(v.unigramDelta).toFixed(3)})</span></div></div>
  <div class="tile"><div class="k">FSQ dimensions won</div>
    <div class="v">${v.dimsWon}/6</div><div class="s">marginals closer to artist · transitions ${v.transWon}/6</div></div>
  <div class="tile"><div class="k">Novel-code mass (OOV)</div>
    <div class="v">${Number.isFinite(r.oovMass.adapter) ? (r.oovMass.adapter * 100).toFixed(1) + '%' : '—'}</div>
    <div class="s">adapter vs ${(r.oovMass.base * 100).toFixed(1)}% base — tokens on codes the artist never used</div></div>
  <div class="tile"><div class="k">Longest replayed training run</div>
    <div class="v">${r.memorization.maxSec.adapter.toFixed(1)}s</div>
    <div class="s">${v.memFlag === 'ok' ? 'no memorization concern (base: ' + r.memorization.maxSec.base.toFixed(1) + 's)' : v.memFlag === 'warn' ? 'borderline — see memorization table' : 'MEMORIZING — adapter replays training data'}</div></div>
</div>

<div class="card">
  <h2>Distance to the artist's code distribution <span class="muted">(Jensen-Shannon, lower = closer)</span></h2>
  ${legend(['base', 'adapter'], true)}
  ${distanceChart}
  <p class="note" style="margin-top:8px">The dashed line is the artist's own within-dataset variability (JS between two halves of the ground truth) — the practical
  "same artist" floor. An adapter bar at or near the floor means its output is statistically inside the artist's own spread.
  Bars: base ${fmt(r.unigram.base)} / adapter ${fmt(r.unigram.adapter)} (floor ${fmt(r.unigram.floor)}) · marginals ${fmt(r.marginalMean.base)} / ${fmt(r.marginalMean.adapter)} (floor ${fmt(r.marginalMean.floor)}) ·
  transitions ${fmt(r.transitionMean.base)} / ${fmt(r.transitionMean.adapter)} (floor ${fmt(r.transitionMean.floor)}). Base↔adapter distance: ${fmt(r.unigram.baseVsAdapter)} unigram.</p>
</div>

<div class="card">
  <h2>Per-dimension distance <span class="muted">(6 FSQ digits of every code)</span></h2>
  ${legend(['base', 'adapter'], true)}
  <h4>Marginal distributions</h4>
  ${dimChart(r.perDimMarginalJs)}
  <h4 style="margin-top:12px">Transition matrices (frame-to-frame movement)</h4>
  ${dimChart(r.perDimTransitionJs)}
</div>

<div class="card">
  <h2>What the code space looks like <span class="muted">(per-dimension level usage)</span></h2>
  ${legend(['gt', 'base', 'adapter'])}
  <div class="panels">${marginalPanels}</div>
</div>

<div class="card">
  <h2>Sequence dynamics</h2>
  ${legend(['gt', 'base', 'adapter'])}
  ${dynChart}
  <table style="margin-top:10px">
    <tr><th></th><th class="num">Ground truth</th><th class="num">Base LM</th><th class="num">Adapter</th></tr>
    <tr><td>Code change rate (frame-to-frame)</td><td class="num">${fmt(r.changeRate.gt)} ± ${fmt(r.changeRate.gtSd)}</td><td class="num">${fmt(r.changeRate.base)} ± ${fmt(r.changeRate.baseSd)}</td><td class="num">${fmt(r.changeRate.adapter)} ± ${fmt(r.changeRate.adapterSd)}</td></tr>
    <tr><td>8-gram repetition rate (structure)</td><td class="num">${fmt(r.repetition.gt)} ± ${fmt(r.repetition.gtSd)}</td><td class="num">${fmt(r.repetition.base)} ± ${fmt(r.repetition.baseSd)}</td><td class="num">${fmt(r.repetition.adapter)} ± ${fmt(r.repetition.adapterSd)}</td></tr>
    <tr><td>Unigram entropy (bits)</td><td class="num">${fmt(r.entropyFull.gt, 2)}</td><td class="num">${fmt(r.entropyFull.base, 2)}</td><td class="num">${fmt(r.entropyFull.adapter, 2)}</td></tr>
    <tr><td>Distinct codes used</td><td class="num">${r.distinctCodes.gt}</td><td class="num">${r.distinctCodes.base}</td><td class="num">${r.distinctCodes.adapter}</td></tr>
    <tr><td>Tokens</td><td class="num">${r.tokenCounts.gt}</td><td class="num">${r.tokenCounts.base}</td><td class="num">${r.tokenCounts.adapter}</td></tr>
  </table>
</div>

<div class="card">
  <h2>Memorization check <span class="muted">(longest common contiguous run vs any training song)</span></h2>
  <p class="note">A style adapter should stay low (a few seconds of coincidental overlap). Tens of seconds means the adapter is
  replaying training sequences — every "closer to the artist" number above would then be memorization, not generalization.
  Same-seed base generations are the control. Adapter max ${r.memorization.maxSec.adapter.toFixed(1)}s / mean ${fmt(r.memorization.meanSec.adapter, 1)}s ·
  base max ${r.memorization.maxSec.base.toFixed(1)}s / mean ${fmt(r.memorization.meanSec.base, 1)}s.</p>
  <table style="margin-top:8px">
    <tr><th>Side</th><th>Prompted with</th><th>Seed</th><th>Longest match in</th><th class="num">Tokens</th><th class="num">Length</th></tr>
    ${memRows}
  </table>
</div>

<div class="card">
  <h2>Per-song breakdown</h2>
  <p class="note">Mean per-dimension marginal JS between each song's own ground truth and the generations prompted with that song's
  caption/lyrics. Green delta = adapter closer than base for that song. "Diverges at" = median first token where the same-seed
  base and adapter generations differ (${fmt(r.divergence.medianSec, 1)}s median overall, range ${fmt(r.divergence.minSec, 1)}–${fmt(r.divergence.maxSec, 1)}s over ${r.divergence.count} pairs).</p>
  <table style="margin-top:8px">
    <tr><th>Song</th><th class="num">Eval len</th><th class="num">Base JS</th><th class="num">Adapter JS</th><th class="num">Δ</th><th class="num">Diverges at</th></tr>
    ${songRows}
  </table>
</div>

<div class="card">
  <h2>Method notes</h2>
  <ul class="note">
    <li>Generations use the trainer's own captions/lyrics from lm_codes.jsonl (pre-tagged with the trigger word), identical seeds on both sides, adapter presence the only variable — the audition A/B discipline.</li>
    <li>Ground truth is truncated to each row's generated duration so distribution comparisons see the same song regions.</li>
    <li>JS divergence is base-2 and needs no smoothing. Finite-sample JS is biased up on sparse supports; base and adapter use identical token budgets so their comparison is unbiased, and the GT floor (computed on half-size splits) is biased conservative (up).</li>
    <li>Verdict thresholds: "toward/away" needs |unigram ΔJS| &gt; 0.005 and ≥4 (≤2) of 6 dimensions won; memorization warns at &gt;5 s, flags at &gt;15 s of contiguous replay.</li>
    <li>Plan fields (BPM, key) are LM-chosen per side, as in production; only caption/lyrics/duration are pinned.</li>
  </ul>
</div>

</main>
<div id="tip"></div>
<script>
  const tip = document.getElementById('tip');
  document.querySelectorAll('[data-tip]').forEach(el => {
    el.addEventListener('mousemove', e => {
      tip.textContent = el.dataset.tip;
      tip.style.left = (e.pageX + 12) + 'px';
      tip.style.top = (e.pageY - 12) + 'px';
      tip.style.opacity = '1';
    });
    el.addEventListener('mouseleave', () => { tip.style.opacity = '0'; });
  });
</script>
</body></html>`;
}

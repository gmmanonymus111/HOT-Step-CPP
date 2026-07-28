#!/usr/bin/env node
/**
 * Stamp trigger words into adapters that were trained before HOT-Step embedded
 * them (every Side-Step-trained adapter, and every HOT-Step one from before
 * 2026-07-28).
 *
 * docs/plans/2026-07-28-adapter-trigger-embedding.md §6
 *
 * Adds `hot_step_trigger`, `hot_step_trigger_position` and
 * `modelspec.trigger_phrase` to a safetensors file's `__metadata__`. Tensor
 * bytes are copied verbatim, so the weights are provably unchanged — only the
 * JSON header grows. Unknown metadata keys are ignored by every other consumer,
 * so a stamped adapter still loads in ComfyUI / PEFT / Side-Step.
 *
 * DRY RUN BY DEFAULT. Nothing is written without --apply.
 *
 *   node server/scripts/stamp-adapter-triggers.mjs
 *   node server/scripts/stamp-adapter-triggers.mjs --apply
 *
 *   --adapters <dir>     adapters root (default: ../../adapters from this file)
 *   --datasets <dir>     a Side-Step corpus root; each <dir>/<name>/dataset.json
 *                        supplies that adapter's REAL custom_tag + tag_position,
 *                        which beats guessing from the filename
 *   --map <file.json>    {"<adapterName>": "<trigger>"} explicit overrides
 *   --position <p>       prepend|append fallback when no dataset says otherwise
 *   --only <lm|dit|all>  which subtree to walk (default all)
 *   --no-backup          skip the .bak copy (not recommended)
 *   --force              restamp adapters that already carry a trigger
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MAX_HEADER_BYTES = 64 * 1024 * 1024;
const SIZE_SUFFIX = /-(?:0\.6B|1\.7B|4B)$/;

// ── args ────────────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
const flag = (name) => argv.includes(`--${name}`);
const opt = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt;
};

const APPLY = flag('apply');
const FORCE = flag('force');
const BACKUP = !flag('no-backup');
const POSITION = opt('position', 'prepend');
const ONLY = opt('only', 'all');
const ADAPTERS_ROOT = path.resolve(opt('adapters', path.join(HERE, '..', '..', 'adapters')));
const DATASETS_DIR = opt('datasets', '');
const MAP_FILE = opt('map', '');

if (POSITION !== 'prepend' && POSITION !== 'append') {
  console.error(`--position must be prepend|append (got "${POSITION}")`);
  process.exit(2);
}

let explicitMap = {};
if (MAP_FILE) {
  try {
    explicitMap = JSON.parse(fs.readFileSync(path.resolve(MAP_FILE), 'utf8'));
  } catch (e) {
    console.error(`cannot read --map ${MAP_FILE}: ${e.message}`);
    process.exit(2);
  }
}

// ── safetensors header I/O ──────────────────────────────────────────────────

function readHeader(file) {
  const fd = fs.openSync(file, 'r');
  try {
    const lenBuf = Buffer.allocUnsafe(8);
    if (fs.readSync(fd, lenBuf, 0, 8, 0) !== 8) return null;
    const headerLen = Number(lenBuf.readBigUInt64LE(0));
    if (!Number.isSafeInteger(headerLen) || headerLen <= 0 || headerLen > MAX_HEADER_BYTES) return null;
    const hdr = Buffer.allocUnsafe(headerLen);
    if (fs.readSync(fd, hdr, 0, headerLen, 8) !== headerLen) return null;
    return { json: JSON.parse(hdr.toString('utf8')), headerLen };
  } catch {
    return null;
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * Rewrite `file` with `metadata` merged into its `__metadata__`.
 *
 * Streams `newLen(8) + newHeader + originalPayload` to a temp file, then swaps.
 * Never edits in place, so an interrupted run leaves the original intact.
 */
function stamp(file, metadata) {
  const head = readHeader(file);
  if (!head) throw new Error('unreadable safetensors header');

  const json = { ...head.json, __metadata__: { ...(head.json.__metadata__ || {}), ...metadata } };
  const newHeader = Buffer.from(JSON.stringify(json), 'utf8');
  // safetensors requires the header to be 8-byte aligned; pad with spaces,
  // which are legal JSON whitespace outside string literals.
  const pad = (8 - (newHeader.length % 8)) % 8;
  const padded = Buffer.concat([newHeader, Buffer.alloc(pad, 0x20)]);
  const lenBuf = Buffer.allocUnsafe(8);
  lenBuf.writeBigUInt64LE(BigInt(padded.length), 0);

  const tmp = `${file}.stamping`;
  const out = fs.openSync(tmp, 'w');
  const src = fs.openSync(file, 'r');
  try {
    fs.writeSync(out, lenBuf);
    fs.writeSync(out, padded);
    const buf = Buffer.allocUnsafe(4 * 1024 * 1024);
    let pos = 8 + head.headerLen;
    for (;;) {
      const n = fs.readSync(src, buf, 0, buf.length, pos);
      if (n <= 0) break;
      fs.writeSync(out, buf, 0, n);
      pos += n;
    }
  } finally {
    fs.closeSync(src);
    fs.closeSync(out);
  }

  if (BACKUP) fs.copyFileSync(file, `${file}.bak`);
  fs.rmSync(file);
  fs.renameSync(tmp, file);
}

// ── discovery ───────────────────────────────────────────────────────────────

/** Every adapter under `root`: PEFT dirs and bare .safetensors files. */
function findAdapters(root, kind) {
  const out = [];
  if (!fs.existsSync(root)) return out;
  for (const e of fs.readdirSync(root, { withFileTypes: true })) {
    if (e.name.startsWith('.')) continue;
    const full = path.join(root, e.name);
    if (e.isDirectory()) {
      const model = path.join(full, 'adapter_model.safetensors');
      if (fs.existsSync(model)) out.push({ name: e.name, file: model, kind });
    } else if (e.isFile() && e.name.endsWith('.safetensors')) {
      out.push({ name: e.name.replace(/\.safetensors$/i, ''), file: full, kind });
    }
  }
  return out;
}

/**
 * `custom_tag` + `tag_position` per dataset name — the ground truth for what an
 * adapter was actually trained with. Reads both the Training Studio's own
 * `dataset_meta.json` and any Side-Step corpus passed via --datasets, whose
 * `dataset.json` carries the same fields under `metadata`.
 */
function loadDatasetTags() {
  const tags = new Map();

  const addStudio = (dsRoot) => {
    if (!fs.existsSync(dsRoot)) return;
    for (const e of fs.readdirSync(dsRoot, { withFileTypes: true })) {
      if (!e.isDirectory()) continue;
      try {
        const meta = JSON.parse(fs.readFileSync(path.join(dsRoot, e.name, 'dataset_meta.json'), 'utf8'));
        const tag = (meta.customTag || meta.custom_tag || '').trim();
        const pos = (meta.tagPosition || meta.tag_position || '').trim();
        if (tag) tags.set(e.name, { tag, pos });
      } catch { /* a dataset without readable meta simply contributes nothing */ }
    }
  };

  const addSideStep = (root) => {
    if (!root || !fs.existsSync(root)) return;
    for (const e of fs.readdirSync(root, { withFileTypes: true })) {
      if (!e.isDirectory()) continue;
      try {
        const j = JSON.parse(fs.readFileSync(path.join(root, e.name, 'dataset.json'), 'utf8'));
        const md = j.metadata || {};
        const tag = (md.custom_tag || '').trim();
        const pos = (md.tag_position || '').trim();
        if (tag) tags.set(e.name, { tag, pos });
      } catch { /* likewise */ }
    }
  };

  addStudio(path.join(HERE, '..', 'data', 'training', 'datasets'));
  if (DATASETS_DIR) addSideStep(path.resolve(DATASETS_DIR));
  return tags;
}

/** explicit map > dataset custom_tag > dir name minus the -<size> suffix. */
function proposeTrigger(name, datasetTags) {
  if (explicitMap[name]) return { trigger: String(explicitMap[name]).trim(), position: POSITION, source: 'map' };
  const bare = name.replace(SIZE_SUFFIX, '');
  const hit = datasetTags.get(name) || datasetTags.get(bare);
  if (hit) {
    // A dataset whose tag_position is "replace" never put the tag in its
    // captions at all (preprocess-run.h:203), so that adapter has no trigger.
    if (hit.pos === 'replace') return { trigger: '', position: '', source: 'replace' };
    return { trigger: hit.tag, position: hit.pos === 'append' ? 'append' : 'prepend', source: 'dataset' };
  }
  return { trigger: bare, position: POSITION, source: 'filename' };
}

// ── run ─────────────────────────────────────────────────────────────────────

const datasetTags = loadDatasetTags();
const targets = [];
if (ONLY === 'all' || ONLY === 'dit') targets.push(...findAdapters(ADAPTERS_ROOT, 'dit'));
if (ONLY === 'all' || ONLY === 'lm') targets.push(...findAdapters(path.join(ADAPTERS_ROOT, 'lm'), 'lm'));

if (!targets.length) {
  console.error(`No adapters found under ${ADAPTERS_ROOT} (--only ${ONLY}).`);
  process.exit(1);
}

const rows = [];
let skipped = 0;
for (const t of targets) {
  const head = readHeader(t.file);
  if (!head) { rows.push({ ...t, trigger: '', source: 'UNREADABLE', action: 'skip' }); continue; }
  const existing = (head.json.__metadata__ || {}).hot_step_trigger;
  if (existing && !FORCE) { skipped++; continue; }
  const { trigger, position, source } = proposeTrigger(t.name, datasetTags);
  rows.push({ ...t, trigger, position, source, action: trigger ? (existing ? 'restamp' : 'stamp') : 'skip' });
}

const pad = (s, n) => String(s).padEnd(n);
console.log(`\nAdapters root: ${ADAPTERS_ROOT}`);
console.log(`${targets.length} adapter(s) found, ${skipped} already stamped (skipped), ${rows.length} to review.`);
console.log(`Fallback position: ${POSITION}   Mode: ${APPLY ? 'APPLY' : 'DRY RUN'}${BACKUP ? '  (keeping .bak)' : ''}`);
console.log(`Dataset tags loaded: ${datasetTags.size}${DATASETS_DIR ? ` (incl. ${DATASETS_DIR})` : ''}\n`);
console.log(`${pad('KIND', 5)} ${pad('ADAPTER', 42)} ${pad('PROPOSED TRIGGER', 28)} ${pad('POS', 8)} ${pad('SOURCE', 9)} ACTION`);
console.log('-'.repeat(110));
for (const r of rows) {
  console.log(`${pad(r.kind, 5)} ${pad(r.name.slice(0, 42), 42)} ${pad(r.trigger.slice(0, 28), 28)} ${pad(r.position || '-', 8)} ${pad(r.source, 9)} ${r.action}`);
}
const bySource = rows.reduce((m, r) => ({ ...m, [r.source]: (m[r.source] || 0) + 1 }), {});
console.log(`\nBy source: ${Object.entries(bySource).map(([k, v]) => `${k}=${v}`).join('  ')}`);

if (!APPLY) {
  console.log(`\nDRY RUN — nothing written. Re-run with --apply once the table above is right.`);
  console.log(`Fix individual rows with --map triggers.json, e.g. {"abba-4B": "abba"}\n`);
  process.exit(0);
}

let ok = 0, failed = 0;
for (const r of rows) {
  if (r.action === 'skip') continue;
  try {
    stamp(r.file, {
      hot_step_trigger: r.trigger,
      hot_step_trigger_position: r.position || POSITION,
      'modelspec.trigger_phrase': r.trigger,
    });
    ok++;
  } catch (e) {
    console.error(`FAILED ${r.name}: ${e.message}`);
    failed++;
  }
}
console.log(`\nStamped ${ok} adapter(s)${failed ? `, ${failed} failed` : ''}.`);
process.exit(failed ? 1 : 0);

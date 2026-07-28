#!/usr/bin/env node
/**
 * Migrate an adapter corpus to the per-base layout (2026-07-28):
 *
 *   lm/<artist>-0.6B  ->  lm-06b/<artist>
 *   lm/<artist>-1.7B  ->  lm-17b/<artist>
 *   lm/<artist>-4B    ->  lm-4b/<artist>
 *   <root>/<dit-dir>  ->  dit-<shorthand>/<dit-dir>   (base read from its own
 *                                                      dit_train_log.json)
 *
 * Everything is a same-volume rename — no data is copied, so this is fast and
 * safe to re-run (already-migrated entries are skipped). DRY RUN by default.
 *
 *   node server/scripts/migrate-adapter-layout.mjs --adapters M:\HOT-Step-CPP\Adapters
 *   node server/scripts/migrate-adapter-layout.mjs --adapters ... --apply
 *
 *   --adapters <dir>    adapters root (default: ../../adapters from this file)
 *   --default-dit <s>   shorthand folder for a DiT adapter dir with no readable
 *                       dit_train_log.json (default: leave in place + warn)
 *   --apply             actually move things
 *
 * Companion of server/src/services/training/adapterLayout.ts — the shorthand
 * map here must stay in sync with DIT_SHORTHANDS there.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const flag = (n) => argv.includes(`--${n}`);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };

const APPLY = flag('apply');
const ROOT = path.resolve(opt('adapters', path.join(HERE, '..', '..', 'adapters')));
const DEFAULT_DIT = opt('default-dit', '');

// Mirror of adapterLayout.ts (keep in sync).
const LM_DIRS = { '0.6B': 'lm-06b', '1.7B': 'lm-17b', '4B': 'lm-4b' };
const DIT_SHORTHANDS = {
  'acestep-v15-merge-base-sft-turbo-xl-thirds': 'dit-xl-thirds',
  'acestep-v15-merge-base-turbo-xl-ta-0.5': 'dit-xl-base-turbo',
  'acestep-v15-xl-sftturbo50': 'dit-xl-sft-turbo',
  'acestep-v15-merge-base-sft-xl-ta-0.5': 'dit-xl-base-sft',
  'acestep-v15-merge-sft-turbo-xl-ta-0.3': 'dit-xl-sft-turbo-ta03',
  'acestep-v15-merge-sft-turbo-xl-ta-0.7': 'dit-xl-sft-turbo-ta07',
  'acestep-v15-xl-base': 'dit-xl-base',
  'acestep-v15-xl-sft': 'dit-xl-sft',
  'acestep-v15-xl-turbo': 'dit-xl-turbo',
  'acestep-v15-base': 'dit-base',
  'acestep-v15-sft': 'dit-sft',
  'acestep-v15-turbo': 'dit-turbo',
  'acestep-v15-sftturbo50': 'dit-sft-turbo',
  'acestep-v15-turbo-continuous': 'dit-turbo-continuous',
  'acestep-v15-turbo-shift1': 'dit-turbo-shift1',
  'acestep-v15-turbo-shift3': 'dit-turbo-shift3',
  'acestep-v15-merge-base-sft-turbo-xl-thirds-convrot-ref': 'dit-xl-thirds-convrot',
  'sa3-dit': 'dit-sa3',
};
const QUANT_RE = /-(BF16|F16|F32|MXFP4|NVFP4|IQ\d[A-Z_]*|Q\d[\w]*)$/i;

function ditShorthand(model) {
  let stem = path.basename(String(model || '')).replace(/\.gguf$/i, '').replace(QUANT_RE, '');
  if (!stem) return '';
  if (DIT_SHORTHANDS[stem]) return DIT_SHORTHANDS[stem];
  const fb = stem.replace(/^acestep-v15-/i, '').replace(/\./g, '');
  return fb ? `dit-${fb}` : '';
}

if (!fs.existsSync(ROOT)) { console.error(`adapters root not found: ${ROOT}`); process.exit(2); }

const moves = [];   // { from, to, note }
const warns = [];

// ── planner-LM adapters ─────────────────────────────────────────────────────
const lmRoot = path.join(ROOT, 'lm');
if (fs.existsSync(lmRoot)) {
  for (const e of fs.readdirSync(lmRoot, { withFileTypes: true })) {
    if (e.name.startsWith('.')) continue;
    const from = path.join(lmRoot, e.name);
    const stem = e.isFile() ? e.name.replace(/\.safetensors$/i, '') : e.name;
    if (e.isFile() && !e.name.endsWith('.safetensors')) { warns.push(`lm/${e.name}: not an adapter — left in place`); continue; }
    const m = /-(0\.6B|1\.7B|4B)$/i.exec(stem);
    if (!m) {
      if (e.isDirectory() && fs.readdirSync(from).length === 0) {
        moves.push({ from, to: '', note: 'empty dir — remove' });
      } else {
        warns.push(`lm/${e.name}: no -<size> suffix — cannot tell the base, left in place`);
      }
      continue;
    }
    const size = ['0.6B', '1.7B', '4B'].find(s => s.toLowerCase() === m[1].toLowerCase());
    const bare = stem.slice(0, -(m[1].length + 1)) + (e.isFile() ? '.safetensors' : '');
    moves.push({ from, to: path.join(ROOT, LM_DIRS[size], bare), note: `lm ${size}` });
  }
}

// ── already-migrated lm-<size> dirs: strip legacy -<size> suffixes ──────────
// A hand-moved corpus (rename lm -> lm-4b) still carries the suffix on every
// child; the parent folder now says the size, so the suffix goes. A child whose
// suffix DISAGREES with its folder is moved to the right size folder instead.
for (const [size, dirName] of Object.entries(LM_DIRS)) {
  const sizeRoot = path.join(ROOT, dirName);
  if (!fs.existsSync(sizeRoot)) continue;
  for (const e of fs.readdirSync(sizeRoot, { withFileTypes: true })) {
    if (e.name.startsWith('.')) continue;
    const from = path.join(sizeRoot, e.name);
    const stem = e.isFile() ? e.name.replace(/\.safetensors$/i, '') : e.name;
    if (e.isFile() && !e.name.endsWith('.safetensors')) continue;
    const m = /-(0\.6B|1\.7B|4B)$/i.exec(stem);
    if (!m) {
      if (e.isDirectory() && fs.readdirSync(from).length === 0) {
        moves.push({ from, to: '', note: 'empty dir — remove' });
      }
      continue;   // already unsuffixed — nothing to do
    }
    const suffixSize = ['0.6B', '1.7B', '4B'].find(s => s.toLowerCase() === m[1].toLowerCase());
    const bare = stem.slice(0, -(m[1].length + 1)) + (e.isFile() ? '.safetensors' : '');
    const targetDir = suffixSize === size ? sizeRoot : path.join(ROOT, LM_DIRS[suffixSize]);
    moves.push({
      from, to: path.join(targetDir, bare),
      note: suffixSize === size ? 'strip suffix' : `RELOCATE — suffix says ${suffixSize}, folder says ${size}`,
    });
  }
}

// ── canonicalise PEFT weight filenames ──────────────────────────────────────
// MUST run (and execute) before the relocation pass below: a dir can need both
// a weights-rename and a relocation, and the rename addresses the old path.
// A dir with adapter_config.json whose weights were renamed <name>.safetensors
// (the old make-the-filename-the-trigger workaround) goes back to the
// adapter_model.safetensors the scanners and loaders look for — the embedded
// trigger has made the rename pointless.
for (const e of fs.readdirSync(ROOT, { withFileTypes: true })) {
  if (!e.isDirectory() || !/^(dit-|lm-)/i.test(e.name)) continue;
  for (const sub of fs.readdirSync(path.join(ROOT, e.name), { withFileTypes: true })) {
    if (!sub.isDirectory() || sub.name.startsWith('.')) continue;
    const dir = path.join(ROOT, e.name, sub.name);
    if (!fs.existsSync(path.join(dir, 'adapter_config.json'))) continue;
    if (fs.existsSync(path.join(dir, 'adapter_model.safetensors'))) continue;
    const st = fs.readdirSync(dir).filter(f => f.endsWith('.safetensors'));
    if (st.length === 1) {
      moves.push({ from: path.join(dir, st[0]), to: path.join(dir, 'adapter_model.safetensors'),
                   note: 'canonicalise weights filename' });
    }
  }
}

// ── dit-* dirs: verify each PEFT child sits under its own base ──────────────
// The folder is a claim about the training base; the adapter's own
// dit_train_log.json is the evidence. Disagreement = move to the right folder.
// Bare .safetensors files record no base and are left where the user put them.
for (const e of fs.readdirSync(ROOT, { withFileTypes: true })) {
  if (!e.isDirectory() || !/^dit-/i.test(e.name)) continue;
  const shorthandDir = path.join(ROOT, e.name);
  for (const sub of fs.readdirSync(shorthandDir, { withFileTypes: true })) {
    if (!sub.isDirectory() || sub.name.startsWith('.')) continue;
    const from = path.join(shorthandDir, sub.name);
    let base = '';
    try {
      const log = JSON.parse(fs.readFileSync(path.join(from, 'dit_train_log.json'), 'utf8'));
      base = log?.config?.dit_name || log?.config?.dit_path || '';
    } catch { continue; }   // no log — nothing to verify against
    const want = base ? ditShorthand(base) : '';
    if (want && want.toLowerCase() !== e.name.toLowerCase()) {
      moves.push({ from, to: path.join(ROOT, want, sub.name), note: `RELOCATE — trained on ${base}` });
    }
  }
}

// ── DiT adapters at the root ────────────────────────────────────────────────
for (const e of fs.readdirSync(ROOT, { withFileTypes: true })) {
  if (e.name.startsWith('.') || e.name === 'lm') continue;
  if (/^(lm-06b|lm-17b|lm-4b|dit-)/i.test(e.name)) continue;   // already migrated trees
  const from = path.join(ROOT, e.name);

  if (e.isDirectory()) {
    const looksAdapter = fs.existsSync(path.join(from, 'adapter_config.json')) ||
      fs.readdirSync(from).some(f => f.endsWith('.safetensors'));
    if (!looksAdapter) { warns.push(`${e.name}: not an adapter dir — left in place`); continue; }
    let base = '';
    try {
      const log = JSON.parse(fs.readFileSync(path.join(from, 'dit_train_log.json'), 'utf8'));
      base = log?.config?.dit_name || log?.config?.dit_path || '';
    } catch { /* no log — fall through */ }
    const shorthand = base ? ditShorthand(base) : DEFAULT_DIT;
    if (!shorthand) { warns.push(`${e.name}: no dit_train_log.json and no --default-dit — left in place`); continue; }
    moves.push({ from, to: path.join(ROOT, shorthand, e.name), note: base ? `dit (${base})` : 'dit (--default-dit)' });
  } else if (e.isFile() && e.name.endsWith('.safetensors')) {
    // A bare DiT adapter at the root records no base anywhere — only move it
    // when the user says where it belongs.
    if (DEFAULT_DIT) moves.push({ from, to: path.join(ROOT, DEFAULT_DIT, e.name), note: 'dit bare (--default-dit)' });
    else warns.push(`${e.name}: bare file with no recorded base — left in place (use --default-dit to move)`);
  }
}

// ── report ──────────────────────────────────────────────────────────────────
console.log(`\nAdapters root: ${ROOT}   Mode: ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);
const pad = (s, n) => String(s).padEnd(n);
for (const mv of moves) {
  const rel = (p) => p ? path.relative(ROOT, p) : '(delete)';
  console.log(`${pad(rel(mv.from), 46)} -> ${pad(rel(mv.to), 40)} ${mv.note}`);
}
for (const w of warns) console.log(`WARN  ${w}`);
console.log(`\n${moves.length} move(s), ${warns.length} warning(s).`);

if (!APPLY) { console.log('DRY RUN — nothing moved. Re-run with --apply.\n'); process.exit(0); }

let ok = 0, failed = 0;
for (const mv of moves) {
  try {
    if (!mv.to) { fs.rmdirSync(mv.from); ok++; continue; }
    if (fs.existsSync(mv.to)) throw new Error(`target exists: ${mv.to}`);
    fs.mkdirSync(path.dirname(mv.to), { recursive: true });
    fs.renameSync(mv.from, mv.to);
    ok++;
  } catch (e) {
    console.error(`FAILED ${mv.from}: ${e.message}`);
    failed++;
  }
}
// Remove the legacy lm/ root if the migration emptied it.
try {
  if (fs.existsSync(lmRoot) && fs.readdirSync(lmRoot).length === 0) { fs.rmdirSync(lmRoot); console.log('removed empty lm/'); }
} catch { /* leave it */ }
console.log(`\nMoved ${ok}${failed ? `, ${failed} FAILED` : ''}.`);
process.exit(failed ? 1 : 0);

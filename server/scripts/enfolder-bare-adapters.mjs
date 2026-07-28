#!/usr/bin/env node
/**
 * Move every bare `<name>.safetensors` in a folder into `<name>/` and rename
 * it to the canonical `adapter_model.safetensors`:
 *
 *   dit-xl-base-turbo/abba.safetensors  ->  dit-xl-base-turbo/abba/adapter_model.safetensors
 *
 * The filename stops carrying the trigger word — that now lives in the file's
 * own __metadata__ (hot_step_trigger, stamped 2026-07-28) — so the name's only
 * remaining job is labelling the folder. Same-volume renames, no data copied,
 * safe to re-run. DRY RUN by default.
 *
 *   node server/scripts/enfolder-bare-adapters.mjs --folder M:\...\dit-xl-base-turbo
 *   node server/scripts/enfolder-bare-adapters.mjs --folder ... --apply
 *
 * NOTE deliberately does NOT write an adapter_config.json: these are
 * LyCORIS-format adapters whose alphas live per-tensor in the file itself; a
 * fabricated PEFT config would mis-scale them.
 */
import fs from 'fs';
import path from 'path';

const argv = process.argv.slice(2);
const APPLY = argv.includes('--apply');
const fi = argv.indexOf('--folder');
const FOLDER = fi >= 0 && fi + 1 < argv.length ? path.resolve(argv[fi + 1]) : '';

if (!FOLDER || !fs.existsSync(FOLDER) || !fs.statSync(FOLDER).isDirectory()) {
  console.error('usage: enfolder-bare-adapters.mjs --folder <dir> [--apply]');
  process.exit(2);
}

const moves = [];
const warns = [];
for (const e of fs.readdirSync(FOLDER, { withFileTypes: true })) {
  if (!e.isFile() || !e.name.endsWith('.safetensors') || e.name.startsWith('.')) continue;
  const stem = e.name.replace(/\.safetensors$/i, '');
  if (stem === 'adapter_model') { warns.push(`${e.name}: already canonical at top level — left alone`); continue; }
  const destDir = path.join(FOLDER, stem);
  const dest = path.join(destDir, 'adapter_model.safetensors');
  if (fs.existsSync(dest)) { warns.push(`${stem}: ${path.basename(destDir)}/adapter_model.safetensors already exists — skipped`); continue; }
  moves.push({ from: path.join(FOLDER, e.name), destDir, dest, stem });
}

console.log(`\nFolder: ${FOLDER}   Mode: ${APPLY ? 'APPLY' : 'DRY RUN'}`);
for (const m of moves.slice(0, 8)) console.log(`  ${m.stem}.safetensors -> ${m.stem}/adapter_model.safetensors`);
if (moves.length > 8) console.log(`  … and ${moves.length - 8} more, same shape`);
for (const w of warns) console.log(`WARN  ${w}`);
console.log(`${moves.length} move(s), ${warns.length} warning(s).`);

if (!APPLY) { console.log('DRY RUN — nothing moved. Re-run with --apply.\n'); process.exit(0); }

let ok = 0, failed = 0;
for (const m of moves) {
  try {
    fs.mkdirSync(m.destDir, { recursive: true });
    fs.renameSync(m.from, m.dest);
    ok++;
  } catch (err) {
    console.error(`FAILED ${m.stem}: ${err.message}`);
    failed++;
  }
}
console.log(`Moved ${ok}${failed ? `, ${failed} FAILED` : ''}.`);
process.exit(failed ? 1 : 0);

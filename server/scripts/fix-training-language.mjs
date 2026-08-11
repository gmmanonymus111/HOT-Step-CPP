#!/usr/bin/env node
/**
 * fix-training-language.mjs — normalize the `language` field in every
 * lm_codes.jsonl from full names to the ISO codes the inference FSM speaks
 * (2026-08-11).
 *
 * WHY. Dataset creation defaulted `default_language` to the literal string
 * 'english' (datasetCreate.ts / datasetBuilder.ts), which flowed
 * dataset -> sidecar -> preprocess __metadata__ -> lm_codes.jsonl -> the
 * training CoT YAML. But metadata-fsm.h constrains `language:` to 51 ISO
 * codes ('en', 'fr', 'zh', ... 'unknown') — 'english' is NOT among them and
 * cannot be emitted at inference. Adapters therefore learned to produce a
 * token sequence the sampler forbids: with the language unpinned the model's
 * probability mass lands on a blocked word and whatever ISO code survives the
 * mask wins. Measured on 4yearstrong_someway, 6 seeds, language unpinned:
 * adapter produced sr/pt/fr on 3 of 6 runs (base LM: 1 of 6).
 *
 * WHAT. A MINIMAL BYTE EDIT: the exact substring `"language":"<name>"` that
 * lm-extract.h:59 writes is replaced with its ISO code. Nothing is
 * re-serialized — the codes arrays, captions and key order are untouched —
 * so the diff is provably confined to this one field. Verified after each
 * file: same line count, every row still parses, every `codes` array
 * identical (length + checksum), language now valid.
 *
 * The audio codes are unaffected by this, so NO re-extract and no retraining
 * is required; existing adapters keep working (they were trained on the wrong
 * language token, which is one line of a five-line CoT — pinning the language
 * at request time is the mitigation for those).
 *
 *   node server/scripts/fix-training-language.mjs            # dry run
 *   node server/scripts/fix-training-language.mjs --apply
 *   node server/scripts/fix-training-language.mjs --apply --tensors <dir>
 */
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const APPLY = argv.includes('--apply');
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const TENSORS = path.resolve(opt('tensors', path.join(HERE, '..', 'data', 'training', 'tensors')));

/** Full name -> ISO code, restricted to the FSM's own 51-value set
 *  (metadata-fsm.h:203-208). Anything already a valid code is left alone. */
const NAME_TO_ISO = {
  arabic: 'ar', azerbaijani: 'az', bulgarian: 'bg', bengali: 'bn', catalan: 'ca',
  czech: 'cs', danish: 'da', german: 'de', greek: 'el', english: 'en',
  spanish: 'es', persian: 'fa', farsi: 'fa', finnish: 'fi', french: 'fr',
  hebrew: 'he', hindi: 'hi', croatian: 'hr', haitian: 'ht', hungarian: 'hu',
  indonesian: 'id', icelandic: 'is', italian: 'it', japanese: 'ja', korean: 'ko',
  latin: 'la', lithuanian: 'lt', malay: 'ms', nepali: 'ne', dutch: 'nl',
  norwegian: 'no', punjabi: 'pa', polish: 'pl', portuguese: 'pt', romanian: 'ro',
  russian: 'ru', sanskrit: 'sa', slovak: 'sk', serbian: 'sr', swedish: 'sv',
  swahili: 'sw', tamil: 'ta', telugu: 'te', thai: 'th', tagalog: 'tl',
  turkish: 'tr', ukrainian: 'uk', urdu: 'ur', vietnamese: 'vi', cantonese: 'yue',
  chinese: 'zh', mandarin: 'zh',
};
const VALID = new Set([...Object.values(NAME_TO_ISO), 'unknown']);

/** Row fingerprint used to prove the edit touched nothing else. */
function fingerprint(text) {
  const rows = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const r = JSON.parse(line);
    rows.push({
      id: r._id ?? r.file ?? '',
      lang: r.language ?? '',
      codesLen: Array.isArray(r.codes) ? r.codes.length : -1,
      codesHash: crypto.createHash('sha1')
        .update(Array.isArray(r.codes) ? r.codes.join(',') : '').digest('hex'),
      capHash: crypto.createHash('sha1').update(String(r.caption ?? '')).digest('hex'),
    });
  }
  return rows;
}

const files = [];
for (const slug of fs.readdirSync(TENSORS)) {
  const slugDir = path.join(TENSORS, slug);
  if (!fs.statSync(slugDir).isDirectory()) continue;
  for (const variant of fs.readdirSync(slugDir)) {
    const f = path.join(slugDir, variant, 'lm_codes.jsonl');
    if (fs.existsSync(f)) files.push(f);
  }
}

console.log(`\ntensors: ${TENSORS}`);
console.log(`lm_codes.jsonl files: ${files.length}   mode: ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);

let backupDir = '';
if (APPLY) {
  const stamp = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
  backupDir = path.join(TENSORS, `_lang_fix_backup_${stamp}`);
  fs.mkdirSync(backupDir, { recursive: true });
  console.log(`backups -> ${backupDir}\n`);
}

let changedFiles = 0, changedRows = 0, skipped = 0, failures = 0;
const seenValues = new Map();

for (const f of files) {
  const before = fs.readFileSync(f, 'utf-8');
  let after = before;
  let rowsHere = 0;

  for (const [name, iso] of Object.entries(NAME_TO_ISO)) {
    const needle = `"language":"${name}"`;
    if (!after.includes(needle)) continue;
    const n = after.split(needle).length - 1;
    after = after.split(needle).join(`"language":"${iso}"`);
    rowsHere += n;
    seenValues.set(name, (seenValues.get(name) || 0) + n);
  }

  if (after === before) { skipped++; continue; }

  // ── verification: everything except `language` must be identical ────────
  let fpBefore, fpAfter;
  try {
    fpBefore = fingerprint(before);
    fpAfter = fingerprint(after);
  } catch (e) {
    console.log(`  FAIL ${f}: unparseable after edit (${e.message}) — skipping`);
    failures++;
    continue;
  }
  let ok = fpBefore.length === fpAfter.length;
  if (ok) {
    for (let i = 0; i < fpBefore.length; i++) {
      const a = fpBefore[i], b = fpAfter[i];
      if (a.id !== b.id || a.codesLen !== b.codesLen || a.codesHash !== b.codesHash || a.capHash !== b.capHash) {
        ok = false;
        break;
      }
      if (!VALID.has(b.lang)) { ok = false; break; }
    }
  }
  if (!ok) {
    console.log(`  FAIL ${f}: verification failed — NOT written`);
    failures++;
    continue;
  }

  changedFiles++;
  changedRows += rowsHere;
  if (APPLY) {
    const rel = path.relative(TENSORS, f).replace(/[\\/]/g, '__');
    fs.copyFileSync(f, path.join(backupDir, rel));
    const tmp = f + '.tmp';
    fs.writeFileSync(tmp, after);
    fs.renameSync(tmp, f);
  }
}

console.log('values replaced:');
for (const [name, n] of [...seenValues.entries()].sort((a, b) => b[1] - a[1])) {
  console.log(`  ${name} -> ${NAME_TO_ISO[name]}   ${n} row(s)`);
}
console.log(`\nfiles changed: ${changedFiles}   rows changed: ${changedRows}`);
console.log(`files already valid: ${skipped}   verification failures: ${failures}`);
if (!APPLY) console.log('\nDRY RUN — nothing written. Re-run with --apply.\n');
else console.log(`\ndone. backups in ${backupDir}\n`);
if (failures > 0) process.exit(1);

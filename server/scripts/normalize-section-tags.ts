#!/usr/bin/env npx tsx
/**
 * Rewrite section tags in existing generation lyrics into the vocabulary the
 * BASE MODEL documents (2026-08-08). Run with `npx tsx` from server/.
 *
 * ACE-Step-1.5 docs/en (Tutorial.md, ace_step_musicians_guide.md) list the tag
 * set and say structure tags may be combined "with `-` for finer control"
 * ([Chorus - anthemic]). Descriptors are therefore CORRECT and are left alone —
 * 12.4% of tags use them and they stay exactly as they are.
 *
 * Only the colon form is wrong. `[Instrumental Break: Guitar Solo]` was an
 * example in the generation prompt, so it got copied; neither the head
 * "Instrumental Break" nor the colon separator appears in the docs. This maps
 * those onto documented heads with a dash:
 *
 *   [Instrumental Break: Guitar Solo]      -> [Guitar Solo]
 *   [Interlude: Twin Guitar Solo]          -> [Guitar Solo]
 *   [Instrumental Break: Saxophone Solo]   -> [Instrumental - Saxophone Solo]
 *   [Instrumental Break]                   -> [Instrumental]
 *
 * [Post-Chorus] and [Interlude] are NOT touched: undocumented, but they appear
 * 408 and 251 times in the real-song training lyrics, so the model has plainly
 * seen them.
 *
 *   npx tsx scripts/normalize-section-tags.ts
 *   npx tsx scripts/normalize-section-tags.ts --apply
 *
 *   --db <file>   SQLite db (default server/data/hotstep.db)
 */
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(HERE, '..', 'package.json'));
const Database = require('better-sqlite3');

const argv = process.argv.slice(2);
const APPLY = argv.includes('--apply');
const opt = (n: string, d?: string) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d;
};
const DB_PATH = path.resolve(opt('db', path.join(HERE, '..', 'data', 'hotstep.db'))!);

/** A whole-line section tag: "[Anything]" alone on its line. */
const TAG_LINE = /^([ \t]*)\[([^\]]{1,60})\][ \t]*$/gm;

export function normalizeTag(inner: string): string | null {
  const t = inner.trim();

  // Colon form only. Dash descriptors are documented and stay untouched.
  const colon = t.match(/^\s*(Instrumental Break|Interlude|Instrumental)\s*:\s*(.+?)\s*$/i);
  if (colon) {
    const desc = colon[2].trim();
    // A guitar solo has its own documented tag; the descriptor adds nothing.
    if (/^(twin |slide |lead |acoustic )?guitar solo$/i.test(desc)) return 'Guitar Solo';
    if (/^piano (solo|interlude)$/i.test(desc)) return 'Piano Interlude';
    return `Instrumental - ${desc}`;
  }

  // Bare undocumented head.
  if (/^instrumental break$/i.test(t)) return 'Instrumental';

  return null;  // already fine
}

const db = new Database(DB_PATH);
const rows = db.prepare('SELECT id, title, lyrics FROM generations').all() as any[];

const changes: Array<{ id: number; title: string; lyrics: string; edits: Map<string, string> }> = [];
const tally = new Map<string, { to: string; n: number }>();

for (const r of rows) {
  const src = String(r.lyrics ?? '');
  if (!src) continue;
  const edits = new Map<string, string>();
  const out = src.replace(TAG_LINE, (whole, indent: string, inner: string) => {
    const to = normalizeTag(inner);
    if (!to) return whole;
    edits.set(inner.trim(), to);
    const key = inner.trim();
    const hit = tally.get(key);
    if (hit) hit.n++; else tally.set(key, { to, n: 1 });
    return `${indent}[${to}]`;
  });
  if (out !== src) changes.push({ id: r.id, title: r.title, lyrics: out, edits });
}

console.log(`\nDB:   ${DB_PATH}`);
console.log(`Mode: ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);
console.log(`songs affected: ${changes.length} of ${rows.length}`);
const totalTags = [...tally.values()].reduce((s, v) => s + v.n, 0);
console.log(`tags rewritten: ${totalTags}\n`);
for (const [from, v] of [...tally.entries()].sort((a, b) => b[1].n - a[1].n)) {
  console.log(`  ${String(v.n).padStart(3)}  [${from}]  ->  [${v.to}]`);
}

if (!changes.length) { console.log('\nNothing to do.\n'); db.close(); process.exit(0); }
if (!APPLY) { console.log('\nDRY RUN — nothing written. Re-run with --apply.\n'); db.close(); process.exit(0); }

const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
const bak = DB_PATH.replace(/\.db$/i, `_backup_tags_${stampNow}.db`);
await db.backup(bak);
console.log(`\nBacked up to ${path.basename(bak)}`);

const upd = db.prepare('UPDATE generations SET lyrics = ? WHERE id = ?');
const tx = db.transaction(() => { for (const c of changes) upd.run(c.lyrics, c.id); });
tx();
db.close();
console.log(`Updated ${changes.length} song(s).`);

// check-mm3-format.ts — validates generated MM3 captions against the OFFICIAL
// template shape.
//
//   cd server && npx tsx scripts/check-mm3-format.ts <dataset-dir> [...more dirs]
//
// Every invariant below was MEASURED over the 1000 vendored upstream templates
// (.claude/skills/mm3-captioning/upstream/templates/), not assumed:
//
//   - zero blank lines, zero leading/trailing whitespace (0/1000 have either)
//   - section headers `Global Metadata` / `Vocal Details` / `Arrangement`,
//     bare on their own lines, in that order
//   - `Basic Attributes: bpm is N. key is K, and scale is S. GENRE.`
//     (998/1000; the other 2 use a <|bpm N-M|> range token). NO time
//     signature — 0/1000 carry one.
//   - the Arrangement labels, including the exact string
//     `Instrument Lifecycle Description (Primary/Secondary Layering):`
//
// MM3 was trained on this shape; a caption that deviates conditions the model
// off-distribution. Run this after changing the MM3 prompt, the fact
// substitution, or after captioning a new dataset.

import fs from 'fs';
import path from 'path';

const dirs = process.argv.slice(2);
if (dirs.length === 0) {
  console.error('usage: npx tsx scripts/check-mm3-format.ts <dataset-dir> [...more dirs]');
  process.exit(2);
}

const SECTIONS = ['Global Metadata', 'Vocal Details', 'Arrangement'];
const FIELD_ORDER = [
  'Basic Attributes:',
  'Global Emotional Progression:',
  'Application Scenarios & Imagery:',
  'Sonics & Production Profile:',
  'Vocal Gender & Timbre:',
  'Vocal Style:',
  'Harmony/Backing Vocals:',
  'Vocal FX:',
  'Instrument Lifecycle Description (Primary/Secondary Layering):',
  'Primary:',
  'Secondary:',
  'Groove & Foundation Progression:',
  'Embellishments, Textures & Spatial FX:',
];
const BA_RE = /^Basic Attributes: bpm is \d+\. key is [A-G][#b]?, and scale is (major|minor)\. .+\.$/;
// Facts can legitimately be absent (no Essentia, unparseable key) — genre-only
// is tolerated as a WARNING, not an error, so a missing-facts dataset is
// visible without failing the whole check.
const BA_GENRE_ONLY_RE = /^Basic Attributes: .+\.$/;

let files = 0, clean = 0, failed = 0, warned = 0;

for (const dir of dirs) {
  // Dataset dirs hold <stem>.mm3.txt; the upstream template dir holds bare
  // .txt files — fall back so the validator can self-test against the 1000
  // official templates (which must all pass by construction).
  let names = fs.readdirSync(dir).filter(n => n.endsWith('.mm3.txt'));
  if (names.length === 0) names = fs.readdirSync(dir).filter(n => n.endsWith('.txt'));
  for (const name of names) {
    files++;
    const text = fs.readFileSync(path.join(dir, name), 'utf8');
    const lines = text.replace(/\r\n/g, '\n').replace(/\n+$/, '').split('\n');
    const errors: string[] = [];
    const warnings: string[] = [];

    if (lines.some(l => l.trim() === '')) errors.push('blank line(s)');
    if (lines.some(l => l !== l.trim())) errors.push('leading/trailing whitespace');

    const sectionIdx = SECTIONS.map(s => lines.findIndex(l => l.trim() === s));
    SECTIONS.forEach((s, i) => { if (sectionIdx[i] < 0) errors.push(`missing section "${s}"`); });
    if (sectionIdx.every(i => i >= 0) && !(sectionIdx[0] < sectionIdx[1] && sectionIdx[1] < sectionIdx[2])) {
      errors.push('sections out of order');
    }

    let cursor = -1;
    for (const label of FIELD_ORDER) {
      const at = lines.findIndex(l => l.trim().startsWith(label));
      if (at < 0) errors.push(`missing "${label.replace(/:$/, '')}"`);
      else if (at < cursor) errors.push(`"${label.replace(/:$/, '')}" out of order`);
      else cursor = at;
    }

    // One line per field: every line must be a section header or start with a
    // known label. A bare label with its prose on the NEXT line, an unknown
    // section ("Chorus: …"), or a decode-loop label run all fail this.
    // Exception, measured upstream: inside the Instrument Lifecycle block the
    // templates use free-form instrument-group labels — Tertiary: (10/1000),
    // Bass:, Synthesizers:, Percussive Accents:, … — so short label-led lines
    // are legal there and only there.
    const ildIdx = lines.findIndex(l => l.trim().startsWith('Instrument Lifecycle Description'));
    const GROUP_LABEL_RE = /^[A-Z][A-Za-z /()&'-]{0,34}: \S/;
    lines.forEach((l, idx) => {
      const t = l.trim();
      if (!t) return;
      if (SECTIONS.includes(t)) return;
      if (FIELD_ORDER.some(lb => t.startsWith(lb))) return;
      if (ildIdx >= 0 && idx > ildIdx && GROUP_LABEL_RE.test(t)) return;
      errors.push(`stray line (not a section or known field): "${t.slice(0, 60)}"`);
    });
    // Upstream files end at the Embellishments field (1000/1000).
    const embIdx = lines.findIndex(l => l.trim().startsWith('Embellishments, Textures & Spatial FX:'));
    if (embIdx >= 0 && lines.slice(embIdx + 1).some(l => l.trim() !== '')) {
      errors.push('content after the Embellishments field');
    }

    const ba = lines.find(l => l.startsWith('Basic Attributes:'));
    if (ba) {
      if (/\d\/\d/.test(ba)) errors.push('time signature in Basic Attributes (0/1000 upstream)');
      if (!BA_RE.test(ba)) {
        if (BA_GENRE_ONLY_RE.test(ba)) warnings.push(`Basic Attributes has no bpm/key facts: "${ba.slice(0, 70)}"`);
        else errors.push(`Basic Attributes off-pattern: "${ba.slice(0, 70)}"`);
      }
    }

    if (errors.length === 0 && warnings.length === 0) { clean++; continue; }
    if (errors.length > 0) failed++; else warned++;
    const tag = errors.length > 0 ? 'FAIL' : 'WARN';
    console.log(`${tag}  ${name}`);
    for (const e of errors) console.log(`      - ${e}`);
    for (const w of warnings) console.log(`      ~ ${w}`);
  }
}

console.log(`\n${files} files: ${clean} clean, ${warned} warned, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);

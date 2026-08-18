// check-moss-genre.ts — guards the genre picker's two known failure modes.
//
//   cd server && npx tsx scripts/check-moss-genre.ts
//
// There is no test runner in server/, so this is a plain script: run it after
// touching GENRE_HINTS or pickGenre. Exit code is the result.
//
// Genre is the single biggest lever on MM3 prompt adherence — Rob's ear test
// identified a generic "Rock" label as the failure mode of the losing arm — and
// the picker has now been wrong in two different ways, so both are pinned here:
//
//   1. ORDERING. An umbrella term reached before a subgenre collapses everything
//      to it. Barry's first version returned on the first hit in MOSS's own
//      Basic Attributes line, turning every track into `Rock`.
//   2. WEAK EVIDENCE. A specific-sounding word used as a production adjective.
//      "blends vocal into ambient space" in a real Daft Punk capture labelled
//      the track `Ambient`, beating the genuine "house" purely because ambient
//      is the more specific term.

import { arbitrateSidecarGenre, pickGenre } from '../src/services/training/mossCaption.js';

let failed = 0;

function check(name: string, got: string, want: string): void {
  const ok = got === want;
  if (!ok) failed++;
  console.log(`${ok ? 'OK  ' : 'FAIL'}  ${name.padEnd(44)} ${got}${ok ? '' : `   (want ${want})`}`);
}

// ── 1. ordering: specific beats umbrella (Barry's validated rock family) ──
check('emo + punk rock beat umbrella Rock', pickGenre('emo energy', 'punk rock', 'Rock'), 'Punk Rock');
check('pop-punk beats punk beats rock', pickGenre('pop punk', 'rock', 'Rock'), 'Pop-Punk');
check('umbrella used when nothing better', pickGenre('a rock song', '', 'Rock'), 'Rock');
check('electro house beats house', pickGenre('pulsing electro house', '', 'Electronic'), 'Electro House');

// ── 2. weak evidence: production prose must not name the genre ────────────
check('"ambient space" is not Ambient', pickGenre('vocal in ambient space', '', 'Pop'), 'Pop');
check('"ambient textures" is not Ambient', pickGenre('washed ambient textures', '', 'Pop'), 'Pop');
check('"ambient pads" is not Ambient', pickGenre('soft ambient pads', '', 'Pop'), 'Pop');
check('genuine ambient still wins', pickGenre('a slow ambient piece', '', 'Pop'), 'Ambient');
check('"classical guitar" is not Classical', pickGenre('nylon classical guitar', '', 'Folk'), 'Folk');
check('"orchestral swells" is not Classical', pickGenre('orchestral swells behind', '', 'Pop'), 'Pop');
check('genuine classical still wins', pickGenre('a classical arrangement', '', 'Pop'), 'Classical');
check('"blues scale" is not Blues', pickGenre('guitar uses the blues scale', '', 'Rock'), 'Rock');
check('genuine blues still wins', pickGenre('a slow blues shuffle', '', 'Rock'), 'Blues');

// ── 3. fallbacks ─────────────────────────────────────────────────────────
check('dataset genre when nothing matches', pickGenre('', '', 'Shoegaze'), 'Shoegaze');
check('final default when nothing at all', pickGenre('', '', ''), 'Alternative Rock');

// ── 4. both observed sources are scanned, not just MOSS ──────────────────
// MOSS's Daft Punk capture names no genre anywhere in its body; the sidecar
// caption is what carries "french house". If only MOSS were scanned this would
// fall through to the weak dataset field.
check('sidecar caption counts as observed', pickGenre('layered synth arpeggios', 'french house loops', 'Electronic'), 'French House');

// ── 5. the genre-first prompt's vocabulary must not collapse to umbrellas ─
// MOSS's line-1 genre now names these outright (americanfootball, 2026-08-18);
// before they were hints, each scanned as its umbrella — Barry's bug returned.
check('post-rock is not Rock', pickGenre('post-rock, ambient rock', '', 'Rock'), 'Post-Rock');
check('midwest emo is not Emo', pickGenre('midwest emo guitars', '', 'Rock'), 'Midwest Emo');
check('post-punk revival is not Punk Rock', pickGenre('indie rock, post-punk revival', '', ''), 'Post-Punk');
check('garage rock is not Rock', pickGenre('strong garage rock influences', '', ''), 'Garage Rock');
check('indie pop is not Pop', pickGenre('indie pop, dream pop', '', 'Indie'), 'Indie Pop');
check('indie folk is not Folk', pickGenre('acoustic indie folk', '', ''), 'Indie Folk');
// \b treats the hyphen as a boundary, so bare \bsoul\b sees "neo-soul".
check('neo-soul is not Soul', pickGenre('smooth neo-soul groove', '', ''), 'Neo-Soul');

// ── 6. sidecar arbitration (labeling Pass A seeds the rip tag first) ──────
// The provider's claim and the sidecar's current value go into ONE scan, so
// hint order arbitrates across sources; no hint match trusts the provider's
// explicit line over the tag; nothing observed writes nothing — NEVER the
// pickGenre default, which must not be able to reach a sidecar.
const arb = (providerGenre: string, providerCaption: string, existing: string) =>
  arbitrateSidecarGenre({ providerGenre, providerCaption, existing }) ?? '(no write)';
check('provider subgenre beats coarse rip tag', arb('indie pop, dream pop', '', 'Indie'), 'Indie Pop');
check('specific user tag beats provider umbrella', arb('indie rock', '', 'Midwest Emo'), 'Midwest Emo');
// 'zeuhl' shares no words with any hint — a genuinely unrecognised claim.
check('unrecognised provider genre still wins', arb('zeuhl', '', 'Indie'), 'zeuhl');
check('caption prose alone can name it', arb('', 'a driving punk rock tune', 'Indie'), 'Punk Rock');
check('nothing observed writes nothing', arb('', '', 'Indie'), '(no write)');
check('no default can reach the sidecar', arb('', '', ''), '(no write)');

console.log(failed ? `\n${failed} FAILED` : '\nall checks passed');
process.exit(failed ? 1 : 0);

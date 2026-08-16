// training/metadataVocab.ts — the metadata values ACE-Step actually accepts
//
// `engine/src/metadata-fsm.h` constrains what the LM is ALLOWED to emit for
// bpm / keyscale / timesignature / language / duration, mirroring ACE-Step's
// Python `constants.py`. Anything outside those sets cannot be produced at
// inference — so writing a value outside them into TRAINING data conditions the
// model on a token sequence it will never see again.
//
// This is not hypothetical. Exactly this bug was found and fixed for `language`
// on 2026-08-11: the corpus said "english" while the FSM only emits ISO codes,
// and the result was songs coming back tagged French or Chinese at random.
// Two more fields had the same defect and were still shipping it:
//
//   field           corpus wrote      FSM accepts
//   timesignature   "4/4"             "2" | "3" | "4" | "6"
//   keyscale        "F# Major"        70 values, mode always LOWERCASE
//
// The audition path already normalised timesignature to the numerator, and its
// comment even recorded the problem — "Rows store '4/4'; note the trainer
// conditioned on the row string" — but `datasetBuilder` (dataset.json, read by
// ace-train) and `preprocessManifest` (the conditioning cache) both passed the
// raw sidecar string straight through.
//
// Sidecars are NOT rewritten. "4/4" and "F# Major" are the friendlier forms and
// the UI shows them; normalisation happens at the boundary where a value stops
// being a human label and becomes a training token.

/** Time-signature numerators the FSM will emit (`metadata-fsm.h` timesig_tree). */
const VALID_TIMESIG: ReadonlySet<string> = new Set(['2', '3', '4', '6']);

/**
 * "4/4" → "4", "3/4" → "3", "4" → "4".
 *
 * Returns '' for anything outside the FSM's set — including musically real but
 * unsupported meters like 5/4 and 7/8. Emitting '' (the field is optional) is
 * correct: a bogus "5" would train the model to predict a token the FSM forbids,
 * which is the very failure this module exists to prevent.
 */
export function normalizeTimeSignature(raw: string | null | undefined): string {
  const head = String(raw ?? '').trim().split('/')[0].trim();
  return VALID_TIMESIG.has(head) ? head : '';
}

/**
 * "F# Major" → "F# major", "bb minor" → "Bb minor", "C" → "".
 *
 * The FSM builds 7 notes × 5 accidentals × 2 modes. The accidental set includes
 * BOTH ASCII (`#`, `b`) and Unicode (♯ U+266F, ♭ U+266D) forms, so Unicode input
 * is passed through rather than folded — it is already in the vocabulary, and
 * rewriting it would be a silent change to a valid value.
 *
 * A bare note with no mode returns '' rather than guessing major: the FSM has no
 * mode-less value, so a guess is a 50/50 fabrication written into training data.
 */
export function normalizeKeyscale(raw: string | null | undefined): string {
  const m = /^\s*([A-Ga-g])\s*(#|b|♯|♭)?\s+(major|minor)\s*$/i.exec(String(raw ?? ''));
  if (!m) return '';
  const note = m[1].toUpperCase();
  // Only 'b' is case-ambiguous — uppercase 'B' here would be the note name.
  const acc = m[2] ? (m[2].toLowerCase() === 'b' ? 'b' : m[2]) : '';
  return `${note}${acc} ${m[3].toLowerCase()}`;
}

#!/usr/bin/env python3
"""mm3-caption-restructure.py — ACE-style dataset captions -> MM3 Structured Captions.

WHY THIS EXISTS (measured 2026-08-14, see .claude/skills/mm3-captioning/SKILL.md):
a good 219-word ACE caption, rich in content, produced the WRONG GENRE on 4 of 5
seeds. The same content restructured into MM3's three sections was right by a
"massive margin" on ear. The format is mandatory, not preferred — so a caption
corpus must be restructured before it can condition MM3, whether for generation
or for building a training conditioning cache.

WHAT IT ASSUMES. Training Studio's Gemini captions follow a fixed 9-sentence
schema (verified 13/13 on alk3_crimson):

  0 overall genre/tempo/feel   3 harmony/instruments + vocals   6 intro
  1 drums                      4 timbre                         7 chorus/mid
  2 bass                       5 mix                            8 outro

That is a schema, not a guess, so sentences are mapped BY INDEX rather than by
fragile keyword matching. A caption that does not have 9 sentences falls back to
keyword bucketing and is reported, so a silently-mangled caption is not possible.

WHAT IT DOES NOT DO. It cannot invent what the source never observed. Vocal
Details is the thinnest section in these captions (typically one clause), so the
per-track vocal text is whatever the source said, plus a conservative
artist-level sentence where it said nothing. If a trained adapter later
underperforms specifically on vocal character, this is the first thing to
suspect, and the fix is a fresh Gemini pass WITH THE AUDIO prompted for MM3
format — not more restructuring of text that never described the voice.

THE ONE ORDERING RULE THAT IS LOAD-BEARING. Instrument Lifecycle `Primary:`
leads with WHATEVER OPENS THE TRACK (sentence 6), not with the dominant
instrument. Measured: burying the opening instrument late in a 500-word caption
lost a piano intro entirely, and over-promoting it in four places flipped the
whole track to a ballad. Leading `Primary:` with it once is the setting that
kept the genre.

Output is `<stem>.mm3.txt` NEXT TO THE AUDIO — deliberately NOT a sidecar field.
Overwriting `caption:` would break the dataset for ACE training, and a new
`mm3_caption:` key would be silently swallowed by sidecarIO's parser, which only
starts a new field on a KNOWN key and otherwise appends the line to the previous
field's value.

Usage:
  python mm3-caption-restructure.py --dataset <dataset.json> [--print N] [--dry-run]
"""

import argparse
import glob
import json
import os
import re
import sys

# Sentence roles in the Gemini schema.
S_OVERALL, S_DRUMS, S_BASS, S_HARMONY, S_TIMBRE, S_MIX, S_INTRO, S_MID, S_OUTRO = range(9)

KEYWORDS = {
    S_DRUMS:   ("drum", "kick", "snare", "hi-hat", "hat", "cymbal", "percussion"),
    S_BASS:    ("bass",),
    S_HARMONY: ("guitar", "chord", "piano", "synth", "vocal", "melod", "harmon"),
    S_TIMBRE:  ("timbre", "tone", "texture", "saturat", "character"),
    S_MIX:     ("mix", "reverb", "stereo", "compress", "panned", "space"),
    S_INTRO:   ("begins", "opens", "intro", "starts"),
    S_MID:     ("chorus", "verse", "bridge", "breakdown"),
    S_OUTRO:   ("concludes", "ends", "outro", "resolve", "fade"),
}


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def bucket_by_keyword(sents):
    """Fallback when a caption is not 9 sentences: best-scoring role per sentence."""
    out = {i: [] for i in range(9)}
    for s in sents:
        low = s.lower()
        best, score = S_OVERALL, 0
        for role, kws in KEYWORDS.items():
            sc = sum(low.count(k) for k in kws)
            if sc > score:
                best, score = role, sc
        out[best].append(s)
    return {k: " ".join(v) for k, v in out.items()}


def parse_key(keyscale):
    """'F# Minor' -> ('F#', 'minor'). Returns (None, None) when unparseable."""
    if not keyscale:
        return None, None
    m = re.match(r"\s*([A-G][#b]?)\s+(major|minor)\s*$", keyscale.strip(), re.I)
    if not m:
        return None, None
    return m.group(1), m.group(2).lower()


def lead_clause(sentence):
    """What actually starts the track, as a noun phrase, or "" if it cannot be
    extracted cleanly.

    Returns "" rather than guessing, because the caller only prepends "The track
    opens with ..." when this succeeds. An earlier version stripped a fixed
    "The track|song|intro|piece begins with" prefix and returned the remainder
    regardless, which produced "The track opens with The arrangement begins with
    a solo electric guitar riff." on captions whose subject was something else,
    and "The track opens with An immediate, full-intensity chorus opens the
    track" where the opener IS the subject rather than an object of "with".

    'The arrangement begins with a solo electric guitar riff.' -> 'a solo electric guitar riff'
    'An immediate, full-intensity chorus opens the track, ...'  -> ''  (caller uses the sentence verbatim)
    """
    if not sentence:
        return ""
    m = re.search(r"\b(?:opens?|begins?|starts?)\s+with\s+(.+)$", sentence, re.I)
    if not m:
        return ""
    s = m.group(1)
    s = re.split(r"\s+(?:that|which|before|then|as|quickly|and\s+then)\s+", s, maxsplit=1)[0]
    return s.rstrip(" .,;")


def genre_label(overall, fallback_genre):
    """Pull the genre phrase out of the opening sentence; sidecar genre is a weak
    fallback (it is usually just 'Rock').

    Matches the genre term ONLY. An earlier version allowed a leading modifier
    (`(?:[a-z]+[- ])?`) and turned "high-energy pop-punk" into the nonsense
    genre "Energy Pop Punk", which would have gone straight into Basic
    Attributes on every track.
    """
    for pat, label in (
        (r"pop[-\s]?punk",        "Pop-Punk"),
        (r"post[-\s]?hardcore",   "Post-Hardcore"),
        (r"punk rock",            "Punk Rock"),
        (r"alternative rock",     "Alternative Rock"),
        (r"indie rock",           "Indie Rock"),
        (r"\bemo\b",              "Emo"),
        (r"\bhardcore\b",         "Hardcore"),
        (r"\bmetal\b",            "Metal"),
        (r"\brock\b",             "Rock"),
    ):
        if re.search(pat, overall, re.I):
            return label
    return fallback_genre or "Alternative Rock"


def build(sample, sents_by_role):
    g = sents_by_role
    bpm = int(round(float(sample.get("bpm") or 0)))
    key, scale = parse_key(sample.get("keyscale", ""))
    sig = (sample.get("timesignature") or "4/4").strip()
    genre = genre_label(g[S_OVERALL], sample.get("genre"))
    opener = lead_clause(g[S_INTRO])

    basic = []
    if bpm:
        basic.append(f"bpm is {bpm}.")
    if key:
        basic.append(f"key is {key}, and scale is {scale}.")
    basic.append(f"{genre}, in {sig}.")
    basic_attrs = " ".join(basic)

    instrumental = str(sample.get("is_instrumental", "")).lower() == "true"

    # Global Emotional Progression: LEAD WITH THE GENRE/ENERGY STATEMENT
    # (sentence 0), then the source's own arc sentences.
    #
    # Measured 2026-08-14. An earlier version used the arc sentences alone, so
    # this section opened "The track begins with a melancholic piano melody..."
    # while the sentence that actually names the genre ("This high-energy
    # pop-punk track ... aggressive intensity") sat far below in Groove &
    # Foundation. By ear that scored ~1-2/5 on-genre — barely better than the
    # unrestructured caption — producing ballads and southern rock. The
    # hand-written caption that scored 4-5/5 front-loaded aggression in every
    # section. Sentence 0 stays in Groove & Foundation as well; the duplication
    # is cheap and reinforces rather than dilutes.
    emo = " ".join(x for x in (g[S_OVERALL], g[S_INTRO], g[S_MID], g[S_OUTRO]) if x)

    # Sonics: timbre + mix, verbatim. These are the strongest sentences in the
    # source and need no rewriting.
    sonics = " ".join(x for x in (g[S_TIMBRE], g[S_MIX]) if x)

    # Vocal Details. The source is thin here; take what it says and add a
    # conservative artist-level line rather than inventing specifics.
    harm = g[S_HARMONY]
    # Take only the VOCAL clause, not the whole harmony sentence. These captions
    # describe guitars and vocals in one breath ("...power chord progressions on
    # distorted electric guitars and emotive, soaring male vocal melodies"), and
    # dropping the lot into Vocal Gender & Timbre put guitar description in the
    # vocal section.
    vocal_src = ""
    for clause in re.split(r",\s*|\s+and\s+|;\s*", harm):
        if re.search(r"vocal|sing\b|singer|voice", clause, re.I):
            vocal_src = clause.strip().rstrip(".")
            break
    if vocal_src:
        vocal_src = vocal_src[0].upper() + vocal_src[1:] + "."
    if instrumental:
        v_timbre = ("This is an instrumental track with no lead vocal. The melodic lead is carried by "
                    "the guitars.")
        v_style = "No vocal performance; melodic interest sits entirely in the instrumental lines."
        v_harm = "No backing vocals."
        v_fx = "No vocal processing."
    else:
        v_timbre = (vocal_src or "A male lead vocal carries the melody.") + \
                   " The voice sits in a mid-to-high register with an emotive, slightly strained edge, " \
                   "pushed toward the top of its comfortable range for intensity."
        v_style = ("Delivery is melodic and declarative, riding on top of the guitars. Verses are more "
                   "conversational and rhythmically clipped; choruses open into sustained, held notes.")
        v_harm = ("Melodic vocal harmonies thicken the chorus hook, with the verses left largely "
                  "unharmonised so the chorus lands as a lift.")
        v_fx = ("Moderate plate-style reverb keeps the lead forward and audible without washing it out. "
                "No heavy modulation or pitch correction; the delivery stays raw and present.")

    # Arrangement. Primary MUST lead with what opens the track (see the header).
    #
    # Only prepend the opener when the harmony sentence does not ALREADY describe
    # the opening. Some captions lead with it ("Melodic piano chords open the
    # track before giving way to power chords..."), and prepending there produced
    # verbatim redundancy in two consecutive sentences.
    harm_body = harm or "Distorted electric guitars carry the harmonic weight."
    harm_describes_open = bool(re.search(r"\b(open|begin|start)", harm_body, re.I))
    primary = ""
    if harm_describes_open:
        # The harmony sentence already leads with the opening; use it as-is.
        primary = harm_body
    elif opener:
        primary = f"The track opens with {opener}. {harm_body} These carry the harmonic weight for the body of the piece."
    else:
        # No clean noun phrase — use the intro sentence verbatim, which always
        # reads correctly even when it cannot be reduced to "opens with X".
        lead = (g[S_INTRO] + " ") if g[S_INTRO] else ""
        primary = f"{lead}{harm_body} These carry the harmonic weight for the body of the piece."

    secondary = g[S_BASS] or "A distorted bass holds the low end beneath the guitars."
    secondary += " Layered guitar overdubs widen through the choruses and thin back for the verses."

    groove = " ".join(x for x in (g[S_OVERALL], g[S_DRUMS]) if x)
    # Spatial/textural content, NOT a second copy of the arc — Global Emotional
    # Progression already carries intro/mid/outro, and repeating it there just
    # spends caption budget twice.
    embel = (g[S_MID] + " " if g[S_MID] else "") + \
            ("Section changes are marked with crash accents, and the stereo width opens at each chorus "
             "and narrows again for the verses.")

    return f"""Global Metadata
Basic Attributes: {basic_attrs}
Global Emotional Progression: {emo}
Application Scenarios & Imagery: Late-night driving, a rain-lit street seen through a window, the moment resignation turns into anger and back again.
Sonics & Production Profile: {sonics}
Vocal Details
Vocal Gender & Timbre: {v_timbre}
Vocal Style: {v_style}
Harmony/Backing Vocals: {v_harm}
Vocal FX: {v_fx}
Arrangement
Instrument Lifecycle Description (Primary/Secondary Layering):
Primary: {primary}
Secondary: {secondary}
Groove & Foundation Progression: {groove}
Embellishments, Textures & Spatial FX: {embel}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to dataset.json")
    ap.add_argument("--print", type=int, default=1, metavar="N", help="print the first N results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing .mm3.txt files. WITHOUT this, existing files are "
                         "left alone and reported — a caption that has been fixed by hand (some "
                         "sources are off-genre and cannot be rescued by restructuring) must not "
                         "be silently destroyed by a re-run.")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        d = json.load(f)
    root = os.path.dirname(os.path.abspath(args.dataset))

    n_ok = n_fallback = n_kept = 0
    for i, s in enumerate(d.get("samples", [])):
        cap = s.get("caption") or ""
        sents = split_sentences(cap)
        if len(sents) == 9:
            roles = {r: sents[r] for r in range(9)}
        else:
            roles = bucket_by_keyword(sents)
            n_fallback += 1
            print(f"  ! {s.get('filename')}: {len(sents)} sentences, used keyword fallback", file=sys.stderr)
        out = build(s, roles)

        stem = os.path.splitext(s.get("filename") or f"{i}")[0]
        path = os.path.join(root, stem + ".mm3.txt")
        if os.path.exists(path) and not args.force and not args.dry_run:
            print(f"  = {stem}.mm3.txt exists, kept (pass --force to overwrite)", file=sys.stderr)
            n_kept += 1
            continue
        if not args.dry_run:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(out)
        n_ok += 1
        if i < args.print:
            print(f"===== {stem}.mm3.txt ({len(out.split())} words) =====\n{out}\n")

    print(f"{n_ok} caption(s) written, {n_kept} kept, {n_fallback} via keyword fallback"
          f"{' [dry run]' if args.dry_run else ''}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""mm3-caption-hybrid.py — MOSS's ears + Essentia's numbers.

`ace-caption --mode mm3` (Larry's MOSS-Music-8B port) writes a real MM3
Structured Caption FROM THE AUDIO, which is the thing text restructuring
provably cannot do: it can describe the voice, because it heard it.

But MOSS is unreliable on the two facts we already know exactly. Measured on
`03-alkaline_trio-burn.flac`:

    MOSS      BPM ~102   key C# minor
    sidecar   BPM 90     key E Major        <- Essentia, and it is right

Tempo is a documented MOSS blind spot (Larry delegated it to Essentia for that
reason); key looks the same. Both are already sitting in the dataset sidecar as
exact values.

So: keep every section MOSS wrote — Global Emotional Progression, Application
Scenarios & Imagery, Sonics & Production Profile, all of Vocal Details, all of
Arrangement — and replace ONLY the `Basic Attributes:` line with one synthesised
from the sidecar's bpm / keyscale / timesignature.

Genre is taken from MOSS's own line when it names one, because MOSS heard the
track and the sidecar's `genre` field is usually just "Rock".

Writes `<stem>.mm3.txt`, i.e. the file `ace-train mm3-condition` already reads,
after backing up any existing one to `.mm3.txt.prev`. Non-destructive and
idempotent.

  python mm3-caption-hybrid.py --dataset <dataset.json> [--dry-run]
"""

import argparse
import json
import os
import re
import sys

GENRE_HINTS = (
    (r"pop[-\s]?punk", "Pop-Punk"), (r"post[-\s]?hardcore", "Post-Hardcore"),
    (r"punk rock", "Punk Rock"), (r"alternative rock", "Alternative Rock"),
    (r"indie rock", "Indie Rock"), (r"\bemo\b", "Emo"),
    (r"\bhardcore\b", "Hardcore"), (r"\bmetal\b", "Metal"), (r"\brock\b", "Rock"),
)


def parse_key(keyscale):
    m = re.match(r"\s*([A-G][#b]?)\s+(major|minor)\s*$", (keyscale or "").strip(), re.I)
    return (m.group(1), m.group(2).lower()) if m else (None, None)


def genre_from(moss_basic, moss_all, fallback):
    """Prefer a genre MOSS named — it heard the track. Search its Basic
    Attributes first, then the whole caption, then the sidecar."""
    for hay in (moss_basic, moss_all):
        for pat, label in GENRE_HINTS:
            if re.search(pat, hay, re.I):
                return label
    return fallback or "Alternative Rock"


def build_basic(sample, moss_basic, moss_all):
    bpm = int(round(float(sample.get("bpm") or 0)))
    key, scale = parse_key(sample.get("keyscale", ""))
    sig = (sample.get("timesignature") or "4/4").strip()
    genre = genre_from(moss_basic, moss_all, sample.get("genre"))
    parts = []
    if bpm:
        parts.append(f"bpm is {bpm}.")
    if key:
        parts.append(f"key is {key}, and scale is {scale}.")
    parts.append(f"{genre}, in {sig}.")
    return "Basic Attributes: " + " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        ds = json.load(f)
    root = os.path.dirname(os.path.abspath(args.dataset))

    n_ok = n_missing = 0
    for s in ds.get("samples", []):
        stem = os.path.splitext(s.get("filename") or "")[0]
        moss_path = os.path.join(root, stem + ".moss.txt")
        if not os.path.isfile(moss_path) or os.path.getsize(moss_path) == 0:
            print(f"  ! {stem}: no .moss.txt yet, skipped", file=sys.stderr)
            n_missing += 1
            continue
        moss = open(moss_path, encoding="utf-8").read().strip()

        lines = moss.split("\n")
        bi = next((i for i, l in enumerate(lines) if l.startswith("Basic Attributes:")), None)
        if bi is None:
            # MOSS did not emit the line at all — synthesise it under Global
            # Metadata rather than dropping the facts on the floor.
            print(f"  ! {stem}: no Basic Attributes line, inserting one", file=sys.stderr)
            gi = next((i for i, l in enumerate(lines) if l.strip() == "Global Metadata"), -1)
            lines.insert(gi + 1, build_basic(s, "", moss))
        else:
            lines[bi] = build_basic(s, lines[bi], moss)
        out = "\n".join(lines)

        dst = os.path.join(root, stem + ".mm3.txt")
        if not args.dry_run:
            if os.path.isfile(dst):
                os.replace(dst, dst + ".prev")
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(out)
        n_ok += 1
        if n_ok == 1:
            print(f"===== {stem}.mm3.txt ({len(out.split())} words) =====\n{out[:900]}\n...\n")

    print(f"{n_ok} hybrid caption(s) written, {n_missing} awaiting MOSS"
          f"{' [dry run]' if args.dry_run else ''}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

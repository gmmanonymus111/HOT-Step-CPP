#!/usr/bin/env python3
"""Convert YuE2's qwen.tiktoken (a raw tiktoken rank file) into the GPT-2
byte-level vocab.json + merges.txt convention that engine/src/bpe.h loads,
plus sidecar files describing what bpe.h's format cannot represent
(special tokens, structural control ids, the codec id space).

Reference source read to derive this (installed package, not modified):
    K:/yue2/.venv/Lib/site-packages/yue2/tokenization_yue2.py
    K:/yue2/.venv/Lib/site-packages/yue2/protocol.py

Usage:
    python yue2-tokenizer-convert.py --tiktoken K:/yue2/models/YuE2-3B/qwen.tiktoken \
        --tokenization-src K:/yue2/.venv/Lib/site-packages/yue2/tokenization_yue2.py \
        --protocol-src K:/yue2/.venv/Lib/site-packages/yue2/protocol.py \
        --out D:/Ace-Step-Latest/hot-step-cpp/models/yue2/tokenizer

CPU-only; does no torch/model work at all, just text-file munging.
"""
import argparse
import ast
import base64
import json
import re
import sys
from pathlib import Path

# ── GPT-2 byte<->unicode encoder (standard "bytes_to_unicode" table). ──────
# This is the exact table build_byte_encoder() constructs in engine/src/bpe.h:
# printable ASCII/Latin-1 stay as themselves, every other byte value is
# remapped to a private codepoint starting at 256, in ascending byte order.
def gpt2_byte_encoder():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(0xA1, 0xAC + 1)) + \
         list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def encode_token_bytes(byte_encoder, token_bytes: bytes) -> str:
    return "".join(byte_encoder[b] for b in token_bytes)


# ── Parse qwen.tiktoken (tiktoken rank file: "<b64 token> <rank>" per line) ─
def load_tiktoken_ranks(path: Path) -> dict[bytes, int]:
    ranks = {}
    for line in path.read_bytes().splitlines():
        if not line:
            continue
        tok_b64, rank = line.split()
        ranks[base64.b64decode(tok_b64)] = int(rank)
    return ranks


# ── Recover the merge that produced each multi-byte token, mirroring the ──
# standard tiktoken "_educational.bpe()" reconstruction algorithm (also used
# by llama.cpp / HF converters for tiktoken-based BPE vocabularies such as
# Qwen's): re-run byte-pair merging on the token's raw bytes, restricted to
# only merges with a strictly lower rank than the token itself. For a vocab
# built by the standard BPE training loop this always collapses to exactly
# two parts, and those two parts are the merge that created this token.
def recover_merge(ranks: dict[bytes, int], token: bytes, max_rank: int):
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i in range(len(parts) - 1):
            pair = parts[i] + parts[i + 1]
            r = ranks.get(pair)
            if r is not None and (min_rank is None or r < min_rank):
                min_idx, min_rank = i, r
        if min_rank is None or min_rank >= max_rank:
            return parts
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]


# ── Pull the pre-tokenizer regex and the special-token list straight out of
# the installed tokenization_yue2.py source, instead of hardcoding them here,
# so this script tracks the installed package rather than a copy that can
# drift. Both are simple, self-contained literals/statements in that file.
def extract_pattern_and_specials(tokenization_src: Path):
    src = tokenization_src.read_text(encoding="utf-8")

    m = re.search(r'pattern\s*=\s*r"([^"]*)"', src)
    if not m:
        raise RuntimeError(f"could not find pre-tokenizer pattern in {tokenization_src}")
    pattern = m.group(1)

    # The specials list is built by three consecutive statements:
    #   specials = ["<|endoftext|>", ...]
    #   specials += [f"<extra_{i}>" for i in range(200)]
    #   specials[204:206] = ["<abc>", "</abc>"]
    # Pull each statement by line prefix and dedent, rather than trying to
    # span them with one regex (source indentation makes that fragile).
    lines = src.splitlines()
    stmt_lines = []
    for prefix in ("specials = [", "specials += [", "specials["):
        found = None
        for line in lines:
            if line.strip().startswith(prefix):
                found = line.strip()
                break
        if found is None:
            raise RuntimeError(f"could not find statement starting with {prefix!r} in {tokenization_src}")
        stmt_lines.append(found)
    ns: dict = {}
    exec(compile(ast.parse("\n".join(stmt_lines)), "<specials>", "exec"), {}, ns)
    specials = ns["specials"]
    return pattern, specials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiktoken", required=True, type=Path, help="path to qwen.tiktoken")
    ap.add_argument("--tokenization-src", required=True, type=Path,
                     help="path to the installed yue2/tokenization_yue2.py")
    ap.add_argument("--protocol-src", type=Path, default=None,
                     help="path to the installed yue2/protocol.py (for the codec/control id map)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    byte_encoder = gpt2_byte_encoder()

    ranks = load_tiktoken_ranks(args.tiktoken)
    n_ordinary = len(ranks)
    print(f"[convert] loaded {n_ordinary} ordinary tokens from {args.tiktoken}", file=sys.stderr)

    pattern, specials = extract_pattern_and_specials(args.tokenization_src)
    print(f"[convert] pre-tokenizer regex: {pattern}", file=sys.stderr)
    print(f"[convert] {len(specials)} special tokens", file=sys.stderr)

    # ── vocab.json: {gpt2-byte-encoded token string: id}, ids 0..n_ordinary-1
    id_to_bytes = {r: b for b, r in ranks.items()}
    vocab = {}
    for rank in range(n_ordinary):
        tb = id_to_bytes[rank]
        vocab[encode_token_bytes(byte_encoder, tb)] = rank
    (args.out / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=0, separators=(",", ":")), encoding="utf-8"
    )

    # ── merges.txt: one "part_a part_b" per line, in rank order (line index
    # doubles as merge priority, exactly how bpe.h's load_bpe_from_files reads it).
    merge_lines = ["#version: 0.2"]
    n_derived = 0
    n_leaf = 0
    for rank in range(n_ordinary):
        tb = id_to_bytes[rank]
        if len(tb) == 1:
            n_leaf += 1
            continue
        parts = recover_merge(ranks, tb, rank)
        if len(parts) != 2:
            raise RuntimeError(
                f"token rank={rank} bytes={tb!r} did not reduce to a single merge "
                f"(got {len(parts)} parts: {parts!r}); tiktoken->GPT2 merge recovery assumption broke"
            )
        a, b = parts
        merge_lines.append(
            f"{encode_token_bytes(byte_encoder, a)} {encode_token_bytes(byte_encoder, b)}"
        )
        n_derived += 1
    (args.out / "merges.txt").write_text("\n".join(merge_lines) + "\n", encoding="utf-8")
    print(f"[convert] wrote {n_leaf} leaf byte tokens + {n_derived} derived merges", file=sys.stderr)

    # ── special_tokens.json: the 208 tiktoken specials, plus (if protocol.py
    # was given) the structural control ids and codec id space that sit past
    # the specials and are NOT text/BPE tokens at all — bpe.h's vocab.json/
    # merges.txt convention has no representation for these; see the summary
    # in docs/plans/yue2/00-oracle-pin.md section 6 for what would need to change.
    special_tokens = {
        "note": "ids 0.." + str(n_ordinary - 1) + " are the ordinary BPE vocab (vocab.json). "
                "These 208 specials occupy the ids immediately after it, in list order.",
        "base_vocab_size": n_ordinary,
        "tokens": [{"id": n_ordinary + i, "text": s} for i, s in enumerate(specials)],
    }
    if args.protocol_src and args.protocol_src.exists():
        psrc = args.protocol_src.read_text(encoding="utf-8")

        def const(name, count=1):
            m = re.search(rf"\b{name}\b(?:\s*,\s*(\w+))?\s*=\s*([0-9,\s]+)", psrc)
            return m

        control_ids = {}
        for m in re.finditer(r"^([A-Z_, ]+)\s*=\s*([0-9, ]+)$", psrc, re.M):
            names = [n.strip() for n in m.group(1).split(",")]
            vals = [v.strip() for v in m.group(2).split(",")]
            if len(names) == len(vals) and all(v.lstrip("-").isdigit() for v in vals):
                for n, v in zip(names, vals):
                    control_ids[n] = int(v)
        special_tokens["protocol_ids"] = control_ids
        special_tokens["protocol_note"] = (
            "MUSIC_START/MUSIC_END/CODEC_OFFSET.. and LATENT_*/PAD ids are structural "
            "control tokens the LM sampler appends/consumes directly as integers; they "
            "are never produced by tokenizing text and have no BPE string form at all."
        )

    (args.out / "special_tokens.json").write_text(
        json.dumps(special_tokens, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── pretokenizer.json: the exact regex string + normalization the
    # reference tokenizer applies before tiktoken's encode_ordinary runs.
    pretok = {
        "regex": pattern,
        "regex_flavor": "PCRE-ish Unicode-property regex (as used by tiktoken); "
                        "\\p{L}/\\p{N} are real Unicode General Category classes, "
                        "not the ASCII/block-range approximation in bpe.h's is_letter/is_digit.",
        "normalization": "NFC (unicodedata.normalize('NFC', text)) applied to the whole "
                          "input string before pre-tokenization/BPE. bpe.h currently does no "
                          "normalization at all.",
        "special_token_handling": "tokenization_yue2.py calls tiktoken's encode_ordinary(), "
                                   "which NEVER special-cases any special-token substring found "
                                   "inside the text (specials are only ever inserted "
                                   "programmatically as raw ints by protocol.py). This differs "
                                   "from bpe.h's bpe_encode(), which hardcodes a literal-substring "
                                   "scan for '<|endoftext|>' and substitutes eos_id for it.",
    }
    (args.out / "pretokenizer.json").write_text(
        json.dumps(pretok, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[convert] wrote vocab.json, merges.txt, special_tokens.json, pretokenizer.json to {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

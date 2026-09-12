#!/usr/bin/env python3
"""Parity harness: reference YuE2 tokenizer vs. a faithful pure-python port
of engine/src/bpe.h's algorithm, run over the converted vocab.json/merges.txt.

Two independent encoders are run over the same corpus:
  1. REFERENCE: yue2.tokenization_yue2.YuE2TextTokenizer (installed package,
     imports tiktoken), i.e. NFC-normalize + tiktoken.encode_ordinary().
  2. BPE_H_PORT: a line-for-line python port of bpe.h's gpt2_pre_tokenize()
     + encode_chunk()/bpe_merge() + its literal-substring "<|endoftext|>"
     special-case, reading ONLY the converted vocab.json/merges.txt (i.e.
     exactly what the C++ engine would produce with these sidecar files).
     No NFC normalization, no real \\p{L}/\\p{N} — this is what bpe.h
     actually does today, byte for byte.

Every mismatch is reported with both token-id sequences so the diff is
inspectable. CPU-only; does not touch the 3B model weights at all.
"""
import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Windows consoles are frequently cp1252; the corpus is deliberately full of
# non-Latin-1 text (CJK, combining accents, emoji), so force UTF-8 stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ════════════════════════════════════════════════════════════════════════
# Part A: faithful python port of engine/src/bpe.h
# ════════════════════════════════════════════════════════════════════════

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


def is_letter(cp: int) -> bool:
    if (ord('A') <= cp <= ord('Z')) or (ord('a') <= cp <= ord('z')):
        return True
    if cp < 0x80:
        return False
    if 0xC0 <= cp <= 0x024F and cp != 0xD7 and cp != 0xF7:
        return True
    if 0x0370 <= cp <= 0x1FFF:
        return True
    if 0x2C00 <= cp <= 0x2DFF:
        return True
    if 0x3040 <= cp <= 0x9FFF:
        return True
    if 0xAC00 <= cp <= 0xD7AF:
        return True
    if 0xF900 <= cp <= 0xFAFF:
        return True
    if cp >= 0x10000:
        return True
    return False


def is_digit(cp: int) -> bool:
    return ord('0') <= cp <= ord('9')


_WS = {ord(' '), ord('\t'), ord('\n'), ord('\r'), 0x0B, 0x0C, 0xA0, 0x2000, 0x2001, 0x2002, 0x200B}


def is_whitespace(cp: int) -> bool:
    return cp in _WS


def is_newline(cp: int) -> bool:
    return cp == ord('\n') or cp == ord('\r')


def gpt2_pre_tokenize(text: str) -> list[str]:
    """Direct port of bpe.h's gpt2_pre_tokenize(), operating on a list of
    unicode codepoints (equivalent to bpe.h decoding utf8 codepoint-by-codepoint)."""
    s = text
    length = len(s)
    i = 0
    chunks: list[str] = []

    while i < length:
        cp = ord(s[i])

        # Rule 1: contractions 's 't 're 've 'm 'll 'd (case-insensitive ASCII fold)
        if (cp == ord("'") or cp == 0x2019) and i + 1 < length:
            rest = s[i + 1:]
            rlen = length - i - 1
            matched = False

            def try_match(suffix: str) -> bool:
                nonlocal i, matched
                slen = len(suffix)
                if rlen >= slen:
                    seg = rest[:slen]
                    if seg.lower() != suffix:
                        return False
                    if rlen > slen:
                        cp2 = ord(rest[slen])
                        if is_letter(cp2):
                            return False
                    chunks.append(s[i:i + 1 + slen])
                    i = i + 1 + slen
                    matched = True
                    return True
                return False

            for suf in ("ll", "re", "ve", "s", "t", "m", "d"):
                if try_match(suf):
                    break
            if matched:
                continue

        # Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+  (primary letter run)
        if is_letter(cp):
            start = i
            i += 1
            while i < length and is_letter(ord(s[i])):
                i += 1
            chunks.append(s[start:i])
            continue

        if not is_newline(cp) and not is_letter(cp) and not is_digit(cp) and not is_whitespace(cp):
            start = i
            after = i + 1
            if after < length and is_letter(ord(s[after])):
                i = after + 1
                while i < length and is_letter(ord(s[i])):
                    i += 1
                chunks.append(s[start:i])
                continue

        # Rule 3: \p{N}  (single digit at a time)
        if is_digit(cp):
            start = i
            while i < length and is_digit(ord(s[i])):
                i += 1
            for j in range(start, i):
                chunks.append(s[j:j + 1])
            continue

        # Rule 5: \s*[\r\n]+
        if is_newline(cp):
            start = i
            while i < length and is_newline(ord(s[i])):
                i += 1
            chunks.append(s[start:i])
            continue

        # Rule 6: whitespace handling
        if is_whitespace(cp):
            start = i
            ws_end = i + 1
            while ws_end < length and is_whitespace(ord(s[ws_end])) and not is_newline(ord(s[ws_end])):
                ws_end += 1
            followed_by_non_ws = ws_end < length and not is_whitespace(ord(s[ws_end])) and not is_newline(ord(s[ws_end]))
            if followed_by_non_ws and ws_end - start > 1:
                trailing = ws_end - 1
                chunks.append(s[start:trailing])
                i = trailing
                continue

            i = start + 1
            handled = False
            if i < length:
                cp2 = ord(s[i])
                if is_letter(cp2):
                    i += 1
                    while i < length and is_letter(ord(s[i])):
                        i += 1
                    chunks.append(s[start:i])
                    handled = True
                elif is_digit(cp2):
                    chunks.append(s[start:i])
                    handled = True
                elif not is_whitespace(cp2) and not is_newline(cp2):
                    pstart = start
                    while i < length:
                        cp3 = ord(s[i])
                        if is_whitespace(cp3) or is_letter(cp3) or is_digit(cp3):
                            break
                        i += 1
                    while i < length and is_newline(ord(s[i])):
                        i += 1
                    chunks.append(s[pstart:i])
                    handled = True
            if handled:
                continue

            # Trailing whitespace (end of string, or followed by more ws/newline)
            i = ws_end
            while i < length and is_whitespace(ord(s[i])):
                i += 1
            chunks.append(s[start:i])
            continue

        # Rule 4: [^\s\p{L}\p{N}]+[\r\n]*  (punctuation/symbol run)
        start = i
        i += 1
        while i < length:
            cp2 = ord(s[i])
            if is_whitespace(cp2) or is_letter(cp2) or is_digit(cp2) or is_newline(cp2):
                break
            i += 1
        while i < length and is_newline(ord(s[i])):
            i += 1
        chunks.append(s[start:i])

    return chunks


class BpeHPort:
    """Loads the converted vocab.json/merges.txt and reproduces bpe.h's
    BPETokenizer + bpe_encode() exactly, including its literal
    '<|endoftext|>' substring special-case and default EOS append."""

    EOS_ID = 151643

    def __init__(self, tokenizer_dir: Path):
        self.byte_encoder = gpt2_byte_encoder()
        self.vocab: dict[str, int] = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
        self.merge_rank: dict[tuple[str, str], int] = {}
        lines = (tokenizer_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
        rank = 0
        for line in lines:
            if not line or line.startswith("#"):
                continue
            a, b = line.split(" ")
            self.merge_rank[(a, b)] = rank
            rank += 1

    def byte_level_encode(self, text: str) -> str:
        return "".join(self.byte_encoder[b] for b in text.encode("utf-8"))

    def bpe_merge(self, symbols: list[str]) -> list[str]:
        work = list(symbols)
        while len(work) > 1:
            best_rank = None
            best_pos = -1
            for i in range(len(work) - 1):
                r = self.merge_rank.get((work[i], work[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pos = r, i
            if best_pos < 0:
                break
            work[best_pos] = work[best_pos] + work[best_pos + 1]
            del work[best_pos + 1]
        return work

    def encode_chunk(self, chunk: str, ids: list[int]):
        encoded = self.byte_level_encode(chunk)
        symbols = list(encoded)  # each python char here is one GPT2-byte-encoded codepoint
        merged = self.bpe_merge(symbols)
        for piece in merged:
            tid = self.vocab.get(piece)
            if tid is not None:
                ids.append(tid)
            else:
                # bpe.h's byte-level fallback: look up each raw byte's own token
                for raw_byte in piece.encode("utf-8"):
                    single = self.byte_encoder[raw_byte]
                    tid2 = self.vocab.get(single)
                    if tid2 is not None:
                        ids.append(tid2)

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        special = "<|endoftext|>"
        pos = 0
        while pos < len(text):
            found = text.find(special, pos)
            segment = text[pos:] if found < 0 else text[pos:found]
            if segment:
                for chunk in gpt2_pre_tokenize(segment):
                    self.encode_chunk(chunk, ids)
            if found < 0:
                break
            ids.append(self.EOS_ID)
            pos = found + len(special)
        if add_eos:
            ids.append(self.EOS_ID)
        return ids


# ════════════════════════════════════════════════════════════════════════
# Part B: test corpus
# ════════════════════════════════════════════════════════════════════════

def build_corpus() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []

    def add(name, text):
        cases.append((name, text))

    # English prose
    add("en_simple", "The quick brown fox jumps over the lazy dog.")
    add("en_contractions", "I can't believe it's already working, they'll say we've won.")
    add("en_contractions2", "She said she'd rather not, but he wouldn't listen.")
    add("en_possessive", "It's the cat's toy, not the dogs' bone.")
    add("en_long", "Verse one begins quietly, then the chorus swells with distorted guitars and a driving bassline.")

    # Chinese
    add("zh_simple", "你好，世界。")
    add("zh_lyrics", "夜色渐深，霓虹灯闪烁在雨后的街道上。")
    add("zh_mixed", "今天天气很好 today is a good day 2026年9月12日")

    # Japanese
    add("ja_simple", "こんにちは世界")
    add("ja_lyrics", "星空の下で君と歩いた、あの夏の記憶が消えない。")
    add("ja_mixed_kana_kanji", "ひらがなとカタカナと漢字が混ざった文章です。")

    # Korean
    add("ko_simple", "안녕하세요 세계")
    add("ko_lyrics", "밤하늘 아래 우리 둘이 걸었던 그 여름날의 기억이 사라지지 않아")

    # Accented lyrics
    add("fr_accents", "Café, déjà vu, à bientôt, l'été résonne où le cœur bat.")
    add("es_accents", "El niño canta bajo la lluvia, mañana será un día más brillante.")
    add("pt_accents", "Ação, coração, não é fácil, é difícil dizer adeus.")
    add("de_umlaut", "Über den Wolken, für immer, schön wie ein Traumbild.")
    # NFC vs NFD forms of the same visible text (accent as combining char)
    nfc_text = unicodedata.normalize("NFC", "café résumé naïve")
    nfd_text = unicodedata.normalize("NFD", "café résumé naïve")
    add("accents_nfc", nfc_text)
    add("accents_nfd_decomposed", nfd_text)

    # Section tags
    add("tag_verse", "[Verse]\nWalking down the empty street tonight")
    add("tag_chorus", "[Chorus]\nWe are burning brighter than the sun")
    add("tag_multi", "[Verse 1]\nLine one\n[Chorus]\nLine two\n[Bridge]\nLine three")
    add("tag_bracket_only", "[Intro][Outro]")

    # ABC notation
    add("abc_simple", "X:1\nT:Test Tune\nM:4/4\nL:1/8\nK:C\nCDEF GABc | cBAG FEDC |")
    add("abc_chords", 'X:1\nK:G\n"G" G2 A2 "D7" B2 c2 | "Em" e2 d2 "C" c2 B2 |')
    add("abc_bars_newlines", "A2B2c2d2|e2f2g2a2|\nb2a2g2f2|e2d2c2B2|\n")
    add("abc_ties_rests", "z4 A2B2 | C3/2D/2 E2 | (3ABC D2 z2 |")

    # Style prompts with commas
    add("style_commas", "energetic, uplifting, pop-rock, female vocals, driving drums, 128 bpm")
    add("style_commas2", "melancholic, slow tempo, piano, strings, rain ambience, lo-fi")

    # Whitespace edge cases
    add("ws_leading_space", " leading space word")
    add("ws_double_space", "double  space  between  words")
    add("ws_tab", "tab\tseparated\twords")
    add("ws_tab_punct", "hello\t!!!")
    add("ws_space_punct", "hello !!!")
    add("ws_trailing_space", "trailing space   ")
    add("ws_newlines", "line one\nline two\n\nline four after blank")
    add("ws_crlf", "windows\r\nline\r\nendings")
    add("ws_ideographic_space", "\u3000\u3000full-width\u3000space\u3000test")
    add("ws_nbsp", "non\u00A0breaking\u00A0space")
    add("ws_only_spaces", "     ")
    add("ws_mixed_runs", "a\n\n\n   \tb")

    # Digits runs
    add("digits_simple", "12345")
    add("digits_mixed", "Track 07, BPM 128, Key Cmaj, year 2026, id00099")
    add("digits_long_run", "1234567890123456789012345")
    add("digits_decimal", "Tempo 128.5 bpm, pitch 440.0 Hz")

    # Emoji
    add("emoji_simple", "Great song! 🎵🔥🎸")
    add("emoji_zwj", "Family time 👨‍👩‍👧‍👦 vibes")
    add("emoji_skin_tone", "Thumbs up 👍🏽 great mix")

    # Contractions edge cases
    add("contraction_capital", "It'S a Test with WEIRD'D casing'Ll happen")
    add("contraction_quote_style", "don\u2019t stop believin\u2019, we\u2019ve only just begun")

    # Special-token-shaped literal text (should NOT be special-cased by the
    # reference tokenizer's encode_ordinary(), unlike bpe.h's hardcoded scan)
    add("literal_endoftext", "before <|endoftext|> after")
    add("literal_im_start", "system prompt <|im_start|>user<|im_end|>")
    add("literal_extra", "token <extra_5> here")
    add("literal_abc_tag", "wrap <abc> notation </abc> tags")

    # Punctuation-heavy / symbol runs
    add("punct_heavy", "Wait... really?! No way -- that's (kind of) amazing!!!")
    add("punct_currency", "Price: $19.99, €18,50, ¥2000, £15.00")
    add("punct_url", "Check https://example.com/path?x=1&y=2 for details")
    add("punct_math", "1+1=2, 3*4=12, a<b<=c, x!=y")

    # Mixed multilingual lyric-style block
    add("mixed_block",
        "[Verse]\nWalking through Tokyo streets, 東京の街を歩く\n"
        "[Chorus]\n안녕 my love, adiós mi amor\n"
        "[Bridge]\n你好，再见，こんにちは")

    # Empty / whitespace-only / single-char edge cases
    add("empty_string", "")
    add("single_char", "a")
    add("single_space", " ")
    add("single_newline", "\n")
    add("single_digit", "7")

    return cases


# ════════════════════════════════════════════════════════════════════════
# Part C: run both encoders + report
# ════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True, type=Path,
                     help="dir with converted vocab.json/merges.txt (yue2-tokenizer-convert.py output)")
    ap.add_argument("--tiktoken", required=True, type=Path, help="path to qwen.tiktoken (for the reference tokenizer)")
    ap.add_argument("--yue2-venv-site-packages", required=True, type=Path,
                     help="K:/yue2/.venv/Lib/site-packages, to import the installed yue2 package")
    ap.add_argument("--json-out", type=Path, default=None, help="optional path to dump full results as JSON")
    ap.add_argument("--dump", type=Path, default=None,
                     help="write {name,text,text_nfc,expected_ids} for the whole corpus to this path, for "
                          "engine/tools/yue2-probe's --tokenizer-check to replay against the C++ tokenizer. "
                          "expected_ids come from the REAL reference encoder (ref_tok.encode, i.e. NFC then "
                          "tiktoken encode_ordinary, no EOS). text_nfc is unicodedata NFC of text -- since the "
                          "engine does no NFC of its own (that's Node's job upstream), a C++-side check should "
                          "tokenize text_nfc and compare against expected_ids, not the raw text field.")
    args = ap.parse_args()

    sys.path.insert(0, str(args.yue2_venv_site_packages))
    from yue2.tokenization_yue2 import YuE2TextTokenizer  # noqa: E402

    ref_tok = YuE2TextTokenizer(str(args.tiktoken))
    port_tok = BpeHPort(args.tokenizer_dir)

    corpus = build_corpus()
    print(f"[check] corpus size: {len(corpus)} strings", file=sys.stderr)

    if args.dump:
        dump_cases = []
        for name, text in corpus:
            expected_ids = ref_tok.encode(text)
            dump_cases.append({
                "name": name,
                "text": text,
                "text_nfc": unicodedata.normalize("NFC", text),
                "expected_ids": expected_ids,
            })
        args.dump.write_text(json.dumps({"cases": dump_cases}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[check] wrote {len(dump_cases)} expected-id cases to {args.dump}", file=sys.stderr)

    results = []
    n_core_mismatch = 0
    n_raw_mismatch = 0

    for name, text in corpus:
        ref_ids = ref_tok.encode(text)                 # NFC + encode_ordinary, no EOS
        port_ids_raw = port_tok.encode(text, add_eos=True)   # bpe.h's actual default behavior
        port_ids_core = port_tok.encode(text, add_eos=False)  # isolates BPE/regex-only mismatches

        raw_match = (ref_ids == port_ids_raw)
        core_match = (ref_ids == port_ids_core)
        if not raw_match:
            n_raw_mismatch += 1
        if not core_match:
            n_core_mismatch += 1

        results.append({
            "name": name,
            "text": text,
            "reference_ids": ref_ids,
            "bpe_h_port_ids_with_default_eos": port_ids_raw,
            "bpe_h_port_ids_no_eos": port_ids_core,
            "match_including_eos_policy": raw_match,
            "match_core_bpe_only": core_match,
        })

    total = len(corpus)
    print(f"\n[check] {total - n_core_mismatch}/{total} match on core BPE/regex tokens (ignoring bpe.h's "
          f"always-appended trailing EOS, which the reference tokenizer never adds)", file=sys.stderr)
    print(f"[check] {total - n_raw_mismatch}/{total} match bpe.h's literal current behavior "
          f"(including its default add_eos=true)", file=sys.stderr)

    print("\n=== CORE MISMATCHES (BPE/regex, EOS policy excluded) ===")
    for r in results:
        if not r["match_core_bpe_only"]:
            print(f"\n[{r['name']}] text={r['text']!r}")
            print(f"  reference     : {r['reference_ids']}")
            print(f"  bpe.h port    : {r['bpe_h_port_ids_no_eos']}")

    print("\n=== ALL RESULTS (including EOS-policy-only differences) marked further below ===")
    for r in results:
        tag = "OK  " if r["match_core_bpe_only"] else "FAIL"
        print(f"{tag} {r['name']}")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[check] wrote full results to {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()

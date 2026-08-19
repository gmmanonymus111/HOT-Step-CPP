"""Read the converted GGUFs back and check structure + KVs against config.json."""
import json
import os
import sys

import gguf
import numpy as np

GGUF_DIR = r"M:\Music Captioners\gguf"
SRC_CFG = sys.argv[1] if len(sys.argv) > 1 else None


def show(path, expect_arch):
    r = gguf.GGUFReader(path)
    arch = None
    kvs = {}
    for f in r.fields.values():
        try:
            v = f.contents()
        except Exception:
            v = None
        kvs[f.name] = v
        if f.name == "general.architecture":
            arch = v
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  arch      : {arch}  (expected {expect_arch})"
          f"{'  OK' if arch == expect_arch else '  ** MISMATCH **'}")
    print(f"  tensors   : {len(r.tensors)}")
    types = {}
    for t in r.tensors:
        types[str(t.tensor_type).split(".")[-1]] = types.get(
            str(t.tensor_type).split(".")[-1], 0) + 1
    print(f"  dtypes    : {types}")
    interesting = [k for k in kvs if k.startswith(("moss.", "qwen3.")) or k in
                   ("general.name",)]
    for k in sorted(interesting):
        v = kvs[k]
        if isinstance(v, (list, tuple)) and len(v) > 8:
            v = f"[{len(v)} items]"
        print(f"  {k:<44} {v}")
    return r, kvs


def main():
    lm_path = os.path.join(GGUF_DIR, "moss-lm-q8_0.gguf")
    aud_path = os.path.join(GGUF_DIR, "moss-aud-f16.gguf")

    lm, lmkv = show(lm_path, "qwen3")
    aud, audkv = show(aud_path, "moss-aud")

    names_lm = {t.name for t in lm.tensors}
    names_aud = {t.name for t in aud.tensors}

    print("\n=== structural checks ===")
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'} {msg}")
        ok = ok and bool(cond)

    check("token_embd.weight" in names_lm, "LM has token_embd.weight")
    check("output.weight" in names_lm, "LM has a separate output.weight (untied head)")
    check("output_norm.weight" in names_lm, "LM has output_norm.weight")
    check(all(f"blk.{i}.attn_q_norm.weight" in names_lm for i in range(36)),
          "LM has Qwen3 QK-norm on all 36 blocks")
    check(len(names_lm) == 399, f"LM tensor count 399 (got {len(names_lm)})")

    check(all(f"aud.conv{i}.weight" in names_aud for i in (1, 2, 3)),
          "audio tower has the 3-conv stem")
    check("aud.stem_proj.weight" in names_aud, "audio tower has stem_proj")
    check(all(f"aud.blk.{i}.attn_q.weight" in names_aud for i in range(32)),
          "audio tower has all 32 encoder blocks")
    check(not any(n.endswith("attn_k.bias") for n in names_aud),
          "no k_proj bias anywhere (Whisper convention preserved)")
    check("aud.adapter.gate.weight" in names_aud, "audio tower has the SwiGLU adapter")
    check(all(f"aud.deepstack.{k}.gate.weight" in names_aud for k in range(3)),
          "audio tower has 3 deepstack mergers")
    check(not any("inv_timescales" in n for n in names_aud),
          "inv_timescales correctly omitted")
    check(len(names_aud) == 502, f"audio tensor count 502 (got {len(names_aud)})")

    # shapes that the C++ graph will assume
    def shape_of(reader, name):
        for t in reader.tensors:
            if t.name == name:
                return tuple(int(x) for x in t.shape)
        return None

    print("\n=== key shapes (ggml order: ne0 fastest) ===")
    for r, n in ((aud, "aud.conv1.weight"), (aud, "aud.conv2.weight"),
                 (aud, "aud.stem_proj.weight"), (aud, "aud.blk.0.attn_q.weight"),
                 (aud, "aud.adapter.gate.weight"), (aud, "aud.adapter.down.weight"),
                 (aud, "aud.deepstack.0.down.weight"),
                 (lm, "token_embd.weight"), (lm, "output.weight"),
                 (lm, "blk.0.attn_q.weight"), (lm, "blk.0.attn_q_norm.weight")):
        print(f"  {n:<34} {shape_of(r, n)}")

    # conv1 must be [KW,KH,IC,OC] = [3,3,1,480] in ggml order for ggml_conv_2d
    c1 = shape_of(aud, "aud.conv1.weight")
    check(c1 == (3, 3, 1, 480),
          f"conv1 is [KW,KH,IC,OC]=[3,3,1,480] for ggml_conv_2d (got {c1})")
    c2 = shape_of(aud, "aud.conv2.weight")
    check(c2 == (3, 3, 480, 480), f"conv2 is [3,3,480,480] (got {c2})")
    sp = shape_of(aud, "aud.stem_proj.weight")
    check(sp == (7680, 1280), f"stem_proj is [7680,1280] = 480*16 -> d_model (got {sp})")

    if SRC_CFG:
        with open(os.path.join(SRC_CFG, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        lc = cfg["language_config"]
        print("\n=== KV vs config.json ===")
        check(lmkv.get("qwen3.block_count") == lc["num_hidden_layers"],
              f"block_count {lmkv.get('qwen3.block_count')} == {lc['num_hidden_layers']}")
        check(lmkv.get("qwen3.embedding_length") == lc["hidden_size"],
              f"embedding_length == {lc['hidden_size']}")
        check(lmkv.get("qwen3.attention.head_count_kv") == lc["num_key_value_heads"],
              f"head_count_kv == {lc['num_key_value_heads']}")
        check(abs(float(audkv.get("moss.audio.tokens_per_second", 0)) - 12.5) < 1e-6,
              "moss.audio.tokens_per_second == 12.5")

    print("\n" + ("ALL CHECKS PASSED" if ok else "** SOME CHECKS FAILED **"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

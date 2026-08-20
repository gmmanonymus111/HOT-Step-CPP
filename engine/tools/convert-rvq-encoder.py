#!/usr/bin/env python3
# convert-rvq-encoder.py: open-RVQ audio->codes encoder (.pt) -> GGUF.
#
# The encoder MiniMax never released. MM3 ships decode-side only for the audio
# token path, so every LM-training dataset needs a community audio->codes
# encoder to produce the `.codes` files `ace-train mm3-condition --codes` and
# the LM trainer consume. This converts one to GGUF so the engine can run it
# natively and the WSL/python exporter leaves the data path.
#
# WHICH CHECKPOINT
# ----------------
# The adopted working encoder (Rob's ear, 2026-08-20): scragnog/
# open-rvq-encoder-minimax-music3-169m-53k-pooled, locally
#   M:\HOT-Step-CPP\_experiments\open-rvq-53k-pooled\v4_pooled_best.pt
# It is PurpleOrc's `V4Encoder` architecture (NOT the SimpleTuner classes) —
# 1088-wide shared encoder, plain linear readouts, causal depth decoder with
# per-codebook prior embeddings. Every checkpoint in that lineage
# (PurpleOrc 53k, Mothersuperior pooled-v4, our hotstep-v1/r2/r3) is the SAME
# architecture, so this converter takes the checkpoint as an argument and the
# engine graph is weights-agnostic. Point --src at whichever encoder is
# current; do not fork the graph per checkpoint.
#
# TENSOR NAMES ARE VERBATIM PYTORCH NAMES. Deliberately: this converter is a
# shape-validating pass-through, and the C++ loader looks up the same strings
# the reference `train_v4.py` module defines. A renaming layer here is a place
# for a silent mismatch to hide, and the parity gate (byte-identical `.codes`
# vs the python exporter) is the only thing standing between a subtle port bug
# and a corpus of quietly wrong training data.
#
# ARCHITECTURE (train_v4.py, reproduced here as the shape contract)
#   conv_in     Conv1d(128 -> 1088, k=7, pad=3)         over DAV latents
#   blocks.0-2  ResBlock(1088, dilation 1/3/9):
#                 h = conv1(gelu(GroupNorm(1,1088)(x))); x + conv2(gelu(h))
#   pool        x = bmm(pool[128,L], x^T) + pos[128,1088]   <- frame pooling
#   transformer 8 x TransformerEncoderLayer(1088, 17 heads, ff 4352,
#                                           gelu, norm_first=True)
#   norm_out    LayerNorm(1088)      applied AFTER the stack (the nn.
#                                    TransformerEncoder itself carries no norm)
#   sem_head    Linear(1088 -> 16384)
#   depth       DepthDecoder: proj 1088->512, sem_emb[16384,512],
#               ac_emb[6144,512] (books 1..6), pos[8,512],
#               2 x TransformerEncoderLayer(512, 8 heads, ff 2048, gelu,
#               norm_first=True), causal 8x8 mask, 7 x Linear(512 -> 1024).
#               Sequence per frame: pos0=ctx, pos1=sem, pos2..7=ac1..ac6;
#               head k reads position k+1.
#
# WINDOWING (also carried as KV — it is part of the model contract, not a
# caller's choice; second-guessing a model's own windowing is how silent
# misalignment happens):
#   LATENT_CHANNELS 128, FRAMES 128, LATENT_WINDOW_MAX 448,
#   RATIO 441/128, CHUNK_FRAMES 200, CHUNK_HOP 100, HOP_LATENTS 345,
#   OWNED_FROM 25
#
# DEPENDENCIES: torch (to read the .pt), numpy, gguf (llama.cpp's gguf-py).
#
# USAGE
#   python engine/tools/convert-rvq-encoder.py \
#     --src "M:/HOT-Step-CPP/_experiments/open-rvq-53k-pooled/v4_pooled_best.pt" \
#     --out models/mm3/mm3-rvq-53kpooled-f32.gguf \
#     --name "open-rvq 53k+pooled (hotstep-adopted 2026-08-20)"

import argparse
import sys

import numpy as np

ARCH = "mm3rvq"

# ── the shape contract (train_v4.py constants) ──────────────────────────────
D_MODEL, N_LAYERS, N_HEADS, FF_MULT = 1088, 8, 17, 4
DEPTH_D, DEPTH_LAYERS, DEPTH_HEADS = 512, 2, 8
SEM_VOCAB, AC_VOCAB, N_AC = 16384, 1024, 7
LATENT_CHANNELS, FRAMES, LATENT_WINDOW_MAX = 128, 128, 448
RATIO_NUM, RATIO_DEN = 441, 128
CHUNK_FRAMES, CHUNK_HOP, HOP_LATENTS, OWNED_FROM = 200, 100, 345, 25
DILATIONS = [1, 3, 9]

FF = D_MODEL * FF_MULT          # 4352
DEPTH_FF = DEPTH_D * FF_MULT    # 2048


def expected_shapes():
    """name -> torch shape. The full inventory; anything missing, mis-shaped or
    left over aborts the conversion. A partially-correct encoder produces codes
    that look plausible and train a subtly wrong adapter."""
    s = {
        "pos":            (1, FRAMES, D_MODEL),
        "conv_in.weight": (D_MODEL, LATENT_CHANNELS, 7),
        "conv_in.bias":   (D_MODEL,),
        "norm_out.weight": (D_MODEL,),
        "norm_out.bias":   (D_MODEL,),
        "sem_head.weight": (SEM_VOCAB, D_MODEL),
        "sem_head.bias":   (SEM_VOCAB,),
    }
    for i in range(len(DILATIONS)):
        s[f"blocks.{i}.norm.weight"]  = (D_MODEL,)
        s[f"blocks.{i}.norm.bias"]    = (D_MODEL,)
        s[f"blocks.{i}.conv1.weight"] = (D_MODEL, D_MODEL, 3)
        s[f"blocks.{i}.conv1.bias"]   = (D_MODEL,)
        s[f"blocks.{i}.conv2.weight"] = (D_MODEL, D_MODEL, 1)
        s[f"blocks.{i}.conv2.bias"]   = (D_MODEL,)

    def encoder_layer(prefix, d, ff):
        return {
            f"{prefix}.self_attn.in_proj_weight":  (3 * d, d),
            f"{prefix}.self_attn.in_proj_bias":    (3 * d,),
            f"{prefix}.self_attn.out_proj.weight": (d, d),
            f"{prefix}.self_attn.out_proj.bias":   (d,),
            f"{prefix}.linear1.weight":            (ff, d),
            f"{prefix}.linear1.bias":              (ff,),
            f"{prefix}.linear2.weight":            (d, ff),
            f"{prefix}.linear2.bias":              (d,),
            f"{prefix}.norm1.weight":              (d,),
            f"{prefix}.norm1.bias":                (d,),
            f"{prefix}.norm2.weight":              (d,),
            f"{prefix}.norm2.bias":                (d,),
        }

    for i in range(N_LAYERS):
        s.update(encoder_layer(f"transformer.layers.{i}", D_MODEL, FF))
    for i in range(DEPTH_LAYERS):
        s.update(encoder_layer(f"depth.tr.layers.{i}", DEPTH_D, DEPTH_FF))
    s.update({
        "depth.pos":            (1, 8, DEPTH_D),
        "depth.proj.weight":    (DEPTH_D, D_MODEL),
        "depth.proj.bias":      (DEPTH_D,),
        "depth.sem_emb.weight": (SEM_VOCAB, DEPTH_D),
        "depth.ac_emb.weight":  ((N_AC - 1) * AC_VOCAB, DEPTH_D),
    })
    for k in range(N_AC):
        s[f"depth.heads.{k}.weight"] = (AC_VOCAB, DEPTH_D)
        s[f"depth.heads.{k}.bias"]   = (AC_VOCAB,)
    return s


def load_state(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # Checkpoints in this lineage are {'model': state_dict, 'step': .., 'metrics': ..};
    # a bare state_dict is also accepted.
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    return {k: v.detach().cpu().float().numpy() for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="V4Encoder .pt checkpoint")
    ap.add_argument("--out", required=True, help="destination .gguf")
    ap.add_argument("--name", default="open-rvq encoder (MiniMax-Music3 169M)")
    ap.add_argument("--quant", choices=["f32", "f16"], default="f32",
                    help="f32 by default: the parity gate compares ARGMAX codes "
                         "against the python exporter, and near-tie logits flip "
                         "under f16. Ship f16 only once parity holds at f32.")
    a = ap.parse_args()

    try:
        import gguf
    except ImportError:
        print("error: gguf (llama.cpp's gguf-py) is required: pip install gguf", file=sys.stderr)
        return 2

    sd = load_state(a.src)
    want = expected_shapes()

    missing = [k for k in want if k not in sd]
    extra   = [k for k in sd if k not in want]
    bad     = [(k, sd[k].shape, want[k]) for k in want if k in sd and tuple(sd[k].shape) != want[k]]
    if missing or extra or bad:
        for k in missing: print(f"  MISSING  {k}  {want[k]}", file=sys.stderr)
        for k in extra:   print(f"  EXTRA    {k}  {sd[k].shape}", file=sys.stderr)
        for k, got, exp in bad: print(f"  SHAPE    {k}  got {got} want {exp}", file=sys.stderr)
        print(f"error: checkpoint does not match the V4Encoder contract "
              f"({len(missing)} missing, {len(extra)} extra, {len(bad)} mis-shaped)", file=sys.stderr)
        return 1

    total = sum(v.size for v in sd.values())
    print(f"[rvq] {a.src}: {len(sd)} tensors, {total/1e6:.1f}M params")

    w = gguf.GGUFWriter(a.out, ARCH)
    w.add_name(a.name)
    w.add_description("Community audio->RVQ-codes encoder for MiniMax-Music3 "
                      "(PurpleOrc V4Encoder architecture)")
    w.add_file_type(gguf.GGMLQuantizationType.F16 if a.quant == "f16"
                    else gguf.GGMLQuantizationType.F32)

    kv = {
        "embedding_length":   D_MODEL,
        "block_count":        N_LAYERS,
        "head_count":         N_HEADS,
        "feed_forward_length": FF,
        "sem_vocab_size":     SEM_VOCAB,
        "ac_vocab_size":      AC_VOCAB,
        "num_acoustic":       N_AC,
        "latent_channels":    LATENT_CHANNELS,
        "frames":             FRAMES,
        "latent_window_max":  LATENT_WINDOW_MAX,
        # windowing contract — the engine must reproduce frame_latent_starts()
        # exactly, so the constants travel with the weights rather than being
        # duplicated as C++ literals that can drift from a future checkpoint.
        "ratio_num":          RATIO_NUM,
        "ratio_den":          RATIO_DEN,
        "chunk_frames":       CHUNK_FRAMES,
        "chunk_hop":          CHUNK_HOP,
        "hop_latents":        HOP_LATENTS,
        "owned_from":         OWNED_FROM,
        "depth.embedding_length":  DEPTH_D,
        "depth.block_count":       DEPTH_LAYERS,
        "depth.head_count":        DEPTH_HEADS,
        "depth.feed_forward_length": DEPTH_FF,
    }
    for k, v in kv.items():
        w.add_uint32(f"{ARCH}.{k}", int(v))
    w.add_array(f"{ARCH}.dilations", [int(d) for d in DILATIONS])
    # Pre-LN everywhere and GELU everywhere — recorded so a future checkpoint
    # that changes either is REJECTED by the loader rather than silently
    # producing wrong codes.
    w.add_string(f"{ARCH}.norm_order", "pre")
    w.add_string(f"{ARCH}.activation", "gelu")

    # 1-D tensors (norms, biases, embeddings) stay F32 even in f16 mode: they
    # are ~2% of the file and the cheapest possible place to lose precision.
    for k in sorted(sd):
        v = sd[k]
        if k in ("pos", "depth.pos"):
            v = v[0]                      # drop the leading batch axis
        arr = v.astype(np.float16) if (a.quant == "f16" and v.ndim >= 2) else v.astype(np.float32)
        w.add_tensor(k, arr)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[rvq] wrote {a.out} ({a.quant})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

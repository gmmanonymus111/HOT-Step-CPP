#!/usr/bin/env python3
"""make-mm3-test-lora.py — synthesise a MiniMax-Music3 flow-DiT LoRA for testing.

There is no MM3 adapter corpus yet, so the merge path in
engine/src/minimax/mm3-adapter.h cannot be validated against a real adapter.
This writes deterministic ones instead, in both on-disk formats, so the plumbing
can be gated without depending on SimpleTuner or a trained checkpoint.

Modes
  zero    every lora_B is exactly 0, so the delta is exactly 0. Merging it MUST
          produce bit-identical audio to merging nothing. This is the strongest
          plumbing gate available without weight introspection: it fails loudly
          if the mapping writes to the wrong tensor, mis-slices a fused qkv,
          corrupts a requant, or silently drops the pending copy.
  known   deterministic pseudo-random A and B at a small magnitude. Merging it
          must CHANGE the audio, reproducibly, and the change must scale with
          MM3_ADAPTER_SCALE.

Formats  (see mm3-adapter.h for why both matter)
  comfy      diffusion_model.* prefix, ComfyUI module names, q/k/v pre-fused
             into .to_qkv — the format SimpleTuner ships by default.
  diffusers  transformer.* prefix, diffusers module names, q/k/v as three
             separate modules — exercises our slice-and-concat path.

No torch, no safetensors package: the container format is a JSON header plus
raw little-endian tensor data, which numpy alone can write.

Usage
  python make-mm3-test-lora.py --out zero.safetensors  --mode zero  --format comfy
  python make-mm3-test-lora.py --out known.safetensors --mode known --format diffusers --rank 8
"""

import argparse
import json
import struct
import sys

import numpy as np

# Geometry, mirroring engine/tools/convert-mm3.py. If these ever disagree with
# the GGUF the engine will reject the adapter on a shape mismatch rather than
# merge something wrong, so a drift here is loud.
DIT_LAYERS = 36
DIT_DIM = 2048
DIT_FF_INNER = 8192
DIT_IN_CH = 128
DIT_CONCAT_CH = DIT_IN_CH * 2 + 2048  # 2304


def dtype_str(a):
    return {np.dtype("float32"): "F32", np.dtype("float16"): "F16"}[a.dtype]


def write_safetensors(path, tensors, metadata=None):
    """tensors: dict name -> np.ndarray (C-contiguous, PyTorch axis order)."""
    header = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    offset = 0
    blobs = []
    for name in tensors:  # insertion order is the file order
        a = np.ascontiguousarray(tensors[name])
        nbytes = a.nbytes
        header[name] = {
            "dtype": dtype_str(a),
            "shape": list(a.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        blobs.append(a.tobytes())
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors requires the data section to start 8-byte aligned.
    pad = (-len(hdr)) % 8
    hdr += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)
    return offset + 8 + len(hdr)


def factors(rng, mode, out_features, in_features, rank):
    """Return (A [rank, in], B [out, rank]) as float32."""
    a = rng.standard_normal((rank, in_features), dtype=np.float32) * 0.02
    if mode == "zero":
        # PEFT initialises B to zero, so a freshly-created untrained adapter is
        # exactly this. A stays non-zero on purpose: it proves the merge really
        # computes B@A rather than short-circuiting on a zero factor.
        b = np.zeros((out_features, rank), dtype=np.float32)
    else:
        b = rng.standard_normal((out_features, rank), dtype=np.float32) * 0.02
    return a, b


def build(fmt, mode, rank, layers, alpha):
    rng = np.random.default_rng(20260814)
    t = {}

    def add(module, out_f, in_f):
        a, b = factors(rng, mode, out_f, in_f, rank)
        t[f"{module}.lora_A.weight"] = a
        t[f"{module}.lora_B.weight"] = b
        t[f"{module}.alpha"] = np.array(alpha, dtype=np.float32)

    if fmt == "comfy":
        root = "diffusion_model.diffusion_transformer"
        add(f"{root}.transformer.project_in", DIT_DIM, DIT_CONCAT_CH)
        add(f"{root}.transformer.project_out", DIT_IN_CH, DIT_DIM)
        for i in layers:
            blk = f"{root}.transformer.layers.{i}"
            # ComfyUI ships q/k/v already fused, in q,k,v row order.
            add(f"{blk}.self_attn.to_qkv", 3 * DIT_DIM, DIT_DIM)
            add(f"{blk}.self_attn.to_out", DIT_DIM, DIT_DIM)
            add(f"{blk}.ff.ff.0.proj", 2 * DIT_FF_INNER, DIT_DIM)
            add(f"{blk}.ff.ff.2", DIT_DIM, DIT_FF_INNER)
    else:
        root = "transformer"
        add(f"{root}.proj_in", DIT_DIM, DIT_CONCAT_CH)
        add(f"{root}.proj_out", DIT_IN_CH, DIT_DIM)
        for i in layers:
            blk = f"{root}.transformer_blocks.{i}"
            # Three separate projections — the engine must assemble them.
            add(f"{blk}.attn.to_q", DIT_DIM, DIT_DIM)
            add(f"{blk}.attn.to_k", DIT_DIM, DIT_DIM)
            add(f"{blk}.attn.to_v", DIT_DIM, DIT_DIM)
            add(f"{blk}.attn.to_out.0", DIT_DIM, DIT_DIM)
            add(f"{blk}.ff_in", 2 * DIT_FF_INNER, DIT_DIM)
            add(f"{blk}.ff_out", DIT_DIM, DIT_FF_INNER)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("zero", "known"), default="zero")
    ap.add_argument("--format", choices=("comfy", "diffusers"), default="comfy")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=None,
                    help="default: equal to rank, i.e. a scaling of 1.0")
    ap.add_argument("--layers", default="all",
                    help="'all' or a comma list, e.g. 0,1,35")
    ap.add_argument("--gate-first", action="store_true",
                    help="stamp swiglu_gate_first=true in __metadata__, so the "
                         "engine must swap the ff_in halves")
    args = ap.parse_args()

    layers = range(DIT_LAYERS) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    alpha = args.rank if args.alpha is None else args.alpha

    tensors = build(args.format, args.mode, args.rank, list(layers), alpha)
    meta = {"format": "pt", "hot_step_test_lora": f"{args.mode}/{args.format}"}
    if args.gate_first:
        meta["swiglu_gate_first"] = "true"

    size = write_safetensors(args.out, tensors, meta)
    n_modules = sum(1 for k in tensors if k.endswith(".lora_A.weight"))
    print(f"wrote {args.out}")
    print(f"  format={args.format} mode={args.mode} rank={args.rank} alpha={alpha}")
    print(f"  {n_modules} LoRA modules, {len(tensors)} tensors, {size/1e6:.1f} MB")
    if args.format == "diffusers":
        # to_q/to_k/to_v collapse onto one fused GGUF tensor.
        expect = 2 + len(list(layers)) * 4
    else:
        expect = 2 + len(list(layers)) * 4
    print(f"  engine should report: merged {expect} tensor(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

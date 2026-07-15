#!/usr/bin/env python3
# convert-comfy-int8.py: ComfyUI int8 (comfy_quant) DiT safetensors -> Q8_0 GGUF
#
# ComfyUI int8 checkpoints (convert_to_quant output) store each quantized
# linear as:
#   <base>.weight        I8  (out, in)         -- quantized weight
#   <base>.weight_scale  F32 scalar or (out,)  -- per-tensor / per-row scale
#   <base>.comfy_quant   U8  json blob         -- {"format": "int8_tensorwise",
#                                                  "convrot": true?,
#                                                  "convrot_groupsize": G?, ...}
# Dequant is w = int8 * scale. Both per-tensor and per-row int8 grids are
# exactly representable in GGUF Q8_0 (fp16 scale + 32x int8 per block; every
# block scale in row r = that row's scale), so we repack bit-faithfully
# instead of dequantizing + requantizing.
#
# ConvRot (--convrot): weights are stored PRE-ROTATED (W' = W @ H_block^T per
# input-dim group, H = regular Hadamard, power-of-4 group size). Handling:
#   - decoder.* weights keep the rotation; the layer is recorded in the GGUF
#     KV "acestep.convrot_map" ("name:group;...") and the engine applies the
#     matching activation rotation at inference (see engine/src/dit-graph.h).
#   - all other components (encoder/tokenizer/detokenizer) are dequantized and
#     UNROTATED offline to BF16 — they run once per generation, so keeping
#     them int8 isn't worth wiring rotation through their graph builders.
#
# Non-quantized tensors follow convert.py conventions (F32 -> BF16 truncate).
# The DiT GGUF must also carry silence_latent + the acestep.* config KVs,
# which ComfyUI files lack -- both are copied from a donor GGUF of the same
# architecture (any convert.py-produced acestep-v15-*.gguf that matches).
#
# Usage:
#   python convert-comfy-int8.py <comfy.safetensors> <donor.gguf> <out.gguf> [--name NAME]

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import gguf  # noqa: E402
import convert  # noqa: E402  (engine/convert.py: read_sf_header, add_metadata)

Q8_0 = gguf.GGMLQuantizationType.Q8_0
BF16 = gguf.GGMLQuantizationType.BF16
QK8_0 = 32  # ggml Q8_0 block size


def log(msg):
    print("[COMFY-INT8] %s" % msg, file=sys.stderr, flush=True)


def f32_to_bf16(arr_f32):
    """Truncate F32 to BF16 (as uint16), matching convert.py."""
    a = np.ascontiguousarray(arr_f32, dtype=np.float32)
    return (a.view(np.uint32) >> 16).astype(np.uint16)


def pack_q8_0(qs, row_scales):
    """Pack int8 weights (rows, cols) + per-row f32 scales into raw Q8_0 blocks.

    Q8_0 block layout: 2-byte fp16 scale followed by 32 int8 values.
    Returns a uint8 array of shape (rows, cols//32 * 34).
    """
    rows, cols = qs.shape
    nb = cols // QK8_0
    d = np.repeat(row_scales.astype(np.float16).reshape(rows, 1, 1), nb, axis=1)
    blocks = np.concatenate([d.view(np.uint8), qs.reshape(rows, nb, QK8_0).view(np.uint8)], axis=2)
    return np.ascontiguousarray(blocks.reshape(rows, nb * (QK8_0 + 2)))


def build_hadamard(size):
    """Regular Hadamard (H4 Kronecker powers, normalized). Power-of-4 sizes.

    Mirrors convert_to_quant utils/convrot.py build_hadamard(); symmetric.
    """
    import math
    if size < 4 or (size & (size - 1)) != 0 or (math.log(size, 4) % 1 != 0):
        raise ValueError("unsupported Hadamard size %d (power of 4 only)" % size)
    H4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64)
    H = H4
    while H.shape[0] < size:
        H = np.kron(H, H4)
    return H / math.sqrt(size)


def unrotate_weight(w, group_size):
    """Undo the offline ConvRot rotation: W = W_rot @ H per input-dim group
    (H symmetric orthogonal, so H^T == H and W_rot @ H recovers W)."""
    out_f, in_f = w.shape
    H = build_hadamard(group_size)
    wg = w.reshape(out_f, in_f // group_size, group_size)
    return (wg @ H).reshape(out_f, in_f).astype(np.float32)


def read_donor(donor_path):
    """Return (config dict, donor name, silence_latent f32 array, {name: shape})."""
    r = gguf.GGUFReader(donor_path)
    fields = {f.name: f for f in r.fields.values()}

    def get_str(key):
        f = fields.get(key)
        return bytes(f.parts[f.data[0]]).decode() if f else None

    cfg_json = get_str("acestep.config_json")
    if not cfg_json:
        log("FATAL: donor %s has no acestep.config_json (not a convert.py DiT GGUF?)" % donor_path)
        sys.exit(1)

    silence = None
    shapes = {}
    for t in r.tensors:
        shapes[t.name] = tuple(reversed([int(d) for d in t.shape]))  # ne order -> torch order
        if t.name == "silence_latent":
            silence = np.asarray(t.data).flatten().reshape(15000, 64).astype(np.float32)

    if silence is None:
        log("FATAL: donor has no silence_latent tensor")
        sys.exit(1)
    return json.loads(cfg_json), get_str("general.name"), silence, shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comfy_st")
    ap.add_argument("donor_gguf")
    ap.add_argument("out_gguf")
    ap.add_argument("--name", default=None, help="general.name for the output (default: derived from output filename)")
    ap.add_argument("--no-runtime-rotation", action="store_true",
                    help="Dequantize + unrotate EVERYTHING to BF16 (no acestep.convrot_map, no engine "
                         "rotation needed). Numerically equivalent reference build — 2x the size; used "
                         "to A/B-validate the engine's runtime rotation path.")
    args = ap.parse_args()

    name = args.name or os.path.basename(args.out_gguf).rsplit(".", 1)[0]

    cfg, donor_name, silence, donor_shapes = read_donor(args.donor_gguf)
    log("donor: %s (%d tensors)" % (donor_name, len(donor_shapes)))

    meta, hdr_size = convert.read_sf_header(args.comfy_st)

    w = gguf.GGUFWriter(args.out_gguf, "acestep-dit", use_temp_file=True)
    w.add_name(name)
    convert.add_metadata(w, cfg, "dit")

    n_q8, n_bf16, n_dequant, total = 0, 0, 0, 0
    shape_mismatches = []
    convrot_map = []  # (tensor name, group size) kept rotated in the GGUF

    with open(args.comfy_st, "rb") as f:

        def read_tensor(tname):
            info = meta[tname]
            f.seek(hdr_size + info["data_offsets"][0])
            return f.read(info["data_offsets"][1] - info["data_offsets"][0]), info

        def layer_quant_config(base):
            key = base + ".comfy_quant"
            if key not in meta:
                return {}
            raw, _ = read_tensor(key)
            try:
                return json.loads(raw.decode())
            except Exception:
                return {}

        for tname in sorted(meta.keys()):
            if tname.endswith(".weight_scale") or tname.endswith(".comfy_quant"):
                continue
            info = meta[tname]
            dtype, shape = info["dtype"], info["shape"]

            donor_shape = donor_shapes.get(tname)
            if donor_shape is None:
                log("  WARNING: %s not in donor GGUF -- writing anyway" % tname)
            elif donor_shape != tuple(shape):
                shape_mismatches.append((tname, tuple(shape), donor_shape))

            raw, _ = read_tensor(tname)

            if dtype == "I8":
                base = tname[: -len(".weight")]
                sraw, sinfo = read_tensor(base + ".weight_scale")
                scales = np.frombuffer(sraw, dtype=np.float32)
                qcfg = layer_quant_config(base)
                rot_group = int(qcfg.get("convrot_groupsize", 0)) if qcfg.get("convrot") else 0

                qs = np.frombuffer(raw, dtype=np.int8).reshape(-1, shape[-1])
                rows, cols = qs.shape
                # scalar scale -> broadcast per-row; per-row scale as-is
                if scales.size == 1:
                    row_scales = np.full(rows, scales[0], dtype=np.float32)
                elif scales.size == rows:
                    row_scales = scales.reshape(rows)
                else:
                    log("  FATAL: %s weight_scale has %d entries for %d rows" % (tname, scales.size, rows))
                    sys.exit(1)

                keep_rotation = rot_group > 0 and tname.startswith("decoder.") and not args.no_runtime_rotation
                packable = len(shape) == 2 and cols % QK8_0 == 0 and not (rot_group > 0 and args.no_runtime_rotation)

                if packable and (rot_group == 0 or keep_rotation):
                    packed = pack_q8_0(qs, row_scales)
                    w.add_tensor(tname, packed, raw_dtype=Q8_0)
                    if keep_rotation:
                        convrot_map.append((tname, rot_group))
                    n_q8 += 1
                    total += packed.nbytes
                else:
                    # dequant fallback; undo rotation so no runtime support needed
                    deq = qs.astype(np.float32) * row_scales[:, None]
                    if rot_group > 0:
                        deq = unrotate_weight(deq, rot_group)
                    w.add_tensor(tname, f32_to_bf16(deq.reshape(shape)), raw_dtype=BF16)
                    n_dequant += 1
                    total += deq.size * 2
            elif dtype == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                w.add_tensor(tname, arr, raw_dtype=BF16)
                n_bf16 += 1
                total += arr.nbytes
            elif dtype == "F32":
                arr = f32_to_bf16(np.frombuffer(raw, dtype=np.float32)).reshape(shape)
                w.add_tensor(tname, arr, raw_dtype=BF16)
                n_bf16 += 1
                total += arr.nbytes
            else:
                log("  FATAL: %s has unsupported dtype %s" % (tname, dtype))
                sys.exit(1)

    w.add_tensor("silence_latent", silence)
    total += silence.nbytes

    if shape_mismatches:
        for tname, got, want in shape_mismatches:
            log("  FATAL: shape mismatch %s: comfy %s vs donor %s" % (tname, got, want))
        sys.exit(1)

    if convrot_map:
        w.add_string("acestep.convrot_map", ";".join("%s:%d" % (n, g) for n, g in convrot_map))
        groups = sorted(set(g for _, g in convrot_map))
        log("convrot: %d rotated decoder weights kept (group sizes %s) -> acestep.convrot_map"
            % (len(convrot_map), groups))

    log("tensors: %d Q8_0, %d BF16, %d dequant-fallback, + silence_latent (%.2f GB)"
        % (n_q8, n_bf16, n_dequant, total / (1 << 30)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    log("wrote %.0f MB -> %s" % (os.path.getsize(args.out_gguf) / (1 << 20), args.out_gguf))


if __name__ == "__main__":
    main()

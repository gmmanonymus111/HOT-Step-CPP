#!/usr/bin/env python3
# convert-comfy-int8.py: ComfyUI int8 (comfy_quant) DiT safetensors -> Q8_0 GGUF
#
# ComfyUI int8_tensorwise checkpoints store each quantized linear as:
#   <base>.weight        I8  (out, in)   -- quantized weight
#   <base>.weight_scale  F32 scalar      -- per-tensor scale
#   <base>.comfy_quant   U8  json blob   -- {"format": "int8_tensorwise"}
# Dequant is w = int8 * scale. A per-tensor int8 grid is exactly representable
# in GGUF Q8_0 (fp16 scale + 32x int8 per block, every block scale = tensor
# scale), so we repack bit-faithfully instead of dequantizing + requantizing.
# Non-quantized tensors follow convert.py conventions (F32 -> BF16 truncate).
#
# The DiT GGUF must also carry silence_latent + the acestep.* config KVs,
# which ComfyUI files lack -- both are copied from a donor GGUF of the same
# architecture (any acestep-v15-xl-*.gguf for XL checkpoints).
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


def f32_to_bf16(arr_u32):
    """Truncate F32 (as uint32) to BF16 (as uint16), matching convert.py."""
    return (arr_u32 >> 16).astype(np.uint16)


def pack_q8_0(qs, scale):
    """Pack int8 weights (rows, cols) + scalar f32 scale into raw Q8_0 blocks.

    Q8_0 block layout: 2-byte fp16 scale followed by 32 int8 values.
    Returns a uint8 array of shape (rows, cols//32 * 34).
    """
    rows, cols = qs.shape
    nb = cols // QK8_0
    d = np.full((rows, nb, 1), np.float16(scale), dtype=np.float16)
    blocks = np.concatenate([d.view(np.uint8), qs.reshape(rows, nb, QK8_0).view(np.uint8)], axis=2)
    return np.ascontiguousarray(blocks.reshape(rows, nb * (QK8_0 + 2)))


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

    with open(args.comfy_st, "rb") as f:
        for tname in sorted(meta.keys()):
            if tname.endswith(".weight_scale") or tname.endswith(".comfy_quant"):
                continue
            info = meta[tname]
            dtype, shape = info["dtype"], info["shape"]
            off0, off1 = info["data_offsets"]

            donor_shape = donor_shapes.get(tname)
            if donor_shape is None:
                log("  WARNING: %s not in donor GGUF -- writing anyway" % tname)
            elif donor_shape != tuple(shape):
                shape_mismatches.append((tname, tuple(shape), donor_shape))

            f.seek(hdr_size + off0)
            raw = f.read(off1 - off0)

            if dtype == "I8":
                sk = tname[: -len(".weight")] + ".weight_scale"
                sinfo = meta[sk]
                f.seek(hdr_size + sinfo["data_offsets"][0])
                scale = np.frombuffer(f.read(4), dtype=np.float32)[0]

                qs = np.frombuffer(raw, dtype=np.int8).reshape(-1, shape[-1])
                if len(shape) == 2 and shape[-1] % QK8_0 == 0:
                    packed = pack_q8_0(qs, scale)
                    w.add_tensor(tname, packed, raw_dtype=Q8_0)
                    n_q8 += 1
                    total += packed.nbytes
                else:
                    deq = (qs.astype(np.float32) * scale).reshape(shape)
                    arr = f32_to_bf16(deq.view(np.uint32) if deq.flags.c_contiguous else np.ascontiguousarray(deq).view(np.uint32))
                    w.add_tensor(tname, arr, raw_dtype=BF16)
                    n_dequant += 1
                    total += arr.nbytes
            elif dtype == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                w.add_tensor(tname, arr, raw_dtype=BF16)
                n_bf16 += 1
                total += arr.nbytes
            elif dtype == "F32":
                arr = f32_to_bf16(np.frombuffer(raw, dtype=np.uint32).reshape(shape))
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

    log("tensors: %d Q8_0, %d BF16, %d dequant-fallback, + silence_latent (%.2f GB)"
        % (n_q8, n_bf16, n_dequant, total / (1 << 30)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    log("wrote %.0f MB -> %s" % (os.path.getsize(args.out_gguf) / (1 << 20), args.out_gguf))


if __name__ == "__main__":
    main()

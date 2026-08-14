#!/usr/bin/env python3
# split-mm3.py: split an mm3-synth-<quant>.gguf bundle into per-component GGUFs.
#
#   mm3-synth-X.gguf  ->  mm3-depth-X.gguf   arch "mm3-depth"   depth.* tensors
#                         mm3-cond-X.gguf    arch "mm3-cond"    cond.*  tensors
#                         mm3-dit-X.gguf     arch "mm3-dit"     dit.*   tensors
#                         mm3-voc-X.gguf     arch "mm3-voc"     voc.*   tensors
#
# Pure passthrough: tensor payloads are copied byte-for-byte in their existing
# quantisation (add_tensor_info + write_tensor_data, the gguf_new_metadata.py
# idiom), so splitting a Q4_K_M bundle yields the exact same Q4_K_M tensors.
# Nothing is re-quantised and no checkpoint access is needed.
#
# KV routing:
#   every file    general.* (architecture replaced, name/description re-labelled),
#                 mm3.model, mm3.converter_version, mm3.source_layout
#   mm3-depth     mm3.depth.*
#   mm3-cond      mm3.cond.*
#   mm3-dit       mm3.dit.* + mm3.flow.*   (the flow schedule drives the DiT)
#   mm3-voc       mm3.voc.*
#   mm3.synth.components is rewritten to the single component each file carries.
#
# By default cond/voc are only emitted from an f16 bundle: they are never
# quantised (same policy as upstream acestep/minimaxmusic experience — the
# bandwidth-bound frontends stay native), so the quant bundles' copies would be
# redundant files nobody should select. --all-components overrides.
#
# Idempotent: existing outputs are skipped unless --force.
#
# USAGE
#   python engine/tools/split-mm3.py models/mm3/mm3-synth-f16.gguf
#   python engine/tools/split-mm3.py --out models/mm3 models/mm3/mm3-synth-*.gguf
#
# DEPENDENCIES: gguf (llama.cpp gguf-py) + numpy, same venv as convert-mm3.py.

import argparse
import os
import re
import sys

import gguf

# role -> (tensor prefix, extra KV prefixes, components label)
ROLES = {
    "depth": (("depth.",), ("mm3.depth.",), "depth"),
    "cond":  (("cond.",),  ("mm3.cond.",),  "cond"),
    "dit":   (("dit.",),   ("mm3.dit.", "mm3.flow."), "dit"),
    "voc":   (("voc.",),   ("mm3.voc.",),   "vocoder"),
}

# KVs copied into every split file (general.architecture is set by the writer).
COMMON_KEYS = (
    "general.license",
    "general.file_type",
    "general.quantization_version",
    "mm3.model",
    "mm3.converter_version",
    "mm3.source_layout",
)

ROLE_NAMES = {
    "depth": "MiniMax-Music3 RVQ depth decoder",
    "cond":  "MiniMax-Music3 condition encoder",
    "dit":   "MiniMax-Music3 flow DiT",
    "voc":   "MiniMax-Music3 vocoder",
}

ROLE_DESCRIPTIONS = {
    "depth": "MiniMax-Music3 RVQ depth decoder (7 acoustic codebooks per frame)",
    "cond":  "MiniMax-Music3 condition encoder (frame hiddens -> DiT conditioning)",
    "dit":   "MiniMax-Music3 flow-matching DiT (latent denoiser)",
    "voc":   "MiniMax-Music3 DAC-style vocoder (latents -> 44.1 kHz stereo)",
}


def log(msg):
    print(msg, flush=True)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def parse_bundle_name(path):
    """mm3-synth-<quant>.gguf -> quant token, else None."""
    m = re.fullmatch(r"mm3-synth-(.+)\.gguf", os.path.basename(path))
    return m.group(1) if m else None


def copy_kv(writer, field):
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def write_split(reader, out_path, role, quant):
    prefixes, kv_prefixes, component = ROLES[role]

    tensors = [t for t in reader.tensors if any(t.name.startswith(p) for p in prefixes)]
    if not tensors:
        die(f"{out_path}: no tensors with prefix {prefixes} in the source bundle")

    w = gguf.GGUFWriter(out_path, arch=f"mm3-{role}")
    w.add_name(ROLE_NAMES[role])
    w.add_description(ROLE_DESCRIPTIONS[role])

    n_kv = 0
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        if name in ("general.name", "general.description", "mm3.synth.components"):
            continue  # re-labelled / rewritten below
        if name in COMMON_KEYS or any(name.startswith(p) for p in kv_prefixes):
            copy_kv(w, field)
            n_kv += 1
    # What this file carries, in the bundle's own vocabulary.
    w.add_array("mm3.synth.components", [component])

    total = 0
    for t in tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        total += t.n_bytes

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in tensors:
        try:
            w.write_tensor_data(t.data, tensor_endianess=reader.endianess)
        except TypeError:  # older gguf-py without the endianess kwarg
            w.write_tensor_data(t.data)
    w.close()

    log(f"  [{role:5s}] {os.path.basename(out_path)}: {len(tensors)} tensors, "
        f"{n_kv + 1} KVs, {total / 1e6:.1f} MB")


def split_bundle(src, out_dir, all_components, force):
    quant = parse_bundle_name(src)
    if quant is None:
        die(f"{src}: not an mm3-synth-<quant>.gguf name; cannot derive the quant token")

    reader = gguf.GGUFReader(src)
    arch_field = reader.get_field("general.architecture")
    arch = arch_field.contents() if arch_field else ""
    if arch != "mm3":
        die(f"{src}: general.architecture is '{arch}', expected 'mm3' (a synth bundle)")

    # cond/voc are never quantised; emit them from the f16 bundle only.
    native = quant.lower() in ("f16", "bf16", "f32")
    roles  = list(ROLES) if (all_components or native) else ["depth", "dit"]

    log(f"{os.path.basename(src)} (quant {quant}) -> {', '.join(roles)}")
    for role in roles:
        out_path = os.path.join(out_dir, f"mm3-{role}-{quant}.gguf")
        if os.path.exists(out_path) and not force:
            log(f"  [{role:5s}] skip: {os.path.basename(out_path)} exists")
            continue
        write_split(reader, out_path, role, quant)


def main():
    ap = argparse.ArgumentParser(description="Split mm3-synth GGUF bundles into per-component files")
    ap.add_argument("bundles", nargs="+", help="mm3-synth-<quant>.gguf file(s)")
    ap.add_argument("--out", default=None, help="output directory (default: alongside each bundle)")
    ap.add_argument("--all-components", action="store_true",
                    help="emit cond/voc from quantised bundles too (default: f16/bf16/f32 only)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    for src in args.bundles:
        if not os.path.exists(src):
            die(f"{src}: not found")
        out_dir = args.out or os.path.dirname(os.path.abspath(src))
        os.makedirs(out_dir, exist_ok=True)
        split_bundle(src, out_dir, args.all_components, args.force)
    log("done")


if __name__ == "__main__":
    main()

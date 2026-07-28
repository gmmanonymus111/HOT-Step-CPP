#!/usr/bin/env python
"""dump_mdx23c_goldens.py — PyTorch reference output for the MDX23C GGML port.

Runs ZFTurbo's no-STFT TFC_TDF_net under torch.no_grad and writes a flat binary
that engine/tools/mdx23c-test.cpp compares against.

    magic "MDXG"
    int32 T, dim_f, cin, n_inst
    f32   input  [T * dim_f * cin]     torch [b, cin, dim_f, T], t fastest
    f32   output [n_inst * cin * dim_f * T]

Memory is bounded: a no_grad forward at the trained T=256 frees activations as
it goes. No tracing anywhere.

The upstream module does `from utils import prefer_target_instrument`, and that
utils.py drags in the whole MSS model zoo (demucs, scnet, omegaconf). We stub
the one function instead — it is three lines — so this needs only torch.

USAGE
    $py = "d:\\Ace-Step-Latest\\hot-step-9000\\.venv\\Scripts\\python.exe"
    & $py scripts\\dump_mdx23c_goldens.py `
        --config <SuperSep>\\models\\config_drumsep_mdx23c.yaml `
        --ckpt   <SuperSep>\\models\\MDX23C-DrumSep-aufr33-jarredou.ckpt `
        --output models\\supersep-ckpt\\goldens_mdx23c.bin
"""

import argparse
import importlib.util
import os
import struct
import sys
import types

import numpy as np
import torch
import yaml

DEFAULT_MSS_REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "MSS_ONNX_TensorRT"))


def prefer_target_instrument(config):
    """Mirror of MSS utils.prefer_target_instrument (avoids importing its utils)."""
    if getattr(config.training, "target_instrument", None):
        return [config.training.target_instrument]
    return config.training.instruments


def load_net_class(mss_repo):
    mod_path = os.path.join(mss_repo, "models_without_stft",
                            "mdx23c_tfc_tdf_v3_no_stft.py")
    if not os.path.isfile(mod_path):
        raise SystemExit(f"Could not find {mod_path}\nPass --mss-repo <checkout>.")
    if mss_repo not in sys.path:
        sys.path.insert(0, mss_repo)
    # Stub `utils` so the module's single import does not pull in the zoo.
    stub = types.ModuleType("utils")
    stub.prefer_target_instrument = prefer_target_instrument
    sys.modules.setdefault("utils", stub)
    spec = importlib.util.spec_from_file_location("mdx23c_no_stft", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TFC_TDF_net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mss-repo", default=DEFAULT_MSS_REPO)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from ml_collections import ConfigDict
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = ConfigDict(yaml.load(f, Loader=yaml.UnsafeLoader))

    Net = load_net_class(os.path.abspath(args.mss_repo))
    model = Net(cfg)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"missing weights: {missing[:6]}")
    if unexpected:
        print(f"[warn] unexpected: {unexpected[:6]}")
    model.eval()

    dim_f = cfg.audio.dim_f
    cin   = cfg.audio.num_channels * 2
    T     = cfg.audio.chunk_size // cfg.audio.hop_length + 1
    n_inst = len(prefer_target_instrument(cfg))

    print(f"[info] T={T} dim_f={dim_f} cin={cin} instruments={n_inst}")

    torch.manual_seed(args.seed)
    x = torch.randn(1, cin, dim_f, T) * 0.05

    with torch.no_grad():
        y = model(x)
    print(f"[info] output shape {tuple(y.shape)} range "
          f"[{y.min():.4f}, {y.max():.4f}]")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(b"MDXG")
        f.write(struct.pack("<4i", T, dim_f, cin, n_inst))
        f.write(np.ascontiguousarray(x[0], dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(y[0], dtype=np.float32).tobytes())

    print(f"[ok] {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()

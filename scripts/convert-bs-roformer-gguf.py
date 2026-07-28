#!/usr/bin/env python
"""convert-bs-roformer-gguf.py — BS-RoFormer .ckpt → GGUF for the HOT-Step engine.

A pure state-dict repack: reads the checkpoint tensor by tensor and writes GGUF.
There is NO graph tracing, so peak memory is roughly the size of the checkpoint
(~270 MB for Leap Xe) rather than the tens of gigabytes that torch.onnx.export
needs to hold a fully unrolled 16-layer axial transformer at T=1722.

Consumed by engine/src/bs-roformer-ggml.h.

TENSOR NAMING
-------------
Flat and index-addressable so the C++ side can build names with snprintf:

    band_split.{b}.norm            [dim_in]            RMSNorm gamma
    band_split.{b}.w               [dim_in, dim]       Linear weight
    band_split.{b}.b               [dim]               Linear bias

    blk.{i}.{time|freq}.rope_freqs [dim_head/2]        rotary inv-freqs
    blk.{i}.{time|freq}.attn_norm  [dim]
    blk.{i}.{time|freq}.qkv        [dim, 3*dim_inner]  fused, no bias
    blk.{i}.{time|freq}.gates_w    [dim, heads]
    blk.{i}.{time|freq}.gates_b    [heads]
    blk.{i}.{time|freq}.out        [dim_inner, dim]    no bias
    blk.{i}.{time|freq}.ff_norm    [dim]
    blk.{i}.{time|freq}.ff1_w      [dim, ff_inner]
    blk.{i}.{time|freq}.ff1_b      [ff_inner]
    blk.{i}.{time|freq}.ff2_w      [ff_inner, dim]
    blk.{i}.{time|freq}.ff2_b      [dim]

    final_norm                     [dim]

    mask.{s}.{b}.w1                [dim, ff_inner]     Linear
    mask.{s}.{b}.b1                [ff_inner]
    mask.{s}.{b}.w2                [ff_inner, dim_in*2]  Linear (GLU halves it)
    mask.{s}.{b}.b2                [dim_in*2]

Shapes above are in GGML order (ne[0] innermost). A torch Linear weight is
(out, in); gguf reverses numpy dims on write, so passing it through unchanged
yields ne = [in, out], which is what ggml_mul_mat wants.

Note `band_split[b]` input width is freqs_per_bands[b] * 2 (stereo) * 2
(complex) — the yaml lists frequency bins, not feature width.

USAGE
-----
    $py = "d:\\Ace-Step-Latest\\hot-step-9000\\.venv\\Scripts\\python.exe"
    & $py tools\\convert-bs-roformer-gguf.py `
        --config models\\supersep-ckpt\\Xe\\leap_xe_config_voc.yaml `
        --ckpt   models\\supersep-ckpt\\Xe\\bs_leap_xe_voc.ckpt `
        --output models\\supersep\\bs_leap_xe_voc-F32.gguf
"""

import argparse
import re
import os

import numpy as np
import torch
import yaml

import gguf

ARCH = "bs-roformer"


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        # freqs_per_bands uses !!python/tuple, so safe_load will not do.
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def load_state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    if sd and all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    return sd


class Repacker:
    """Pulls named tensors out of the state dict, tracking what was consumed."""

    def __init__(self, sd, writer):
        self.sd = sd
        self.writer = writer
        self.used = set()
        self.written = 0

    def put(self, dst, src, expect_dims=None):
        if src not in self.sd:
            raise SystemExit(f"missing tensor in checkpoint: {src}")
        t = self.sd[src]
        if expect_dims is not None and t.dim() != expect_dims:
            raise SystemExit(
                f"{src}: expected {expect_dims}-D, got {tuple(t.shape)}")
        arr = t.detach().to(torch.float32).contiguous().numpy()
        self.writer.add_tensor(dst, arr)
        self.used.add(src)
        self.written += 1

    def report(self):
        leftover = sorted(set(self.sd) - self.used)
        if leftover:
            raise SystemExit(
                f"{len(leftover)} checkpoint tensors were not written, e.g. "
                f"{leftover[:6]}\nThe converter does not understand this "
                "checkpoint layout — refusing to emit a partial model.")
        print(f"[ok] wrote {self.written} tensors, none left over")


def convert(cfg, sd, out_path):
    mc = cfg["model"]
    dim = mc["dim"]
    depth = mc["depth"]
    heads = mc["heads"]
    dim_head = mc["dim_head"]
    n_stems = mc.get("num_stems", 1)
    n_fft = mc["stft_n_fft"]
    hop = mc["stft_hop_length"]
    win = mc.get("stft_win_length", n_fft)
    stereo = bool(mc.get("stereo", True))
    n_ch = 2 if stereo else 1
    mlp_mult = mc.get("mlp_expansion_factor", 4)
    chunk = cfg["audio"]["chunk_size"]

    n_freqs = n_fft // 2 + 1

    if "freqs_per_bands" in mc:
        # BS-RoFormer: contiguous bands that tile the spectrum exactly.
        arch = "bs"
        freqs_per_bands = list(mc["freqs_per_bands"])
        if sum(freqs_per_bands) != n_freqs:
            raise SystemExit(
                f"freqs_per_bands sums to {sum(freqs_per_bands)}, expected "
                f"{n_freqs} (n_fft//2+1)")
    else:
        # Mel-Band RoFormer: OVERLAPPING bands from a mel filterbank. The bands
        # do not tile — the caller gathers freq_indices before the graph and
        # scatters the mask back afterwards (see mel_band_tables.inc), so the
        # per-band widths come from how many bins each mel filter touches and
        # their sum is the *gathered* length, not n_freqs.
        arch = "mel"
        from librosa import filters
        num_bands = mc["num_bands"]
        fb = filters.mel(sr=mc.get("sample_rate", 44100), n_fft=n_fft, n_mels=num_bands)
        fb = np.asarray(fb)
        # Matches MelBandRoformer.__init__: force the first/last bins on so
        # every frequency is covered by at least one band.
        fb[0][0] = 1.0
        fb[-1, -1] = 1.0
        per_band = (fb > 0)
        if not per_band.any(axis=0).all():
            raise SystemExit("mel filterbank leaves some frequencies uncovered")
        freqs_per_bands = per_band.sum(axis=1).tolist()

    n_bands = len(freqs_per_bands)
    # Feature width per band: bins * channels * 2 (real/imag).
    band_widths = [int(f) * n_ch * 2 for f in freqs_per_bands]

    dim_inner = heads * dim_head
    ff_inner = dim * mlp_mult
    target = cfg.get("training", {}).get("target_instrument", "unknown")

    print(f"[info] dim={dim} depth={depth} heads={heads} dim_head={dim_head}")
    print(f"[info] bands={n_bands} stems={n_stems} target={target}")
    print(f"[info] n_fft={n_fft} hop={hop} chunk={chunk} -> T={chunk // hop + 1}")

    if mc.get("linear_transformer_depth", 0) != 0:
        raise SystemExit("linear_transformer_depth != 0 is not supported")
    if mc.get("time_transformer_depth", 1) != 1 or \
       mc.get("freq_transformer_depth", 1) != 1:
        raise SystemExit("only time/freq_transformer_depth == 1 is supported")
    if mc.get("skip_connection", False):
        raise SystemExit("skip_connection=True is not supported")

    # Structural variations between checkpoints, detected from the state dict
    # rather than trusted from the yaml (the Mel-Band Karaoke config claims
    # mask_estimator_depth 2 but its MLPs actually have 3 Linears).
    has_out_norm   = any(re.match(r"layers\.\d+\.\d+\.norm\.", k) for k in sd)
    has_final_norm = any(k.startswith("final_norm") for k in sd)
    mask_idx = sorted({int(k.split(".")[5]) for k in sd
                       if k.startswith("mask_estimators.") and k.endswith(".weight")})
    mask_layers = len(mask_idx)
    if mask_layers == 0:
        raise SystemExit("no mask_estimators found in checkpoint")

    print(f"[info] arch={arch} mask_layers={mask_layers} (indices {mask_idx}) "
          f"out_norm={has_out_norm} final_norm={has_final_norm}")

    w = gguf.GGUFWriter(out_path, ARCH)

    w.add_string("bs_roformer.arch", arch)
    w.add_uint32("bs_roformer.mask_layers", mask_layers)
    w.add_bool("bs_roformer.has_out_norm", has_out_norm)
    w.add_bool("bs_roformer.has_final_norm", has_final_norm)
    w.add_uint32("bs_roformer.dim", dim)
    w.add_uint32("bs_roformer.depth", depth)
    w.add_uint32("bs_roformer.heads", heads)
    w.add_uint32("bs_roformer.dim_head", dim_head)
    w.add_uint32("bs_roformer.dim_inner", dim_inner)
    w.add_uint32("bs_roformer.ff_inner", ff_inner)
    w.add_uint32("bs_roformer.n_bands", n_bands)
    w.add_uint32("bs_roformer.n_stems", n_stems)
    w.add_uint32("bs_roformer.n_channels", n_ch)
    w.add_uint32("bs_roformer.n_fft", n_fft)
    w.add_uint32("bs_roformer.hop_length", hop)
    w.add_uint32("bs_roformer.win_length", win)
    w.add_uint32("bs_roformer.chunk_size", chunk)
    w.add_array("bs_roformer.band_widths", band_widths)
    w.add_string("bs_roformer.target_instrument", str(target))

    r = Repacker(sd, w)

    # ── Band split: 90 × [RMSNorm(dim_in) -> Linear(dim_in, dim)] ──────────
    for b in range(n_bands):
        p = f"band_split.to_features.{b}"
        r.put(f"band_split.{b}.norm", f"{p}.0.gamma", 1)
        r.put(f"band_split.{b}.w",    f"{p}.1.weight", 2)
        r.put(f"band_split.{b}.b",    f"{p}.1.bias", 1)

    # ── Axial transformer body ────────────────────────────────────────────
    # layers[i][0] = time transformer, layers[i][1] = freq transformer.
    # Each has depth 1, so exactly one (Attention, FeedForward) pair, and
    # norm_output is False (no trailing per-Transformer norm — confirmed by
    # the checkpoint's tensor count reconciling exactly without one).
    for i in range(depth):
        for axis_idx, axis in ((0, "time"), (1, "freq")):
            src = f"layers.{i}.{axis_idx}.layers.0"
            dst = f"blk.{i}.{axis}"
            r.put(f"{dst}.rope_freqs", f"{src}.0.rotary_embed.freqs", 1)
            r.put(f"{dst}.attn_norm",  f"{src}.0.norm.gamma", 1)
            r.put(f"{dst}.qkv",        f"{src}.0.to_qkv.weight", 2)
            r.put(f"{dst}.gates_w",    f"{src}.0.to_gates.weight", 2)
            r.put(f"{dst}.gates_b",    f"{src}.0.to_gates.bias", 1)
            r.put(f"{dst}.out",        f"{src}.0.to_out.0.weight", 2)
            # FeedForward Sequential: 0=RMSNorm 1=Linear 2=GELU 3=Dropout
            #                         4=Linear 5=Dropout
            r.put(f"{dst}.ff_norm",    f"{src}.1.net.0.gamma", 1)
            r.put(f"{dst}.ff1_w",      f"{src}.1.net.1.weight", 2)
            r.put(f"{dst}.ff1_b",      f"{src}.1.net.1.bias", 1)
            r.put(f"{dst}.ff2_w",      f"{src}.1.net.4.weight", 2)
            r.put(f"{dst}.ff2_b",      f"{src}.1.net.4.bias", 1)

            if has_out_norm:
                r.put(f"{dst}.out_norm", f"layers.{i}.{axis_idx}.norm.gamma", 1)

    if has_final_norm:
        r.put("final_norm", "final_norm.gamma", 1)

    # ── Mask estimators: per stem, per band ───────────────────────────────
    # to_freqs[b] = Sequential(MLP, GLU); MLP = Sequential(Linear, Tanh, Linear)
    for s in range(n_stems):
        for b in range(n_bands):
            p = f"mask_estimators.{s}.to_freqs.{b}.0"
            d = f"mask.{s}.{b}"
            for n, idx in enumerate(mask_idx):
                r.put(f"{d}.w{n + 1}", f"{p}.{idx}.weight", 2)
                r.put(f"{d}.b{n + 1}", f"{p}.{idx}.bias", 1)

    r.report()

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[ok] {out_path} ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Model yaml")
    ap.add_argument("--ckpt", required=True, help="Checkpoint .ckpt")
    ap.add_argument("--output", required=True, help="Destination .gguf")
    args = ap.parse_args()

    cfg = read_config(args.config)
    sd = load_state_dict(args.ckpt)
    print(f"[info] checkpoint: {len(sd)} tensors, "
          f"{sum(v.numel() for v in sd.values()):,} params")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    convert(cfg, sd, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""dump_bs_roformer_goldens.py — PyTorch reference activations for the GGML port.

Runs ZFTurbo's no-STFT BS-RoFormer under torch.no_grad at a SHORT time length
and dumps intermediate tensors, so engine/src/bs-roformer-ggml.h can be
validated stage by stage instead of only at the output.

MEMORY
------
Deliberately runs at T=256 (~3 s of audio) rather than the model's trained
T=1722. A no_grad forward frees activations as it goes, so peak is bounded by
the largest few tensors (~23 MB each at T=256) plus the 268 MB of weights.
Numerics do not depend on sequence length, so a short T validates the graph
just as well. Do NOT raise --time-steps to 1722 "to be thorough" — that is not
what broke before (tracing was), but there is no reason to pay for it either.

WHAT IT DUMPS
-------------
Full tensors (for elementwise comparison):
    input           [1, T, F*C*2]
    band_split      [1, T, n_bands, dim]
    layer_00        [1, T, n_bands, dim]   after layers[0] (time + freq)
    layer_01        [1, T, n_bands, dim]
    layer_last      [1, T, n_bands, dim]
    final_norm      [1, T, n_bands, dim]
    mask            [1, S, F*C, T, 2]      the graph's actual output

Per-layer summary stats for all `depth` layers (mean/std/absmax), enough to
localise which layer a divergence starts in without storing 16 full tensors.

USAGE
-----
    $py = "d:\\Ace-Step-Latest\\hot-step-9000\\.venv\\Scripts\\python.exe"
    & $py tools\\dump_bs_roformer_goldens.py `
        --config models\\supersep-ckpt\\Xe\\leap_xe_config_voc.yaml `
        --ckpt   models\\supersep-ckpt\\Xe\\bs_leap_xe_voc.ckpt `
        --output models\\supersep-ckpt\\goldens_voc.npz
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import yaml

DEFAULT_MSS_REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "MSS_ONNX_TensorRT"))


def load_no_stft_class(mss_repo, arch):
    """Import the amputated model class for `arch` ('bs' or 'mel')."""
    fname, cls = (("bs_roformer_no_stft.py", "BSRoformer") if arch == "bs"
                  else ("mel_band_roformer_no_stft.py", "MelBandRoformer"))
    mod_path = os.path.join(mss_repo, "models_without_stft", fname)
    if not os.path.isfile(mod_path):
        raise SystemExit(f"Could not find {mod_path}\nPass --mss-repo <checkout>.")
    if mss_repo not in sys.path:
        sys.path.insert(0, mss_repo)
    spec = importlib.util.spec_from_file_location(fname[:-3], mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls)


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def build_model(BSRoformer, cfg, ckpt_path):
    kwargs = dict(cfg["model"])
    kwargs["flash_attn"] = False        # match what the GGML graph computes
    kwargs["use_torch_checkpoint"] = False
    model = BSRoformer(**kwargs)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    if sd and all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"missing weights: {missing[:6]}")
    if unexpected:
        print(f"[warn] unexpected: {unexpected[:6]}")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mss-repo", default=DEFAULT_MSS_REPO)
    ap.add_argument("--time-steps", type=int, default=256,
                    help="T to run at (default 256; see MEMORY in the docstring)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = read_config(args.config)
    mc = cfg["model"]
    arch = "bs" if "freqs_per_bands" in mc else "mel"
    ModelCls = load_no_stft_class(os.path.abspath(args.mss_repo), arch)
    model = build_model(ModelCls, cfg, args.ckpt)

    n_ch = 2 if mc.get("stereo", True) else 1
    depth = mc["depth"]
    T = args.time_steps

    torch.manual_seed(args.seed)
    # Scaled to roughly the magnitude of a real STFT bin so activations land in
    # a representative range rather than an artificially tiny one.
    if arch == "bs":
        # BSRoformer.forward takes the flattened [b, t, (f c)] directly.
        in_dim = (mc["stft_n_fft"] // 2 + 1) * n_ch * 2
        x = torch.randn(1, T, in_dim) * 0.05
        model_in = x
    else:
        # MelBandRoformer.forward takes [b, f, t, c] and flattens it itself,
        # where f is the GATHERED index count (bands overlap), so derive it
        # from the model's own freq_indices buffer rather than from n_fft.
        n_gathered = int(model.freq_indices.numel())
        in_dim = n_gathered * 2
        model_in = torch.randn(1, n_gathered, T, 2) * 0.05
        # Flattened view is what the C++ side feeds: [t, (f c)]
        x = model_in.permute(0, 2, 1, 3).reshape(1, T, in_dim).contiguous()
    print(f"[info] arch={arch} in_dim={in_dim}")

    out = {}
    stats = []

    def store(name, t):
        out[name] = t.detach().to(torch.float32).contiguous().numpy()

    # Forward hooks capture intermediates without touching the model source.
    captured = {}

    def hook(tag):
        def fn(_mod, _inp, output):
            captured[tag] = output.detach()
        return fn

    handles = [model.band_split.register_forward_hook(hook("band_split"))]
    # Mel-Band Karaoke has norm_output=True on each Transformer and NO
    # final_norm; hooking the last layer stands in for it there.
    has_final_norm = not isinstance(model.final_norm, torch.nn.Identity) \
        if hasattr(model, "final_norm") else False
    if has_final_norm:
        handles.append(model.final_norm.register_forward_hook(hook("final_norm")))
    for i, block in enumerate(model.layers):
        # block is ModuleList([time_transformer, freq_transformer]); hooking the
        # freq transformer captures the state after the full layer.
        handles.append(block[-1].register_forward_hook(hook(f"L{i}")))

    print(f"[info] running T={T}, in_dim={in_dim}, depth={depth}")
    with torch.no_grad():
        mask = model(model_in)

    for h in handles:
        h.remove()

    store("input", x)
    store("mask", mask)
    store("band_split", captured["band_split"])
    # With no final_norm the C++ stage depth+1 is a no-op passthrough, so the
    # last layer's output is exactly what it should produce.
    store("final_norm", captured["final_norm"] if has_final_norm
          else captured[f"L{depth - 1}"])
    store("layer_00", captured["L0"])
    store("layer_01", captured["L1"])
    store("layer_last", captured[f"L{depth - 1}"])

    for i in range(depth):
        t = captured[f"L{i}"].float()
        stats.append([t.mean().item(), t.std().item(), t.abs().max().item()])
    out["layer_stats"] = np.asarray(stats, dtype=np.float32)  # [depth, 3]

    out["meta"] = np.asarray([T, in_dim, depth, mask.shape[1], mask.shape[2]],
                             dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output, **out)

    print(f"[ok] {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")
    print(f"[ok] mask shape {tuple(mask.shape)}  "
          f"range [{mask.min():.4f}, {mask.max():.4f}]")
    print("[ok] per-layer absmax: " +
          " ".join(f"{s[2]:.2f}" for s in stats))


if __name__ == "__main__":
    main()

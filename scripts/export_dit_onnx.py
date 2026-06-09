"""Export AceStep DiT decoder from safetensors to ONNX for native TRT.

Self-contained script: doesn't require the 'acestep' package. Loads the model
definition directly from the model directory's modeling file.

The exported ONNX matches HOT-Step's dit-trt.h I/O layout:
  - input_latents:  [B, T, 192]  fp32  (context[128] + xt[64], pre-concatenated)
  - enc_hidden:     [B, S, 2048] fp32
  - t:              [B]          fp32
  - t_r:            [B]          fp32
  - velocity:       [B, T, 64]   fp32  (output)

Usage:
  python scripts/export_dit_onnx.py ^
      --model-dir models/acestep-v15-merge-sft-turbo-xl-ta-0.7 ^
      --output-dir models/onnx/dit-stream
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import shutil
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


# ── Traceable Lambda replacement ────────────────────────────────────────────
class _Transpose12(nn.Module):
    """Drop-in for Lambda(lambda x: x.transpose(1, 2))."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


# ── Export wrapper ──────────────────────────────────────────────────────────
class DiTForTRTExport(nn.Module):
    """Wraps AceStepDiTModel for ONNX export matching dit-trt.h I/O layout.

    Takes pre-concatenated input_latents [B, T, 192] and exposes t/t_r
    as separate inputs, matching the C++ engine's tensor layout.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        self._replace_lambdas()
        self.decoder.config._attn_implementation = "sdpa"
        self._patch_decoder_for_trace()

    def _replace_lambdas(self) -> None:
        for seq in (self.decoder.proj_in, self.decoder.proj_out):
            for i, mod in enumerate(seq):
                if type(mod).__name__ == "Lambda":
                    seq[i] = _Transpose12()

    def _patch_decoder_for_trace(self) -> None:
        """Monkey-patch decoder forward for ONNX traceability."""
        try:
            import transformers.integrations.sdpa_attention as _sdpa_mod
            _sdpa_mod.use_gqa_in_sdpa = lambda *args, **kwargs: False
        except (ImportError, AttributeError):
            pass

        decoder = self.decoder
        sliding_window = decoder.config.sliding_window
        layer_types = decoder.config.layer_types

        time_embed_dim = decoder.time_embed.time_proj.out_features // 6

        def _patched_time_embed_forward(self_te, t):
            t_freq = self_te.timestep_embedding(t, self_te.in_channels)
            temb = self_te.linear_1(t_freq.to(t.dtype))
            temb = self_te.act1(temb)
            temb = self_te.linear_2(temb)
            timestep_proj = self_te.time_proj(self_te.act2(temb)).reshape(-1, 6, time_embed_dim)
            return temb, timestep_proj

        decoder.time_embed.forward = types.MethodType(
            _patched_time_embed_forward, decoder.time_embed)
        decoder.time_embed_r.forward = types.MethodType(
            _patched_time_embed_forward, decoder.time_embed_r)

        def _export_forward(
            self_dec, hidden_states, timestep, timestep_r,
            attention_mask, encoder_hidden_states, encoder_attention_mask,
            context_latents, use_cache=None, past_key_values=None,
            cache_position=None, position_ids=None, output_attentions=False,
            return_hidden_states=None, custom_layers_config=None,
            enable_early_exit=False, **flash_attn_kwargs,
        ):
            temb_t, timestep_proj_t = self_dec.time_embed(timestep)
            temb_r, timestep_proj_r = self_dec.time_embed_r(timestep - timestep_r)
            temb = temb_t + temb_r
            timestep_proj = timestep_proj_t + timestep_proj_r

            hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
            hidden_states = self_dec.proj_in(hidden_states)
            encoder_hidden_states = self_dec.condition_embedder(encoder_hidden_states)

            seq_len_pat = hidden_states.shape[1]
            cache_position = torch.arange(seq_len_pat, device=hidden_states.device)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = self_dec.rotary_emb(hidden_states, position_ids)

            indices = cache_position
            diff = indices.unsqueeze(0) - indices.unsqueeze(1)
            sw_mask = torch.where(
                torch.abs(diff) <= sliding_window,
                torch.zeros(1, device=hidden_states.device, dtype=hidden_states.dtype),
                torch.full((1,), torch.finfo(hidden_states.dtype).min,
                           device=hidden_states.device, dtype=hidden_states.dtype),
            )
            sw_mask = sw_mask.unsqueeze(0).unsqueeze(0)

            for i, layer_module in enumerate(self_dec.layers):
                attn_mask = sw_mask if layer_types[i] == "sliding_attention" else None
                layer_outputs = layer_module(
                    hidden_states, position_embeddings, timestep_proj, attn_mask,
                    position_ids, None, False, False, cache_position,
                    encoder_hidden_states, None,
                )
                hidden_states = layer_outputs[0]

            shift, scale = (self_dec.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            hidden_states = (self_dec.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
            hidden_states = self_dec.proj_out(hidden_states)
            return (hidden_states, None)

        decoder.forward = types.MethodType(_export_forward, decoder)

    def forward(
        self,
        input_latents: torch.Tensor,   # [B, T, 192]
        enc_hidden: torch.Tensor,       # [B, S, 2048]
        t: torch.Tensor,                # [B]
        t_r: torch.Tensor,              # [B]
    ) -> torch.Tensor:
        context_latents = input_latents[..., :128]
        hidden_states = input_latents[..., 128:]
        outputs = self.decoder(
            hidden_states=hidden_states, timestep=t, timestep_r=t_r,
            attention_mask=None, encoder_hidden_states=enc_hidden,
            encoder_attention_mask=None, context_latents=context_latents,
            use_cache=False, past_key_values=None, output_attentions=False,
        )
        return outputs[0]


# ── Load model from directory ────────────────────────────────────────────────

def _load_module_from_file(name: str, filepath: str):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_model(model_dir: str, device: str = "cpu"):
    """Load AceStepConditionGenerationModel from a safetensors directory.

    Handles the 'acestep' module dependency by loading the config class
    from the DEMON project or from a standalone copy.
    """
    model_dir = Path(model_dir).resolve()

    # Load config.json
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    # ── Bootstrap the 'acestep' package from DEMON project ──────────────
    # The model dir's shim files (configuration_acestep_v15.py, apg_guidance.py)
    # import from 'acestep.models.common.*', which maps to DEMON's 'acestep.models.*'.
    # We add DEMON to sys.path AND create a 'common' alias pointing to 'models'.
    demon_root = Path("d:/Ace-Step-Latest/Demon")
    if not demon_root.exists():
        raise RuntimeError(
            "DEMON project not found at d:/Ace-Step-Latest/Demon — "
            "needed for model class definitions")

    if str(demon_root) not in sys.path:
        sys.path.insert(0, str(demon_root))
    print(f"[Export] Using DEMON project for acestep package: {demon_root}")

    # Import the real acestep package so its __init__.py runs
    import acestep
    import acestep.models

    # The model-dir shims import from 'acestep.models.common.*' but DEMON
    # puts everything directly in 'acestep.models.*'. Create a 'common'
    # alias that points to the real 'acestep.models' module.
    sys.modules["acestep.models.common"] = sys.modules["acestep.models"]

    # Now load the config
    from acestep.models.configuration_acestep_v15 import AceStepConfig
    config = AceStepConfig(**config_dict)
    print(f"[Export] Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")

    # Register the config module under its short name too (the modeling file
    # tries 'from configuration_acestep_v15 import AceStepConfig' as fallback)
    config_shim = types.ModuleType("configuration_acestep_v15")
    config_shim.AceStepConfig = AceStepConfig
    sys.modules["configuration_acestep_v15"] = config_shim

    # Register apg_guidance similarly (modeling file does 'from apg_guidance import ...')
    import acestep.models.apg_guidance as _real_apg
    sys.modules["apg_guidance"] = _real_apg
    # Also alias 'acestep.models.common.apg_guidance' → the real module
    sys.modules["acestep.models.common.apg_guidance"] = _real_apg
    sys.modules["acestep.models.common.configuration_acestep_v15"] = sys.modules[
        "acestep.models.configuration_acestep_v15"]

    # Load modeling file
    modeling_file = model_dir / "modeling_acestep_v15_xl_base.py"
    if not modeling_file.exists():
        raise FileNotFoundError(f"Modeling file not found: {modeling_file}")

    print(f"[Export] Loading model definition from {modeling_file.name}...")

    # Add model_dir to path for local imports
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    model_mod = _load_module_from_file("_acestep_modeling", str(modeling_file))
    model_cls = model_mod.AceStepConditionGenerationModel

    # Create model on CPU (meta device doesn't work with ResidualFSQ)
    print(f"[Export] Creating model on CPU (this uses ~10GB RAM temporarily)...")
    model = model_cls(config)

    # Load safetensors weights
    from safetensors.torch import load_file

    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            weight_files = set(index["weight_map"].values())
            state_dict = {}
            for wf in sorted(weight_files):
                print(f"[Export] Loading shard: {wf}")
                shard = load_file(str(model_dir / wf), device="cpu")
                state_dict.update(shard)
        else:
            raise FileNotFoundError(f"No model.safetensors or index file in {model_dir}")
    else:
        fsize = safetensors_path.stat().st_size / (1 << 20)
        print(f"[Export] Loading weights: {safetensors_path.name} ({fsize:.0f} MB)...")
        state_dict = load_file(str(safetensors_path), device="cpu")

    print(f"[Export] Loading {len(state_dict)} tensors into model...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Export] WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[Export] WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    del state_dict  # Free RAM

    # Only move the decoder to GPU (we don't need encoders/tokenizer for export)
    model.decoder = model.decoder.to(device)
    model.eval()

    return model, config


def export_onnx(
    model_dir: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 1,
    seq_len: int = 750,
    enc_len: int = 200,
    opset: int = 17,
):
    """Export DiT decoder to ONNX in FP32."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Export] Loading model from {model_dir}...")
    model, config = load_model(model_dir, device=device)

    # Extract the decoder (DiT model)
    decoder = model.decoder
    param_count = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"[Export] Decoder: {config.num_hidden_layers} layers, "
          f"hidden={config.hidden_size}, {param_count:.0f}M params")

    # Wrap for export
    wrapper = DiTForTRTExport(decoder).float().to(device).eval()
    print("[Export] Exporting in FP32 precision")

    B, T, S = batch_size, seq_len, enc_len
    print(f"[Export] Trace shapes: B={B}, T={T}, S={S}")

    example_inputs = (
        torch.randn(B, T, 192, device=device, dtype=torch.float32),
        torch.randn(B, S, 2048, device=device, dtype=torch.float32),
        torch.full((B,), 0.5, device=device, dtype=torch.float32),
        torch.full((B,), 0.5, device=device, dtype=torch.float32),
    )

    input_names = ["input_latents", "enc_hidden", "t", "t_r"]
    output_names = ["velocity"]

    dynamic_axes = {
        "input_latents": {0: "batch", 1: "seq_len"},
        "enc_hidden":    {0: "batch", 1: "enc_len"},
        "t":             {0: "batch"},
        "t_r":           {0: "batch"},
        "velocity":      {0: "batch", 1: "seq_len"},
    }

    onnx_path = output_dir / "dit.onnx"

    with torch.no_grad():
        print("[Export] Running test forward pass...")
        test_out = wrapper(*example_inputs)
        print(f"[Export] Test output shape: {test_out.shape}, "
              f"mean={test_out.float().mean():.6f}, "
              f"std={test_out.float().std():.6f}")

        if torch.isnan(test_out).any():
            print("[Export] ERROR: test output contains NaN! Aborting.")
            return None

        print("[Export] Exporting to ONNX (this may take a few minutes)...")
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    size_mb = onnx_path.stat().st_size / (1 << 20)
    print(f"[Export] ONNX saved to {onnx_path} ({size_mb:.1f} MB)")

    # Copy config
    model_dir_path = Path(model_dir)
    for fname in ["config.json", "silence_latent.pt"]:
        src = model_dir_path / fname
        if src.exists():
            shutil.copy2(str(src), str(output_dir / fname))
            print(f"[Export] Copied {fname}")

    print(f"\n{'='*60}")
    print(f"[Export] SUCCESS! ONNX model at: {onnx_path}")
    print(f"[Export] Size: {size_mb:.1f} MB")
    print(f"{'='*60}")

    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AceStep DiT to ONNX")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=750)
    parser.add_argument("--enc-len", type=int, default=200)
    parser.add_argument("--opset", type=int, default=17)

    args = parser.parse_args()
    export_onnx(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        enc_len=args.enc_len,
        opset=args.opset,
    )

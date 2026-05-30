#!/usr/bin/env python3
"""
export_cond_enc.py — Export AceStep Condition Encoder to ONNX.

The condition encoder takes outputs from the text encoder (text_hidden + lyric_embed)
and reference audio features (timbre_feats), and produces enc_hidden for DiT cross-attention.

Internal architecture:
  - text_projector: Linear(1024→2048, no bias) — projects text encoder output
  - lyric_encoder: Linear(1024→2048)+bias → 8-layer bidirectional Qwen3 → RMSNorm
  - timbre_encoder: Linear(64→2048)+bias → [CLS prepend] → 4-layer bidir Qwen3 → RMSNorm → position[0]
  - cat(lyric_out, timbre_out, text_proj_out) → enc_hidden [B, S_total, 2048]

Usage:
    python export_cond_enc.py --model-dir <path-to-DiT-safetensors> --output <output.onnx>
"""

import argparse
import os
import sys
import time
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class CondEncoderWrapper(nn.Module):
    """Wrapper for ONNX export that simplifies the condition encoder interface.
    
    For inference (batch_size=1), we simplify:
    - No pack_sequences sorting (all tokens are valid, no padding)
    - Timbre: single reference, so unpack is trivial (just unsqueeze)
    - Output is simple cat(lyric, timbre, text_proj)
    
    ONNX inputs:
        text_hidden:  [B, S_text, 1024]  fp16 — from text encoder
        lyric_embed:  [B, S_lyric, 1024] fp16 — from embedding table lookup
        timbre_feats: [B, S_ref, 64]     fp16 — from VAE encoder (or zeros)
        has_timbre:   [1]                int64 — 1 if timbre is present, 0 if not
    
    ONNX output:
        enc_hidden:   [B, S_total, 2048] fp16 — packed conditioning
    """
    
    def __init__(self, cond_encoder):
        super().__init__()
        self.text_projector = cond_encoder.text_projector
        self.lyric_encoder = cond_encoder.lyric_encoder
        self.timbre_encoder = cond_encoder.timbre_encoder
    
    def forward(self, text_hidden, lyric_embed, timbre_feats, has_timbre):
        """
        Forward pass with simplified interface for ONNX export.
        
        Note: For ONNX tracing, has_timbre must be a tensor, not a Python bool.
        We use torch.where / masking to handle the conditional timbre path.
        """
        B = text_hidden.shape[0]
        
        # 1) Text projection: [B, S_text, 1024] → [B, S_text, 2048]
        text_proj = self.text_projector(text_hidden)
        
        # 2) Lyric encoding: [B, S_lyric, 1024] → 8L bidir Qwen3 → [B, S_lyric, 2048]
        S_lyric = lyric_embed.shape[1]
        lyric_mask = torch.ones(B, S_lyric, device=lyric_embed.device, dtype=torch.long)
        lyric_out = self.lyric_encoder(
            inputs_embeds=lyric_embed,
            attention_mask=lyric_mask,
        )
        if hasattr(lyric_out, 'last_hidden_state'):
            lyric_out = lyric_out.last_hidden_state
        else:
            lyric_out = lyric_out[0]
        
        # 3) Timbre encoding: [B, S_ref, 64] → 4L bidir Qwen3 → position[0] → [B, 1, 2048]
        # For ONNX: we always run the timbre path but zero out if has_timbre=0
        S_ref = timbre_feats.shape[1]
        timbre_mask = torch.ones(1, S_ref, device=timbre_feats.device, dtype=torch.long)
        # refer_audio_order_mask: all 0s means everything belongs to batch 0
        order_mask = torch.zeros(1, device=timbre_feats.device, dtype=torch.long)
        
        # Reshape for timbre encoder: expects [N_packed, S_ref, 64]
        timbre_input = timbre_feats  # [B, S_ref, 64]
        timbre_embs, timbre_embs_mask = self.timbre_encoder(
            refer_audio_acoustic_hidden_states_packed=timbre_input,
            refer_audio_order_mask=order_mask,
            attention_mask=timbre_mask,
        )
        # timbre_embs: [B, 1, 2048] — CLS token output per batch
        
        # 4) Concatenate: [lyric, timbre, text_proj]
        # When has_timbre=0, skip timbre in the concatenation
        # For ONNX compatibility, always cat but mask the timbre contribution
        ht = has_timbre[0]
        if ht > 0:
            enc_hidden = torch.cat([lyric_out, timbre_embs, text_proj], dim=1)
        else:
            enc_hidden = torch.cat([lyric_out, text_proj], dim=1)
        
        return enc_hidden


class CondEncoderWrapperFixed(nn.Module):
    """Fixed version that always includes timbre (simplifies ONNX graph).
    
    For inference, timbre is always present (silence latent as zero timbre).
    This avoids dynamic control flow in the ONNX graph.
    
    ONNX inputs:
        text_hidden:  [B, S_text, 1024]  fp16
        lyric_embed:  [B, S_lyric, 1024] fp16
        timbre_feats: [B, S_ref, 64]     fp16 (zeros if no reference)
    
    ONNX output:
        enc_hidden:   [B, S_total, 2048] fp16 where S_total = S_lyric + 1 + S_text
    """
    
    def __init__(self, cond_encoder):
        super().__init__()
        self.text_projector = cond_encoder.text_projector
        self.lyric_encoder = cond_encoder.lyric_encoder
        self.timbre_encoder = cond_encoder.timbre_encoder
    
    def forward(self, text_hidden, lyric_embed, timbre_feats):
        B = text_hidden.shape[0]
        
        # 1) Text projection
        text_proj = self.text_projector(text_hidden)
        
        # 2) Lyric encoding
        S_lyric = lyric_embed.shape[1]
        lyric_mask = torch.ones(B, S_lyric, device=lyric_embed.device, dtype=torch.long)
        lyric_out = self.lyric_encoder(
            inputs_embeds=lyric_embed,
            attention_mask=lyric_mask,
        )
        if hasattr(lyric_out, 'last_hidden_state'):
            lyric_out = lyric_out.last_hidden_state
        else:
            lyric_out = lyric_out[0]
        
        # 3) Timbre encoding — always run, zeros produce a neutral timbre embedding
        S_ref = timbre_feats.shape[1]
        timbre_mask = torch.ones(B, S_ref, device=timbre_feats.device, dtype=torch.long)
        order_mask = torch.zeros(B, device=timbre_feats.device, dtype=torch.long)
        # The timbre encoder expects [N_total, ...] packed format
        # For B=1, N_total=1, so just pass through
        timbre_embs, _ = self.timbre_encoder(
            refer_audio_acoustic_hidden_states_packed=timbre_feats,
            refer_audio_order_mask=order_mask,
            attention_mask=timbre_mask,
        )
        # timbre_embs: [B, 1, 2048]
        
        # 4) Concatenate: lyric + timbre + text_proj
        enc_hidden = torch.cat([lyric_out, timbre_embs, text_proj], dim=1)
        
        return enc_hidden


def load_model(model_dir: str, device: str = "cuda", dtype=torch.float16):
    """Load the AceStep model and extract the condition encoder."""
    model_dir = Path(model_dir)
    
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Monkey-patch transformers auto_docstring
    try:
        import transformers.utils.auto_docstring as _ad
        _ad.auto_docstring = lambda *a, **kw: (lambda cls: cls)
    except Exception:
        pass
    
    sys.path.insert(0, str(model_dir))
    
    from modeling_acestep_v15_xl_base import AceStepConditionEncoder
    from configuration_acestep_v15 import AceStepConfig
    
    import json
    with open(model_dir / "config.json") as f:
        config_dict = json.load(f)
    
    config = AceStepConfig(**config_dict)
    config._attn_implementation = "sdpa"
    
    print(f"[export_cond_enc] Loading model from {model_dir}...")
    t0 = time.time()
    
    cond_encoder = AceStepConditionEncoder(config)
    
    # Load weights — filter to encoder.* prefix
    from safetensors.torch import load_file
    st_path = model_dir / "model.safetensors"
    state_dict = load_file(str(st_path))
    
    cond_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            cond_state_dict[k[len("encoder."):]] = v
    
    missing, unexpected = cond_encoder.load_state_dict(cond_state_dict, strict=False)
    if missing:
        print(f"[export_cond_enc] Warning: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"[export_cond_enc] Warning: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    
    cond_encoder = cond_encoder.to(device=device, dtype=dtype)
    cond_encoder.eval()
    
    t1 = time.time()
    n_params = sum(p.numel() for p in cond_encoder.parameters()) / 1e6
    print(f"[export_cond_enc] Model loaded in {t1-t0:.1f}s ({n_params:.0f}M params)")
    print(f"[export_cond_enc] text_hidden_dim={config.text_hidden_dim}, hidden_size={config.hidden_size}")
    
    return cond_encoder, config


def export_onnx(cond_encoder, config, output_path: str, opset: int = 18):
    """Export the condition encoder to ONNX."""
    device = next(cond_encoder.parameters()).device
    dtype = next(cond_encoder.parameters()).dtype
    
    wrapper = CondEncoderWrapperFixed(cond_encoder)
    wrapper.eval()
    
    # Dummy inputs
    B = 1
    S_text = 64
    S_lyric = 128
    S_ref = 8   # 8 frames of reference audio (short clip)
    
    dummy_text_hidden = torch.randn(B, S_text, config.text_hidden_dim, device=device, dtype=dtype)
    dummy_lyric_embed = torch.randn(B, S_lyric, config.text_hidden_dim, device=device, dtype=dtype)
    dummy_timbre_feats = torch.randn(B, S_ref, config.timbre_hidden_dim, device=device, dtype=dtype)
    
    print(f"[export_cond_enc] Tracing with shapes: text={list(dummy_text_hidden.shape)}, "
          f"lyric={list(dummy_lyric_embed.shape)}, timbre={list(dummy_timbre_feats.shape)}")
    
    # Test forward
    print("[export_cond_enc] Testing forward pass...")
    with torch.no_grad():
        test_out = wrapper(dummy_text_hidden, dummy_lyric_embed, dummy_timbre_feats)
    expected_S = S_lyric + 1 + S_text  # lyric + timbre(1) + text
    print(f"[export_cond_enc] Output shape: {list(test_out.shape)} "
          f"(expected [{B}, {expected_S}, {config.hidden_size}])")
    
    # Export
    print(f"[export_cond_enc] Exporting to ONNX (opset {opset})...")
    t0 = time.time()
    
    torch.onnx.export(
        wrapper,
        (dummy_text_hidden, dummy_lyric_embed, dummy_timbre_feats),
        output_path,
        opset_version=opset,
        input_names=["text_hidden", "lyric_embed", "timbre_feats"],
        output_names=["enc_hidden"],
        dynamic_axes={
            "text_hidden":  {0: "batch", 1: "text_seq"},
            "lyric_embed":  {0: "batch", 1: "lyric_seq"},
            "timbre_feats": {0: "batch", 1: "timbre_seq"},
            "enc_hidden":   {0: "batch", 1: "enc_seq"},
        },
        do_constant_folding=True,
        export_params=True,
    )
    
    t1 = time.time()
    file_size = os.path.getsize(output_path)
    print(f"[export_cond_enc] Exported to {output_path}")
    print(f"[export_cond_enc] File size: {file_size/1e6:.1f} MB")
    print(f"[export_cond_enc] Export time: {t1-t0:.1f}s")


def export_null_cond_emb(model_dir: str, output_path: str):
    """Export null_condition_emb as raw float32 binary."""
    from safetensors.torch import load_file
    model_dir = Path(model_dir)
    st_path = model_dir / "model.safetensors"
    
    state_dict = load_file(str(st_path))
    key = "null_condition_emb"
    if key not in state_dict:
        print(f"[export_cond_enc] WARNING: {key} not found, skipping")
        return
    
    vec = state_dict[key].detach().cpu().float().numpy().flatten()
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(vec)))
        f.write(vec.tobytes())
    
    print(f"[export_cond_enc] null_condition_emb: [{len(vec)}] -> {output_path} ({len(vec)*4} bytes)")


def verify_onnx(onnx_path: str, cond_encoder, config):
    """Verify ONNX output matches PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export_cond_enc] onnxruntime not installed, skipping verification")
        return
    
    device = next(cond_encoder.parameters()).device
    dtype = next(cond_encoder.parameters()).dtype
    wrapper = CondEncoderWrapperFixed(cond_encoder)
    wrapper.eval()
    
    B, S_text, S_lyric, S_ref = 1, 32, 64, 8
    text_hidden = torch.randn(B, S_text, config.text_hidden_dim, device=device, dtype=dtype)
    lyric_embed = torch.randn(B, S_lyric, config.text_hidden_dim, device=device, dtype=dtype)
    timbre_feats = torch.randn(B, S_ref, config.timbre_hidden_dim, device=device, dtype=dtype)
    
    with torch.no_grad():
        ref_out = wrapper(text_hidden, lyric_embed, timbre_feats).cpu().float().numpy()
    
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    ort_out = sess.run(None, {
        "text_hidden": text_hidden.cpu().float().numpy(),
        "lyric_embed": lyric_embed.cpu().float().numpy(),
        "timbre_feats": timbre_feats.cpu().float().numpy(),
    })[0]
    
    max_diff = np.max(np.abs(ref_out - ort_out))
    mean_diff = np.mean(np.abs(ref_out - ort_out))
    print(f"[export_cond_enc] Verification: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    if max_diff < 0.05:
        print("[export_cond_enc] PASS: ONNX output matches PyTorch")
    else:
        print("[export_cond_enc] WARNING: Large difference — may need investigation")


def main():
    parser = argparse.ArgumentParser(description="Export AceStep condition encoder to ONNX")
    parser.add_argument("--model-dir", required=True,
                        help="Path to the DiT model directory (contains encoder weights)")
    parser.add_argument("--output", default=None,
                        help="Output ONNX file (default: models/onnx/cond_encoder.onnx)")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    if args.output is None:
        onnx_dir = Path(args.model_dir).parent / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(onnx_dir / "cond_encoder.onnx")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dir = os.path.dirname(args.output)
    
    # Load model
    cond_encoder, config = load_model(args.model_dir, device=args.device)
    
    # Export ONNX
    export_onnx(cond_encoder, config, args.output, opset=args.opset)
    
    # Export null_condition_emb
    null_cond_path = os.path.join(output_dir, "null_condition_emb.bin")
    export_null_cond_emb(args.model_dir, null_cond_path)
    
    # Verify
    if args.verify:
        verify_onnx(args.output, cond_encoder, config)
    
    print("[export_cond_enc] Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TRT-LLM Engine Builder for HOT-Step LM (Qwen3-4B)

Builds a TRT-LLM engine from the HuggingFace model, cached for reuse.
Engine is GPU-architecture specific — must be rebuilt per GPU.

Usage (Docker):
  docker run --rm --gpus all --ipc=host \
    -v D:\Ace-Step-Latest\hot-step-cpp:/workspace \
    nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
    python3 /workspace/tools/trtllm-build/build_trtllm_engine.py

Options:
  --model       Path to HF model directory (default: auto-detect)
  --output      Output engine directory (default: models/onnx/lm-4B/trtllm-engine-{GPU}/)
  --max-seq-len Maximum sequence length (default: 8192)
  --force       Rebuild even if engine already exists
"""

import argparse
import os
import sys
import time
import shutil
import subprocess


def get_gpu_name():
    """Get GPU name for engine cache path."""
    try:
        import torch
        name = torch.cuda.get_device_name(0)
        # Sanitize for filesystem: "NVIDIA GeForce RTX 5090" -> "RTX5090"
        name = name.replace("NVIDIA ", "").replace("GeForce ", "")
        name = name.replace(" ", "")
        return name
    except Exception:
        return "unknown"


def detect_model_dir():
    """Auto-detect model directory (Docker vs WSL vs Windows)."""
    candidates = [
        "/workspace/models/acestep-5Hz-lm-4B",    # Docker mount
        "/mnt/d/Ace-Step-Latest/hot-step-cpp/models/acestep-5Hz-lm-4B",  # WSL
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    print("ERROR: Could not find model directory. Use --model to specify.")
    sys.exit(1)


def detect_output_dir(workspace_root):
    """Auto-detect output directory."""
    gpu = get_gpu_name()
    # models/onnx/lm-4B/trtllm-engine-{GPU}/
    candidates = [
        f"/workspace/models/onnx/lm-4B/trtllm-engine-{gpu}",     # Docker
        f"/mnt/d/Ace-Step-Latest/hot-step-cpp/models/onnx/lm-4B/trtllm-engine-{gpu}",  # WSL
    ]
    for c in candidates:
        parent = os.path.dirname(c)
        if os.path.exists(os.path.dirname(parent)):
            return c
    return candidates[0]


def build_engine(model_dir, output_dir, max_seq_len, max_batch_size=1):
    """Build TRT-LLM engine from HF model."""
    
    print("=" * 60)
    print("BUILDING TRT-LLM ENGINE")
    print("=" * 60)
    print(f"  Model:        {model_dir}")
    print(f"  Output:       {output_dir}")
    print(f"  Max seq len:  {max_seq_len}")
    print(f"  Max batch:    {max_batch_size}")
    print(f"  GPU:          {get_gpu_name()}")
    print()

    # Import TRT-LLM
    try:
        from tensorrt_llm._tensorrt_engine import LLM as TrtLLM
        from tensorrt_llm import BuildConfig
        print(f"  TRT-LLM version: {__import__('tensorrt_llm').__version__}")
    except ImportError as e:
        print(f"ERROR: Cannot import TRT-LLM: {e}")
        print("Run this script inside the TRT-LLM Docker container.")
        sys.exit(1)

    # Build config
    build_config = BuildConfig(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_num_tokens=max_seq_len,
    )

    print("\nBuilding engine (this takes 60-120 seconds)...")
    t0 = time.perf_counter()

    llm = TrtLLM(
        model=model_dir,
        build_config=build_config,
        dtype="bfloat16",
        kv_cache_config={"free_gpu_memory_fraction": 0.3},
    )

    t_build = time.perf_counter() - t0
    print(f"Engine built in {t_build:.1f}s")

    # Save engine
    os.makedirs(output_dir, exist_ok=True)
    llm.save(output_dir)
    print(f"Saved to: {output_dir}")

    # Report size
    total_size = 0
    for f in os.listdir(output_dir):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            total_size += sz
            print(f"  {f}: {sz / 1024 / 1024:.1f} MiB")
    print(f"  Total: {total_size / 1024 / 1024:.1f} MiB")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Build TRT-LLM engine for HOT-Step LM")
    parser.add_argument("--model", type=str, default=None, help="HF model directory")
    parser.add_argument("--output", type=str, default=None, help="Output engine directory")
    parser.add_argument("--max-seq-len", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--max-batch-size", type=int, default=1, help="Maximum batch size")
    parser.add_argument("--force", action="store_true", help="Rebuild even if engine exists")
    args = parser.parse_args()

    model_dir = args.model or detect_model_dir()
    output_dir = args.output or detect_output_dir(model_dir)

    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    # Check if engine already exists
    if os.path.exists(os.path.join(output_dir, "config.json")) and not args.force:
        print(f"Engine already exists at: {output_dir}")
        print("Use --force to rebuild.")
        return

    build_engine(model_dir, output_dir, args.max_seq_len, args.max_batch_size)
    print("\n✓ Engine build complete!")


if __name__ == "__main__":
    main()

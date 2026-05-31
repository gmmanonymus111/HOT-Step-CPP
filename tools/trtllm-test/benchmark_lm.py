#!/usr/bin/env python3
"""
TRT-LLM Benchmark for HOT-Step LM (Qwen3-4B)

Tests Phase 1 (text/caption generation) performance comparing:
  1. PyTorch baseline (transformers)
  2. TensorRT-LLM

Usage (from WSL2):
  # First run with --build to create the TRT-LLM engine:
  python benchmark_lm.py --build --benchmark

  # Subsequent runs (reuse cached engine):
  python benchmark_lm.py --benchmark

  # Just build the engine without benchmarking:
  python benchmark_lm.py --build

Model path: Reads from the Windows filesystem via /mnt/d/...
"""

import argparse
import os
import sys
import time
import json

# Model path — auto-detect Docker vs WSL2
_DOCKER_MODEL = "/workspace/models/acestep-5Hz-lm-4B"
_WSL_MODEL = "/mnt/d/Ace-Step-Latest/hot-step-cpp/models/acestep-5Hz-lm-4B"
MODEL_DIR = _DOCKER_MODEL if os.path.exists(_DOCKER_MODEL) else _WSL_MODEL
ENGINE_DIR = "/workspace/tools/trtllm-test/engine-qwen3-4b"

# Generation params matching HOT-Step Phase 1
_MAX_NEW_TOKENS = [200]  # mutable container to avoid global keyword issues
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.9
SEED = 42

# Test prompt — uses the REAL Qwen3 chat template from HOT-Step Phase 1
# Format: <|im_start|>system\n# Instruction\n{instruction}\n\n<|im_end|>\n<|im_start|>user\n# Caption\n{caption}\n\n# Lyric\n{lyrics}\n<|im_end|>\n<|im_start|>assistant\n
TEST_PROMPT = """<|im_start|>system
# Instruction
Generate audio semantic tokens based on the given conditions:

<|im_end|>
<|im_start|>user
# Caption
A synthwave electronic track with driving bass and shimmering arpeggios. Energetic and uplifting mood, 120 BPM, C major, 4/4 time signature. Features synthesizer, drum machine, and bass.

# Lyric
[Verse]
Neon lights are calling
Through the midnight rain
Digital horizons
Breaking every chain

[Chorus]
We're riding on the frequency
Electric hearts align
Through the noise and static
Our signal starts to shine
<|im_end|>
<|im_start|>assistant
"""


def benchmark_pytorch():
    """Baseline: HuggingFace transformers with PyTorch"""
    print("\n" + "=" * 60)
    print("BENCHMARK: PyTorch (transformers)")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {MODEL_DIR}...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Tokenize
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to("cuda")
    prompt_len = inputs.input_ids.shape[1]
    print(f"  Prompt tokens: {prompt_len}")

    # Warmup (1 short generation)
    print("Warming up...")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            do_sample=True,
        )
    torch.cuda.synchronize()

    # Benchmark
    print(f"Generating {_MAX_NEW_TOKENS[0]} tokens...")
    torch.manual_seed(SEED)
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS[0],
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            do_sample=True,
        )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    gen_tokens = output.shape[1] - prompt_len
    elapsed = t_end - t_start
    tok_per_sec = gen_tokens / elapsed

    # Decode output
    generated_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    print(f"\n--- Results ---")
    print(f"  Generated: {gen_tokens} tokens in {elapsed:.3f}s")
    print(f"  Throughput: {tok_per_sec:.1f} tok/s")
    print(f"\n--- FULL OUTPUT (PyTorch) ---")
    print(generated_text)
    print("--- END OUTPUT ---")

    # Cleanup to free VRAM for next benchmark
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return {
        "engine": "pytorch",
        "tokens": gen_tokens,
        "time_s": elapsed,
        "tok_per_sec": tok_per_sec,
        "output": generated_text[:500],
    }


def build_trtllm_engine():
    """Build TRT-LLM engine from HuggingFace model"""
    print("\n" + "=" * 60)
    print("BUILDING TRT-LLM ENGINE")
    print("=" * 60)

    try:
        import tensorrt_llm
        print(f"TRT-LLM version: {tensorrt_llm.__version__}")
    except ImportError:
        print("ERROR: tensorrt_llm not installed!")
        print("Install with: pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com")
        sys.exit(1)

    from tensorrt_llm._tensorrt_engine import LLM as TrtLLM
    from tensorrt_llm import BuildConfig

    print(f"Model: {MODEL_DIR}")
    print(f"Engine output: {ENGINE_DIR}")

    os.makedirs(ENGINE_DIR, exist_ok=True)

    t0 = time.perf_counter()

    # Build config — optimize for single-user, single-GPU, low latency
    build_config = BuildConfig(
        max_batch_size=1,
        max_seq_len=512,           # Only need ~300 tokens (prompt + generation)
        max_num_tokens=512,
    )

    # The LLM class handles conversion + engine build automatically
    llm = TrtLLM(
        model=MODEL_DIR,
        build_config=build_config,
        dtype="bfloat16",
        kv_cache_config={"free_gpu_memory_fraction": 0.1},
    )

    # Save the engine for reuse
    llm.save(ENGINE_DIR)

    build_time = time.perf_counter() - t0
    print(f"Engine built in {build_time:.1f}s")
    print(f"Saved to: {ENGINE_DIR}")

    return llm


def benchmark_trtllm(llm=None):
    """Benchmark TRT-LLM inference"""
    print("\n" + "=" * 60)
    print("BENCHMARK: TensorRT-LLM")
    print("=" * 60)

    from tensorrt_llm._tensorrt_engine import LLM as TrtLLM
    from tensorrt_llm import SamplingParams

    # Load engine if not provided
    if llm is None:
        if not os.path.exists(ENGINE_DIR):
            print(f"ERROR: No engine found at {ENGINE_DIR}")
            print("Run with --build first!")
            sys.exit(1)

        print(f"Loading engine from {ENGINE_DIR}...")
        t0 = time.perf_counter()
        llm = TrtLLM(model=ENGINE_DIR)
        load_time = time.perf_counter() - t0
        print(f"Engine loaded in {load_time:.1f}s")

    # Warmup
    print("Warming up...")
    warmup_params = SamplingParams(
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        max_tokens=10,
        seed=SEED,
    )
    _ = llm.generate(TEST_PROMPT, warmup_params)

    # Benchmark
    print(f"Generating {_MAX_NEW_TOKENS[0]} tokens...")
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        max_tokens=_MAX_NEW_TOKENS[0],
        seed=SEED,
    )

    t_start = time.perf_counter()
    output = llm.generate(TEST_PROMPT, sampling_params)
    t_end = time.perf_counter()

    gen_tokens = len(output.outputs[0].token_ids)
    elapsed = t_end - t_start
    tok_per_sec = gen_tokens / elapsed
    generated_text = output.outputs[0].text

    print(f"\n--- Results ---")
    print(f"  Generated: {gen_tokens} tokens in {elapsed:.3f}s")
    print(f"  Throughput: {tok_per_sec:.1f} tok/s")
    print(f"\n--- FULL OUTPUT (TRT-LLM) ---")
    print(generated_text)
    print("--- END OUTPUT ---")

    return {
        "engine": "trtllm",
        "tokens": gen_tokens,
        "time_s": elapsed,
        "tok_per_sec": tok_per_sec,
        "output": generated_text[:500],
    }


def main():
    parser = argparse.ArgumentParser(description="TRT-LLM benchmark for HOT-Step LM")
    parser.add_argument("--build", action="store_true", help="Build TRT-LLM engine")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--pytorch-only", action="store_true", help="Only run PyTorch benchmark")
    parser.add_argument("--trtllm-only", action="store_true", help="Only run TRT-LLM benchmark")
    parser.add_argument("--max-tokens", type=int, default=_MAX_NEW_TOKENS[0], help="Max tokens to generate")
    parser.add_argument("--quantize", choices=["bf16", "fp16", "int8", "int4_awq", "fp8"],
                       default="bf16", help="Quantization for TRT-LLM engine")
    args = parser.parse_args()

    _MAX_NEW_TOKENS[0] = args.max_tokens

    if not args.build and not args.benchmark:
        parser.print_help()
        print("\nSpecify --build, --benchmark, or both!")
        sys.exit(1)

    # Verify model exists
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model not found at {MODEL_DIR}")
        print("Make sure the Windows path is accessible via /mnt/d/")
        sys.exit(1)

    print(f"Model: {MODEL_DIR}")
    print(f"Max tokens: {_MAX_NEW_TOKENS[0]}")

    results = []
    llm = None

    # Build engine
    if args.build:
        llm = build_trtllm_engine()

    # Run benchmarks
    if args.benchmark:
        if not args.trtllm_only:
            try:
                r = benchmark_pytorch()
                results.append(r)
            except Exception as e:
                print(f"PyTorch benchmark failed: {e}")

        if not args.pytorch_only:
            try:
                r = benchmark_trtllm(llm)
                results.append(r)
            except Exception as e:
                print(f"TRT-LLM benchmark failed: {e}")

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        for r in results:
            print(f"  {r['engine']:>10}: {r['tok_per_sec']:>7.1f} tok/s  ({r['tokens']} tokens in {r['time_s']:.3f}s)")

        if len(results) == 2:
            speedup = results[1]["tok_per_sec"] / results[0]["tok_per_sec"]
            print(f"\n  Speedup: {speedup:.2f}x")

    # Save results
    results_file = "/tmp/trtllm-benchmark-results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

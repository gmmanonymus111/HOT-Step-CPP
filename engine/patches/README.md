# engine/patches

`bf16-out-prod.patch` teaches ggml-cuda's `out_prod` to accept a **BF16 `src0`**, which is what lets the DiT trainer's `--mirror bf16` keep frozen trainable-layer weights in BF16 instead of promoting the whole mirror to F32. `engine/ggml/` is a git submodule, so a submodule update silently reverts it.

`mm-backward.patch` adds an **env-gated alternative formulation** for the `MUL_MAT` activation gradient in `ggml.c`'s `ggml_compute_backward`. Upstream emits `out_prod(src0, transpose(grad))`, and ggml-cuda's `OUT_PROD` is F32-only — which forces the frozen weight to F32 and drags the *forward* `mul_mat` onto TF32 tensor cores too. With `GGML_BACKWARD_MM=1` set, the backward becomes `mul_mat(cont(transpose(src0)), grad)` instead: mathematically and shape-wise identical, but dtype-agnostic, so a BF16 weight rides real BF16 tensor cores in both directions with no dequant. Measured on an RTX 5090 at ~1.7–1.8× per layer per step. With the env var unset the emitted graph is byte-identical to upstream.

`cpy-q-occupancy.patch` fixes the **launch geometry of ggml-cuda's quant→F32 copies**. Upstream launches them as `<<<ne, 1>>>` — one CUDA *block* per *element*, one thread inside it — while the kernel body indexes `i = tid*qk` and returns on `i >= ne`. With `qk = 32` that means 31 of every 32 blocks exist only to hit the guard, and each surviving thread runs alone in a 32-lane warp; residency caps around 32 blocks/SM. The patch changes the launch shape and nothing else (the index expression already reads `blockDim.x`, so each thread gets exactly the element it would have had as a lone block — identical work, identical output), matching how the F16/F32 scalar copies in the same file already launch.

This matters because MM3 LM training on a quantized base is QLoRA-style dequantize-per-matmul: `qwen3_f32()` emits an in-graph `ggml_cast`, which becomes a `CPY q8_0 → F32` node, which is this kernel. **The failure mode is silent** — without the patch the numbers are still correct, just far slower — so `verify-hooks.ps1` Hook 9 greps for the marker.

The sibling F32→quant launches are deliberately left alone: they share the 1-thread-per-block shape but already use the correct block *count* (`ne/qk`), so they waste occupancy without wasting blocks, and they are an inference path (quantized KV cache) rather than the training path this was measured against.

`ace-train`'s `--bwd <outprod|mm>` sets that env var; the Training Studio defaults both trainers to `mm`.

Reapply from the repo root — **apply all of them**, they touch disjoint files (`ggml-cuda/*.cu` vs `ggml.c`) so order does not matter:

```sh
for p in engine/patches/*.patch; do git apply --verbose "$p"; done
```

Verify they are still in place: `powershell -File engine\verify-hooks.ps1` (Hook 7 greps `out-prod.cu`, Hook 8 greps `ggml.c`, both for their HOT-Step marker comments). CI reapplies them in the "Apply engine patches" step of every build job, using the same glob loop.

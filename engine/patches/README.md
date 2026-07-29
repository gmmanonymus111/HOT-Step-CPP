# engine/patches

`bf16-out-prod.patch` teaches ggml-cuda's `out_prod` to accept a **BF16 `src0`**, which is what lets the DiT trainer's `--mirror bf16` keep frozen trainable-layer weights in BF16 instead of promoting the whole mirror to F32. `engine/ggml/` is a git submodule, so a submodule update silently reverts it.

`mm-backward.patch` adds an **env-gated alternative formulation** for the `MUL_MAT` activation gradient in `ggml.c`'s `ggml_compute_backward`. Upstream emits `out_prod(src0, transpose(grad))`, and ggml-cuda's `OUT_PROD` is F32-only — which forces the frozen weight to F32 and drags the *forward* `mul_mat` onto TF32 tensor cores too. With `GGML_BACKWARD_MM=1` set, the backward becomes `mul_mat(cont(transpose(src0)), grad)` instead: mathematically and shape-wise identical, but dtype-agnostic, so a BF16 weight rides real BF16 tensor cores in both directions with no dequant. Measured on an RTX 5090 at ~1.7–1.8× per layer per step. With the env var unset the emitted graph is byte-identical to upstream.

`ace-train`'s `--bwd <outprod|mm>` sets that env var; the Training Studio defaults both trainers to `mm`.

Reapply from the repo root — **apply all of them**, they touch disjoint files (`ggml-cuda/*.cu` vs `ggml.c`) so order does not matter:

```sh
for p in engine/patches/*.patch; do git apply --verbose "$p"; done
```

Verify they are still in place: `powershell -File engine\verify-hooks.ps1` (Hook 7 greps `out-prod.cu`, Hook 8 greps `ggml.c`, both for their HOT-Step marker comments). CI reapplies them in the "Apply engine patches" step of every build job, using the same glob loop.

# engine/patches

`bf16-out-prod.patch` teaches ggml-cuda's `out_prod` to accept a **BF16 `src0`**, which is what lets the DiT trainer's `--mirror bf16` keep frozen trainable-layer weights in BF16 instead of promoting the whole mirror to F32. `engine/ggml/` is a git submodule, so a submodule update silently reverts it.

Reapply from the repo root: `git apply engine/patches/bf16-out-prod.patch`

Verify it is still in place: `powershell -File engine\verify-hooks.ps1` (Hook 7 greps `out-prod.cu` for the HOT-Step marker comment).

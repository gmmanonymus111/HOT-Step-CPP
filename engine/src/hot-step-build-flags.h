#pragma once
// hot-step-build-flags.h — build-time feature flags for special hardware targets.
//
// The cuda12-volta release variant builds with -DHOT_STEP_DISABLE_FA (here) and
// -DGGML_CUDA_FORCE_CUBLAS (ggml). Both are needed because ggml's modern
// mma-based kernels have no device code for Volta (sm_70):
//   * flash_attn_ext_f16  → disabled here, so the engine takes the manual
//                           attention path (works on any arch).
//   * mul_mat_q (MMQ)     → FORCE_CUBLAS makes ggml use cuBLAS instead.
// A normal build leaves both on (FA, MMQ) for Turing+ performance.

#ifdef HOT_STEP_DISABLE_FA
static constexpr bool HOT_STEP_FA_DISABLED = true;
#else
static constexpr bool HOT_STEP_FA_DISABLED = false;
#endif

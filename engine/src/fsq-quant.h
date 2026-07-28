#pragma once
// fsq-quant.h — HOT-Step: the ONE reference-conformant FSQ scalar quantizer.
//
// ─── HOT-STEP HOOK — DO NOT LET AN UPSTREAM SYNC REVERT THIS ────────────────
// fsq-detok.h includes this file, and fsq-tok.h (which includes fsq-detok.h)
// no longer defines fsq_encode_index itself. If an upstream sync restores
// acestep.cpp's own copies of `fsq_encode_index` / `fsq_decode_index` /
// `FSQ_LEVELS` you will get a *redefinition* compile error — that is the
// intended loud failure. Fix it by deleting the upstream copies again, not by
// dropping this include. `engine/verify-hooks.ps1` checks for the include.
//
// ─── WHY (the conformance bug this file fixes) ──────────────────────────────
// The checkpoints are tokenized by the model's own AceStepAudioTokenizer, whose
// `quantizer` is a vector_quantize_pytorch `ResidualFSQ`. In vqp >= 1.27.21
// (Side-Step ships 1.28.0) `ResidualFSQ.__init__` hard-wires
// `preserve_symmetry=True` and `bound_hard_clamp=True`, and soft-clamps its
// input before the residual layers:
//
//   residual_fsq.py:116-123   soft_clamp_input_value = 1 + 1/(levels - 1)
//   residual_fsq.py:193-195   x = tanh(x / clamp_value) * clamp_value
//   residual_fsq.py:228-234   layer(residual / scale), scale = levels**0 = 1
//                             for quantizer 0 (ACE uses num_quantizers == 1)
//   finite_scalar_quantization.py:149-157  symmetry_preserving_bound(
//                             z, hard_clamp=True):
//                               w = clamp(z, -1, 1)
//                               level = floor((L-1)*(w+1)/2 + 0.5)
//   finite_scalar_quantization.py:183-188  _scale_and_shift (preserve_symmetry)
//                               -> index = sum_d level_d * basis_d
//
// The pre-fix engine encoder used plain `tanh(z)` with no soft clamp, i.e. the
// OLD vqp bound. It agrees with the reference only near z = 0: measured against
// a PyTorch dump of the tokenizer's own pre-quantization projections it
// reproduced only 40.19% of the reference indices (always off by +-1 level),
// versus 100.00% (13046/13046) for the formula below.
//
// DECODE conformance: with preserve_symmetry=True the reference inverse is
//   finite_scalar_quantization.py:190-195
//     _scale_and_shift_inverse(level) = level * (2 / (L - 1)) - 1
// which is what fsq_decode_index implements below, evaluated in the same float32
// order as torch (int32 level * float32 step - 1). NOTE the OTHER branch of that
// function, `(level - L//2) / (L//2)`, is the non-preserve_symmetry one and is
// NOT what ACE uses — do not "fix" the even-L dims (8,8,8) towards it.
//
// basis = exclusive cumprod(levels) = [1, 8, 64, 512, 2560, 12800]
//   (finite_scalar_quantization.py:91), matching the `mult *= L` / `stride *= L`
//   loops below.

#include <cmath>

// FSQ constants (levels of the ACE-Step 1.5 audio tokenizer's single FSQ stage)
static const int FSQ_NDIMS     = 6;
static const int FSQ_LEVELS[6] = { 8, 8, 8, 5, 5, 5 };

// FSQ encode: FSQ_NDIMS raw `quantizer.project_in` outputs -> flat codebook index.
//
// Reference path: ResidualFSQ soft clamp -> FSQ(bound_hard_clamp=True)
//                 symmetry_preserving_bound -> codes_to_indices.
static int fsq_encode_index(const float * raw_vals) {
    int index = 0;
    int mult  = 1;
    for (int d = 0; d < FSQ_NDIMS; d++) {
        const int   L   = FSQ_LEVELS[d];
        const float Lm1 = (float) (L - 1);

        // ResidualFSQ soft clamp: x = tanh(x / c) * c   with c = 1 + 1/(L-1)
        const float c = 1.0f + 1.0f / Lm1;
        float       y = tanhf(raw_vals[d] / c) * c;

        // FSQ(bound_hard_clamp=True): tanh is replaced by a hard clamp
        if (y < -1.0f) {
            y = -1.0f;
        }
        if (y > 1.0f) {
            y = 1.0f;
        }

        // symmetry_preserving_bound + _scale_and_shift collapse to this level
        int code = (int) floorf(Lm1 * (y + 1.0f) / 2.0f + 0.5f);
        if (code < 0) {
            code = 0;
        }
        if (code >= L) {
            code = L - 1;
        }

        index += code * mult;
        mult *= L;
    }
    return index;
}

// FSQ decode: flat codebook index -> FSQ_NDIMS normalized floats in [-1, 1].
//
// Reference path: FSQ.indices_to_level_indices -> _scale_and_shift_inverse with
// preserve_symmetry=True, i.e. level * (2 / (L - 1)) - 1. Kept in this exact
// evaluation order (step first, then multiply) so the float32 result is
// bit-identical to torch's `zhat * (2. / (self._levels - 1)) - 1.`.
static void fsq_decode_index(int index, float * out) {
    int stride = 1;
    for (int d = 0; d < FSQ_NDIMS; d++) {
        const int   L         = FSQ_LEVELS[d];
        const int   level_idx = (index / stride) % L;
        const float step      = 2.0f / (float) (L - 1);
        out[d]                = (float) level_idx * step - 1.0f;
        stride *= L;
    }
}

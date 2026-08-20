#!/usr/bin/env python3
# rvq-encoder-fixture.py: golden forward/codes fixture for the native RVQ encoder.
#
# The native port (`ace-train mm3-codes`) must reproduce the reference
# V4Encoder exactly enough that the ARGMAX codes are identical — a code that
# differs is a training-corpus corruption that no downstream ear test can
# attribute. Rather than discovering that at the end of a 13-track export, this
# writes a small deterministic fixture the C++ side can check against on every
# build, one rung at a time:
#
#   rung 1  encoder features   [128, 1088]   cosine / max-abs vs reference
#   rung 2  semantic logits    [128, 16384]  and their argmax
#   rung 3  depth greedy codes [128, 7]      EXACT match required
#
# The input is synthetic-but-deterministic (seeded normal latents + a real pool
# matrix from the model's own windowing), so the fixture needs no audio, no DAV
# encoder and no GPU — it isolates the encoder graph from the data path. The
# END-TO-END gate is separate and stricter: encode the same tracks with the
# python exporter and the native binary and diff the .codes files byte for byte.
#
# Format (little-endian, one file, so the C++ reader is 20 lines):
#   magic "RVQFIX01"
#   u32 n_lat, u32 n_ch, u32 n_frames, u32 d_model, u32 sem_vocab, u32 n_ac
#   f32 latents  [n_lat, n_ch]          (row-major, matches the torch input)
#   f32 pool     [n_frames, n_lat]
#   f32 feats    [n_frames, d_model]
#   f32 sem_logits [n_frames, sem_vocab]
#   i32 codes    [n_frames, 1 + n_ac]   (semantic argmax + greedy acoustic)
#
# USAGE
#   python engine/tools/rvq-encoder-fixture.py \
#     --ckpt "M:/HOT-Step-CPP/_experiments/open-rvq-53k-pooled/v4_pooled_best.pt" \
#     --train-v4 "M:/HOT-Step-CPP/_experiments/open-rvq-53k-pooled" \
#     --out engine/tests/fixtures/rvq-encoder-53kpooled.fix

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-v4", required=True,
                    help="directory holding the checkpoint's own train_v4.py "
                         "(the architecture is defined there, not here)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-frames-total", type=int, default=600,
                    help="pretend the track is this many frames; the fixture "
                         "window is taken from frame 0 (the pool matrix then "
                         "carries a realistic, non-uniform span layout)")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.train_v4)))
    import torch
    from train_v4 import (  # noqa: E402
        FRAMES, LATENT_WINDOW_MAX, SEM_VOCAB, N_AC, V4Encoder,
        frame_latent_starts, pool_matrix,
    )

    torch.manual_seed(a.seed)
    model = V4Encoder().eval()
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)

    # The model's OWN windowing, so the pool matrix has the real fractional
    # span structure rather than a uniform one that would hide an off-by-one.
    starts = frame_latent_starts(a.n_frames_total)
    bounds = starts[0:FRAMES + 1] - starts[0]
    n_lat = int(bounds[-1])
    assert n_lat <= LATENT_WINDOW_MAX, f"{n_lat} > {LATENT_WINDOW_MAX}"

    rng = np.random.default_rng(a.seed)
    lat = np.zeros((LATENT_WINDOW_MAX, 128), dtype=np.float32)
    # DAV latents are roughly unit-scale; the exact distribution does not matter
    # for a graph-parity check, only that it is deterministic and non-degenerate.
    lat[:n_lat] = rng.standard_normal((n_lat, 128), dtype=np.float32)
    pool = pool_matrix(bounds)

    with torch.no_grad():
        feats, sem_logits = model(torch.from_numpy(lat[None]), torch.from_numpy(pool[None]))
        c0 = sem_logits.argmax(-1)                                  # [1, 128]
        ac = model.depth.greedy(feats.flatten(0, 1), c0.flatten(0, 1))  # [128, 7]

    feats = feats[0].numpy().astype(np.float32)
    sem_logits = sem_logits[0].numpy().astype(np.float32)
    codes = np.concatenate([c0[0].numpy()[:, None], ac.numpy()], axis=1).astype(np.int32)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"RVQFIX01")
        f.write(struct.pack("<6I", LATENT_WINDOW_MAX, 128, FRAMES, feats.shape[1], SEM_VOCAB, N_AC))
        for arr in (lat, pool, feats, sem_logits):
            f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(codes, dtype=np.int32).tobytes())

    print(f"[fix] {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"[fix] latents {lat.shape} ({n_lat} real rows), pool {pool.shape}")
    print(f"[fix] feats  mean {feats.mean():+.6f}  std {feats.std():.6f}  "
          f"absmax {np.abs(feats).max():.6f}")
    print(f"[fix] logits mean {sem_logits.mean():+.6f}  std {sem_logits.std():.6f}")
    print(f"[fix] semantic codes: {codes[:8, 0].tolist()} … "
          f"({len(set(codes[:, 0].tolist()))} unique of {FRAMES})")
    print(f"[fix] frame 0 acoustic: {codes[0, 1:].tolist()}")
    # Argmax margin: how close the top-2 semantic logits are. A tiny margin
    # means f16 (or a different GEMM order) can legitimately flip that frame —
    # worth knowing BEFORE calling a code mismatch a port bug.
    top2 = np.partition(sem_logits, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    print(f"[fix] argmax margin: min {margin.min():.4f}  p05 {np.percentile(margin,5):.4f}  "
          f"median {np.median(margin):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

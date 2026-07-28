#!/usr/bin/env python
"""goldens_to_bin.py — flatten dump_bs_roformer_goldens.py output for C++.

engine/tools/bs-roformer-test.cpp reads a flat little-endian binary rather than
an .npz, because parsing zip+npy in C++ buys nothing here.

Layout:
    magic "BSRG"                     4 bytes
    int32 T, in_dim, depth, dim, n_bands, n_stems
    f32   input      [in_dim * T]
    f32   band_split [dim * n_bands * T]
    f32   layer_00   [dim * n_bands * T]
    f32   layer_01   [dim * n_bands * T]
    f32   layer_last [dim * n_bands * T]
    f32   final_norm [dim * n_bands * T]
    f32   mask       [n_stems * in_dim * T]

Hidden-state arrays arrive from torch as [1, T, n_bands, dim] and are written
in GGML memory order (dim fastest, then n_bands, then T) so the C++ side can
compare against its debug tensor elementwise without reindexing.

The mask arrives as [1, S, fs, T, 2] and is already in the engine's order.

USAGE
    $py = "d:\\Ace-Step-Latest\\hot-step-9000\\.venv\\Scripts\\python.exe"
    & $py tools\\goldens_to_bin.py `
        --input  models\\supersep-ckpt\\goldens_voc.npz `
        --output models\\supersep-ckpt\\goldens_voc.bin
"""

import argparse
import struct

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    z = np.load(args.input)
    T, in_dim, depth, n_stems, fs = (int(v) for v in z["meta"])

    band_split = z["band_split"]          # [1, T, n_bands, dim]
    _, t_chk, n_bands, dim = band_split.shape
    assert t_chk == T, f"T mismatch {t_chk} vs {T}"

    print(f"T={T} in_dim={in_dim} depth={depth} dim={dim} "
          f"bands={n_bands} stems={n_stems} fs={fs}")

    expect = T * n_bands * dim

    def hidden(name):
        # Rank varies: band_split/final_norm are hooked outside the pack and
        # come back [1, T, n_bands, dim]; the per-layer hooks sit on the freq
        # transformer, which sees the packed [(b t), n_bands, dim]. Batch is 1,
        # so both flatten to the same GGML order (dim fastest, n_bands, T) —
        # just flatten whatever rank arrived and check the count.
        a = np.ascontiguousarray(z[name], dtype=np.float32).reshape(-1)
        if a.size != expect:
            raise SystemExit(
                f"{name}: {a.size} elements, expected {expect} "
                f"(shape {z[name].shape})")
        return a

    with open(args.output, "wb") as f:
        f.write(b"BSRG")
        f.write(struct.pack("<6i", T, in_dim, depth, dim, n_bands, n_stems))

        # input is [1, T, in_dim]; GGML wants in_dim fastest then T — same order.
        f.write(np.ascontiguousarray(z["input"][0], dtype=np.float32).tobytes())

        for name in ("band_split", "layer_00", "layer_01", "layer_last", "final_norm"):
            f.write(hidden(name).tobytes())

        # mask [1, S, fs, T, 2] -> linear [s][fs][t][2], already correct
        f.write(np.ascontiguousarray(z["mask"][0], dtype=np.float32).tobytes())

    import os
    print(f"[ok] {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()

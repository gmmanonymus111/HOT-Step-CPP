"""Reproduce the MOSS audio tower from the CONVERTED GGUF, in numpy, and score it against
the fp32 reference fixtures.

Purpose is threefold:
  1. Prove convert-moss.py is correct end-to-end (right tensors, right layouts, nothing
     transposed) BEFORE any C++ is written.
  2. Serve as the executable spec for the ggml graph — every step below maps 1:1 to a
     ggml op, and the constants are the ones the C++ must use.
  3. Establish the parity floor. The mm3-backend bar is >=0.999 corr per module vs the
     bf16 dumps, or >=0.9999 vs an fp32 CPU rerun. These fixtures ARE the fp32 rerun, and
     the GGUF is f16, so expect ~0.9999 and treat anything below 0.999 as a real bug.

  python numpy_ref.py
"""
import json
import os
import sys

import gguf
import numpy as np

GGUF = r"M:\Music Captioners\gguf\moss-aud-f16.gguf"
FIX = r"M:\Music Captioners\fixtures"


def load_gguf(path):
    r = gguf.GGUFReader(path)
    kv = {}
    for f in r.fields.values():
        try:
            kv[f.name] = f.contents()
        except Exception:
            pass
    T = {}
    for t in r.tensors:
        # GGUF stores ne0-fastest; numpy/torch order is the reverse.
        shape = tuple(int(x) for x in t.shape)[::-1]
        a = np.asarray(t.data)
        T[t.name] = a.reshape(shape).astype(np.float32)
    return kv, T


def load_fix(track, name, shape=None):
    p = os.path.join(FIX, track, name + ".f32")
    a = np.fromfile(p, dtype="<f4")
    if shape is not None:
        a = a.reshape(shape)
    return a


def score(ref, got, label):
    ref = ref.astype(np.float64).ravel()
    got = got.astype(np.float64).ravel()
    if ref.shape != got.shape:
        print(f"  {label:<26} SHAPE MISMATCH ref{ref.shape} got{got.shape}")
        return False
    corr = float(np.corrcoef(ref, got)[0, 1])
    rel = float(np.sqrt(np.mean((ref - got) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-12))
    mark = "OK  " if corr >= 0.999 else "FAIL"
    print(f"  {mark} {label:<26} corr={corr:.6f}  relRMSE={rel:.2e}")
    return corr >= 0.999


# --- ops, each one a ggml op in the C++ ------------------------------------------------

def conv2d_s2p1(x, w, b):
    """Conv2d(k=3, stride=2, pad=1). x [C_in,H,W], w [C_out,C_in,3,3] -> [C_out,H',W'].

    ggml equivalent: ggml_conv_2d(ctx, kernel, x, 1,1 -> no: s0=2,s1=2, p0=1,p1=1, d=1,1).
    NOTE for the port: ggml_conv_2d forces its im2col to F16. mm3's vocoder/cond/dav graphs
    all hand-roll an F32 im2col for exactly this reason; the conv stem should too."""
    cin, H, W = x.shape
    cout = w.shape[0]
    xp = np.zeros((cin, H + 2, W + 2), dtype=np.float32)
    xp[:, 1:-1, 1:-1] = x
    Ho, Wo = (H + 1) // 2, (W + 1) // 2
    # im2col: [cin*3*3, Ho*Wo]
    cols = np.empty((cin * 9, Ho * Wo), dtype=np.float32)
    idx = 0
    for kh in range(3):
        for kw in range(3):
            patch = xp[:, kh:kh + 2 * Ho:2, kw:kw + 2 * Wo:2]
            cols[idx * cin:(idx + 1) * cin, :] = patch.reshape(cin, -1)
            idx += 1
    # reorder rows to (cin, kh, kw) matching w.reshape(cout, cin*9)
    cols = cols.reshape(9, cin, Ho * Wo).transpose(1, 0, 2).reshape(cin * 9, Ho * Wo)
    y = w.reshape(cout, cin * 9) @ cols
    y += b[:, None]
    return y.reshape(cout, Ho, Wo)


def layer_norm(x, w, b, eps):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def gelu(x):
    # transformers ACT2FN["gelu"] is the exact erf form, not the tanh approximation.
    from math import sqrt
    from scipy.special import erf  # noqa
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def gelu_np(x):
    # erf without scipy: use numpy's vectorised math via the identity erf(x)=2*Phi(x*sqrt2)-1
    # implemented with the Abramowitz-Stegun-free route -> just use math.erf elementwise is
    # too slow, so use the tanh-free formulation via np.vectorize fallback.
    try:
        from scipy.special import erf as _erf
        return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))
    except ImportError:
        # numpy has no erf; this rational approximation is accurate to ~1e-7, which is far
        # below the f16 noise floor we are measuring against.
        s = np.sign(x)
        a = np.abs(x) / np.sqrt(2.0)
        t = 1.0 / (1.0 + 0.3275911 * a)
        y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                    - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
        return 0.5 * x * (1.0 + s * y)


def sinusoids(seq_len, dim, max_timescale=10000.0):
    """MUST be recomputed, not loaded — inv_timescales is persistent=False upstream and is
    deliberately absent from the GGUF."""
    half = dim // 2
    log_inc = np.log(max_timescale) / (half - 1)
    inv = np.exp(-log_inc * np.arange(half, dtype=np.float64))
    t = np.arange(seq_len, dtype=np.float64)[:, None] * inv[None, :]
    return np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32)


def attention(x, T, p, n_head):
    d = x.shape[-1]
    hd = d // n_head
    scaling = hd ** -0.5
    q = (x @ T[p + "attn_q.weight"].T + T[p + "attn_q.bias"]) * scaling
    k = x @ T[p + "attn_k.weight"].T                      # no bias (Whisper convention)
    v = x @ T[p + "attn_v.weight"].T + T[p + "attn_v.bias"]
    L = x.shape[0]
    q = q.reshape(L, n_head, hd).transpose(1, 0, 2)
    k = k.reshape(L, n_head, hd).transpose(1, 0, 2)
    v = v.reshape(L, n_head, hd).transpose(1, 0, 2)
    s = q @ k.transpose(0, 2, 1)                          # [H,L,L]
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    a = e / e.sum(axis=-1, keepdims=True)
    o = (a @ v).transpose(1, 0, 2).reshape(L, d)
    return o @ T[p + "attn_out.weight"].T + T[p + "attn_out.bias"]


def swiglu(x, T, p):
    g = x @ T[p + "gate.weight"].T
    u = x @ T[p + "up.weight"].T
    return (g / (1.0 + np.exp(-g))) * u @ T[p + "down.weight"].T


def main():
    kv, T = load_gguf(GGUF)
    n_layer = int(kv["moss.audio.block_count"])
    d_model = int(kv["moss.audio.embedding_length"])
    n_head = int(kv["moss.audio.head_count"])
    eps = float(kv["moss.audio.layer_norm_eps"])
    ds_layers = [int(x) for x in kv["moss.audio.deepstack_encoder_layers"]]
    print(f"gguf: {n_layer} layers, d_model {d_model}, heads {n_head}, "
          f"deepstack at {ds_layers}")

    tracks = [d for d in os.listdir(FIX) if os.path.isdir(os.path.join(FIX, d))]
    all_ok = True

    for track in tracks:
        with open(os.path.join(FIX, track, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        print(f"\n=== {track} ===")
        mel = load_fix(track, "mel", tuple(man["mel"]["shape"]))
        n_aud = man["_n_audio_tokens"]

        # 1) conv stem: [1,128,T] -> [480,16,T/8]
        x = mel[None, :, :]
        for i in (1, 2, 3):
            x = conv2d_s2p1(x, T[f"aud.conv{i}.weight"], T[f"aud.conv{i}.bias"])
            x = gelu_np(x)
        # 2) [C,F,T'] -> [T',C*F]; torch does permute(0,3,1,2).flatten(2) on [B,C,F,T]
        C, F, Tt = x.shape
        h = x.transpose(2, 0, 1).reshape(Tt, C * F)
        h = h @ T["aud.stem_proj.weight"].T + T["aud.stem_proj.bias"]
        if h.shape[0] > n_aud:
            h = h[:n_aud]
        h = h + sinusoids(h.shape[0], d_model)

        # 3) 32 pre-norm Whisper encoder layers, capturing the deepstack taps
        caps = {}
        for i in range(n_layer):
            p = f"aud.blk.{i}."
            r = h
            h = layer_norm(h, T[p + "attn_norm.weight"], T[p + "attn_norm.bias"], eps)
            h = r + attention(h, T, p, n_head)
            r = h
            h = layer_norm(h, T[p + "ffn_norm.weight"], T[p + "ffn_norm.bias"], eps)
            h = gelu_np(h @ T[p + "ffn_up.weight"].T + T[p + "ffn_up.bias"])
            h = r + (h @ T[p + "ffn_down.weight"].T + T[p + "ffn_down.bias"])
            if i in ds_layers:
                caps[i] = h.copy()

        enc = layer_norm(h, T["aud.norm.weight"], T["aud.norm.bias"], eps)

        ok = True
        ok &= score(load_fix(track, "encoder_out", (n_aud, d_model)), enc, "encoder_out")
        for k, li in enumerate(ds_layers):
            ok &= score(load_fix(track, f"encoder_deepstack_{k}", (n_aud, d_model)),
                        caps[li], f"encoder_deepstack_{k} (L{li})")

        # 4) adapter + deepstack mergers (SwiGLU, no bias)
        ad = swiglu(enc, T, "aud.adapter.")
        ok &= score(load_fix(track, "adapter_out", (n_aud, 4096)), ad, "adapter_out")
        for k, li in enumerate(ds_layers):
            m = swiglu(caps[li], T, f"aud.deepstack.{k}.")
            ok &= score(load_fix(track, f"deepstack_merger_{k}", (n_aud, 4096)), m,
                        f"deepstack_merger_{k}")
        all_ok &= ok

    print("\n" + ("ALL MODULES WITHIN PARITY" if all_ok else "** PARITY FAILURE **"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

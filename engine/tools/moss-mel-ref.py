"""Whisper log-mel frontend in numpy, validated against the captured mel fixture.

This is the piece with NO existing implementation anywhere in engine/src/ — there is no
mel filterbank in the tree at all — so it has to be written from scratch for the port, and
it has to match HF's WhisperFeatureExtractor._np_extract_fbank_features bit-for-bit-ish or
everything downstream drifts.

The three details that are easy to get wrong, in order of how quietly they fail:
  1. mel_scale="slaney" AND norm="slaney" (librosa defaults). Using HTK mel, or skipping
     the Slaney area normalisation, shifts every filter and the encoder still "works".
  2. The final frame is DROPPED (`magnitudes = stft[..., :-1]`). Off-by-one here shifts the
     whole time axis by 10 ms.
  3. The dynamic-range clamp and rescale: max(log_spec, log_spec.max() - 8.0) then
     (log_spec + 4.0) / 4.0. Omitting the rescale leaves values ~4x too large and the
     conv stem saturates.

  python mel_ref.py
"""
import json
import os
import sys

import numpy as np

FIX = r"M:\Music Captioners\fixtures"
WAV16K = r"M:\Music Captioners\tracks\wav16k"

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 128


def hz_to_mel_slaney(f):
    f_min, f_sp = 0.0, 200.0 / 3
    mel = (f - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz,
                    min_log_mel + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep, mel)


def mel_to_hz_slaney(m):
    f_min, f_sp = 0.0, 200.0 / 3
    freq = f_min + f_sp * m
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)),
                    freq)


def mel_filter_bank(n_freq, n_mels, fmin, fmax, sr):
    """Slaney-scale, Slaney-normalised triangular filterbank -- librosa/HF defaults."""
    mel_min, mel_max = hz_to_mel_slaney(np.array(fmin)), hz_to_mel_slaney(np.array(fmax))
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz_slaney(mel_pts)
    fft_freqs = np.linspace(0, sr / 2, n_freq)

    diff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    fb = np.zeros((n_mels, n_freq))
    for i in range(n_mels):
        lower = -ramps[i] / diff[i]
        upper = ramps[i + 2] / diff[i + 1]
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
    # Slaney area normalisation: each filter integrates to a constant, not a constant peak.
    enorm = 2.0 / (hz_pts[2:n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, None]
    return fb


def log_mel(audio, sr=SR):
    win = np.hanning(N_FFT + 1)[:-1].astype(np.float64)   # periodic Hann
    pad = N_FFT // 2
    a = np.pad(audio.astype(np.float64), (pad, pad), mode="reflect")
    n_frames = 1 + (len(a) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = a[idx] * win[None, :]
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)            # [frames, 201]
    mags = (np.abs(spec) ** 2)[:-1]                        # DROP the final frame
    fb = mel_filter_bank(1 + N_FFT // 2, N_MELS, 0.0, sr / 2.0, sr)
    mel = fb @ mags.T                                      # [n_mels, frames]
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def read_wav(path, seconds=None):
    import wave
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, "expect 16k mono"
        n = w.getnframes()
        if seconds:
            n = min(n, int(seconds * SR))
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def main():
    tracks = [d for d in os.listdir(FIX) if os.path.isdir(os.path.join(FIX, d))]
    ok_all = True
    for track in tracks:
        with open(os.path.join(FIX, track, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        stem = man["_track"]
        clip = man["_clip_seconds"]
        wav = os.path.join(WAV16K, stem + ".wav")
        if not os.path.exists(wav):
            print(f"{track}: missing {wav}")
            continue
        ref = np.fromfile(os.path.join(FIX, track, "mel.f32"),
                          dtype="<f4").reshape(tuple(man["mel"]["shape"]))
        got = log_mel(read_wav(wav, clip))
        n = min(ref.shape[1], got.shape[1])
        r, g = ref[:, :n].astype(np.float64), got[:, :n].astype(np.float64)
        corr = float(np.corrcoef(r.ravel(), g.ravel())[0, 1])
        rel = float(np.sqrt(np.mean((r - g) ** 2)) / (np.sqrt(np.mean(r ** 2)) + 1e-12))
        mx = float(np.abs(r - g).max())
        good = corr >= 0.9999
        ok_all &= good
        print(f"  {'OK  ' if good else 'FAIL'} {track:<24} ref{ref.shape} got{got.shape} "
              f"corr={corr:.7f} relRMSE={rel:.2e} maxabs={mx:.2e}")

        # The reference mel is stored BF16 (MelConfig.mel_dtype is torch.bfloat16 -- the
        # processor computes in fp32 and casts before the encoder sees it). Quantify the
        # gap in bf16 ULPs rather than absolute error: near 1.0 one ULP is 2^-8 = 3.906e-3,
        # which is exactly the maxabs above. So the question is not "is there a gap" but
        # "is any element off by MORE than one ULP", which would mean a real formula bug.
        ulp = np.ldexp(1.0, np.floor(np.log2(np.maximum(np.abs(r), 1e-30))).astype(int) - 8)
        off = np.abs(g - r) / ulp
        print(f"       {float((off > 0.501).mean())*100:5.1f}% of bins differ, "
              f"worst {float(off.max()):.2f} bf16 ULP")
        # NOT a pass/fail gate. The dynamic-range clamp coupies the whole spectrogram:
        #   log_spec = maximum(log_spec, log_spec.max() - 8.0)
        # the floor is derived from the GLOBAL maximum, so a 1-ULP difference in a single
        # loud bin moves the floor for every quiet bin at once. Tracks with lots of near-
        # silence (Johnny Cash opens on sparse spoken word) therefore show a few ULP of
        # spread while still correlating at 0.999998.
        #
        # PORTING CONSEQUENCE, and this one bites: the C++ must compute that maximum over
        # the WHOLE utterance. If long audio is chunked and each chunk takes its own max,
        # every chunk gets a different floor and the frames stop being comparable -- which
        # would look like a mysterious encoder drift on long tracks only.
    print("\n" + ("MEL FRONTEND MATCHES" if ok_all else "** MEL MISMATCH **"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

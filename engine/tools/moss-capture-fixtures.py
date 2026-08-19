"""Capture MOSS-Music parity fixtures on CPU in fp32, for the GGML port.

Runs in WSL where transformers is 4.57.1 — MOSS's exact target version — so NO compatibility
shims are involved. That matters: the earlier Windows run needed four patches for
transformers 5.15, and the prime suspect for its bad output was the cache_position shim in
the decode loop, not the forward pass. This script therefore does two things:

  1. Dumps every intermediate the GGML port needs to validate against, in fp32 (the stricter
     of the two parity bars in the mm3-backend skill: >=0.9999 vs an fp32 CPU rerun).
  2. Runs a deliberately naive greedy decode — full re-forward per step, no KV cache, audio
     re-supplied every step so the mask can never go stale. If THAT produces good captions,
     the decode shim was the bug and the forward pass is sound, which is the assumption the
     whole port rests on.

  .venv/bin/python capture_fixtures.py --clip 30 --steps 24
"""
import argparse
import json
import os
import sys
import time

ROOT = "/mnt/m/Music Captioners"
OUT = os.path.join(ROOT, "fixtures")
WAV16K = os.path.join(ROOT, "tracks", "wav16k")

os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoProcessor, WhisperConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL = os.path.expanduser("~/weights/MOSS-Music-8B-Instruct")

PROMPT = (
    "Analyse this music track and describe it in detail: genre and energy, the drum and "
    "percussion pattern, bass, harmony and lead instrumentation, vocal style and timbre, "
    "production and mix character, and how the arrangement develops from intro to ending. "
    "State the tempo in BPM, the key and scale, and the time signature. Be specific and "
    "concrete about what you actually hear."
)

TRACKS = [
    "daftpunk_discovery__01 One More Time",
    "johnnycash_american4__01 - Johnny Cash - The Man Comes Around",
]


def save(name, t, manifest):
    """Write a tensor as raw little-endian f32 plus a manifest entry."""
    a = t.detach().to(torch.float32).cpu().numpy()
    a = np.ascontiguousarray(a)
    path = os.path.join(OUT, name + ".f32")
    a.astype("<f4").tofile(path)
    manifest[name] = {"shape": list(a.shape), "dtype": "float32",
                      "file": os.path.basename(path),
                      "min": float(a.min()), "max": float(a.max()),
                      "mean": float(a.mean()), "absmean": float(np.abs(a).mean())}
    print(f"    saved {name:<28} {str(list(a.shape)):<20} absmean={np.abs(a).mean():.6f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=int, default=30, help="seconds of audio to use")
    ap.add_argument("--steps", type=int, default=24, help="greedy tokens to decode")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | CPU fp32",
          flush=True)

    os.makedirs(OUT, exist_ok=True)

    print("loading processor + model (fp32, CPU) ...", flush=True)
    t0 = time.time()
    # enable_time_marker=True is MANDATORY and is NOT the default.
    # processing_moss_music.py's from_pretrained does
    #     enable_time_marker = kwargs.pop("enable_time_marker", False)
    # while its own __init__ signature defaults it True. SGLang's processor always
    # emits the markers, so a fixture captured without them guards a prompt the
    # model was never trained on -- which is exactly the bug that made the HF path
    # answer "Arabic pop" where SGLang answers "Eurodance".
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True,
                                              enable_time_marker=True)
    if not getattr(processor, "enable_time_marker", False):
        raise SystemExit("capture_fixtures: time markers did NOT enable — refusing to "
                         "write fixtures that would enshrine the marker-less prompt")
    # config.json's auto_map registers only AutoConfig/AutoProcessor — never AutoModel — so
    # AutoModel cannot resolve MossMusicModel on ANY transformers version. Not a 5.x issue.
    model_cls = get_class_from_dynamic_module("modeling_moss_music.MossMusicModel", MODEL)
    # MossMusicConfig stores audio_config as a raw dict, but the encoder does attribute
    # access on it and hands it to WhisperEncoderLayer. This is NOT a transformers-5.x
    # issue — it fails identically on 4.57.1, the version MOSS targets. WhisperConfig takes
    # the Whisper keys by name and absorbs the MOSS extras as plain attributes.
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if isinstance(config.audio_config, dict):
        config.audio_config = WhisperConfig(**config.audio_config)

    # No device_map: it pulls in accelerate, and everything is on CPU here anyway.
    model = model_cls.from_pretrained(MODEL, config=config, trust_remote_code=True,
                                      torch_dtype=torch.float32)
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    audio_token_id = processor.audio_token_id

    for stem in TRACKS:
        wav = os.path.join(WAV16K, stem + ".wav")
        if not os.path.exists(wav):
            print(f"MISSING {wav}", flush=True)
            continue
        short = stem.split("__")[0]
        print(f"\n=== {short} ===", flush=True)

        audio, sr = sf.read(wav, dtype="float32")
        assert sr == 16000, sr
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio[: args.clip * sr]

        manifest = {"_track": stem, "_clip_seconds": args.clip,
                    "_model": "OpenMOSS-Team/MOSS-Music-8B-Instruct",
                    "_dtype": "float32", "_device": "cpu",
                    "_transformers": __import__("transformers").__version__,
                    "_prompt": PROMPT}

        inputs = processor(text=PROMPT, audios=[audio], return_tensors="pt")
        input_ids = inputs["input_ids"]
        audio_mask = input_ids == audio_token_id
        manifest["_n_audio_tokens"] = int(audio_mask.sum())
        manifest["_n_input_ids"] = int(input_ids.shape[1])
        manifest["_audio_token_id"] = int(audio_token_id)
        print(f"  input_ids {tuple(input_ids.shape)}  audio_tokens {int(audio_mask.sum())} "
              f"({int(audio_mask.sum())/args.clip:.2f}/s)", flush=True)

        # ---- mel, straight off the processor -------------------------------------
        save("mel", inputs["audio_data"][0], manifest)
        save("input_ids", input_ids[0].to(torch.float32), manifest)

        # ---- hooks on every boundary the port must match --------------------------
        caps = {}

        def grab(key):
            def hook(_m, _i, out):
                caps[key] = out[0] if isinstance(out, tuple) else out
            return hook

        handles = [
            model.audio_adapter.register_forward_hook(grab("adapter_out")),
        ]
        for i, mod in enumerate(model.deepstack_audio_merger_list):
            handles.append(mod.register_forward_hook(grab(f"deepstack_merger_{i}")))

        enc_out = {}

        def enc_hook(_m, _i, out):
            enc_out["last"] = out.last_hidden_state
            enc_out["hidden"] = out.hidden_states
        handles.append(model.audio_encoder.register_forward_hook(enc_hook))

        def lm_pre(_m, a, kw):
            if kw.get("inputs_embeds") is not None:
                caps["spliced_embeds"] = kw["inputs_embeds"]
        handles.append(model.language_model.register_forward_pre_hook(lm_pre,
                                                                     with_kwargs=True))

        fwd = dict(input_ids=input_ids,
                   attention_mask=inputs["attention_mask"],
                   audio_data=inputs["audio_data"].to(torch.float32),
                   audio_data_seqlens=inputs["audio_data_seqlens"],
                   audio_input_mask=audio_mask,
                   use_cache=False)

        print("  forward pass ...", flush=True)
        t1 = time.time()
        with torch.inference_mode():
            out = model(**fwd)
        print(f"  forward in {time.time()-t1:.0f}s", flush=True)

        save("encoder_out", enc_out["last"][0], manifest)
        for i, h in enumerate(enc_out["hidden"] or []):
            save(f"encoder_deepstack_{i}", h[0], manifest)
        save("adapter_out", caps["adapter_out"][0], manifest)
        for i in range(len(model.deepstack_audio_merger_list)):
            k = f"deepstack_merger_{i}"
            if k in caps:
                save(k, caps[k][0], manifest)
        if "spliced_embeds" in caps:
            save("spliced_embeds", caps["spliced_embeds"][0], manifest)
        save("logits_last", out.logits[0, -1], manifest)

        top = torch.topk(out.logits[0, -1], 10)
        manifest["_first_token_top10"] = [
            {"id": int(i), "logit": float(v),
             "tok": processor.tokenizer.decode([int(i)])}
            for v, i in zip(top.values, top.indices)]
        print("  first-token top5: " +
              ", ".join(f"{d['tok']!r}({d['logit']:.2f})"
                        for d in manifest["_first_token_top10"][:5]), flush=True)

        for h in handles:
            h.remove()

        # ---- naive greedy decode: no cache, audio re-fed, mask never stale --------
        print(f"  greedy decode ({args.steps} tokens, no KV cache) ...", flush=True)
        ids = input_ids.clone()
        t2 = time.time()
        with torch.inference_mode():
            for step in range(args.steps):
                m = ids == audio_token_id
                o = model(input_ids=ids,
                          attention_mask=torch.ones_like(ids),
                          audio_data=inputs["audio_data"].to(torch.float32),
                          audio_data_seqlens=inputs["audio_data_seqlens"],
                          audio_input_mask=m,
                          use_cache=False)
                nxt = int(o.logits[0, -1].argmax())
                ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
                if nxt == 151645:
                    break
        text = processor.tokenizer.decode(ids[0, input_ids.shape[1]:],
                                          skip_special_tokens=True)
        el = time.time() - t2
        manifest["_greedy_no_cache"] = text
        manifest["_greedy_seconds"] = round(el, 1)
        print(f"  decode in {el:.0f}s ({el/max(1,args.steps):.1f}s/token)", flush=True)
        print(f"  >>> {text}", flush=True)

        d = os.path.join(OUT, short)
        os.makedirs(d, exist_ok=True)
        for k in list(manifest):
            if not k.startswith("_"):
                src = os.path.join(OUT, manifest[k]["file"])
                if os.path.exists(src):
                    os.replace(src, os.path.join(d, manifest[k]["file"]))
        with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"  -> {d}", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()

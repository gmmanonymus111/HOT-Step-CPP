#!/usr/bin/env python3
# convert-moss.py: MOSS-Music-8B safetensors -> GGUF for the HOT-Step captioning backend.
#
# Produces two files into --out:
#
#   moss-lm-<quant>.gguf   arch "qwen3"     -- the 8B LM, written to llama.cpp Qwen3
#                                              conventions (token_embd.weight,
#                                              blk.N.attn_q.weight, ...) so a
#                                              llama.cpp-grade loader consumes it
#                                              structurally. 398 + 1 tensors.
#   moss-aud-<quant>.gguf  arch "moss-aud"  -- Whisper-style audio encoder + the SwiGLU
#                                              adapter + the 3 deepstack mergers,
#                                              prefixed aud.*. 502 tensors.
#
# WHY TWO FILES
# -------------
# Same reasoning as convert-mm3.py: the LM file stays structurally *pure* Qwen3 because
# llama.cpp's loader rejects unknown tensors, so nothing audio-specific may live in it.
# Everything else is ours to shape. Unlike MM3 there are only two roles, so there is no
# bundle-then-split step.
#
# QUANTISATION POLICY
# -------------------
# --quant applies to the LM only. The audio file is ALWAYS f16 (norms/biases f32).
# Measured on the fp32 reference: encoder_out has absmean 0.554 but adapter_out has
# absmean 426 -- a ~770x magnitude jump across the adapter. Audio encoders are already
# the quantisation-sensitive half of a multimodal model, and that dynamic range is a
# standing invitation to lose the plot. It is only ~1.3 GB at f16; do not "optimise" it.
#
# NOT WRITTEN: audio_encoder.embed_positions.inv_timescales. It is present in the
# checkpoint but registered persistent=False upstream, so HF itself reports it as an
# UNEXPECTED key and recomputes it in __init__. The runtime must recompute it with the
# standard Whisper formula (see moss.audio.max_timescale below). Storing it would invite
# someone to load it and silently diverge if the formula ever differs.
#
# DEPENDENCIES: numpy + gguf (llama.cpp's gguf-py). No torch: BF16 shards are read with a
# small built-in mmap reader, the same approach convert-mm3.py takes.
#
# USAGE
#   python engine/tools/convert-moss.py \
#          --src "M:/Music Captioners/hf-cache/hub/models--OpenMOSS-Team--MOSS-Music-8B-Instruct/snapshots/<hash>" \
#          --out engine/models/moss --quant q8_0

import argparse
import json
import mmap
import os
import struct
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("need gguf (llama.cpp's gguf-py): pip install gguf")


def log(m):
    print(m, flush=True)


def die(m):
    sys.exit(f"convert-moss: {m}")


# ---------------------------------------------------------------------------
# Minimal safetensors reader (BF16-capable, no torch)
# ---------------------------------------------------------------------------

_ST_DTYPE = {
    "F64": (np.float64, 8), "F32": (np.float32, 4), "F16": (np.float16, 2),
    "BF16": (None, 2), "I64": (np.int64, 8), "I32": (np.int32, 4),
    "I16": (np.int16, 2), "I8": (np.int8, 1), "U8": (np.uint8, 1),
    "BOOL": (np.bool_, 1),
}


class Safetensors:
    """mmap over one or more .safetensors shards, exposing name -> float32 ndarray."""

    def __init__(self, paths):
        self.entries = {}   # name -> (path, dtype, shape, begin, end)
        self._maps = {}
        for p in paths:
            with open(p, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(hlen).decode("utf-8"))
            base = 8 + hlen
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                b, e = meta["data_offsets"]
                self.entries[name] = (p, meta["dtype"], tuple(meta["shape"]),
                                      base + b, base + e)

    def _map(self, path):
        if path not in self._maps:
            f = open(path, "rb")
            self._maps[path] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        return self._maps[path][1]

    def __contains__(self, name):
        return name in self.entries

    def names(self):
        return set(self.entries)

    def get(self, name):
        if name not in self.entries:
            die(f"missing tensor {name}")
        path, dt, shape, b, e = self.entries[name]
        mm = self._map(path)
        raw = mm[b:e]
        if dt == "BF16":
            # bf16 is the top 16 bits of an fp32; widen by shifting left, no table needed.
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32)
        else:
            npdt, _sz = _ST_DTYPE.get(dt, (None, None))
            if npdt is None:
                die(f"{name}: unsupported safetensors dtype {dt}")
            arr = np.frombuffer(raw, dtype=npdt).astype(np.float32)
        return arr.reshape(shape)

    def close(self):
        for f, mm in self._maps.values():
            mm.close()
            f.close()


# ---------------------------------------------------------------------------
# Tensor policy + writer  (mirrors convert-mm3.py so both files behave alike)
# ---------------------------------------------------------------------------

F32 = "f32"      # never quantised: norms, biases
HALF = "half"    # f16 even under --quant q8_0
QUANT = "quant"  # q8_0 under --quant q8_0, f16 otherwise

F16_MAX = 65504.0


class Bundle:
    """Accumulates tensors for one GGUF, converting each to its FINAL storage dtype at
    put() time rather than hoarding float32 until write().

    convert-mm3.py keeps everything as float32 and converts in write(); for an 8B LM that
    is a ~32 GB peak, which is fine on an idle box and fails the moment anything else is
    using the machine (it OOM'd here next to a WSL job). Converting eagerly drops the peak
    to the size of the output file -- ~8.5 GB at q8_0 -- for no loss of behaviour."""

    def __init__(self, arch, quant):
        self.arch = arch
        self.quant = quant
        self.kv = []
        self.tensors = []      # (name, payload, raw_shape|None, raw_dtype|None)
        self.counts = {"f32": 0, "f16": 0, "q8_0": 0}
        self._seen = set()

    def meta(self, fn, *args):
        self.kv.append((fn, args))

    def put(self, name, arr, policy):
        if name in self._seen:
            die(f"duplicate output tensor {name}")
        self._seen.add(name)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        ne0 = arr.shape[-1] if arr.ndim else 1

        if policy == F32 or arr.ndim < 2:
            self.tensors.append((name, arr, None, None))
            self.counts["f32"] += 1
            return

        peak = float(np.abs(arr).max()) if arr.size else 0.0
        if peak > F16_MAX:
            log(f"  {name}: |w|max={peak:.1f} exceeds f16 range, storing F32")
            self.tensors.append((name, arr, None, None))
            self.counts["f32"] += 1
            return

        if self.quant == "q8_0" and policy == QUANT and ne0 % 32 == 0:
            # No raw_shape: gguf derives the logical shape from the quantised byte rows
            # and would misread one we supplied.
            q = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            self.tensors.append((name, q, None, gguf.GGMLQuantizationType.Q8_0))
            self.counts["q8_0"] += 1
        else:
            # raw_dtype labels, it does not convert -- hand it real f16 bytes.
            h = arr.astype(np.float16).view(np.uint16)
            self.tensors.append((name, h, arr.shape, gguf.GGMLQuantizationType.F16))
            self.counts["f16"] += 1

    def write(self, path):
        w = gguf.GGUFWriter(path, self.arch)
        for fn, args in self.kv:
            getattr(w, fn)(*args)
        for name, payload, raw_shape, raw_dtype in self.tensors:
            if raw_dtype is None:
                w.add_tensor(name, payload)
            elif raw_shape is None:
                w.add_tensor(name, payload, raw_dtype=raw_dtype)
            else:
                w.add_tensor(name, payload, raw_shape=raw_shape, raw_dtype=raw_dtype)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        n, size = self.counts, os.path.getsize(path) / 1e9
        log(f"wrote {os.path.basename(path)}: {len(self.tensors)} tensors "
            f"({n['f32']} F32, {n['f16']} F16, {n['q8_0']} Q8_0), {size:.2f} GB")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer(b, src_dir, vocab_size):
    """MOSS-Instruct ships vocab.json + merges.txt + added_tokens.json and NO
    tokenizer.json -- i.e. the *slow* Qwen2Tokenizer (`use_fast=False` in its own
    processor). This is MM3 trap #2 verbatim: `tokenizer.ggml.pre = qwen2` in the
    llama.cpp sense means {1,3} digit grouping, whereas the slow tokenizer uses classic
    GPT-2 single-digit pre-tokenisation. The engine's bpe.h implements the latter; the
    KV below is only a label, so the C++ side must not switch behaviour on it."""
    vp = os.path.join(src_dir, "vocab.json")
    mp = os.path.join(src_dir, "merges.txt")
    ap = os.path.join(src_dir, "added_tokens.json")
    for p in (vp, mp):
        if not os.path.exists(p):
            die(f"tokenizer: missing {p}")

    with open(vp, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    added = {}
    if os.path.exists(ap):
        with open(ap, "r", encoding="utf-8") as f:
            added = json.load(f)          # {content: id}

    reverse = {i: s for s, i in vocab.items()}
    added_ids = set()
    for content, i in added.items():
        reverse[i] = content
        added_ids.add(i)

    tokens, toktypes = [], []
    n_pad = 0
    for i in range(vocab_size):
        if i in reverse:
            tokens.append(reverse[i])
            toktypes.append(gguf.TokenType.CONTROL if i in added_ids
                            else gguf.TokenType.NORMAL)
        else:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
            n_pad += 1

    merges = []
    with open(mp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#version"):
                continue
            merges.append(line)

    b.meta("add_tokenizer_model", "gpt2")
    b.meta("add_tokenizer_pre", "qwen2")
    b.meta("add_token_list", tokens)
    b.meta("add_token_types", toktypes)
    b.meta("add_token_merges", merges)
    b.meta("add_bos_token_id", 151643)     # <|endoftext|>
    b.meta("add_eos_token_id", 151645)     # <|im_end|>
    b.meta("add_pad_token_id", 151643)
    b.meta("add_add_bos_token", False)
    log(f"tokenizer: {len(tokens)} tokens ({len(vocab)} BPE + {len(added)} added "
        f"+ {n_pad} unused), {len(merges)} merges")


# ---------------------------------------------------------------------------
# LM  (llama.cpp Qwen3 naming)
# ---------------------------------------------------------------------------

def build_lm(st, cfg, src_dir, quant):
    lc = cfg["language_config"]
    n_layer = lc["num_hidden_layers"]
    n_embd = lc["hidden_size"]
    b = Bundle("qwen3", quant)

    b.meta("add_name", "MOSS-Music-8B")
    b.meta("add_block_count", n_layer)
    b.meta("add_context_length", lc["max_position_embeddings"])
    b.meta("add_embedding_length", n_embd)
    b.meta("add_feed_forward_length", lc["intermediate_size"])
    b.meta("add_head_count", lc["num_attention_heads"])
    b.meta("add_head_count_kv", lc["num_key_value_heads"])
    b.meta("add_key_length", lc["head_dim"])
    b.meta("add_value_length", lc["head_dim"])
    b.meta("add_layer_norm_rms_eps", lc["rms_norm_eps"])
    b.meta("add_rope_freq_base", lc["rope_parameters"]["rope_theta"]
           if "rope_parameters" in lc else lc["rope_theta"])
    b.meta("add_vocab_size", lc["vocab_size"])
    b.meta("add_file_type", gguf.LlamaFileType.ALL_F32)

    b.put("token_embd.weight", st.get("language_model.embed_tokens.weight"), QUANT)
    b.put("output_norm.weight", st.get("language_model.norm.weight"), F32)
    # tie_word_embeddings is false -- lm_head is a genuinely separate tensor. A stock
    # Qwen3 GGUF that omits `output.weight` is NOT interchangeable with this one.
    b.put("output.weight", st.get("lm_head.weight"), QUANT)

    for i in range(n_layer):
        s = f"language_model.layers.{i}."
        d = f"blk.{i}."
        b.put(d + "attn_norm.weight", st.get(s + "input_layernorm.weight"), F32)
        b.put(d + "ffn_norm.weight", st.get(s + "post_attention_layernorm.weight"), F32)
        b.put(d + "attn_q.weight", st.get(s + "self_attn.q_proj.weight"), QUANT)
        b.put(d + "attn_k.weight", st.get(s + "self_attn.k_proj.weight"), QUANT)
        b.put(d + "attn_v.weight", st.get(s + "self_attn.v_proj.weight"), QUANT)
        b.put(d + "attn_output.weight", st.get(s + "self_attn.o_proj.weight"), QUANT)
        # Qwen3 QK-norm: per-head RMSNorm over head_dim, applied before RoPE.
        b.put(d + "attn_q_norm.weight", st.get(s + "self_attn.q_norm.weight"), F32)
        b.put(d + "attn_k_norm.weight", st.get(s + "self_attn.k_norm.weight"), F32)
        b.put(d + "ffn_gate.weight", st.get(s + "mlp.gate_proj.weight"), QUANT)
        b.put(d + "ffn_up.weight", st.get(s + "mlp.up_proj.weight"), QUANT)
        b.put(d + "ffn_down.weight", st.get(s + "mlp.down_proj.weight"), QUANT)

    build_tokenizer(b, src_dir, lc["vocab_size"])
    return b


# ---------------------------------------------------------------------------
# Audio encoder + adapter + deepstack
# ---------------------------------------------------------------------------

def build_aud(st, cfg):
    ac = cfg["audio_config"]
    lc = cfg["language_config"]
    n_layer = ac["encoder_layers"]
    b = Bundle("moss-aud", "f16")

    ds_idx = list(ac.get("deepstack_encoder_layer_indexes") or [])
    n_inject = cfg.get("deepstack_num_inject_layers")
    if n_inject is not None:
        ds_idx = ds_idx[: int(n_inject)]

    b.meta("add_name", "MOSS-Music-8B audio tower")
    b.meta("add_uint32", "moss.audio.block_count", n_layer)
    b.meta("add_uint32", "moss.audio.embedding_length", ac["d_model"])
    b.meta("add_uint32", "moss.audio.head_count", ac["encoder_attention_heads"])
    b.meta("add_uint32", "moss.audio.feed_forward_length", ac["encoder_ffn_dim"])
    b.meta("add_uint32", "moss.audio.n_mels", ac["num_mel_bins"])
    b.meta("add_uint32", "moss.audio.output_dim", ac["output_dim"])
    b.meta("add_uint32", "moss.audio.downsample_hidden", ac["downsample_hidden_size"])
    b.meta("add_uint32", "moss.audio.downsample_rate", ac["downsample_rate"])
    b.meta("add_float32", "moss.audio.layer_norm_eps", float(ac["layer_norm_eps"]))
    # Measured exactly: mel is 100 fps (hop 160 @ 16 kHz) and the three stride-2 convs
    # divide by 8, giving 12.5 audio tokens/second. A 320.84 s track -> 4011 tokens.
    b.meta("add_float32", "moss.audio.tokens_per_second", 12.5)
    b.meta("add_uint32", "moss.audio.sample_rate", 16000)
    b.meta("add_uint32", "moss.audio.n_fft", 400)
    b.meta("add_uint32", "moss.audio.hop_length", 160)
    # embed_positions.inv_timescales is deliberately NOT stored -- recompute it:
    #   inv = exp(-log(max_timescale)/(d_model/2 - 1) * arange(d_model/2))
    #   pos = concat(sin(t*inv), cos(t*inv))
    b.meta("add_float32", "moss.audio.max_timescale", 10000.0)
    b.meta("add_array", "moss.audio.deepstack_encoder_layers",
           [int(x) for x in ds_idx])
    b.meta("add_uint32", "moss.audio.deepstack_inject_layers", len(ds_idx))
    b.meta("add_uint32", "moss.audio.adapter_hidden", cfg["adapter_hidden_size"])
    b.meta("add_uint32", "moss.audio.lm_embedding_length", lc["hidden_size"])
    b.meta("add_uint32", "moss.audio.audio_token_id", 151654)
    b.meta("add_uint32", "moss.audio.audio_bos_id", 151669)
    b.meta("add_uint32", "moss.audio.audio_eos_id", 151670)
    b.meta("add_file_type", gguf.LlamaFileType.ALL_F32)

    # Conv stem. PyTorch Conv2d weight is [OC, IC, KH, KW]; GGUF/ggml reverse the dim
    # order, so those same bytes present to ggml_conv_2d as [KW, KH, IC, OC] -- which is
    # exactly its kernel layout. Nothing needs transposing (same finding as mdx23c-ggml.h).
    for k in ("conv1", "conv2", "conv3"):
        b.put(f"aud.{k}.weight", st.get(f"audio_encoder.{k}.weight"), HALF)
        b.put(f"aud.{k}.bias", st.get(f"audio_encoder.{k}.bias"), F32)

    b.put("aud.stem_proj.weight", st.get("audio_encoder.stem_proj.weight"), HALF)
    b.put("aud.stem_proj.bias", st.get("audio_encoder.stem_proj.bias"), F32)
    b.put("aud.norm.weight", st.get("audio_encoder.layer_norm.weight"), F32)
    b.put("aud.norm.bias", st.get("audio_encoder.layer_norm.bias"), F32)

    for i in range(n_layer):
        s = f"audio_encoder.layers.{i}."
        d = f"aud.blk.{i}."
        b.put(d + "attn_norm.weight", st.get(s + "self_attn_layer_norm.weight"), F32)
        b.put(d + "attn_norm.bias", st.get(s + "self_attn_layer_norm.bias"), F32)
        b.put(d + "attn_q.weight", st.get(s + "self_attn.q_proj.weight"), HALF)
        b.put(d + "attn_q.bias", st.get(s + "self_attn.q_proj.bias"), F32)
        # Whisper convention: k_proj has NO bias. Do not synthesise a zero one --
        # the graph must simply skip the add.
        b.put(d + "attn_k.weight", st.get(s + "self_attn.k_proj.weight"), HALF)
        b.put(d + "attn_v.weight", st.get(s + "self_attn.v_proj.weight"), HALF)
        b.put(d + "attn_v.bias", st.get(s + "self_attn.v_proj.bias"), F32)
        b.put(d + "attn_out.weight", st.get(s + "self_attn.out_proj.weight"), HALF)
        b.put(d + "attn_out.bias", st.get(s + "self_attn.out_proj.bias"), F32)
        b.put(d + "ffn_norm.weight", st.get(s + "final_layer_norm.weight"), F32)
        b.put(d + "ffn_norm.bias", st.get(s + "final_layer_norm.bias"), F32)
        b.put(d + "ffn_up.weight", st.get(s + "fc1.weight"), HALF)
        b.put(d + "ffn_up.bias", st.get(s + "fc1.bias"), F32)
        b.put(d + "ffn_down.weight", st.get(s + "fc2.weight"), HALF)
        b.put(d + "ffn_down.bias", st.get(s + "fc2.bias"), F32)

    # SwiGLU, no biases: down(silu(gate(x)) * up(x)).
    for tag, prefix in [("adapter", "audio_adapter")] + [
            (f"deepstack.{k}", f"deepstack_audio_merger_list.{k}")
            for k in range(len(ds_idx))]:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            short = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}[proj]
            b.put(f"aud.{tag}.{short}.weight", st.get(f"{prefix}.{proj}.weight"), HALF)

    return b


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF snapshot dir for MOSS-Music-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="f16", choices=("f16", "q8_0"),
                    help="LM only; the audio tower is always f16")
    ap.add_argument("--components", default="lm,aud")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    with open(os.path.join(src, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "moss_music":
        die(f"{src}: model_type is {cfg.get('model_type')!r}, expected 'moss_music'")

    shards = sorted(p for p in os.listdir(src) if p.endswith(".safetensors"))
    if not shards:
        die(f"{src}: no .safetensors shards")
    st = Safetensors([os.path.join(src, p) for p in shards])
    log(f"source: {len(shards)} shards, {len(st.names())} tensors")

    os.makedirs(args.out, exist_ok=True)
    want = {c.strip() for c in args.components.split(",") if c.strip()}
    consumed = set()

    if "lm" in want:
        log("building LM ...")
        b = build_lm(st, cfg, src, args.quant)
        for i in range(cfg["language_config"]["num_hidden_layers"]):
            s = f"language_model.layers.{i}."
            consumed |= {s + x for x in (
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")}
        consumed |= {"language_model.embed_tokens.weight", "language_model.norm.weight",
                     "lm_head.weight"}
        b.write(os.path.join(args.out, f"moss-lm-{args.quant}.gguf"))

    if "aud" in want:
        log("building audio tower (always f16) ...")
        b = build_aud(st, cfg)
        consumed |= {n for n in st.names() if n.startswith(
            ("audio_encoder.", "audio_adapter.", "deepstack_audio_merger_list."))}
        b.write(os.path.join(args.out, "moss-aud-f16.gguf"))

    # Refuse to emit a partial model silently: anything unaccounted for is a bug in the
    # mapping above, not a curiosity. inv_timescales is the one intentional omission.
    if want >= {"lm", "aud"}:
        leftover = st.names() - consumed
        leftover.discard("audio_encoder.embed_positions.inv_timescales")
        if leftover:
            die(f"{len(leftover)} source tensors not consumed, e.g. "
                f"{sorted(leftover)[:5]}")
        log("all source tensors accounted for "
            "(inv_timescales intentionally omitted -- recomputed at runtime)")

    st.close()
    log("done")


if __name__ == "__main__":
    main()

---
name: mm3-backend
description: Maps HOT-Step's native MiniMax-Music3 backend — engine port modules, endpoints, server/UI integration, parity/fixture infrastructure, and the hard-won trap list. Use when working on anything MM3 — engine/src/minimax/, backends/minimax/, /mm3/* endpoints, the backend toggle/capability gating, MM3 model files or Model Manager entries, debugging MM3 generations, MM3 performance work, or extending MM3 features (covers, training, Lyric Studio).
---

# MiniMax-Music3 backend

Native C++/GGML port of [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
built 2026-08-13 (release day) as HOT-Step's second generation backend behind an N-backend
abstraction. **Status: rudimentary text2music only** (caption + lyrics + duration + seed);
no covers/repaint/stems/adapters/training. Output = raw 44.1 kHz stereo WAV (app norm is 48 k —
post-chain steps that hardcode 48 k are skipped for MM3).

Deep docs (local, gitignored): `docs/plans/multi-backend-architecture.md` (architecture plan,
day-0 findings, op inventory) and `docs/plans/mm3-gguf-layout.md` (GGUF contract + loader
addendum). Caption format: the **mm3-captioning** skill.

## Model + pipeline (25 fps frames; every module parity-proven vs the diffusers reference)

```
caption+lyrics → Qwen2 BPE → Global LM 8.59B (Qwen3 arch, semantic codes @ ids 151675–168058,
  EOS 151670, AR CFG 1.5 as persistent 2-row batch) → per frame: RVQ depth decoder 0.6B
  (7 acoustic codebooks) → frame_hiddens [F,8,4096] → per 200-frame window (hop 100):
  condition encoder 25M (×3.4453125 nearest resample) → flow DiT 2.4B (30 Euler steps,
  CFG 1.7, zeros-cond uncond as separate pass) → vocoder 54M (DAC-style, ×512 → 44.1 kHz)
  → overlap-crop stitch
```

## File map

| Piece | Where |
|---|---|
| Engine modules | `engine/src/minimax/` — `mm3-model.h` (loader/residency), `mm3-tokenizer.h`, `mm3-lm-graph.h`, `mm3-ar-loop.h`, `mm3-sample.h`, `mm3-depth-graph.h`, `mm3-cond-graph.h`, `mm3-dit-graph.h`, `mm3-vocoder-graph.h`, `mm3-pipeline.h` (e2e + chunking), `mm3-request.h` (prompt assembly/hygiene), `mm3-job.h` (job queue + VRAM arbitration), `mm3-server.h` (endpoints) |
| Hooks | one include in `engine/tools/hot-step-server.cpp` (+ `mm3_register_routes`/`mm3_register_job_routes` call sites); checked by `engine/verify-hooks.ps1` (hooks 4/4b/4c) |
| Server backend | `server/src/services/backends/` — `types.ts` (EngineBackend + capability manifest), `registry.ts`, `ace/`, `minimax/{client,index,generate}.ts`; routes `server/src/routes/backends.ts`; generation branch at top of `runGeneration` in `routes/generate.ts` |
| UI | `stores/backendStore.ts`, `hooks/useCapabilities.ts`, `global-bar/BackendToggle.tsx` (hidden until ≥2 backends), `shared/BackendCapabilityGate.tsx` (studio guards), gating in `GlobalParamBar.tsx` |
| Models | **5-way split since 2026-08-14** (ported from ServeurpersoCom/minimaxmusic.cpp): `models/mm3/mm3-{lm,depth,cond,dit,voc}-<quant>.gguf` (archs `qwen3` / `mm3-{depth,cond,dit,voc}`), legacy `mm3-synth-*` bundles still load (fill any role; split file wins per quant token). Per-role quant mixing (LM Q8_0 + DiT Q4_K_M is the headline combo); a DiT/adapter swap reloads only cond+dit+voc — LM stays warm. cond/voc are **never quantised** (f16 only). Hosted `scragnog/MiniMax-Music3-GGUF`; registry role `mm3`, packs rebuilt on split components in `server/src/data/model-registry.json` |
| Converter | `engine/tools/convert-mm3.py` (safetensors→GGUF bundle; folds weight-norm; refuses pruned/int8_convrot) then `engine/tools/split-mm3.py` (byte-exact bundle→5-way split; idempotent; cond/voc only from native bundles) |
| Fixtures / parity | `D:\Ace-Step-Latest\mm3-weights\fixtures\` (manifest.json + raw f32 dumps + reference WAVs), seed-spread study in `..\seed-spread-2026-08-13\`; venvs: `.venv-convert` (numpy/gguf), `.venv-ref` (patched diffusers @ dafe3733 — `patch_venv.py --restore`; `capture_fixtures.py --replay` rebuilds dumps without rerunning the model) |

## Engine endpoints (:8085 via app, standalone tests on :8086)

`GET /mm3/props` (files/config/loaded/limits — **blocks while an MM3 generation runs**; always
call with ~2.5 s timeout and keep last-known-good), `POST /mm3/warm` / `POST /mm3/unload`
(idempotent; unload frees weights+KV), `POST /mm3/synth` (production, rides the same FIFO GPU
worker as ACE `/synth`; standard `/job?id=` progress/cancel/result; request contract documented
in `mm3-request.h`/`mm3-job.h`), `GET /mm3/job?id=` (MM3-vocabulary progress, never blocks),
`POST /mm3/tokenize-check` (cold-capable; 5000-token limit), plus deprecated bring-up endpoints
(`/mm3/voc-decode`, `/mm3/dit-forward`, `/mm3/flow-sample`, `/mm3/depth-frame`,
`/mm3/cond-encode`, `/mm3/lm-plan`, `/mm3/synth-e2e`) kept for parity work — they run GPU work
on httplib threads; never build production paths on them.

**Standalone launch gotcha:** ace-server exits `0xC0000135` with zero output unless
`engine/trtllm-libs` + `engine/deps/tensorrt_libs` are prepended to PATH (aceEngineProcess.ts
does this; `engine/server.cmd` does not).

## The trap list (each cost real debugging — do not relearn)

1. **ComfyUI's wrapper NEGATES the DiT output; the diffusers reference (and our port) does not.**
   `mm3.dit.output_negated` in the GGUF records Comfy's behavior. Do not "fix" the sign.
2. **`tokenizer.ggml.pre = qwen2` is misleading** — the reference uses the *slow* Qwen2Tokenizer
   (single-digit regex = classic GPT-2 pre-tokenization, which `bpe.h` implements). Matching the
   KV's llama.cpp meaning ({1,3} digit grouping) breaks token parity.
3. **Scheduler sigmas must replicate float32 `linspace(1, 1/30, 30)` rounding** — deriving
   `i/steps` is wrong in the 7th digit and it matters.
4. **AR iteration 0 is fed back but never emitted** (emitted frame j = iteration j+1). A
   one-frame indexing slip degrades conditioning parity 49×.
5. **The semantic code embeds via the LM's `token_embd`, not `depth.audio_embd`.**
6. Caption hygiene: **`splitlines()` for caption, `split("\n")` for lyrics** — mixing them leaks
   a trailing `\n` into the template. Empty lyrics → we substitute `[instrumental]` (the
   reference *rejects* empty; this substitution is a HOT-Step decision).
7. Condition resample is **plain `nearest`, not `nearest-exact`** (differs on 199/689 positions).
8. **Never use `std::normal_distribution`** for reproducible noise (stdlib-dependent bytes) —
   `mm3_fill_noise` uses splitmix64 + Box-Muller.
9. GGUFs live in the **`models/mm3/` subdir** deliberately: the ACE registry scan globs only the
   models root (unknown-arch warnings + 17 GB header reparse per boot if placed there).
10. **Single-seed spectral/genre judgments are meaningless** — the reference's own 11-seed spread
    spans 272× in flatness and wanders off-genre with minimal captions. Structured 3-section
    captions (mm3-captioning skill) are the adherence lever. Compare distributions, not takes.
11. VRAM: f16 stack ≈ 22.5 GB + KV (288 kB/position) + ~3 GB compute headroom. Engine-side
    arbitration evicts idle ACE modules before MM3 warm; Node-side `releaseVram()` handles the
    reverse on backend switch and before ACE gens. ~600 MB stays in the CUDA pool after unload
    (returns on process exit — not a leak).
12. The LM GGUF is **not interchangeable with stock Qwen3-8B GGUFs** (extended 200 k vocab,
    untied head) and llama.cpp alone cannot run music generation.

## Performance budget (RTX 5090, f16, 12 s clip ≈ 12.4 s wall ≈ 1.0× realtime)

AR 25.5 ms/frame (LM step 15.3 — bandwidth-bound; depth 9.2 — launch-bound tiny matmuls, 37 % of
AR for 7 % of params) · flow 2.2 s/window · vocoder 85 ms/window. Speed levers in order:
**q8_0 LM** (~2× LM step + smaller download; re-validate by ear — quant can flip borderline
codes), **depth-decoder kernel fusion**, TRT much later. Known quality morsel: our synth on
identical codes measures ~18 % lower spectral flatness than the reference (unresolved, minor).

## Validation bar for MM3 changes

Forced-replay parity against the fixtures (never sampled-path comparisons — RNG can't match
torch). Established floors: per-module ≥ 0.999 corr vs the bf16 dumps (the dumps' own floor,
~1.6e-2 relRMSE) or ≥ 0.9999 vs an fp32 CPU rerun of the reference module. Full-clip replay:
0.9988. If a change should be bit-neutral, prove it with the deterministic seeds.

## Lyric timestamps: use Whisper, not attention (measured 2026-08-14)

ACE derives LRC from its **DiT's lyric cross-attention**. MM3 has no analogue:
its flow DiT has no cross-attention and never sees lyrics — conditioning is
channel concatenation from the condition encoder. The only place lyric tokens
and audio frames coexist is the **LM decode loop**, so that was probed
(`MM3_ALIGN_DUMP=1`, `MM3_ALIGN_FILE=<path>`; forces the manual F32 attention
path because flash fuses the softmax away).

Result: **viable at LINE level, which is the granularity that matters.**
`lrc_align()` emits lines ("consensus → DTW → sentence grouping → LRC text"),
not words, so line onset is the bar. Three heads track the lyric across every
test clip — **L12/H27, L19/H7, L24/H29** — and a naive DTW over their consensus
gives median line-onset error **0.83 s (indie)** and **0.71 s (synth)**, all
lines within 2 s. Folk was inconclusive (only 2 lines matched, and Whisper
itself renders that clip as a single 20 s segment).

Errors skew consistently NEGATIVE — attention leads the audio by ~0.6–0.8 s,
which is expected (the LM attends to a token as it begins generating that
content) and is a constant offset worth calibrating out, not noise.

Do NOT judge this at word level. An earlier pass did, called it unusable, and
was wrong twice over: it compared "fraction through BPE tokens" against
"fraction through words" — curves that differ even for a PERFECT alignment,
because tokens-per-word varies — and it used a naive DTW over 3 heads rather
than `lrc_align()`'s consensus denoising with ACE's 7-head-scale config.

Traps, each of which produced a wrong answer first:
1. **55 % of all 1152 heads sit permanently on one structural token.** Any head
   ranking must reward MOVEMENT — scoring "monotonic" as `delta >= 0` counts a
   pinned head as perfectly monotonic (it scored 0.98 and ranked first).
2. **Single-clip results do not generalise.** The best head on one clip (L16/H13)
   did not make the top 14 across three. Rank by the WORST clip, never the mean.
3. **Whisper `base` is not adequate ground truth for sung vocals** — it returned
   11 word timings for a folk clip `large-v3-turbo` transcribes as one 10-word
   line, which made attention look 7 s wrong. Validate against `large-v3-turbo`.
4. **Capture requires the manual attention path on EVERY layer, not just the
   three that are read.** This is the expensive, counter-intuitive one. Leaving
   the other 33 layers on flash is cheaper (9.8 vs 11.2 ms/step) and produces
   identical audio — but WRONG alignment: same seed, selective capture stamped
   the lines at 0.9/3.6/7.5/10.3 s where the vocal sits at 0.1/10.2/15.1/19.4,
   while all-manual gave 2.2/9.9/16.5/19.5. The captured layers' attention
   depends on whether the layers feeding them ran flash or manual by far more
   than that difference is supposed to matter. Not understood; do not "optimise"
   it back without re-validating against Whisper.
   Net cost is ~+49 % on the LM step, so roughly +20 % on a long render, and
   only when the LRC toggle is on. Still cheaper than a full Whisper pass.
5. The dump is **MM3ALIGN2**: an ASCII header line, then `tokens` × int32 lyric
   token ids, then f32 in `[frame][layer][head][token]` order. v1 omitted the
   ids, which is what forced the bogus token-progress-vs-word-progress
   comparison. Resolve ids to text via `tokenizer.ggml.tokens` in the LM GGUF.

**SHIPPED** (`features.lyricTimestamps: true`, request field `get_lrc`). The
engine emits LRC on `Job::result_lrc` → the existing `x-lrc-text` header → the
server writes `<uuid>.lrc`, same contract ACE uses. Measured against Whisper:
median line error **0.39 s** over the matched lines of two clips (synth
+0.20/+0.52/+0.62/−0.02, indie +2.12/−0.26/+1.36/+0.14).

`lrc_align()` is NOT used, and that was measured rather than assumed: fed the
same captured attention it stamps every line in the first half of the song, and
sweeping violence_level 2.0 → 0.0 changes nothing. A plain monotonic DTW over
the head-averaged matrix, grouped on newline tokens, is what works — see
`mm3-align.h`. `lrc_align()` stays untouched for ACE.

## Training: DiT yes, LM never (assessed 2026-08-14)

Full teardown + staged plan: `docs/plans/2026-08-14-mm3-training-feasibility.md` (local).

**MM3 ships decode-side only for the audio-token path — there is no audio → code encoder.**
Verified from the HF tensor inventory, not inferred: `qwen_7B/` (18.5 GB) is the **LM +
RVQ depth decoder in MiniMax's original `AbabForCausalLM` format** (`model.layers.*` +
`model.audio_decoder.*` with 7 audio heads, `embed_tokens [200000,4096]`), matching
`language_model/` + `rvq_depth_decoder/` to ~5 MB. It is **NOT a music tokenizer** — an earlier
note here said it was, which wrongly made LM training look possible. Consequences:

1. **LM / planner adapters are BLOCKED.** Code-SFT needs ground-truth codes per song; none
   exist and none can be derived. (ACE's LM training works only because FSQ encodes audio→codes.)
2. **DiT conditioning cannot be teacher-forced.** The only way to get `frame_hiddens` for a
   training clip is to AR-sample the LM from its caption/lyrics — so condition and target come
   from *different music*, matching only in caption/lyrics/duration. A timbre/production adapter
   can still work (low sigma is well-posed); the risk is the DiT learning to ignore
   `encoder_hidden_states`, i.e. prompt/lyric adherence. Measure it with the LRC probe above.

The encoder that IS needed and IS released: **`dav.pth` (492 MB) — the DAV encoder**, audio →
128-ch latents @ 86.13 Hz, L/R through shared weights, **mean only** (`logs_proj` unused). Our
local `comfy/vae/minimax_music3_dav.safetensors` is **decoder-only**, so it is a new download.

SimpleTuner ([#3074](https://github.com/bghira/SimpleTuner/pull/3074)/#3075, merged 2026-08-14)
trains the 2.4B flow DiT only, targets `to_q/to_k/to_v/to_out.0/ff_in/ff_out/proj_in/proj_out`,
plain rectified-flow loss, and defines the de-facto **ComfyUI MM3 LoRA format**. No published
evidence anyone has trained a *good* MM3 LoRA yet — treat it as a reference implementation, not
as validation. Two traps for loading those LoRAs: we **fuse q,k,v** into `dit.blk.N.attn_qkv`,
and our `mm3.dit.glu_order = value_gate` must be reconciled with their `swiglu_gate_first`
metadata. `convert-mm3.py:build_dit` already resolves both ComfyUI and diffusers names.

Recommended first step is **load, don't train**: add an `mm3` target to the adapter system,
train one LoRA in SimpleTuner, and hear whether it works before building a native trainer.

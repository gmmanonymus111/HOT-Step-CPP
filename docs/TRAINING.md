# Training System — Architecture & Continuation Guide

*The complete map of HOT-Step's native training system (Training Studio). Written as the handoff/continuation doc: everything an agent or contributor needs to keep working on this subsystem in a fresh session. Built 2026-07-27/28; all measurements from an RTX 5090 (32 GB, sm_120).*

## What it is

End-to-end adapter training **entirely in C++/GGML — no Python anywhere**:

```
Dataset creation → Tensor preprocessing → LM LoRA training (0.6B/1.7B/4B)
                                        → DiT LoRA training (LoKR planned)
                                        → Pure-LM audition (A/B, no DiT influence)
```

Side-Step (`D:\Ace-Step-Latest\Side-Step`, local) is the Python reference implementation this system reaches parity with; its dev sanctioned porting. Dataset/sidecar formats are byte-compatible with Side-Step; tensor caches deliberately are not (safetensors, not pickle).

## Component map

| Piece | Engine | Server | UI |
|---|---|---|---|
| Dataset creation | — (uses ace-server `/understand` optionally, legacy) | `routes/training.ts` + `services/training/{sidecarIO,datasetScan,labelStore,essentiaClient,enhanceService,captionPrompt,datasetBuilder,labelingQueue}.ts` | `training-studio/{DatasetList,NewDatasetWizard,SampleGrid,SampleDrawer,LabelPanel,EnhancePanel,BuildPanel}.tsx` |
| Preprocess | `ace-train preprocess` (`engine/src/train/preprocess-*.h`, `st-write.h`) | `preprocessRunner.ts`, `preprocessStatus.ts`, `aceTrain.ts` | `Preprocess{Panel,OptionsForm,VariantCard}.tsx` |
| LM training | `ace-train train-lm` (`lm-*.h`: graph/optim/ckpt/bf16/vram/data/extract/export/selftest/train-run) | `trainLmRunner.ts`, `trainLmStatus.ts` | `TrainPanel.tsx`, `TrainLmForm.tsx`, `TrainingChart.tsx`, `TrainingRunStats.tsx` |
| DiT training | `ace-train train-dit` (`dit-*.h`: same family) | `trainDitRunner.ts`, `trainDitStatus.ts` | `TrainDitForm.tsx` (+ shared chart/stats) |
| Audition | `POST /codes-decode` on **ace-server** (`hot-step-server.cpp`, ~174 lines) | `audition{Service,Runner,Store}.ts` | `AuditionCard.tsx`, `AuditionPlayer.tsx`, `LmAdapterPicker.tsx` |
| Engine lifecycle | — | `services/aceEngineProcess.ts` (stop/restart with epoch-guarded respawn cancellation) | engine-paused banners |

Frozen contracts (types, routes, JSONL event schemas, CLI) are duplicated verbatim between `server/src/services/training/types.ts` and `ui/src/services/trainingApi.ts` — **kept in sync by hand, deliberately**.

Job model: one global promise-chain queue (`labelingQueue.ts`), SSE streams with replayable capped buffers, `_meta.json` persistence, `TrainingMetricEvent` for training numbers. GPU jobs stop the ace-server child first (`stopEngine`, default on) and restart it in a `finally`.

## Key design facts (the ones that bite)

**Data & labeling**
- Sidecars (`<stem>.txt`) live **next to the audio**; `dataset.json` in the source folder — Side-Step-compatible on purpose. Studio-private state (audio_codes, provenance, raw analyzer results) lives in `server/data/training/datasets/<slug>/labels/`.
- Labeling is engine-free: Essentia (BPM/key, cached by size+mtime) ∥ Genius lyrics (relaxed collab matching — "Electric Callboy & BABYMETAL" broke exact primary-artist equality) → Gemini captions **with the audio attached** (96 kbps MP3 inline; the local caption is *omitted* from the prompt to avoid anchoring). `ace-understand` is out of the default flow (weak captions, hallucinated lyrics, J-pop prior); still reachable via API `useUnderstand:true`.
- Dataset language is **declared, not detected** (`default_language`, forced-write on every label pass).
- Lyrics *absence* never writes `is_instrumental` — only lyrics presence writes `false`.

**FSQ (critical correctness area)**
- `engine/src/fsq-quant.h` is the single source of truth. The reference path is **ResidualFSQ `preserve_symmetry`**: soft clamp `c=1+1/(L−1)` → `tanh(z/c)·c` → hard clamp → `floor((L−1)(w+1)/2+0.5)`. NOT `FSQ.bound` (that branch never executes in vqp). Encode verified 13046/13046 vs the checkpoint's own tokenizer.
- `verify-hooks.ps1` has 2 FSQ hooks — upstream syncs must not revert this.
- Side-Step's stored `lm_codes.jsonl` are a **bf16 ceiling** (~85% identity); engine-F32 vs fp32-reference is the honest gate (99.0%).

**LM trainer**
- Hand-rolled loop (`lm-optim.h`): ggml-opt can't grad-clip and its grad accessor crashes with dynamic graphs. Persistent grad accumulators indexed by forward-node order, **in-graph** global-norm clip at 1.0, AdamW nodes, Side-Step-exact LR schedule (`lr(0)=0`, 5% warmup, cosine→0.1).
- ggml gap catalogue: `out_prod` F32-only (the central constraint), no backward for flash-attn / `SET_ROWS` (KV cache) / fused swiglu; CE labels are dense one-hot; `CONCAT` has no backward (use `ACC`); rms_norm backward is wrong on **non-contiguous (permuted) inputs** — norm before permute.
- 4B fits ~12 GB via per-layer checkpoint segments + a transient per-layer F32 window + chunked-CE head (BF16 embedding). Naive path preserved byte-identical for 0.6B/1.7B.
- `--weights bf16` (experimental): backward surgery rewrites `out_prod(W_f32,gT)` → `mul_mat(cont(transpose(W_bf16)),g)`. **1.256× at 4B** measured; gradient cos ~0.9992 vs F32 (BF16 rounding compounds sub-linearly with depth; partly ggml-cuda's dst-BF16 rounding, only fixable by a vendor patch). Zero vendor patches shipped; `engine/ggml` submodule is clean upstream.
- `--batch` intentionally **not built** — measured amortisable overhead below the bar where it matters (4B: 9.3% < 10%); flag exits 2 citing the numbers.
- Defaults (Side-Step parity, from Rob's real runs): target loss **4.0** (not 0.4!), GA 2, epochs 75, milestone step 1.0, lr 1e-4.
- Self-test: `ace-train train-lm --self-test` — T1–T13 (+T14–T17 bf16, report-not-gate). T5/T4 finite differences run **TF32-off in a child process** (cuBLAS TF32 noise defeats central differences; gradients were always right). Known pre-existing 4B rough edges: T11e/T12 slightly over bars (BF16-dtype-resolution class).

**DiT trainer**
- Trainable graph is **bit-identical** to the inference forward (17 debug-tensor diff = 0.0). Unfused load via `g_dit_load_no_fuse` (mirrors `g_qwen3_load_no_fuse`; the zero-stub-adapter trick is dead).
- F32 mirror streamed from the GGUF via CPU-backend load (a GPU-copy mirror transiently costs +5–8 GB and OOMs cards the steady state fits).
- Crop-window training (random window/step). No flash-attn backward ⇒ O(S²·Nh) retained softmax ⇒ full-song full-depth ≈ 68 GB: **impossible**; crop ~1000–1250 frames max at 24 GB full depth; `--layers` top-K ladder for smaller cards; <16 GB refused.
- Product loss = flow_snr (window-normalized — batch-1 mean normalization is silently a no-op, the trap) + channel_balance from `channel_stats.json`; lr default **5e-4** (1e-4 measurably does not train).
- Defaults: epochs 400, r128/α256, genre ratio 30%, target-MLP **on** (needs `--no-target-mlp`-aware binary — see Pending). DiT target-loss 0.4 requires Side-Step-length runs; epoch cap is the practical stop (measured floor ~0.9 at 200 steps).
- Adapter output: PEFT dir, loads through **both** `adapter-merge.h` and runtime paths (round-trip reproduces trainer loss; that check is a permanent gate).

**Trigger words (embedded in the adapter)**
- The tag was always trained in — `preprocess-run.h:192-204` bakes `custom_tag` into the caption before text encoding, and `lm-extract.h` re-applies it — but export used to drop it, leaving inference to guess the trigger from the filename (wrong for our adapters: the dir is `<name>-<size>`, not the tag).
- Both trainers now write `hot_step_trigger`, `hot_step_trigger_position` and `modelspec.trigger_phrase` into the adapter's safetensors `__metadata__`. Tensors and `adapter_config.json` are untouched, so ComfyUI/PEFT/Side-Step load it exactly as before. **Not** `adapter_config.json` — PEFT does `LoraConfig(**json)` and unknown keys are a version-dependent TypeError.
- Source of the value: `--trigger`/`--trigger-position` flags, else the variant's `preprocess_meta.json` (`custom_tag`/`tag_position`). The **variant meta, not the dataset row**, is authoritative — a dataset's tag can be edited after preprocessing, in which case the tensors (and the adapter) carry the old one. `tag_position: replace` embeds nothing: that path never puts the tag in the caption at all.
- Generation-side resolution lives in `services/generation/triggerWords.ts` and runs **server-side** for every caller: per adapter, manual override → embedded → filename fallback → none. A stack can mix prepend and append. The gate widened from "a DiT adapter is loaded" to "any adapter", so planner-LM adapters now contribute their trigger too.
- Legacy corpus: `server/scripts/stamp-adapter-triggers.mjs`, dry-run by default. Verified on a copy — 132 MB payload SHA-256 identical, header stays 8-byte aligned, `.bak` kept.

**Adapter layout (per-base + per-run, 2026-07-28)**
- `<adapters>/lm-06b|lm-17b|lm-4b/<artist>/<run>/` and `<adapters>/dit-<shorthand>/<artist>/<run>/`, where `<run>` = `YYYY-MM-DD_HH-MM-SS` (logs/ convention) — retraining an artist never overwrites an earlier adapter. Artist names carry **no** `-<size>` suffix; the parent folder says the base.
- Single source of truth: `server/src/services/training/adapterLayout.ts` (size slugs, the confirmed DiT shorthand map `xl-thirds`/`xl-base-turbo`/`xl-sft-turbo`/…, run stamps, latest-run resolution). `migrate-adapter-layout.mjs` moves an old corpus; its shorthand map must stay in sync.
- Writes go through `lmRunDirFor`/`ditRunDirFor` (fresh stamped dir); reads through `adapterDirFor`/`adapterDitDirFor` (newest run → unversioned artist dir → legacy flat `lm/<name>-4B` / root DiT dir). Two legacy forms are read-everywhere, written-never.
- Scanners: `GET /api/adapters/lm` walks all lm-* roots + legacy `lm/` (entries carry `lmSize`, `run`, `trigger`) — this is what the global-bar planner list, Lyric Studio's preset picker and the Training Studio picker all consume. `POST /api/adapters/scan` descends into `dit-*` folders (and their run subdirs) only.
- The audition's base-LM pinning now derives the size from the `lm-<size>` parent folder (suffix kept as legacy fallback) — do not reintroduce suffix-only derivation.

**Audition (pure-LM preview)**
- `POST /codes-decode`: codes → `detok_ggml_decode` → tail-call the existing VAE decode worker (layouts are byte-identical; zero changes to `/vae`). Deterministic; full song ≈ 3.1 s warm.
- A/B = two `/lm` calls, same explicit `lm_seed`, base side **must** byte-match an adapterless run (V5 hard gate). Base LM auto-derived from the adapter's `-<size>` suffix (`pickLmFor`) — never let the engine's `resolve_name` fallback pick (sticky 0.6B met a 4B adapter: "36 layers but model has 28").
- LM-echo sideband trap: never forward the `/lm` reply into another request — build requests fresh (see `generation-request-flow` skill).

## Build & verification rules

- `ace-train`-only changes: `cmake --build engine/build --config Release --target ace-train` — safe with the app running **if no ace-train.exe process is live** (training runs spawn it). New `train/*.h` headers need **no CMake change**.
- `hot-step-server.cpp` (ace-server) changes: full `dev-rebuild.bat` cycle (app goes down; Rob restarts with `dev.bat`). Never `build.cmd` directly, never `--clean-first`.
- Self-tests are the regression net: `ace-train train-lm --self-test`, `ace-train train-dit --self-test`, `ace-train spike …` (gemmbench, bf16layer, dit2 — the measurement harnesses that justified every design call; keep them).
- GPU discipline for agents: `nvidia-smi` before runs, stay under ~29 GB total, never kill Rob's python/node/ace-server, bounded runs while he's working.
- TypeScript: `server` `npx tsc --noEmit`; `ui` `npx tsc -p tsconfig.app.json --noEmit` (one pre-existing error in `globalParamsStore.ts:449` — `lmAdapter` vs `lmAdapters` in `types.ts:347` — is NOT ours).

## Measured reference numbers

| Thing | Number |
|---|---|
| LM 0.6B full E2E (17 songs, 16 epochs) | ~70 s wall |
| LM 4B, checkpointed | ~12.4 GB trainer-owned, ~1.06–1.34 s/micro-step; a real run auto-stops ~epoch 29 at target 4.0 ≈ 12 min (bf16: ~9.5 min) |
| DiT full-depth crop 750 | ~147 ms/step, ~23 GB; crop ceiling ~1000–1250 frames @24 GB |
| Preprocess | ~6 GB VRAM (vae-chunk 384; 1024 costs ~17 GB), ~3 s/song |
| Codes decode (audition) | ~200 ms warm 30 s clip; ~3.1 s full song |
| TF32 vs BF16 GEMM (5090) | ~95 vs ~165 TFLOP/s; end-to-end lever 1.256× at 4B |

## Pending / open decisions (Rob's list)

1. ~~One `dev-rebuild` owed~~ — **done**: the 2026-07-28 17:05 rebuild (during the SuperSep work) postdates every one of those source edits; `ace-train train-dit --help` shows `--no-target-mlp` and `ace-server.exe` carries the audition fixes.
1b. **Trigger stamper awaits Rob's approval**: `node server/scripts/stamp-adapter-triggers.mjs --datasets D:\Ace-Step-Latest\Datasets-LoRA-LoKR` prints the proposed table for 179 adapters (28 from real dataset.json ground truth, 151 from the filename). Nothing is written without `--apply`. Also outstanding: the per-adapter **manual trigger override** UI — the server already accepts `triggerSpecs` entries with `source:'override'` and a `path`, but no control emits them yet.
2. **bf16 listen test**: train twin LM adapters (f32-window vs `--weights bf16`), A/B via audition; ship-call by ear.
3. **LoKR for DiT** (`dit-adapter.h` has the parameterization seam ready) — Rob's preferred adapter type.
4. **ConvRot bases**: refused by the trainer pending a convrot spike (`…-convrot-*` GGUFs exist).
5. **BF16 `out_prod` vendor patch**: would ~2.5× DiT crop @24 GB, unlock 12 GB cards, and remove the bf16-lever's dst-rounding error — requires forking the ggml submodule (CI re-inits from upstream; see the rejected-P1 analysis in the levers plan).
6. **Top-K DiT adapter quality** on small cards: runs, but musical usefulness unmeasured.
7. Micro-batching: closed with measurements; revisit only if the graph-build overhead picture changes.
8. UI `minVram` hint string may be optimistic at r128+MLP defaults.

## Where the deep documents live

`docs/plans/` is **gitignored (local-only)** — on Rob's machine the full design/implementation plans exist: `2026-07-27-training-studio-design.md` (master design), `-dataset-studio-implementation.md`, `-preprocess-implementation.md`, `-lm-trainer-implementation.md`, `2026-07-28-lm-4b-training.md`, `-dit-trainer-implementation.md`, `-lm-speed-levers.md`, `-codes-preview.md`. Each contains the frozen contracts and verification ladders. The commit history (`git log --oneline --grep=training`) carries the measured numbers per milestone.

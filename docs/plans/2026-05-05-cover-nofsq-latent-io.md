# Cover-NoFSQ Toggle & Latent Import/Export Implementation Plan

> **For Antigravity:** REQUIRED SUB-SKILL: Load executing-plans to implement this plan task-by-task.

**Goal:** Expose the `cover-nofsq` mode in Cover Studio UI and implement full latent import/export across the C++, Node.js, and React layers.

**Architecture:** Two features, five tasks. Feature A (cover-nofsq toggle) is UI-only — the engine already supports it. Feature B (latent I/O) threads through every layer: C++ `synth_batch_run` captures latents, the C++ server stores them in the Job struct and serves them via `GET /job?latent=1`, the Node.js generation route wraps raw latents in a custom **HSLAT** (HOT-Step LATent) file format with JSON metadata, the DB gains a `latent_url` column, and the UI gains download + import controls that auto-populate from embedded metadata.

**HSLAT File Format:**
```
┌──────────────────────────────────────────────────┐
│ Magic: "HSLAT\x01"              (6 bytes)        │ ← identifies file + format version
│ JSON length: uint32_le           (4 bytes)        │ ← byte count of metadata string
│ JSON metadata: UTF-8 string      (variable)       │ ← BPM, key, lyrics, seed, etc.
│ Padding: zeros to 256-byte boundary              │ ← aligns latent data for GPU
│ Latent data: float32[T × 64]    (T×256 bytes)   │ ← raw post-DiT tensor
└──────────────────────────────────────────────────┘
```
The C++ engine only ever sees raw float32 — all HSLAT header creation/parsing happens in Node.js via a new `latentFormat.ts` service. On import, files without the `HSLAT` magic are treated as raw float32 (upstream-compatible).

**Tech Stack:** C++ (hot-step-server.cpp, synth-batch-runner.h), TypeScript/Node.js (aceClient.ts, generate.ts, download.ts, database.ts), React/TSX (CoverStudio.tsx, DownloadModal.tsx)

---

## Feature A: Cover-NoFSQ Toggle

### Task 1: Add NoFSQ toggle to Cover Studio UI

**Files:**
- Modify: `ui/src/components/cover-studio/CoverStudio.tsx`

**Context:** The engine already supports `task_type: "cover-nofsq"`. The server route at [generate.ts:283](file:///D:/Ace-Step-Latest/hot-step-cpp/server/src/routes/generate.ts#L283) already includes `cover-nofsq` in the `isCoverTask` list. All that's needed is a UI toggle that switches the `taskType` field in the generation params.

**Step 1: Add state and persistence**

Add a `noFsq` boolean state with persistence, right after `coverNoiseStrength` (line ~58):

```tsx
const [noFsq, setNoFsq] = useState(() => restore<boolean>('noFsq', false));
```

Add the persist effect alongside the others (after line ~124):

```tsx
useEffect(() => { persist('noFsq', noFsq); }, [noFsq]);
```

**Step 2: Use noFsq in generation params**

In `handleGenerate`, change line 328 from:

```tsx
taskType: 'cover',
```

to:

```tsx
taskType: noFsq ? 'cover-nofsq' : 'cover',
```

**Step 3: Add UI toggle to ArtistSettingsPanel**

Pass `noFsq` and `setNoFsq` as props to `ArtistSettingsPanel`. Add a small toggle switch in the cover settings section (near the Audio Cover Strength slider). The toggle label should be "NoFSQ Mode" with a tooltip: "Skip FSQ quantization for more faithful covers that stay closer to the source structure."

Style: Use a compact pill-toggle matching the existing `advancedMode` toggle pattern from `SourcePanel`.

**Step 4: Commit**

```powershell
cd D:\Ace-Step-Latest\hot-step-cpp
git add ui/src/components/cover-studio/
git commit -m "feat(cover-studio): add cover-nofsq toggle"
```

---

## Feature B: Latent Import/Export

### Task 2: C++ Engine — Capture latents in synth_batch_run and serve via Job

**Files:**
- Modify: `engine/tools/synth-batch-runner.h`
- Modify: `engine/tools/hot-step-server.cpp`

**Context:** Upstream's `synth_batch_run` has a `latents_out` parameter; our version has `lrc_out` instead. We need both. The `ace_synth_job_get_latent(job, track_idx, &T)` API is already available — it returns a pointer to the post-DiT latent tensor `[T, 64]` that lives until `ace_synth_job_free()`.

#### Step 1: Add `latents_out` parameter to `synth_batch_run`

Current signature at [synth-batch-runner.h:28-37](file:///D:/Ace-Step-Latest/hot-step-cpp/engine/tools/synth-batch-runner.h#L28-L37):

```cpp
static int synth_batch_run(AceSynth * ctx,
                           std::vector<std::vector<AceRequest>> & groups,
                           const float * src_audio, int src_len,
                           const float * ref_audio, int ref_len,
                           AceAudio * audio_out,
                           std::string * lrc_out = nullptr,
                           bool (*cancel)(void *) = nullptr,
                           void * cancel_data = nullptr);
```

New signature — add `src_latents`, `ref_latents`, and `latents_out`:

```cpp
static int synth_batch_run(AceSynth * ctx,
                           std::vector<std::vector<AceRequest>> & groups,
                           const float * src_audio, int src_len,
                           const float * src_latents, int src_T_latent,
                           const float * ref_audio, int ref_len,
                           const float * ref_latents, int ref_T_latent,
                           AceAudio * audio_out,
                           std::string * lrc_out = nullptr,
                           std::vector<std::vector<float>> * latents_out = nullptr,
                           bool (*cancel)(void *) = nullptr,
                           void * cancel_data = nullptr);
```

Update the body:
1. Pass `src_latents, src_T_latent, ref_latents, ref_T_latent` through to `ace_synth_job_run_dit()` instead of `nullptr, 0`.
2. After Phase 1 (DiT), before Phase 2 (VAE), capture latents if `latents_out != nullptr`:

```cpp
if (latents_out) {
    latents_out->resize((size_t) off);
    for (int g = 0; g < n_groups; g++) {
        const int gn = (int) groups[g].size();
        for (int i = 0; i < gn; i++) {
            int T = 0;
            const float * src = ace_synth_job_get_latent(jobs[g], i, &T);
            (*latents_out)[audio_off[g] + i].assign(src, src + (size_t) T * 64);
        }
    }
}
```

#### Step 2: Update hot-step-server.cpp `synth_worker` call site

Current call at [hot-step-server.cpp:973-974](file:///D:/Ace-Step-Latest/hot-step-cpp/engine/tools/hot-step-server.cpp#L973-L974):

```cpp
const int rc = synth_batch_run(ctx, groups, src_interleaved, src_len,
                               ref_interleaved, ref_len, audio.data(),
                               lrc_results.data(), server_cancel_job, (void *) &job->cancel);
```

Update to:

```cpp
std::vector<std::vector<float>> captured_latents;
const int rc = synth_batch_run(ctx, groups,
                               src_interleaved, src_len,
                               src_lat_ptr, src_T_latent,
                               ref_interleaved, ref_len,
                               nullptr, 0,  // ref_latents (not yet exposed)
                               audio.data(),
                               lrc_results.data(),
                               &captured_latents,
                               server_cancel_job, (void *) &job->cancel);
```

Where `src_lat_ptr` and `src_T_latent` are parsed from multipart (see Step 3).

After the synth completes, store the first track's latent in the job:

```cpp
if (!captured_latents.empty() && !captured_latents[0].empty()) {
    job->result_latent = std::move(captured_latents[0]);
}
```

#### Step 3: Add `result_latent` field to Job struct

At [hot-step-server.cpp:235-247](file:///D:/Ace-Step-Latest/hot-step-cpp/engine/tools/hot-step-server.cpp#L235-L247), add:

```cpp
std::vector<float> result_latent;  // post-DiT latent [T*64], empty if not captured
```

#### Step 4: Serve latent via `GET /job?id=N&latent=1`

In the job GET handler at [hot-step-server.cpp:1697-1724](file:///D:/Ace-Step-Latest/hot-step-cpp/engine/tools/hot-step-server.cpp#L1697-L1724), add a new branch before the `?result=1` check:

```cpp
// ?latent=1: return raw latent bytes (float32, [T*64])
if (req.has_param("latent") && req.get_param_value("latent") == "1") {
    if (job->status.load() != 1 || job->result_latent.empty()) {
        json_error(res, 404, "Latent not available");
        return;
    }
    res.set_content(
        reinterpret_cast<const char *>(job->result_latent.data()),
        job->result_latent.size() * sizeof(float),
        "application/octet-stream");
    return;
}
```

#### Step 5: Parse `src_latents` from multipart in `handle_synth`

In the multipart parsing section at [hot-step-server.cpp:1108-1161](file:///D:/Ace-Step-Latest/hot-step-cpp/engine/tools/hot-step-server.cpp#L1108-L1161), after the `ref_audio` parsing, add:

```cpp
float * src_lat_buf = nullptr;
int src_T_latent = 0;

if (req.form.has_file("src_latents")) {
    auto file = req.form.get_file("src_latents");
    if (!file.content.empty()) {
        if (file.content.size() % (64 * sizeof(float)) != 0) {
            json_error(res, 400, "src_latents size must be a multiple of 256 bytes (64 * float32)");
            return;
        }
        src_T_latent = (int)(file.content.size() / (64 * sizeof(float)));
        src_lat_buf = (float *) malloc(file.content.size());
        memcpy(src_lat_buf, file.content.data(), file.content.size());
        fprintf(stderr, "[Server] Source latents: T=%d (%.2fs @ 25Hz)\n",
                src_T_latent, (float)src_T_latent / 25.0f);
    }
}
```

And update `synth_worker` signature + lambda capture to pass `src_lat_buf` and `src_T_latent`. Free `src_lat_buf` in the worker alongside `src_interleaved`.

#### Step 6: Update `synth_worker` signature

Add `float * src_latents, int src_T_latent` parameters to `synth_worker`, and ensure `free(src_latents)` is called on all exit paths.

#### Step 7: Build and test

```powershell
cd D:\Ace-Step-Latest\hot-step-cpp
cmd /c dev-rebuild.bat
```

> [!IMPORTANT]
> This is the only task that requires an engine rebuild. All subsequent tasks are Node.js/UI only.

#### Step 8: Commit

```powershell
git add engine/tools/synth-batch-runner.h engine/tools/hot-step-server.cpp
git commit -m "feat(engine): capture post-DiT latents, serve via /job?latent=1, accept src_latents input"
```

---

### Task 3: Node.js — Fetch and save latents, add DB column

**Files:**
- Modify: `server/src/services/aceClient.ts`
- Modify: `server/src/routes/generate.ts`
- Modify: `server/src/routes/download.ts`
- Modify: `server/src/db/database.ts`

#### Step 1: Add `getJobLatent()` to aceClient

After `getJobResult` in [aceClient.ts:280-284](file:///D:/Ace-Step-Latest/hot-step-cpp/server/src/services/aceClient.ts#L280-L284), add:

```typescript
/** GET /job?id=N&latent=1 — fetch captured post-DiT latent (raw float32 bytes).
 *  Returns null if no latent was captured. */
async getJobLatent(jobId: string): Promise<Buffer | null> {
  try {
    const res = await fetch(`${BASE}/job?id=${jobId}&latent=1`, {
      signal: AbortSignal.timeout(TIMEOUT_RESULT),
    });
    if (!res.ok) return null;
    const buf = await res.arrayBuffer();
    return buf.byteLength > 0 ? Buffer.from(buf) : null;
  } catch {
    return null;
  }
},
```

#### Step 2: Add `latent_url` column migration in database.ts

In the `songsMigrations` array at [database.ts:167-172](file:///D:/Ace-Step-Latest/hot-step-cpp/server/src/db/database.ts#L167-L172), add:

```typescript
{
  check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='latent_url'`,
  alter: `ALTER TABLE songs ADD COLUMN latent_url TEXT DEFAULT ''`,
},
```

#### Step 3: Fetch and save latent in generate.ts

After saving the audio file at [generate.ts:686-688](file:///D:/Ace-Step-Latest/hot-step-cpp/server/src/routes/generate.ts#L686-L688), add latent capture:

```typescript
// Fetch and save companion latent file (post-DiT representation)
let latentUrl = '';
try {
  const latentBuf = await aceClient.getJobLatent(synthJobId);
  if (latentBuf && latentBuf.length > 0) {
    const latentFilename = filename.replace(/\.[^.]+$/, '.latent');
    const latentPath = path.join(config.data.audioDir, latentFilename);
    fs.writeFileSync(latentPath, latentBuf);
    latentUrl = `/audio/${latentFilename}`;
    logGeneration(job.id, 'INFO',
      `[Latent] Track ${trackIdx + 1}: saved ${latentFilename} (${(latentBuf.length / 1024).toFixed(0)} KB, T=${latentBuf.length / 256} frames)`);
  }
} catch (latErr: any) {
  logGeneration(job.id, 'DEBUG', `[Latent] Track ${trackIdx + 1}: capture skipped: ${latErr.message}`);
}
```

Store the first track's `latentUrl` for use in the DB insert.

#### Step 4: Update DB INSERT to include `latent_url`

At [generate.ts:874-884](file:///D:/Ace-Step-Latest/hot-step-cpp/server/src/routes/generate.ts#L874-L884), add `latent_url` to the INSERT statement:

```typescript
getDb().prepare(`
  INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                     duration, bpm, key_scale, time_signature, tags, dit_model,
                     generation_params, mastered_audio_url, latent_url)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
`).run(
  songId, job.userId, title, trackLyrics, style, trackCaption,
  audioUrl, duration, bpm, keyScale, timeSignature,
  JSON.stringify([]), aceReq.synth_model || '', JSON.stringify(job.params),
  trackMastered, latentUrl,
);
```

#### Step 5: Add latent download to download.ts

In the download route at [download.ts:86](file:///D:/Ace-Step-Latest/hot-step-cpp/server/src/routes/download.ts#L86), add a `version=latent` branch early in the handler:

```typescript
// Latent download — raw binary, no conversion
if (version === 'latent') {
  const latentUrl = song?.latent_url;
  if (!latentUrl) {
    res.status(404).json({ error: 'No latent file available for this track' });
    return;
  }
  const latentFilename = path.basename(latentUrl);
  const latentPath = path.join(config.data.audioDir, latentFilename);
  if (!fs.existsSync(latentPath)) {
    res.status(404).json({ error: 'Latent file not found on disk' });
    return;
  }
  const downloadName = `${songTitle || 'track'}.latent`;
  res.setHeader('Content-Type', 'application/octet-stream');
  res.setHeader('Content-Disposition', `attachment; filename="${downloadName}"`);
  res.setHeader('Content-Length', fs.statSync(latentPath).size);
  fs.createReadStream(latentPath).pipe(res);
  return;
}
```

#### Step 6: Add latent upload route

Add `/api/upload/latent` in the upload route (or create inline in generate.ts). The latent file is a raw binary blob with no header — just validate that `size % 256 === 0` (each frame is 64 floats × 4 bytes = 256 bytes).

#### Step 7: Commit

```powershell
git add server/src/
git commit -m "feat(server): latent capture, save, download, and upload endpoints"
```

---

### Task 4: Node.js — aceClient multipart with src_latents

**Files:**
- Modify: `server/src/services/aceClient.ts`
- Modify: `server/src/routes/generate.ts`

#### Step 1: Extend `submitSynthMultipart` to accept latent buffer

Add an optional `srcLatents?: Buffer` parameter to [aceClient.ts:186-244](file:///D:/Ace-Step-Latest/hot-step-cpp/server/src/services/aceClient.ts#L186-L244):

```typescript
async submitSynthMultipart(
  request: AceRequest | AceRequest[],
  srcAudio?: Buffer,
  refAudio?: Buffer,
  srcLatents?: Buffer,         // <-- NEW
  format: string = 'wav16',
  keepLoaded = false,
): Promise<string> {
```

After the `ref_audio` part addition, add:

```typescript
// Source latents (raw float32, replaces VAE encode of source audio)
if (srcLatents) {
  addPart('src_latents', srcLatents, 'application/octet-stream', 'source.latent');
}
```

#### Step 2: Wire latent import through generate.ts

In the cover task source audio section at [generate.ts:480-509](file:///D:/Ace-Step-Latest/hot-step-cpp/server/src/routes/generate.ts#L480-L509), add latent loading:

```typescript
// Source latent (alternative to source audio — skips VAE encode)
let srcLatentBuf: Buffer | undefined;
if (isCoverTask && job.params.sourceLatentUrl) {
  const latentUrl = job.params.sourceLatentUrl;
  const latentPath = latentUrl.startsWith('/audio/')
    ? path.join(config.data.audioDir, latentUrl.replace('/audio/', ''))
    : path.isAbsolute(latentUrl) ? latentUrl : path.join(config.data.dir, latentUrl);
  if (fs.existsSync(latentPath)) {
    srcLatentBuf = fs.readFileSync(latentPath);
    // Validate: must be a multiple of 256 bytes (64 floats per frame)
    if (srcLatentBuf.length % 256 !== 0) {
      logGeneration(job.id, 'WARNING', `[Latent] Invalid latent file size (${srcLatentBuf.length} bytes), ignoring`);
      srcLatentBuf = undefined;
    } else {
      logGeneration(job.id, 'INFO',
        `[Latent] Source latent loaded: ${latentPath} (${srcLatentBuf.length / 256} frames, ${(srcLatentBuf.length / 256 / 25).toFixed(1)}s)`);
    }
  }
}
```

Update the `submitSynthMultipart` call to pass `srcLatentBuf`:

```typescript
synthJobId = await aceClient.submitSynthMultipart(
  synthReq, srcAudioBuf, refAudioBuf, srcLatentBuf, synthFormat, coResident
);
```

#### Step 3: Commit

```powershell
git add server/src/services/aceClient.ts server/src/routes/generate.ts
git commit -m "feat(server): wire latent import through synth multipart pipeline"
```

---

### Task 5: UI — Latent download in DownloadModal + import in Cover Studio

**Files:**
- Modify: `ui/src/components/shared/DownloadModal.tsx`
- Modify: `ui/src/components/cover-studio/CoverStudio.tsx`
- Modify: `ui/src/components/cover-studio/SourcePanel.tsx`

#### Step 1: Add "Latent" version option to DownloadModal

In [DownloadModal.tsx](file:///D:/Ace-Step-Latest/hot-step-cpp/ui/src/components/shared/DownloadModal.tsx), extend the `DownloadVersion` type:

```typescript
type DownloadVersion = 'original' | 'mastered' | 'both' | 'latent';
```

Add a fourth radio/button option labeled "💾 Latent (raw neural)" with description "Raw post-DiT representation. Can be re-imported for faster cover generation."

The latent download uses `version=latent` on the existing download endpoint.

#### Step 2: Add latent import to CoverStudio

Add state:
```tsx
const [sourceLatentUrl, setSourceLatentUrl] = useState('');
```

In `handleGenerate`, if `sourceLatentUrl` is set, pass it in the params:
```tsx
if (sourceLatentUrl) params.sourceLatentUrl = sourceLatentUrl;
```

#### Step 3: Add latent import UI to SourcePanel

Add a small "Import Latent" dropzone/button that accepts `.latent` files. When a file is selected:
1. Upload to `/api/upload/latent`
2. Set `sourceLatentUrl` to the returned URL
3. Show a badge indicating "Latent source loaded" instead of audio waveform

The latent replaces the need for source audio encoding (VAE pass is skipped). The user still needs to provide lyrics and other metadata.

#### Step 4: Commit

```powershell
git add ui/src/components/
git commit -m "feat(ui): latent download in DownloadModal, latent import in Cover Studio"
```

---

## Verification Plan

### Automated Tests
1. **Engine build**: `cmd /c dev-rebuild.bat` — must compile without errors
2. **Server start**: `npm run dev` in `server/` — must start without DB migration errors
3. **Latent size validation**: Generate a track, verify `.latent` file exists alongside `.wav` and that `size % 256 === 0`

### Manual Verification
1. **Cover-nofsq**: Toggle on, generate a cover → verify engine log shows `cover-nofsq` task type
2. **Latent export**: Generate any track → download latent from DownloadModal → verify file is valid float32 binary
3. **Latent import**: Upload exported latent as source in Cover Studio → generate cover → verify engine log shows `src_latents: T=...` (VAE encode skipped)
4. **Backward compat**: Generate a normal (non-cover) track → verify no regression in audio output or response handling

# User Isolation Audit Summary — HOT-Step CPP

## Completed
- ✅ Songs/library (user_id scoped)
- ✅ Generation queue (user_id scoped, ownership checks on cancel/reset)
- ✅ Song Builder (user_id scoped)
- ✅ Seeds, profiles, settings, cover art (user_id scoped)
- ✅ Mastering references (user_id scoped)
- ✅ Late library update for songs (auth-logged-in event refresh)

## Remaining (in priority order)

### 1. Lyric Studio (Lireek) — CRITICAL
- **Risk:** All users share artists, albums, profiles, generations, presets.
- **Fix:** Add user_id to all Lireek tables, scope all queries.
- **Session file:** `SESSION-TASK-lyric-studio-isolation.md`

### 2. Training Studio — CRITICAL
- **Risk:** Shared datasets, training jobs, adapters, GPU hours.
- **Fix:** Add user_id to tables, prefix disk paths with user_id, scope adapter scans.
- **Queue fix:** Job cancellation ownership check.
- **Library fix:** UI refresh on login for training data.
- **Session file:** `SESSION-TASK-training-studio-isolation.md`

### 3. Stem Studio + SuperSep — HIGH
- **Risk:** Shared stem jobs, results, cancellation.
- **Fix:** Add userId to jobs, prefix disk paths, ownership checks.
- **Queue fix:** Cancel/delete ownership check.
- **Library fix:** UI refresh on login for stem jobs.
- **Session file:** `SESSION-TASK-stem-studio-isolation.md`

### 4. MIDI Studio — MEDIUM
- **Risk:** Shared transcription jobs, results, HF token.
- **Fix:** Add userId to jobs, prefix disk paths, per-user HF token.
- **Queue fix:** Cancel/delete ownership check.
- **Library fix:** UI refresh on login for MIDI jobs.
- **Session file:** `SESSION-TASK-midi-studio-isolation.md`

### 5. Cover Studio / Analyze — LOW
- **Risk:** Likely already scoped via song_id, but needs verification.
- **Fix:** Audit cache keying, fix if shared.
- **Session file:** `SESSION-TASK-cover-studio-isolation.md`

## Common Patterns Across All Fixes

### Database
- Add `user_id TEXT NOT NULL` to all relevant tables.
- Migrate existing rows to admin user's ID.
- Add indexes on user_id.

### Routes
- Extract userId: `const userId = req.user!.userId;`
- SELECT: `WHERE user_id = ?`
- INSERT: include user_id
- UPDATE/DELETE: `WHERE user_id = ?`

### Queue Access
- Cancel/delete: verify `job.userId === req.user.userId || req.user.role === 'admin'`
- Return 403 if not authorized.

### Disk Paths
- Prefix with user_id: `data/<feature>/<userId>/<jobId>/`

### Late Library Update
- Listen for `auth-logged-in` event in UI.
- Re-fetch feature data on login.
- Invalidate cached state on auth transitions.

## Process
1. Open a new session for each task (to avoid context overflow).
2. Paste the corresponding SESSION-TASK file as the initial prompt.
3. Implement, test, commit.
4. Mark complete in this file.
5. Move to next task.

## Notes
- No auto-build. Use GitHub Actions for build verification.
- All changes experimental until tested.
- Single RTX 3090 for testing (no multi-GPU).
- Alex is the contributor (not owner Rob/scragnog).
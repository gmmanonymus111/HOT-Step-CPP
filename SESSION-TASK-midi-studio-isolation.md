# Session Task: User-Isolate MIDI Studio + Fix Queue/Library Issues

## Context
HOT-Step CPP has multi-user auth. MIDI Studio (audio→MIDI transcription) is NOT scoped per user — all users share jobs, results, and can cancel each other's transcriptions. No late-library-update fix for MIDI results either.

## Goal
1. Scope all MIDI Studio jobs/results per user.
2. Fix queue access: only job owner or admin can cancel/delete.
3. Fix late library update: MIDI results appear immediately after login.

## Current State
- Routes: `server/src/routes/midiStudio.ts`
- Protected by `requireAuth` at app level, but NO user_id filtering.
- Job storage:
  - In-memory: `jobs = new Map<string, MidiJob>()`
  - Disk: `data/midi/<jobId>/` for results (.mid files, events)
- MidiJob has `songId?` but no `userId`.
- No ownership checks on cancel/delete.
- HF token storage: check if per-user or global (should be per-user).

## Plan

### 1. Job object enrichment
- Add `userId: string` to MidiJob interface.
- On job creation: `userId = req.user!.userId`.

### 2. Disk path scoping
- Change from `data/midi/<jobId>/` to `data/midi/<userId>/<jobId>/`.
- Update jobDir() and all read/write paths.
- Migration: existing MIDI jobs → admin user's namespace.

### 3. Route-level enforcement
- List jobs (`GET /jobs`): filter by `req.user.userId`.
- Get job status/progress: verify `job.userId === req.user.userId || admin`.
- Cancel job: verify ownership → 403 if not owner/admin.
- Delete job: verify ownership → 403 if not owner/admin.
- Download .mid: verify ownership.
- SSE stream: verify ownership.

### 4. Queue access fix
- Pattern (same as generation queue):
  ```ts
  const caller = req.user;
  if (!caller || (caller.userId !== job.userId && caller.role !== 'admin')) {
    res.status(403).json({ error: 'Forbidden' });
    return;
  }
  ```
- Apply to: cancel, delete, any job-modifying endpoint.

### 5. HF token scoping
- Check `server/src/services/muscriptor.ts` for HF token storage.
- If global, make per-user: store in users table or separate mapping.
- Each user manages their own HF token for gated model downloads.

### 6. Late library update fix
MIDI results don't go through song-created events. Fix:
- UI listens for MIDI job completion via SSE (already exists in MidiStudio).
- On login (auth-logged-in event), re-fetch MIDI job history.
- If UI caches MIDI jobs, invalidate on auth transitions.
- Check UI components: `ui/src/components/midi-studio/` for state management.

### 7. Job persistence (optional)
Current in-memory Map loses jobs on restart. Consider:
- Store job metadata in SQLite with user_id.
- On restart, reload active jobs from DB.
- Out of scope if too complex — note as future task.

### 8. Files to modify
- `server/src/routes/midiStudio.ts`
- `server/src/services/muscriptor.ts` (HF token scoping)
- UI: `ui/src/components/midi-studio/` (verify login refresh)
- Possibly `server/src/db/database.ts` if adding job persistence

### 9. Testing checklist
- Two users: each starts transcriptions — verify they only see their own.
- Verify cancel/delete checks ownership (403 for non-owner).
- Verify disk paths are user-prefixed.
- Verify HF tokens are per-user.
- Verify login refresh shows correct jobs/results.
- Run `npx tsc --noEmit` in server/ and ui/ — must pass.

## Constraints
- No auto-build. Use GitHub Actions.
- All changes experimental until tested.
- Single RTX 3090 for testing.
- Commit after verification.

## Deliverable
- MidiJob includes userId.
- All jobs/results scoped by user_id.
- Cancel/delete checks ownership (403 for non-owner/non-admin).
- Disk paths user-prefixed.
- HF tokens per-user.
- UI refreshes on login.
- TypeScript compiles cleanly.
- Commit with descriptive message.
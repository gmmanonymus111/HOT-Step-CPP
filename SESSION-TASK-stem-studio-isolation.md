# Session Task: User-Isolate Stem Studio + SuperSep + Fix Queue/Library Issues

## Context
HOT-Step CPP has multi-user auth. Stem Studio (generative extraction) and SuperSep (neural separation) are NOT scoped per user — all users share jobs, results, and can cancel each other's work. No late-library-update fix for stem results either.

## Goal
1. Scope all Stem Studio and SuperSep jobs/results per user.
2. Fix queue access: only job owner or admin can cancel/delete.
3. Fix late library update: stem results appear immediately after login.

## Current State
- Stem Studio routes: `server/src/routes/stemStudio.ts`
- SuperSep routes: `server/src/routes/supersep.ts`
- Protected by `requireAuth` at app level, but NO user_id filtering.
- Job storage:
  - In-memory: `jobs = new Map<string, StemJob>()` (lost on restart)
  - Disk: `data/stems/<jobId>/` for results
- No user_id in job objects or paths.
- No ownership checks on cancel/delete.

## Plan

### 1. Job object enrichment
- Add `userId: string` to StemJob interface.
- On job creation: `userId = req.user!.userId`.
- Same for SuperSep jobs (check if separate job type or shared).

### 2. Disk path scoping
- Change from `data/stems/<jobId>/` to `data/stems/<userId>/<jobId>/`.
- Update all read/write paths.
- Migration: existing stems → admin user's namespace (or leave flat but tag metadata).

### 3. Route-level enforcement
- List jobs: filter by `req.user.userId` (or show all if admin).
- Get job status: verify `job.userId === req.user.userId || admin`.
- Cancel job: verify ownership → 403 if not owner/admin.
- Delete job: verify ownership → 403 if not owner/admin.
- Download stems: verify ownership.

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

### 5. Late library update fix
Stem results don't go through song-created events. Fix:
- UI listens for stem job completion via SSE (already exists?).
- On login (auth-logged-in event), re-fetch stem job history.
- If UI caches stem jobs, invalidate on auth transitions.
- Check UI components: `ui/src/components/stem-studio/` for state management.

### 6. Job persistence (optional but recommended)
Current in-memory Map loses jobs on restart. Consider:
- Store job metadata in SQLite with user_id.
- On restart, reload active jobs from DB.
- Out of scope for this session if too complex — note as future task.

### 7. Files to modify
- `server/src/routes/stemStudio.ts`
- `server/src/routes/supersep.ts`
- UI: `ui/src/components/stem-studio/` (verify login refresh)
- Possibly `server/src/db/database.ts` if adding job persistence

### 8. Testing checklist
- Two users: each starts stem jobs — verify they only see their own.
- Verify cancel/delete checks ownership (403 for non-owner).
- Verify disk paths are user-prefixed.
- Verify login refresh shows correct jobs/results.
- Run `npx tsc --noEmit` in server/ and ui/ — must pass.

## Constraints
- No auto-build. Use GitHub Actions.
- All changes experimental until tested.
- Single RTX 3090 for testing.
- Commit after verification.

## Deliverable
- StemJob includes userId.
- All jobs/results scoped by user_id.
- Cancel/delete checks ownership (403 for non-owner/non-admin).
- Disk paths user-prefixed.
- UI refreshes on login.
- TypeScript compiles cleanly.
- Commit with descriptive message.
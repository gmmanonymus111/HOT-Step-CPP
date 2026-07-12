# Session Task: User-Isolate Training Studio

## Context
HOT-Step CPP has multi-user auth. Training Studio is NOT scoped per user — all users share datasets, pipelines, training jobs, adapters, preprocess caches, audition previews. This is critical: training jobs are expensive (GPU hours), datasets may be personal, and adapters are valuable IP.

## Goal
Scope all Training Studio data per user. Each user sees only their own datasets, pipelines, jobs, adapters, and previews.

## Current State
- Routes: `server/src/routes/training.ts`
- Services: `server/src/services/training/*.ts`
- Protected by `requireAuth` at app level, but NO user_id filtering.
- Data storage:
  - SQLite: datasets table (one row per dataset), pipelines, jobs
  - Disk: `server/data/training/datasets/<slug>/labels/` (studio-private state)
  - Disk: adapters in `<adapters>/lm-*/<artist>/<run>/` and `<adapters>/dit-*/<artist>/<run>/`
  - Disk: tensor caches, preprocess outputs
- No user_id in any training-related table or path currently.

## Plan

### 1. Database migration
- Add `user_id TEXT NOT NULL` to:
  - datasets table
  - pipelines table (if exists)
  - jobs table (if exists)
  - Any other training-related tables
- Migrate existing rows to admin user's ID.
- Add indexes on user_id.

### 2. Route-level enforcement
- Every training route must filter by `req.user.userId`.
- Pattern: `const userId = req.user!.userId;` then scope all queries.
- CRUD:
  - SELECT: WHERE user_id = ?
  - INSERT: include user_id
  - UPDATE/DELETE: WHERE user_id = ?

### 3. Disk path scoping (CRITICAL)
Current paths are flat/shared. Options:

**Option A (recommended):** Prefix disk paths with user_id:
- `server/data/training/datasets/<userId>/<slug>/labels/`
- `server/data/training/pipelines/<userId>/`
- `server/data/training/jobs/<userId>/`
- Adapters: `<adapters>/lm-*/<userId>/<artist>/<run>/`

**Option B:** Keep flat paths but store user_id in metadata files and filter on access.

Option A is cleaner, prevents accidental leakage, and matches the user-isolation model.

### 4. Adapter scoping
- Adapter scan endpoints (`GET /api/adapters/lm`, `POST /api/adapters/scan`) must filter by user.
- Each user only sees their own adapters.
- Migration: existing adapters → admin user's namespace.

### 5. Queue access fix (training jobs)
- Training job cancellation/deletion must check ownership.
- Only job owner or admin can cancel/delete.
- Same pattern as generation queue: check `req.user.userId === job.userId || req.user.role === 'admin'`.

### 6. Late library update fix (training outputs)
- Training outputs (adapters, datasets) don't go through the song-created event flow.
- Ensure UI re-fetches training data after:
  - Login (same auth-logged-in event pattern)
  - Job completion (SSE already exists, verify it triggers UI refresh)
- If UI caches training state, invalidate on auth transitions.

### 7. Files to modify
- `server/src/routes/training.ts`
- `server/src/services/training/*.ts`
- `server/src/db/database.ts` (migration)
- Adapter layout: `server/src/services/training/adapterLayout.ts`
- UI training studio components (verify they use user-scoped endpoints)

### 8. Testing checklist
- Two users: each creates datasets, starts training — verify isolation.
- Verify adapter scans show only user's adapters.
- Verify job cancellation respects ownership.
- Verify disk paths are user-prefixed.
- Verify login refresh shows correct data.
- Run `npx tsc --noEmit` in server/ — must pass.

## Constraints
- No auto-build. Use GitHub Actions.
- All changes experimental until tested.
- Single RTX 3090 for testing.
- Commit after verification.

## Deliverable
- All training tables have user_id.
- All disk paths prefixed with user_id.
- All queries/operations scoped by user_id.
- Job cancellation checks ownership.
- UI refreshes on login.
- TypeScript compiles cleanly.
- Commit with descriptive message.
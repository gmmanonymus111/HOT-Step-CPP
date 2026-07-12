# Session Task: User-Isolate Lyric Studio (Lireek)

## Context
HOT-Step CPP has multi-user auth. Songs and generation queue are properly scoped per user. Lyric Studio (Lireek) is NOT — all users share the same artists, albums, profiles, generations, presets, etc.

## Goal
Scope all Lyric Studio data per user. Each user sees only their own artists, albums, profiles, generations, presets, playlists, etc.

## Current State
- Routes: `server/src/routes/lireek.ts` + `server/src/routes/lireek/*.ts`
- Protected by `requireAuth` at app level, but NO user_id filtering in queries.
- Uses SQLite database (check `server/src/db/` for schema).
- Tables likely include: artists, albums, profiles, generations, presets, playlists, etc.
- No user_id column in any table currently.

## Plan

### 1. Database migration
- Add `user_id TEXT NOT NULL` column to all Lireek tables.
- Migrate existing rows to the admin user's ID (same pattern as songs migration).
- Add indexes on `user_id` for performance.
- Create migration in `server/src/db/database.ts` or a separate migration file.

### 2. Route-level enforcement
- Every Lireek route must filter by `req.user.userId`.
- Pattern: `const userId = req.user!.userId;` then append `WHERE user_id = ?` to all queries.
- CRUD operations:
  - SELECT: filter by user_id
  - INSERT: include user_id from req.user
  - UPDATE: include `WHERE user_id = ?` to prevent cross-user updates
  - DELETE: include `WHERE user_id = ?`

### 3. Files to modify
- `server/src/routes/lireek.ts`
- All files in `server/src/routes/lireek/`
- Any services in `server/src/services/` that access Lireek tables directly
- DB schema/migration in `server/src/db/database.ts`

### 4. Edge cases
- Import/migration tools (e.g., HOT-Step 9000 import) — must import into current user's scope.
- Bulk operations — must filter by user_id.
- Any shared/reference data (if exists) — decide if truly shared or per-user.

### 5. Testing checklist
- Create two users (admin + test user).
- Each creates artists, albums, generations — verify they don't see each other's data.
- Verify existing data appears only under admin.
- Verify import/migration tools scope to current user.
- Run `npx tsc --noEmit` in server/ — must pass.

## Constraints
- No auto-build. Use GitHub Actions for build verification.
- All changes experimental until tested.
- Single RTX 3090 for any local testing (no multi-GPU).
- Commit after verification.

## Deliverable
- All Lireek tables have user_id column.
- All queries scoped by user_id.
- Migration runs on first auth-enabled launch.
- TypeScript compiles cleanly.
- Commit with descriptive message.
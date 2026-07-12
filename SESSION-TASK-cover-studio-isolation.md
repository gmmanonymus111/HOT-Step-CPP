# Session Task: User-Isolate Cover Studio / Analyze + Verify Queue/Library

## Context
HOT-Step CPP has multi-user auth. Cover Studio uses the analyze route for audio analysis (BPM, key, etc.). Songs are already per-user, but analysis cache/results may be shared. Need to verify and fix.

## Goal
1. Verify Cover Studio/Analyze is properly scoped per user (via song ownership).
2. If analysis cache is shared, scope it per user.
3. Verify queue access and late library update are not issues here.

## Current State
- Routes: `server/src/routes/analyze.ts`
- Protected by `requireAuth` at app level.
- Cover Studio workflow:
  - User uploads audio or selects a library song.
  - Analyze extracts BPM, key, energy, timbre.
  - Results used for cover generation.
- Songs are per-user, so selecting a library song is already scoped.
- Question: is analysis cache keyed by song_id (safe) or by audio hash/filename (potentially shared)?

## Plan

### 1. Audit analyze.ts
- Check how analysis results are stored/cached.
- If keyed by song_id: already scoped (songs are per-user).
- If keyed by audio hash/filename: potentially shared — fix needed.
- Check if uploaded reference tracks are scoped (should be, via mastering routes).

### 2. Fix if needed
- If cache is shared:
  - Add user_id to cache key or storage.
  - Or scope by song_id only (don't cache across users).
- Verify uploaded tracks for analysis are per-user.

### 3. Queue access check
- Cover generation uses the main generation queue (already fixed).
- Verify no separate cover-specific queue exists.

### 4. Late library update check
- Cover results are songs → already fixed via auth-logged-in event.
- Verify no separate cover result caching in UI.

### 5. Files to check/modify
- `server/src/routes/analyze.ts`
- `server/src/routes/coverArt.ts` (if relevant)
- UI: `ui/src/components/cover-studio/` (verify no shared state)

### 6. Testing checklist
- Two users: analyze same audio file — verify results don't leak.
- Verify cover generation uses scoped queue.
- Verify login refresh shows correct covers.
- Run `npx tsc --noEmit` in server/ and ui/ — must pass.

## Constraints
- No auto-build. Use GitHub Actions.
- All changes experimental until tested.
- Single RTX 3090 for testing.
- Commit after verification (or note if no changes needed).

## Deliverable
- Analysis cache scoped per user (or proven safe via song_id keying).
- No shared state between users.
- Queue access and library update verified working.
- TypeScript compiles cleanly.
- Commit if changes made, or close as "verified safe" if no changes needed.
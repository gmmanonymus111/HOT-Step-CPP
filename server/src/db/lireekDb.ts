// lireekDb.ts — Query functions for Lyric Studio / Lireek tables
//
// These tables now live in the unified hotstep.db (previously lireek.db).
// All functions use getDb() from database.ts — there is no separate connection.
//
// USER ISOLATION: every function requires a userId parameter. All queries are
// scoped to that user — no cross-user data leakage.

import { getDb } from './database.js';

// ── Legacy exports (no-ops, kept for compatibility during transition) ────────
// initLireekDb/closeLireekDb are no longer needed — the tables are created
// and migrated in database.ts initDb(). These are kept temporarily so callers
// that import them don't break at compile time.
/** @deprecated Tables are now in hotstep.db — no separate init needed */
export function initLireekDb(): void {
  // No-op — tables created in initDb()
}
/** @deprecated Tables are now in hotstep.db — no separate close needed */
export function closeLireekDb(): void {
  // No-op — closed by closeDb()
}
/** @deprecated Use getDb() directly */
export function getLireekDb(): ReturnType<typeof getDb> {
  return getDb();
}


// ── Artists ──────────────────────────────────────────────────────────────────

export function getOrCreateArtist(userId: string, name: string): Record<string, any> {
  const db = getDb();
  const existing = db.prepare(
    'SELECT * FROM artists WHERE name = ? COLLATE NOCASE AND user_id = ?'
  ).get(name, userId) as any;
  if (existing) return existing;

  const now = new Date().toISOString();
  const result = db.prepare(
    'INSERT INTO artists (name, user_id, created_at) VALUES (?, ?, ?)'
  ).run(name, userId, now);
  return { id: result.lastInsertRowid, name, user_id: userId, created_at: now, image_url: null, genius_id: null };
}

export function listArtists(userId: string): Record<string, any>[] {
  return getDb().prepare(
    `SELECT a.*, COUNT(ls.id) AS lyrics_set_count
     FROM artists a LEFT JOIN lyrics_sets ls ON ls.artist_id = a.id AND ls.user_id = ?
     WHERE a.user_id = ?
     GROUP BY a.id ORDER BY a.name`
  ).all(userId, userId) as any[];
}

export function deleteArtist(userId: string, id: number): boolean {
  const result = getDb().prepare('DELETE FROM artists WHERE id = ? AND user_id = ?').run(id, userId);
  return result.changes > 0;
}

export function updateArtistImage(userId: string, id: number, imageUrl: string | null): void {
  getDb().prepare('UPDATE artists SET image_url = ? WHERE id = ? AND user_id = ?').run(imageUrl, id, userId);
}

export function updateArtistGeniusId(userId: string, id: number, geniusId: number | null): void {
  getDb().prepare('UPDATE artists SET genius_id = ? WHERE id = ? AND user_id = ?').run(geniusId, id, userId);
}

export function getArtist(userId: string, id: number): Record<string, any> | undefined {
  return getDb().prepare('SELECT * FROM artists WHERE id = ? AND user_id = ?').get(id, userId) as any;
}

/** Case-insensitive artist lookup that never creates — preview code must not
 *  leave rows behind for an export the user then cancels. */
export function findArtistByName(userId: string, name: string): Record<string, any> | null {
  return (getDb().prepare(
    'SELECT * FROM artists WHERE name = ? COLLATE NOCASE AND user_id = ?'
  ).get(name, userId) as any) ?? null;
}


// ── Lyrics Sets ─────────────────────────────────────────────────────────────

export function saveLyricsSet(
  userId: string, artistId: number, album: string | null, maxSongs: number, songs: any[],
  imageUrl?: string | null,
): Record<string, any> {
  const now = new Date().toISOString();
  const songsJson = JSON.stringify(songs);
  const result = getDb().prepare(
    'INSERT INTO lyrics_sets (artist_id, album, max_songs, songs, image_url, user_id, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?)'
  ).run(artistId, album, maxSongs, songsJson, imageUrl ?? null, userId, now);
  return {
    id: result.lastInsertRowid, artist_id: artistId, album, max_songs: maxSongs,
    total_songs: songs.length, image_url: imageUrl ?? null, user_id: userId, fetched_at: now,
  };
}

/** Case-insensitive artist+album match — the update-in-place target for a
 *  Training Studio export. `songs` stays parsed out like getLyricsSets(). */
export function findLyricsSetByAlbum(userId: string, artistId: number, album: string): Record<string, any> | null {
  const row = getDb().prepare(
    'SELECT * FROM lyrics_sets WHERE artist_id = ? AND album = ? COLLATE NOCASE AND user_id = ?'
  ).get(artistId, album, userId) as any;
  if (!row) return null;
  const songs = JSON.parse(row.songs);
  const { songs: _, ...rest } = row;
  return { ...rest, total_songs: songs.length };
}

/** Replace a set's whole song list in place (re-export from Training Studio).
 *  Album is rewritten too so an override can fix casing/spelling; the image is
 *  only touched when a non-null one is passed — never cleared. */
export function replaceLyricsSetSongs(
  userId: string, id: number, album: string | null, songs: any[], imageUrl?: string | null,
): Record<string, any> | null {
  const now = new Date().toISOString();
  const db = getDb();
  db.prepare(
    'UPDATE lyrics_sets SET album = ?, max_songs = ?, songs = ?, fetched_at = ? WHERE id = ? AND user_id = ?'
  ).run(album, songs.length, JSON.stringify(songs), now, id, userId);
  if (imageUrl) {
    db.prepare('UPDATE lyrics_sets SET image_url = ? WHERE id = ? AND user_id = ?').run(imageUrl, id, userId);
  }
  return getLyricsSet(userId, id);
}

export function getLyricsSets(userId: string, artistId?: number): Record<string, any>[] {
  const db = getDb();
  const query = artistId
    ? db.prepare(
        `SELECT ls.*, a.name as artist_name FROM lyrics_sets ls
         JOIN artists a ON a.id = ls.artist_id
         WHERE ls.artist_id = ? AND ls.user_id = ?
         ORDER BY ls.fetched_at DESC`
      )
    : db.prepare(
        `SELECT ls.*, a.name as artist_name FROM lyrics_sets ls
         JOIN artists a ON a.id = ls.artist_id
         WHERE ls.user_id = ?
         ORDER BY ls.fetched_at DESC`
      );

  const rows = (artistId ? query.all(artistId, userId) : query.all(userId)) as any[];
  return rows.map(r => {
    const songs = JSON.parse(r.songs);
    const { songs: _, ...rest } = r;
    return { ...rest, total_songs: songs.length };
  });
}

export function getLyricsSet(userId: string, id: number): Record<string, any> | null {
  const row = getDb().prepare(
    `SELECT ls.*, a.name as artist_name FROM lyrics_sets ls
     JOIN artists a ON a.id = ls.artist_id
     WHERE ls.id = ? AND ls.user_id = ?`
  ).get(id, userId) as any;
  if (!row) return null;
  row.songs = JSON.parse(row.songs);
  row.total_songs = row.songs.length;
  return row;
}

export function deleteLyricsSet(userId: string, id: number): boolean {
  return getDb().prepare('DELETE FROM lyrics_sets WHERE id = ? AND user_id = ?').run(id, userId).changes > 0;
}

export function removeSongFromSet(userId: string, lyricsSetId: number, songIndex: number): Record<string, any> | null {
  const db = getDb();
  const row = db.prepare('SELECT songs FROM lyrics_sets WHERE id = ? AND user_id = ?').get(lyricsSetId, userId) as any;
  if (!row) return null;
  const songs = JSON.parse(row.songs);
  if (songIndex < 0 || songIndex >= songs.length) return null;
  songs.splice(songIndex, 1);
  db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ? AND user_id = ?').run(JSON.stringify(songs), lyricsSetId, userId);
  return getLyricsSet(userId, lyricsSetId);
}

export function editSongInSet(userId: string, lyricsSetId: number, songIndex: number, newLyrics: string): Record<string, any> | null {
  const db = getDb();
  const row = db.prepare('SELECT songs FROM lyrics_sets WHERE id = ? AND user_id = ?').get(lyricsSetId, userId) as any;
  if (!row) return null;
  const songs = JSON.parse(row.songs);
  if (songIndex < 0 || songIndex >= songs.length) return null;
  songs[songIndex].lyrics = newLyrics;
  db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ? AND user_id = ?').run(JSON.stringify(songs), lyricsSetId, userId);
  return getLyricsSet(userId, lyricsSetId);
}

export function addSongToSet(userId: string, lyricsSetId: number, song: { title: string; album?: string; lyrics: string }): Record<string, any> | null {
  const db = getDb();
  const row = db.prepare('SELECT songs FROM lyrics_sets WHERE id = ? AND user_id = ?').get(lyricsSetId, userId) as any;
  if (!row) return null;
  const songs = JSON.parse(row.songs);
  songs.push(song);
  db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ? AND user_id = ?').run(JSON.stringify(songs), lyricsSetId, userId);
  return getLyricsSet(userId, lyricsSetId);
}

export function updateLyricsSetImage(userId: string, id: number, imageUrl: string | null): void {
  getDb().prepare('UPDATE lyrics_sets SET image_url = ? WHERE id = ? AND user_id = ?').run(imageUrl, id, userId);
}


// ── Profiles ────────────────────────────────────────────────────────────────

export function saveProfile(
  userId: string, lyricsSetId: number, provider: string, model: string, profileData: any,
): Record<string, any> {
  const now = new Date().toISOString();
  const result = getDb().prepare(
    'INSERT INTO profiles (lyrics_set_id, provider, model, profile_data, user_id, created_at) VALUES (?, ?, ?, ?, ?, ?)'
  ).run(lyricsSetId, provider, model, JSON.stringify(profileData), userId, now);
  return {
    id: result.lastInsertRowid, lyrics_set_id: lyricsSetId,
    provider, model, profile_data: profileData, user_id: userId, created_at: now,
  };
}

export function getProfiles(userId: string, lyricsSetId?: number): Record<string, any>[] {
  const db = getDb();
  const query = lyricsSetId
    ? db.prepare('SELECT * FROM profiles WHERE lyrics_set_id = ? AND user_id = ? ORDER BY created_at DESC')
    : db.prepare('SELECT * FROM profiles WHERE user_id = ? ORDER BY created_at DESC');
  const rows = (lyricsSetId ? query.all(lyricsSetId, userId) : query.all(userId)) as any[];
  return rows.map(r => ({ ...r, profile_data: JSON.parse(r.profile_data) }));
}

export function getProfile(userId: string, id: number): Record<string, any> | null {
  const row = getDb().prepare('SELECT * FROM profiles WHERE id = ? AND user_id = ?').get(id, userId) as any;
  if (!row) return null;
  row.profile_data = JSON.parse(row.profile_data);
  return row;
}

export function deleteProfile(userId: string, id: number): boolean {
  return getDb().prepare('DELETE FROM profiles WHERE id = ? AND user_id = ?').run(id, userId).changes > 0;
}

export function updateProfileData(userId: string, id: number, profileData: any): void {
  getDb().prepare('UPDATE profiles SET profile_data = ? WHERE id = ? AND user_id = ?').run(JSON.stringify(profileData), id, userId);
}


// ── Generations ─────────────────────────────────────────────────────────────

export interface SaveGenerationParams {
  userId: string;
  profileId: number;
  provider: string;
  model: string;
  lyrics: string;
  extraInstructions?: string;
  title?: string;
  subject?: string;
  bpm?: number;
  key?: string;
  caption?: string;
  duration?: number;
  systemPrompt?: string;
  userPrompt?: string;
  parentGenerationId?: number | null;
}

export function saveGeneration(p: SaveGenerationParams): Record<string, any> {
  const now = new Date().toISOString();
  const result = getDb().prepare(
    `INSERT INTO generations
     (profile_id, provider, model, extra_instructions, title, subject, bpm, key, caption, duration, lyrics, system_prompt, user_prompt, parent_generation_id, user_id, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    p.profileId, p.provider, p.model, p.extraInstructions ?? null,
    p.title ?? '', p.subject ?? '', p.bpm ?? 0, p.key ?? '', p.caption ?? '', p.duration ?? 0,
    p.lyrics, p.systemPrompt ?? '', p.userPrompt ?? '', p.parentGenerationId ?? null, p.userId, now,
  );
  return {
    id: result.lastInsertRowid, profile_id: p.profileId, provider: p.provider,
    model: p.model, extra_instructions: p.extraInstructions ?? null,
    title: p.title ?? '', subject: p.subject ?? '', bpm: p.bpm ?? 0,
    key: p.key ?? '', caption: p.caption ?? '', duration: p.duration ?? 0,
    lyrics: p.lyrics, system_prompt: p.systemPrompt ?? '', user_prompt: p.userPrompt ?? '',
    parent_generation_id: p.parentGenerationId ?? null, user_id: p.userId, created_at: now,
  };
}

export function getGenerations(userId: string, profileId?: number, lyricsSetId?: number): Record<string, any>[] {
  const db = getDb();
  if (profileId) {
    return db.prepare(
      `SELECT g.* FROM generations g JOIN profiles p ON p.id = g.profile_id
       WHERE g.profile_id = ? AND g.user_id = ? ORDER BY g.created_at DESC`
    ).all(profileId, userId) as any[];
  }
  if (lyricsSetId) {
    return db.prepare(
      `SELECT g.* FROM generations g
       JOIN profiles p ON p.id = g.profile_id
       WHERE p.lyrics_set_id = ? AND g.user_id = ? ORDER BY g.created_at DESC`
    ).all(lyricsSetId, userId) as any[];
  }
  return db.prepare('SELECT * FROM generations WHERE user_id = ? ORDER BY created_at DESC').all(userId) as any[];
}

export function getAllGenerationsWithContext(userId: string): Record<string, any>[] {
  return getDb().prepare(
    `SELECT g.*, a.name AS artist_name, ls.album, ls.artist_id
     FROM generations g
     JOIN profiles p ON p.id = g.profile_id
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     WHERE g.user_id = ?
     ORDER BY g.created_at DESC`
  ).all(userId) as any[];
}

export function getGeneration(userId: string, id: number): Record<string, any> | null {
  return (getDb().prepare('SELECT * FROM generations WHERE id = ? AND user_id = ?').get(id, userId) as any) ?? null;
}

export function updateGenerationMetadata(
  userId: string, id: number, bpm: number, key: string, caption: string, duration: number = 0,
): void {
  getDb().prepare(
    'UPDATE generations SET bpm = ?, key = ?, caption = ?, duration = ? WHERE id = ? AND user_id = ?'
  ).run(bpm, key, caption, duration, id, userId);
}

export function updateGenerationFields(userId: string, id: number, fields: Record<string, any>): void {
  const allowed = ['title', 'subject', 'lyrics', 'bpm', 'key', 'caption', 'duration', 'extra_instructions'];
  const sets: string[] = [];
  const values: any[] = [];
  for (const [k, v] of Object.entries(fields)) {
    if (allowed.includes(k)) {
      sets.push(`${k} = ?`);
      values.push(v);
    }
  }
  if (sets.length === 0) return;
  values.push(id, userId);
  getDb().prepare(`UPDATE generations SET ${sets.join(', ')} WHERE id = ? AND user_id = ?`).run(...values);
}

export function deleteGeneration(userId: string, id: number): boolean {
  return getDb().prepare('DELETE FROM generations WHERE id = ? AND user_id = ?').run(id, userId).changes > 0;
}

export function purgeProfilesAndGenerations(userId: string): { generations_deleted: number; profiles_deleted: number } {
  const db = getDb();
  const genResult = db.prepare('DELETE FROM generations WHERE user_id = ?').run(userId);
  const profResult = db.prepare('DELETE FROM profiles WHERE user_id = ?').run(userId);
  return { generations_deleted: genResult.changes, profiles_deleted: profResult.changes };
}

export function purgeGenerationsOnly(userId: string): { generations_deleted: number } {
  const result = getDb().prepare('DELETE FROM generations WHERE user_id = ?').run(userId);
  return { generations_deleted: result.changes };
}

export function purgeProfilesOnly(userId: string): { profiles_deleted: number; generations_deleted: number } {
  const db = getDb();
  // Generations depend on profiles via FK, so delete generations first
  const genResult = db.prepare('DELETE FROM generations WHERE user_id = ?').run(userId);
  const profResult = db.prepare('DELETE FROM profiles WHERE user_id = ?').run(userId);
  return { profiles_deleted: profResult.changes, generations_deleted: genResult.changes };
}


// ── Settings ────────────────────────────────────────────────────────────────
// Settings are now per-user. Keys are scoped by user_id.

export function getSetting(userId: string, key: string, defaultValue = ''): string {
  const row = getDb().prepare('SELECT value FROM settings WHERE key = ? AND user_id = ?').get(key, userId) as any;
  return row?.value ?? defaultValue;
}

export function setSetting(userId: string, key: string, value: string): void {
  getDb().prepare(
    'INSERT INTO settings (key, value, user_id) VALUES (?, ?, ?) ON CONFLICT(key, user_id) DO UPDATE SET value = excluded.value'
  ).run(key, value, userId);
}


// ── Album Presets ───────────────────────────────────────────────────────────

export function getPreset(userId: string, lyricsSetId: number): Record<string, any> | null {
  return (getDb().prepare('SELECT * FROM album_presets WHERE lyrics_set_id = ? AND user_id = ?').get(lyricsSetId, userId) as any) ?? null;
}

export function getAllPresets(userId: string): Record<string, any>[] {
  return getDb().prepare(
    `SELECT ap.*, a.name as artist_name, ls.album, ls.artist_id
     FROM album_presets ap
     JOIN lyrics_sets ls ON ls.id = ap.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     WHERE ap.user_id = ?
     ORDER BY ap.created_at DESC`
  ).all(userId) as any[];
}

export function upsertPreset(userId: string, lyricsSetId: number, data: {
  adapterPath?: string | null;
  adapterScale?: number | null;
  adapterGroupScales?: any;
  referenceTrackPath?: string | null;
  audioCoverStrength?: number | null;
  lmAdapterPath?: string | null;
  lmAdapterScale?: number | null;
}): Record<string, any> {
  const db = getDb();
  const existing = getPreset(userId, lyricsSetId);
  const groupScalesJson = data.adapterGroupScales ? JSON.stringify(data.adapterGroupScales) : null;

  if (existing) {
    db.prepare(
      `UPDATE album_presets SET adapter_path = ?, adapter_scale = ?, adapter_group_scales = ?,
       reference_track_path = ?, audio_cover_strength = ?, lm_adapter_path = ?, lm_adapter_scale = ?
       WHERE lyrics_set_id = ? AND user_id = ?`
    ).run(
      data.adapterPath ?? null, data.adapterScale ?? null, groupScalesJson,
      data.referenceTrackPath ?? null, data.audioCoverStrength ?? null,
      data.lmAdapterPath ?? null, data.lmAdapterScale ?? null, lyricsSetId, userId,
    );
  } else {
    db.prepare(
      `INSERT INTO album_presets (lyrics_set_id, adapter_path, adapter_scale, adapter_group_scales, reference_track_path, audio_cover_strength, lm_adapter_path, lm_adapter_scale, user_id)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      lyricsSetId, data.adapterPath ?? null, data.adapterScale ?? null, groupScalesJson,
      data.referenceTrackPath ?? null, data.audioCoverStrength ?? null,
      data.lmAdapterPath ?? null, data.lmAdapterScale ?? null, userId,
    );
  }
  return getPreset(userId, lyricsSetId)!;
}

export function deletePreset(userId: string, lyricsSetId: number): boolean {
  return getDb().prepare('DELETE FROM album_presets WHERE lyrics_set_id = ? AND user_id = ?').run(lyricsSetId, userId).changes > 0;
}


// ── Audio Generations ───────────────────────────────────────────────────────

export function linkAudioGeneration(userId: string, generationId: number, jobId: string): Record<string, any> {
  const now = new Date().toISOString();
  const result = getDb().prepare(
    'INSERT INTO audio_generations (generation_id, hotstep_job_id, user_id, created_at) VALUES (?, ?, ?, ?)'
  ).run(generationId, jobId, userId, now);
  return { id: result.lastInsertRowid, generation_id: generationId, hotstep_job_id: jobId, user_id: userId, created_at: now };
}

export function getAudioGenerations(userId: string, generationId: number): Record<string, any>[] {
  return getDb().prepare(
    `SELECT ag.*, s.mastered_audio_url
     FROM audio_generations ag
     LEFT JOIN songs s ON s.audio_url = ag.audio_url
     WHERE ag.generation_id = ? AND ag.user_id = ? ORDER BY ag.created_at DESC`
  ).all(generationId, userId) as any[];
}

export function resolveAudioGeneration(userId: string, jobId: string, audioUrl: string, coverUrl?: string): void {
  getDb().prepare(
    'UPDATE audio_generations SET audio_url = ?, cover_url = ? WHERE hotstep_job_id = ? AND user_id = ?'
  ).run(audioUrl, coverUrl ?? null, jobId, userId);
}

export function deleteAudioGeneration(userId: string, id: number): boolean {
  return getDb().prepare('DELETE FROM audio_generations WHERE id = ? AND user_id = ?').run(id, userId).changes > 0;
}

/** Delete audio_generations rows matching the given hotstep job IDs (used when songs are deleted from the main library). */
export function deleteAudioGenerationsByJobIds(userId: string, jobIds: string[]): number {
  if (jobIds.length === 0) return 0;
  const placeholders = jobIds.map(() => '?').join(',');
  return getDb().prepare(`DELETE FROM audio_generations WHERE hotstep_job_id IN (${placeholders}) AND user_id = ?`).run(...jobIds, userId).changes;
}

export function getRecentGenerationsWithAudio(userId: string, limit = 50): Record<string, any>[] {
  return getDb().prepare(
    `SELECT g.title AS song_title, g.subject, g.caption, g.lyrics, g.duration,
       g.created_at AS ag_created_at, g.id AS generation_id,
       a.name AS artist_name, a.image_url AS artist_image, a.id AS artist_id,
       ls.album, ls.id AS lyrics_set_id,
       ag.id AS ag_id, ag.audio_url, ag.cover_url, ag.hotstep_job_id,
       s.mastered_audio_url
     FROM audio_generations ag
     JOIN generations g ON g.id = ag.generation_id
     JOIN profiles p ON p.id = g.profile_id
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     LEFT JOIN songs s ON s.audio_url = ag.audio_url
     WHERE ag.audio_url IS NOT NULL AND ag.user_id = ?
     ORDER BY ag.created_at DESC
     LIMIT ?`
  ).all(userId, limit) as any[];
}
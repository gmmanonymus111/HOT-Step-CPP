// geniusService.ts — Lyrics acquisition via the Genius API
//
// Port of Python genius_service.py.
// Uses fetch() for the Genius REST API and cheerio for HTML parsing.

import * as cheerio from 'cheerio';
import { config } from '../../config.js';

const API_ROOT = 'https://api.genius.com';

const BROWSER_HEADERS: Record<string, string> = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
  'Accept-Language': 'en-US,en;q=0.9',
};

export interface SongLyrics {
  title: string;
  album: string | null;
  lyrics: string;
}

export interface LyricsSearchResponse {
  artist: string;
  album: string | null;
  songs: SongLyrics[];
  total_songs: number;
}


// ── Helpers ─────────────────────────────────────────────────────────────────

function getAuthHeaders(): Record<string, string> {
  const token = config.lireek.geniusAccessToken;
  if (!token) {
    throw new Error('GENIUS_ACCESS_TOKEN is not set. Please add it to your .env file.');
  }
  return { Authorization: `Bearer ${token}` };
}

function cleanLyrics(raw: string): string {
  if (!raw) return '';
  // Remove contributor/title header (captures up to first section header)
  let text = raw.replace(/^\d+\s*Contributors?.*?Lyrics.*?(?=\[)/is, '');
  // Simpler header strip if no section header found
  text = text.replace(/^\d+\s*Contributors?.*?Lyrics\s*\n?/is, '');
  // Remove 'You might also like'
  text = text.replace(/You might also like\s*/g, '');
  // Remove trailing 'Embed'
  text = text.replace(/\d*Embed$/, '').trim();
  // Collapse 3+ consecutive blank lines into 2
  text = text.replace(/\n{3,}/g, '\n\n');
  return text.trim();
}

async function scrapeLyrics(songUrl: string): Promise<string | null> {
  try {
    const resp = await fetch(songUrl, {
      headers: BROWSER_HEADERS,
      redirect: 'follow',
    });
    if (!resp.ok) {
      console.warn(`[Genius] Failed to scrape ${songUrl}: ${resp.status}`);
      return null;
    }

    const html = await resp.text();
    const $ = cheerio.load(html.replace(/<br\/?>/gi, '\n'));

    // Modern Genius layout: data-lyrics-container divs
    const containers = $('div[data-lyrics-container="true"]');
    if (containers.length) {
      const parts: string[] = [];
      containers.each((_, el) => { parts.push($(el).text()); });
      return parts.join('\n');
    }

    // Fallback: root lyrics div
    const root = $('div.lyrics, div[class*="Lyrics__Root"]');
    if (root.length) return root.text();

    console.warn(`[Genius] No lyrics div found on: ${songUrl}`);
    return null;
  } catch (err) {
    console.warn(`[Genius] Scrape error for ${songUrl}:`, err);
    return null;
  }
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));


// ── Authenticated Genius API calls ──────────────────────────────────────────

async function apiSearch(query: string, perPage = 20): Promise<any[]> {
  const headers = getAuthHeaders();
  const url = `${API_ROOT}/search?q=${encodeURIComponent(query)}&per_page=${perPage}`;
  const resp = await fetch(url, { headers });
  if (!resp.ok) throw new Error(`Genius search failed: ${resp.status}`);
  const data = await resp.json() as any;
  return data.response.hits;
}

async function apiGetArtistSongs(artistId: number, perPage = 20, sort = 'popularity'): Promise<any[]> {
  const headers = getAuthHeaders();
  const url = `${API_ROOT}/artists/${artistId}/songs?per_page=${perPage}&page=1&sort=${sort}`;
  const resp = await fetch(url, { headers });
  if (!resp.ok) throw new Error(`Genius artist songs failed: ${resp.status}`);
  const data = await resp.json() as any;
  return data.response.songs;
}

async function apiGetAlbumTracks(albumId: number): Promise<any[]> {
  const headers = getAuthHeaders();
  const tracks: any[] = [];
  let page: number | null = 1;

  while (page) {
    const url = `${API_ROOT}/albums/${albumId}/tracks?per_page=50&page=${page}`;
    const resp = await fetch(url, { headers });
    if (!resp.ok) throw new Error(`Genius album tracks failed: ${resp.status}`);
    const data = await resp.json() as any;
    tracks.push(...data.response.tracks);
    page = data.response.next_page ?? null;
  }
  return tracks;
}

async function apiGetArtistDetails(artistId: number): Promise<any> {
  const headers = getAuthHeaders();
  const resp = await fetch(`${API_ROOT}/artists/${artistId}`, { headers });
  if (!resp.ok) throw new Error(`Genius artist details failed: ${resp.status}`);
  const data = await resp.json() as any;
  return data.response.artist;
}

async function getArtistIdFromUrl(url: string): Promise<number | null> {
  try {
    const resp = await fetch(url, { headers: BROWSER_HEADERS, redirect: 'follow' });
    if (!resp.ok) return null;
    const html = await resp.text();
    const match = html.match(/\\?"artist_id\\?":\s*(\d+)/) ?? html.match(/content="genius:\/\/artists\/(\d+)"/);
    return match ? parseInt(match[1], 10) : null;
  } catch { return null; }
}

async function findArtistId(artistName: string): Promise<number | null> {
  const hits = await apiSearch(artistName, 5);
  const nameLower = artistName.toLowerCase();
  for (const hit of hits) {
    const primary = hit.result?.primary_artist;
    if (primary?.name?.toLowerCase() === nameLower) return primary.id;
  }
  return hits[0]?.result?.primary_artist?.id ?? null;
}

// Patterns indicating bonus/non-original tracks
const BONUS_PATTERN = /\([^)]*(?:Demo|Live|Outtake|Cassette|Remix|Acoustic|Remaster|Session)[^)]*\)/i;
function isBonusTrack(title: string): boolean {
  return BONUS_PATTERN.test(title);
}

function slugifyForGenius(name: string): string {
  return name.replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-');
}

async function findAlbumIdByPage(albumName: string, artistName: string): Promise<number | null> {
  const slug = `${slugifyForGenius(artistName)}/${slugifyForGenius(albumName)}`;
  const url = `https://genius.com/albums/${slug}`;
  const artistLower = artistName.toLowerCase();

  try {
    const resp = await fetch(url, { headers: BROWSER_HEADERS, redirect: 'follow' });
    if (!resp.ok) return null;
    const html = await resp.text();
    const $ = cheerio.load(html);
    const links = $('div.chart_row a.u-display_block').toArray();
    if (!links.length) return null;

    for (const link of links.slice(0, 3)) {
      const h3 = $(link).find('h3');
      let titleText = (h3.length ? h3.text() : $(link).text()).trim();
      titleText = titleText.replace(/\s*Lyrics$/, '').trim();
      if (!titleText) continue;

      const hits = await apiSearch(`${titleText} ${artistName}`, 5);
      for (const hit of hits) {
        const result = hit.result ?? {};
        const songId = result.id;
        if (!songId) continue;
        if (result.primary_artist?.name?.toLowerCase() !== artistLower) continue;

        const headers = getAuthHeaders();
        const sResp = await fetch(`${API_ROOT}/songs/${songId}`, { headers });
        if (!sResp.ok) continue;
        const sData = await sResp.json() as any;
        const album = sData.response.song.album;
        if (album && albumName.toLowerCase().includes(album.name?.toLowerCase())) {
          return album.id;
        }
      }
    }
  } catch {}
  return null;
}

async function findAlbumId(albumName: string, artistName: string): Promise<number | null> {
  const headers = getAuthHeaders();
  const query = `${albumName} ${artistName}`;
  const hits = await apiSearch(query, 10);
  const albumLower = albumName.toLowerCase();
  const artistLower = artistName.toLowerCase();

  // Sort: artist matches first
  const sorted = [...hits].sort((a, b) => {
    const aMatch = a.result?.primary_artist?.name?.toLowerCase() === artistLower ? 1 : 0;
    const bMatch = b.result?.primary_artist?.name?.toLowerCase() === artistLower ? 1 : 0;
    return bMatch - aMatch;
  });

  const candidates: { id: number; name: string }[] = [];
  const seenIds = new Set<number>();

  for (const hit of sorted) {
    const songId = hit.result?.id;
    if (!songId || candidates.length >= 3) break;

    try {
      const resp = await fetch(`${API_ROOT}/songs/${songId}`, { headers });
      if (!resp.ok) continue;
      const data = await resp.json() as any;
      const album = data.response.song.album;
      if (album && album.name?.toLowerCase().includes(albumLower)) {
        if (!seenIds.has(album.id)) {
          seenIds.add(album.id);
          candidates.push({ id: album.id, name: album.name });
        }
      }
    } catch {}
  }

  if (candidates.length) {
    // Prefer exact name match, then shortest name (less likely deluxe/expanded)
    const exact = candidates.find(c => c.name.toLowerCase() === albumLower);
    if (exact) return exact.id;
    candidates.sort((a, b) => a.name.length - b.name.length);
    return candidates[0].id;
  }

  // Fallback: direct page scrape
  return findAlbumIdByPage(albumName, artistName);
}

async function scrapeAlbumPageTracks(
  albumName = '', artistName = '', url?: string,
): Promise<{ title: string; url: string }[]> {
  if (!url) {
    const slug = `${slugifyForGenius(artistName)}/${slugifyForGenius(albumName)}`;
    url = `https://genius.com/albums/${slug}`;
  }
  try {
    const resp = await fetch(url, { headers: BROWSER_HEADERS, redirect: 'follow' });
    if (!resp.ok) return [];
    const html = await resp.text();
    const $ = cheerio.load(html);

    const tracks: { title: string; url: string }[] = [];
    $('div.chart_row a.u-display_block').each((_, el) => {
      let href = $(el).attr('href') ?? '';
      if (!href) return;
      if (!href.startsWith('http')) href = `https://genius.com${href}`;
      const h3 = $(el).find('h3');
      let title = (h3.length ? h3.text() : $(el).text()).trim();
      title = title.replace(/\s*Lyrics$/, '').trim();
      if (title) tracks.push({ title, url: href });
    });
    return tracks;
  } catch { return []; }
}


// ── Public API ──────────────────────────────────────────────────────────────

export async function fetchLyrics(
  artistName: string,
  albumName?: string | null,
  maxSongs = 10,
): Promise<LyricsSearchResponse> {
  const scrapeDelay = 300;
  const songs: SongLyrics[] = [];
  let artistId: number | null = null;

  // Handle Genius artist URL input
  if (artistName.includes('genius.com/artists/')) {
    const urlMatch = artistName.match(/(https?:\/\/(?:www\.)?genius\.com\/artists\/[^\s]+)/);
    const url = urlMatch ? urlMatch[1] : artistName.trim();
    artistId = await getArtistIdFromUrl(url);
    if (artistId) {
      try {
        const details = await apiGetArtistDetails(artistId);
        if (details.name) artistName = details.name;
      } catch {}
    }
  }

  // Handle album URL input
  let albumUrl: string | undefined;
  if (albumName && albumName.includes('genius.com/albums/')) {
    const urlMatch = albumName.match(/(https?:\/\/(?:www\.)?genius\.com\/albums\/[^\s]+)/);
    albumUrl = urlMatch ? urlMatch[1] : albumName.trim();
    const parts = albumUrl.replace(/\/$/, '').split('/');
    if (parts.length >= 2) {
      albumName = parts[parts.length - 1].replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      if (!artistId) {
        const artistSlug = parts[parts.length - 2];
        artistName = artistSlug.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      }
    }
  }

  if (albumName) {
    console.log(`[Genius] Fetching album '${albumName}' by '${artistName}'`);

    let albumIdResolved: number | null = null;
    if (!albumUrl) {
      albumIdResolved = await findAlbumId(albumName, artistName);
    }

    if (!albumUrl && albumIdResolved != null) {
      // API-based album track fetch
      const tracks = await apiGetAlbumTracks(albumIdResolved);
      const seenTitles = new Set<string>();

      for (const track of tracks) {
        const songInfo = track.song ?? {};
        const url = songInfo.url;
        const title = songInfo.title ?? 'Unknown';
        if (!url || songInfo.lyrics_state !== 'complete' || songInfo.instrumental) continue;
        if (isBonusTrack(title)) continue;
        const baseTitle = title.replace(/\s*\(.*\)/, '').trim().toLowerCase();
        if (seenTitles.has(baseTitle)) continue;
        seenTitles.add(baseTitle);

        try {
          await sleep(scrapeDelay);
          const raw = await scrapeLyrics(url);
          if (raw) songs.push({ title, album: albumName, lyrics: cleanLyrics(raw) });
        } catch (err) {
          console.warn(`[Genius] Failed to scrape '${title}':`, err);
        }
      }
    } else {
      // Fallback: scrape tracks from album page
      const pageTracks = await scrapeAlbumPageTracks(albumName, artistName, albumUrl);
      for (const pt of pageTracks) {
        if (isBonusTrack(pt.title)) continue;
        try {
          await sleep(scrapeDelay);
          const raw = await scrapeLyrics(pt.url);
          if (raw) songs.push({ title: pt.title, album: albumName, lyrics: cleanLyrics(raw) });
        } catch (err) {
          console.warn(`[Genius] Failed to scrape '${pt.title}':`, err);
        }
      }
    }
  } else {
    // General artist search (no specific album)
    console.log(`[Genius] Fetching up to ${maxSongs} songs by '${artistName}'`);
    if (!artistId) artistId = await findArtistId(artistName);
    if (!artistId) {
      throw new Error(`Could not find artist '${artistName}' on Genius.`);
    }

    const apiSongs = await apiGetArtistSongs(artistId, maxSongs);
    for (const songInfo of apiSongs.slice(0, maxSongs)) {
      const url = songInfo.url;
      const title = songInfo.title ?? 'Unknown';
      const songAlbum = songInfo.album?.name ?? null;
      if (!url) continue;

      try {
        await sleep(scrapeDelay);
        const raw = await scrapeLyrics(url);
        if (raw) songs.push({ title, album: songAlbum, lyrics: cleanLyrics(raw) });
      } catch (err) {
        console.warn(`[Genius] Failed to scrape '${title}':`, err);
      }
    }
  }

  if (!songs.length) {
    throw new Error(
      `No lyrics found for '${artistName}'` +
      (albumName ? ` – album '${albumName}'` : '') +
      '. Please check the spelling and try again.',
    );
  }

  return { artist: artistName, album: albumName ?? null, songs, total_songs: songs.length };
}

/**
 * Search for a single song's lyrics on Genius.
 */
/** Lowercase, strip parentheticals/brackets and punctuation, collapse spaces. */
function normalizeTitle(s: string): string {
  return s.toLowerCase()
    .replace(/\([^)]*\)|\[[^\]]*\]/g, ' ')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export async function searchSongLyrics(
  artist: string, title: string,
  opts?: { relaxed?: boolean },
): Promise<{ title: string; lyrics: string; url: string } | null> {
  const hits = await apiSearch(`${title} ${artist}`, 5);
  const artistLower = artist.toLowerCase().trim();
  const titleNorm = normalizeTitle(title);

  for (const hit of hits) {
    const result = hit.result ?? {};
    const primaryLower = (result.primary_artist?.name ?? '').toLowerCase().trim();
    if (opts?.relaxed) {
      // Collabs and feats break exact matching ("Electric Callboy & BABYMETAL",
      // "Artist feat. X") — accept containment either way, but then require the
      // titles to agree so a popular unrelated song can't slip in.
      const artistOk = !!primaryLower
        && (primaryLower.includes(artistLower) || artistLower.includes(primaryLower));
      if (!artistOk) continue;
      const resultNorm = normalizeTitle(result.title ?? '');
      const titleOk = !!titleNorm && !!resultNorm
        && (resultNorm.includes(titleNorm) || titleNorm.includes(resultNorm));
      if (!titleOk) continue;
    } else if (primaryLower !== artistLower) {
      continue;
    }
    const songUrl = result.url;
    if (!songUrl) continue;

    const raw = await scrapeLyrics(songUrl);
    if (raw) {
      return { title: result.title ?? title, lyrics: cleanLyrics(raw), url: songUrl };
    }
  }
  return null;
}

/**
 * Refresh an artist's image URL from Genius.
 * Returns the image URL or null if not found.
 */
export async function getArtistImageUrl(artistName: string): Promise<string | null> {
  try {
    const id = await findArtistId(artistName);
    if (!id) return null;
    const details = await apiGetArtistDetails(id);
    return details.image_url ?? null;
  } catch { return null; }
}

/**
 * Get an album cover image URL from Genius.
 */
export async function getAlbumImageUrl(albumName: string, artistName: string): Promise<string | null> {
  try {
    const albumId = await findAlbumId(albumName, artistName);
    if (!albumId) return null;
    const headers = getAuthHeaders();
    const resp = await fetch(`${API_ROOT}/albums/${albumId}`, { headers });
    if (!resp.ok) return null;
    const data = await resp.json() as any;
    return data.response.album?.cover_art_url ?? null;
  } catch { return null; }
}

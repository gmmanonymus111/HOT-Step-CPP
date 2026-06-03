/**
 * GenerateAllModal.tsx — Confirmation modal for "Generate All Audio".
 *
 * Fetches all written songs (generations) across all artists/albums,
 * shows a summary of how many songs will be queued, and on confirm
 * enqueues every generation into the audio generation queue.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { X, Loader2, AlertTriangle, Music, Zap, Users, Disc3 } from 'lucide-react';
import { lireekApi } from '../../services/lireekApi';
import type { Artist, Generation, Profile } from '../../services/lireekApi';
import { enqueueAudioGen } from '../../stores/audioGenQueueStore';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParamsStore } from '../../context/GlobalParamsContext';

interface GenerateAllModalProps {
  open: boolean;
  onClose: () => void;
  artists: Artist[];
  showToast?: (msg: string) => void;
}

interface GenerateAllStats {
  totalSongs: number;
  artistCount: number;
  albumCount: number;
  generations: (Generation & { artist_id: number; artist_name: string; album: string })[];
  profiles: Profile[];
}

export const GenerateAllModal: React.FC<GenerateAllModalProps> = ({
  open, onClose, artists, showToast,
}) => {
  const { token } = useAuth();
  const globalParams = useGlobalParamsStore();

  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<GenerateAllStats | null>(null);
  const [enqueuing, setEnqueuing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });

  // Load stats when modal opens
  useEffect(() => {
    if (!open) return;
    setStats(null);
    setEnqueuing(false);
    setProgress({ current: 0, total: 0 });

    const loadStats = async () => {
      setLoading(true);
      try {
        const [genRes, profileRes] = await Promise.all([
          lireekApi.listAllGenerations(),
          lireekApi.listProfiles(),
        ]);

        // listAllGenerations may return raw array or { generations }
        const gens: any[] = Array.isArray(genRes) ? genRes : (genRes.generations || []);
        const profiles = Array.isArray(profileRes) ? profileRes : (profileRes.profiles || []);

        const artistIds = new Set(gens.map((g: any) => g.artist_id));
        const albumNames = new Set(gens.map((g: any) => `${g.artist_id}-${g.album || 'Unknown'}`));

        // Sort deterministically: artist name A→Z, then song title A→Z within each artist.
        // This lets the user see exactly where a batch stopped and resume from there.
        gens.sort((a: any, b: any) => {
          const artistCmp = (a.artist_name || '').localeCompare(b.artist_name || '', undefined, { sensitivity: 'base' });
          if (artistCmp !== 0) return artistCmp;
          return (a.title || '').localeCompare(b.title || '', undefined, { sensitivity: 'base' });
        });

        setStats({
          totalSongs: gens.length,
          artistCount: artistIds.size,
          albumCount: albumNames.size,
          generations: gens,
          profiles,
        });
      } catch (err: any) {
        showToast?.(`Failed to load data: ${err.message}`);
        onClose();
      } finally {
        setLoading(false);
      }
    };

    loadStats();
  }, [open]);

  const handleConfirm = useCallback(async () => {
    if (!stats || !token) return;
    setEnqueuing(true);
    setProgress({ current: 0, total: stats.totalSongs });

    const paramsSnapshot = globalParams.getGlobalParams();
    const profileMap = new Map(stats.profiles.map(p => [p.id, p]));
    const artistMap = new Map(artists.map(a => [a.id, a]));

    let queued = 0;
    let skipped = 0;

    for (let i = 0; i < stats.generations.length; i++) {
      const gen = stats.generations[i];
      const profile = profileMap.get(gen.profile_id);
      if (!profile) {
        skipped++;
        setProgress({ current: i + 1, total: stats.totalSongs });
        continue;
      }

      const artist = artistMap.get(gen.artist_id);

      try {
        await enqueueAudioGen(gen, {
          artistId: gen.artist_id || 0,
          artistName: gen.artist_name || artist?.name || 'Unknown',
          artistImageUrl: artist?.image_url || '',
          profileId: profile.id,
          lyricsSetId: profile.lyrics_set_id,
        }, paramsSnapshot, token);
        queued++;
      } catch (err) {
        skipped++;
      }

      setProgress({ current: i + 1, total: stats.totalSongs });
    }

    const parts = [];
    if (queued > 0) parts.push(`${queued} song${queued !== 1 ? 's' : ''} queued`);
    if (skipped > 0) parts.push(`${skipped} skipped`);
    showToast?.(parts.join(', ') || 'Done');

    setEnqueuing(false);
    onClose();
  }, [stats, token, artists, globalParams, showToast, onClose]);

  if (!open) return null;

  // Estimate time: ~3 min per song is a rough average for generation
  const estimatedMinutes = stats ? Math.ceil(stats.totalSongs * 3) : 0;
  const estimatedHours = Math.floor(estimatedMinutes / 60);
  const estimatedMins = estimatedMinutes % 60;
  const estimatedStr = estimatedHours > 0
    ? `~${estimatedHours}h ${estimatedMins}m`
    : `~${estimatedMins}m`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 dark:bg-black/60 backdrop-blur-sm">
      <div className="bg-white dark:bg-zinc-900 rounded-2xl border border-zinc-300 dark:border-white/10 shadow-2xl w-[480px] max-h-[80vh] flex flex-col animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/5">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-400" />
            <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Generate All Audio</h2>
          </div>
          <button
            onClick={onClose}
            disabled={enqueuing}
            className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors disabled:opacity-50"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-12 gap-3">
              <Loader2 className="w-8 h-8 text-pink-400 animate-spin" />
              <p className="text-sm text-zinc-500">Loading generation data…</p>
            </div>
          ) : stats && !enqueuing ? (
            <>
              {/* Warning */}
              <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-950/30 border border-amber-500/20">
                <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-amber-200/90 leading-relaxed">
                  <p className="font-semibold text-amber-300 mb-1">This is a massive operation!</p>
                  <p>
                    This will queue <strong>every written song</strong> from every album,
                    from every artist for audio generation. Each song will be processed
                    sequentially through the audio engine.
                  </p>
                </div>
              </div>

              {/* Stats grid */}
              <div className="grid grid-cols-3 gap-3">
                <div className="flex flex-col items-center p-3 rounded-xl bg-white/[0.03] border border-zinc-200 dark:border-white/5">
                  <Music className="w-5 h-5 text-pink-400 mb-1.5" />
                  <span className="text-2xl font-bold text-white">{stats.totalSongs}</span>
                  <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Songs</span>
                </div>
                <div className="flex flex-col items-center p-3 rounded-xl bg-white/[0.03] border border-zinc-200 dark:border-white/5">
                  <Users className="w-5 h-5 text-blue-400 mb-1.5" />
                  <span className="text-2xl font-bold text-white">{stats.artistCount}</span>
                  <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Artists</span>
                </div>
                <div className="flex flex-col items-center p-3 rounded-xl bg-white/[0.03] border border-zinc-200 dark:border-white/5">
                  <Disc3 className="w-5 h-5 text-green-400 mb-1.5" />
                  <span className="text-2xl font-bold text-white">{stats.albumCount}</span>
                  <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Albums</span>
                </div>
              </div>

              {/* Estimated time */}
              {stats.totalSongs > 0 && (
                <div className="text-center text-sm text-zinc-500">
                  Estimated time: <span className="font-semibold text-zinc-300">{estimatedStr}</span>
                  <span className="text-[10px] ml-1 text-zinc-600">(~3 min/song avg)</span>
                </div>
              )}

              {stats.totalSongs === 0 && (
                <div className="text-center py-4">
                  <p className="text-sm text-zinc-500">No written songs found to generate.</p>
                  <p className="text-xs text-zinc-600 mt-1">
                    Use the Bulk Operations panel to generate lyrics first.
                  </p>
                </div>
              )}
            </>
          ) : enqueuing ? (
            <div className="flex flex-col items-center justify-center py-8 gap-4">
              <Loader2 className="w-8 h-8 text-pink-400 animate-spin" />
              <div className="text-center">
                <p className="text-sm text-white font-medium">
                  Queuing songs… {progress.current}/{progress.total}
                </p>
                <div className="w-48 h-1.5 bg-zinc-800 rounded-full mt-3 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-pink-500 to-amber-500 rounded-full transition-all duration-300"
                    style={{ width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%` }}
                  />
                </div>
              </div>
            </div>
          ) : null}
        </div>

        {/* Footer */}
        {!loading && stats && !enqueuing && (
          <div className="px-6 py-4 border-t border-zinc-200 dark:border-white/5 flex items-center justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-400 hover:text-zinc-800 dark:text-zinc-200 hover:bg-white/5 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirm}
              disabled={stats.totalSongs === 0}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all disabled:opacity-30 bg-gradient-to-r from-amber-600 to-pink-600 hover:from-amber-500 hover:to-pink-500 text-white shadow-lg shadow-amber-500/20 hover:shadow-amber-500/30 hover:scale-[1.02] active:scale-[0.98]"
            >
              <Zap className="w-4 h-4" />
              Generate All {stats.totalSongs} Song{stats.totalSongs !== 1 ? 's' : ''}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

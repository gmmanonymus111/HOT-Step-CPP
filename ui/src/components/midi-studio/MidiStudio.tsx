// MidiStudio.tsx — MIDI Studio orchestrator (audio → MIDI transcription)
//
// Pick any track from the library, transcribe it to multi-instrument MIDI,
// preview the result as a piano roll, and download the .mid.
//
// Transcription is powered by MuScriptor (github.com/muscriptor/muscriptor),
// developed by Kyutai & Mirelo — full attribution in the footer card.
// The inference engine is being ported to native C++ (ace-midi, see
// docs/plans/muscriptor-cpp-port.md); until it lands the server reports
// enginePending and the transcribe action is disabled.

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Piano, FolderOpen, Search, X, Download, Trash2, ChevronDown,
  ChevronUp, CheckCircle2, ExternalLink, Music, KeyRound, Loader2, Hammer,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../context/AuthContext';
import { songApi } from '../../services/api';
import type { Song } from '../../types';
import {
  getMidiStatus, submitTranscription, listMidiJobs, deleteMidiJob,
  getMidiFileUrl, saveHfToken, HF_MODEL_URLS, HF_TOKEN_SETTINGS_URL,
  type MidiStudioStatus, type MidiJobSummary, type MuscriptorModel,
} from '../../services/midiStudioApi';
import { PianoRoll } from './PianoRoll';

const MODEL_INFO: Array<{ id: MuscriptorModel; params: string }> = [
  { id: 'small', params: '103M' },
  { id: 'medium', params: '307M' },
  { id: 'large', params: '1.4B' },
];

export const MidiStudio: React.FC = () => {
  const { t } = useTranslation();
  const { token } = useAuth();

  const [status, setStatus] = useState<MidiStudioStatus | null>(null);

  // ── HF access token (gated weights) ──
  const [hfTokenInput, setHfTokenInput] = useState('');
  const [hfTokenBusy, setHfTokenBusy] = useState(false);
  const [hfTokenError, setHfTokenError] = useState('');

  // ── Source selection ──
  const [showLibrary, setShowLibrary] = useState(false);
  const [librarySearch, setLibrarySearch] = useState('');
  const [librarySongs, setLibrarySongs] = useState<Song[]>([]);
  const [sourceAudioUrl, setSourceAudioUrl] = useState('');
  const [sourceName, setSourceName] = useState('');
  const [sourceSongId, setSourceSongId] = useState<string | undefined>(undefined);
  const [model, setModel] = useState<MuscriptorModel>(() =>
    (localStorage.getItem('hs-midi-model') as MuscriptorModel) || 'small');

  // ── Jobs ──
  const [jobs, setJobs] = useState<MidiJobSummary[]>([]);
  const [expandedJob, setExpandedJob] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState('');

  const refreshStatus = useCallback(() => {
    getMidiStatus().then(setStatus).catch(() => {});
  }, []);
  const refreshJobs = useCallback(() => {
    listMidiJobs().then(setJobs).catch(() => {});
  }, []);

  useEffect(() => { refreshStatus(); refreshJobs(); }, [refreshStatus, refreshJobs]);

  // Load library songs when picker opens
  useEffect(() => {
    if (showLibrary && token) {
      songApi.list(token).then(({ songs }) => setLibrarySongs(songs.filter(s => s.audioUrl))).catch(() => {});
    }
  }, [showLibrary, token]);

  const filteredSongs = useMemo(() => {
    if (!librarySearch.trim()) return librarySongs;
    const q = librarySearch.toLowerCase();
    return librarySongs.filter(s =>
      s.title?.toLowerCase().includes(q) ||
      s.artistName?.toLowerCase().includes(q) ||
      s.style?.toLowerCase().includes(q)
    );
  }, [librarySongs, librarySearch]);

  const handleSelectSong = useCallback((song: Song) => {
    setSourceAudioUrl(song.audioUrl || song.audio_url || '');
    setSourceName(song.title || 'Library Track');
    setSourceSongId(song.id);
    setShowLibrary(false);
    setSubmitError('');
  }, []);

  const handleModelChange = (m: MuscriptorModel) => {
    setModel(m);
    try { localStorage.setItem('hs-midi-model', m); } catch { /* ignore */ }
  };

  const handleTranscribe = async () => {
    if (!sourceAudioUrl) return;
    setSubmitError('');
    try {
      await submitTranscription({ sourceAudioUrl, sourceFileName: sourceName, songId: sourceSongId, model });
      refreshJobs();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleSaveHfToken = async (tok: string) => {
    setHfTokenError('');
    setHfTokenBusy(true);
    try {
      await saveHfToken(tok);
      setHfTokenInput('');
      refreshStatus();
    } catch (err) {
      setHfTokenError(err instanceof Error ? err.message : String(err));
    } finally {
      setHfTokenBusy(false);
    }
  };

  const handleDelete = async (jobId: string) => {
    try {
      await deleteMidiJob(jobId);
      if (expandedJob === jobId) setExpandedJob(null);
      setJobs(js => js.filter(j => j.id !== jobId));
    } catch { /* refresh will reconcile */ }
    refreshJobs();
  };

  const enginePending = status?.enginePending ?? true;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-8 flex flex-col gap-6">

        {/* ── Header ── */}
        <div>
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-white flex items-center gap-3">
            <Piano size={26} className="text-purple-500 dark:text-purple-400" />
            {t('midiStudio.title')}
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">{t('midiStudio.subtitle')}</p>
        </div>

        {/* ── Engine port banner ── */}
        {enginePending && (
          <div className="rounded-xl border border-sky-300/60 dark:border-sky-500/20 bg-sky-50 dark:bg-sky-500/5 p-4 flex items-start gap-3">
            <Hammer size={18} className="text-sky-500 mt-0.5 flex-shrink-0" />
            <div className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
              <div className="text-sm font-semibold text-zinc-900 dark:text-white mb-1">{t('midiStudio.portTitle')}</div>
              {t('midiStudio.portBody')}
            </div>
          </div>
        )}

        {/* ── Hugging Face model access (weights are gated) ── */}
        {status && (
          <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4">
            <div className="text-sm font-semibold text-zinc-900 dark:text-white mb-1 flex items-center gap-2">
              <KeyRound size={15} className="text-purple-500 dark:text-purple-400" />
              {t('midiStudio.hfAccessTitle')}
              {status.hfTokenSet && (
                <span className="flex items-center gap-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 size={12} /> {t('midiStudio.hfTokenSaved')}
                </span>
              )}
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">{t('midiStudio.hfAccessBody')}</p>
            <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs">
              {(Object.keys(HF_MODEL_URLS) as MuscriptorModel[]).map(m => (
                <a key={m} href={HF_MODEL_URLS[m]} target="_blank" rel="noreferrer"
                  className="flex items-center gap-1 text-purple-600 dark:text-purple-400 hover:underline capitalize">
                  <ExternalLink size={11} /> {m}
                </a>
              ))}
              <a href={HF_TOKEN_SETTINGS_URL} target="_blank" rel="noreferrer"
                className="flex items-center gap-1 text-purple-600 dark:text-purple-400 hover:underline">
                <ExternalLink size={11} /> {t('midiStudio.hfGetToken')}
              </a>
            </div>
            <div className="mt-3 flex items-center gap-2 flex-wrap">
              <input
                type="password"
                value={hfTokenInput}
                onChange={e => setHfTokenInput(e.target.value)}
                placeholder={t('midiStudio.hfTokenPlaceholder')}
                className="flex-1 min-w-[220px] px-3 py-2 rounded-lg text-xs bg-zinc-100 dark:bg-black/30 border border-zinc-200 dark:border-white/10 text-zinc-900 dark:text-white placeholder-zinc-400 outline-none focus:border-purple-500/40"
              />
              <button
                onClick={() => handleSaveHfToken(hfTokenInput)}
                disabled={hfTokenBusy || !hfTokenInput.trim()}
                className="px-4 py-2 rounded-lg text-xs font-semibold bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {hfTokenBusy ? <Loader2 size={13} className="animate-spin" /> : t('midiStudio.hfTokenSave')}
              </button>
              {status.hfTokenSet && (
                <button
                  onClick={() => handleSaveHfToken('')}
                  disabled={hfTokenBusy}
                  className="px-3 py-2 rounded-lg text-xs font-medium text-zinc-500 dark:text-zinc-400 hover:text-red-500 border border-zinc-200 dark:border-white/10 transition-colors"
                >
                  {t('midiStudio.hfTokenClear')}
                </button>
              )}
            </div>
            {hfTokenError && <div className="mt-2 text-xs text-red-500 dark:text-red-400">{hfTokenError}</div>}
            <p className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-2">{t('midiStudio.hfTokenNote')}</p>
          </div>
        )}

        {/* ── New transcription ── */}
        <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4">
          <div className="text-sm font-semibold text-zinc-900 dark:text-white mb-3">{t('midiStudio.newTranscription')}</div>

          {/* Source picker */}
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={() => setShowLibrary(v => !v)}
              className="px-3 py-2 rounded-lg text-xs font-medium flex items-center gap-2 bg-zinc-100 dark:bg-white/5 hover:bg-zinc-200 dark:hover:bg-white/10 text-zinc-700 dark:text-zinc-300 transition-colors"
            >
              <FolderOpen size={14} /> {t('midiStudio.chooseFromLibrary')}
            </button>
            {sourceName ? (
              <span className="flex items-center gap-2 text-sm text-zinc-800 dark:text-zinc-200">
                <Music size={14} className="text-purple-500 dark:text-purple-400" />
                {sourceName}
                <button onClick={() => { setSourceAudioUrl(''); setSourceName(''); setSourceSongId(undefined); }}
                  className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200">
                  <X size={14} />
                </button>
              </span>
            ) : (
              <span className="text-xs text-zinc-400 dark:text-zinc-500">{t('midiStudio.noTrackSelected')}</span>
            )}
          </div>

          {/* Inline library panel */}
          {showLibrary && (
            <div className="mt-3 rounded-lg border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20">
              <div className="p-2 border-b border-zinc-200 dark:border-white/5 flex items-center gap-2">
                <Search size={14} className="text-zinc-400 flex-shrink-0" />
                <input
                  value={librarySearch}
                  onChange={e => setLibrarySearch(e.target.value)}
                  placeholder={t('midiStudio.searchLibrary')}
                  className="flex-1 bg-transparent text-sm text-zinc-900 dark:text-white placeholder-zinc-400 outline-none"
                />
              </div>
              <div className="max-h-56 overflow-y-auto">
                {filteredSongs.length === 0 && (
                  <div className="px-3 py-4 text-xs text-zinc-400 dark:text-zinc-500">{t('midiStudio.libraryEmpty')}</div>
                )}
                {filteredSongs.map(song => (
                  <button
                    key={song.id}
                    onClick={() => handleSelectSong(song)}
                    className="w-full text-left px-3 py-2 flex items-center justify-between gap-3 hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors"
                  >
                    <span className="text-sm text-zinc-800 dark:text-zinc-200 truncate">{song.title || t('midiStudio.untitled')}</span>
                    <span className="text-[11px] text-zinc-400 flex-shrink-0">
                      {song.duration ? `${Math.floor(Number(song.duration) / 60)}:${String(Math.floor(Number(song.duration) % 60)).padStart(2, '0')}` : ''}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Model select */}
          <div className="mt-4">
            <div className="text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-2">{t('midiStudio.model')}</div>
            <div className="flex gap-2 flex-wrap">
              {MODEL_INFO.map(mi => (
                <button
                  key={mi.id}
                  onClick={() => handleModelChange(mi.id)}
                  className={`px-3 py-2 rounded-lg text-xs border transition-colors ${
                    model === mi.id
                      ? 'border-purple-500/60 bg-purple-500/10 text-purple-600 dark:text-purple-300'
                      : 'border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-400 hover:border-zinc-300 dark:hover:border-white/20'
                  }`}
                >
                  <span className="font-semibold capitalize">{mi.id}</span>
                  <span className="opacity-60 ml-1.5">{mi.params}</span>
                  <span className="block text-[10px] opacity-60 mt-0.5">{t(`midiStudio.model_${mi.id}`)}</span>
                </button>
              ))}
            </div>
            <p className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-2">{t('midiStudio.firstRunNote')}</p>
          </div>

          <div className="mt-4 flex items-center gap-3">
            <button
              onClick={handleTranscribe}
              disabled={enginePending || !sourceAudioUrl}
              title={enginePending ? t('midiStudio.portTitle') : undefined}
              className="px-5 py-2.5 rounded-lg text-sm font-semibold bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Piano size={16} /> {t('midiStudio.transcribe')}
            </button>
            {enginePending && <span className="text-xs text-sky-600 dark:text-sky-400">{t('midiStudio.portShort')}</span>}
            {submitError && <span className="text-xs text-red-500 dark:text-red-400">{submitError}</span>}
          </div>
        </div>

        {/* ── Transcriptions list ── */}
        <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4">
          <div className="text-sm font-semibold text-zinc-900 dark:text-white mb-3">{t('midiStudio.transcriptions')}</div>
          {jobs.length === 0 && (
            <div className="text-xs text-zinc-400 dark:text-zinc-500 py-2">{t('midiStudio.noJobs')}</div>
          )}
          <div className="flex flex-col gap-2">
            {jobs.map(job => (
              <div key={job.id} className="rounded-lg border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20 px-3 py-2.5">
                <div className="flex items-center gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-zinc-800 dark:text-zinc-200 truncate">{job.sourceFileName}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-200 dark:bg-white/10 text-zinc-500 dark:text-zinc-400 capitalize flex-shrink-0">{job.model}</span>
                    </div>
                    <div className="flex items-center gap-1.5 mt-0.5 text-[11px] text-emerald-600 dark:text-emerald-400">
                      <CheckCircle2 size={11} />
                      {t('midiStudio.doneSummary', { notes: job.noteCount, seconds: Math.round(job.durationSec) })}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 flex-shrink-0">
                    <button
                      onClick={() => setExpandedJob(e => e === job.id ? null : job.id)}
                      className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors"
                      title={t('midiStudio.preview')}
                    >
                      {expandedJob === job.id ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
                    </button>
                    <a
                      href={getMidiFileUrl(job.id)}
                      className="p-1.5 rounded-md text-zinc-500 hover:text-purple-600 dark:hover:text-purple-400 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors"
                      title={t('midiStudio.downloadMid')}
                      download
                    >
                      <Download size={15} />
                    </a>
                    <button
                      onClick={() => handleDelete(job.id)}
                      className="p-1.5 rounded-md text-zinc-500 hover:text-red-500 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors"
                      title={t('midiStudio.delete')}
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                </div>
                {expandedJob === job.id && <PianoRoll jobId={job.id} />}
              </div>
            ))}
          </div>
        </div>

        {/* ── Attribution ── */}
        <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20 p-4 text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">
          <div className="font-semibold text-zinc-700 dark:text-zinc-300 mb-1">{t('midiStudio.creditsTitle')}</div>
          <p>
            {t('midiStudio.creditsBody')}{' '}
            <span className="text-zinc-700 dark:text-zinc-300">
              Simon Rouard, Michael Krause, Axel Roebel, Carl-Johann Simon-Gabriel, Alexandre Défossez
            </span>.
          </p>
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
            <a href="https://github.com/muscriptor/muscriptor" target="_blank" rel="noreferrer"
              className="flex items-center gap-1 text-purple-600 dark:text-purple-400 hover:underline">
              <ExternalLink size={11} /> GitHub
            </a>
            <a href="https://arxiv.org/abs/2607.08168" target="_blank" rel="noreferrer"
              className="flex items-center gap-1 text-purple-600 dark:text-purple-400 hover:underline">
              <ExternalLink size={11} /> {t('midiStudio.paper')}
            </a>
            <a href="https://muscriptor.kyutai.org/" target="_blank" rel="noreferrer"
              className="flex items-center gap-1 text-purple-600 dark:text-purple-400 hover:underline">
              <ExternalLink size={11} /> {t('midiStudio.demo')}
            </a>
          </div>
          <p className="mt-2 opacity-80">{t('midiStudio.licenseNote')}</p>
        </div>
      </div>
    </div>
  );
};

export default MidiStudio;

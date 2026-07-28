// BatchImportWizard.tsx — "Import multiple…" batch pipeline entry point
//
// Step 1: pick a root folder, list its immediate subdirectories, scan each
// SEQUENTIALLY for audio files (avoid hammering the server), render a
// checkbox row per folder. Step 2: pick which of the five pipeline stages to
// run. Submit starts one server-side pipeline and hands off to the Monitor
// phase. Stage OPTIONS come from the (not-yet-built) defaults config page —
// this wizard only picks which stages run, per plan §5.1.

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AlertTriangle, FolderOpen, FolderPlus, Loader2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { adapterApi } from '../../services/api';
import { PIPELINE_STAGES, scanPreview, type PipelineStage } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { FolderPicker } from './FolderPicker';

interface FolderRow {
  path: string;
  name: string;
  audioFiles: number | null;   // null = not scanned yet
  scanning: boolean;
  scanError: string | null;
  alreadyImported: boolean;
  checked: boolean;
}

const STAGE_LABEL_KEYS: Record<PipelineStage, string> = {
  label: 'trainingStudio.batch.stageLabel',
  build: 'trainingStudio.batch.stageBuild',
  preprocess: 'trainingStudio.batch.stagePreprocess',
  'train-dit': 'trainingStudio.batch.stageTrainDit',
  'train-lm': 'trainingStudio.batch.stageTrainLm',
};

function defaultStages(): Record<PipelineStage, boolean> {
  return Object.fromEntries(PIPELINE_STAGES.map(s => [s, true])) as Record<PipelineStage, boolean>;
}

function basename(dir: string): string {
  return dir.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? dir;
}

interface BatchImportWizardProps {
  open: boolean;
  onClose: () => void;
}

export const BatchImportWizard: React.FC<BatchImportWizardProps> = ({ open, onClose }) => {
  const { t } = useTranslation();
  const datasets = useTrainingStore(s => s.datasets);
  const startPipeline = useTrainingStore(s => s.startPipeline);
  const setPhase = useTrainingStore(s => s.setPhase);

  const [rootDir, setRootDir] = useState('');
  const [rootPickerOpen, setRootPickerOpen] = useState(false);
  const [addPickerOpen, setAddPickerOpen] = useState(false);
  const [rows, setRows] = useState<FolderRow[]>([]);
  const [loadingRoot, setLoadingRoot] = useState(false);
  const [rootError, setRootError] = useState<string | null>(null);
  const [stages, setStages] = useState<Record<PipelineStage, boolean>>(defaultStages);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Bumped on every root reload / modal open so a stale sequential scan loop
  // (and any in-flight browse/scan) stops touching state after the user moves
  // on to a different root or closes and reopens the wizard.
  const genRef = useRef(0);

  useEffect(() => {
    if (!open) return;
    genRef.current++;
    setRootDir(''); setRows([]); setRootError(null); setSubmitError(null); setStages(defaultStages());
  }, [open]);

  const alreadyImported = useCallback((dir: string) => {
    const lower = dir.toLowerCase();
    return datasets.some(d => d.sourceDir.toLowerCase() === lower);
  }, [datasets]);

  const scanRow = useCallback(async (gen: number, path: string) => {
    if (gen !== genRef.current) return;
    setRows(prev => prev.map(r => (r.path === path ? { ...r, scanning: true, scanError: null } : r)));
    try {
      const preview = await scanPreview(path, true);
      if (gen !== genRef.current) return;
      setRows(prev => prev.map(r => (r.path === path
        ? { ...r, scanning: false, audioFiles: preview.audioFiles, checked: r.checked && preview.audioFiles > 0 }
        : r)));
    } catch (err) {
      if (gen !== genRef.current) return;
      setRows(prev => prev.map(r => (r.path === path
        ? {
            ...r, scanning: false, audioFiles: 0, checked: false,
            scanError: err instanceof Error ? err.message : 'Scan failed',
          }
        : r)));
    }
  }, []);

  // Scans one row at a time, in order, so a 20-folder batch does not fire 20
  // concurrent recursive-readdirs at the server.
  const scanSequentially = useCallback(async (gen: number, paths: string[]) => {
    for (const path of paths) {
      if (gen !== genRef.current) return;
      await scanRow(gen, path);
    }
  }, [scanRow]);

  const loadRoot = useCallback(async (dir: string) => {
    const gen = ++genRef.current;
    setRootDir(dir);
    setRootError(null);
    setLoadingRoot(true);
    setRows([]);
    try {
      const result = await adapterApi.browse(dir, 'trainingAudio');
      if (gen !== genRef.current) return;
      const subdirs = result.entries.filter(e => e.type === 'dir' && e.name !== '..');
      const initial: FolderRow[] = subdirs.map(e => ({
        path: e.path,
        name: e.name,
        audioFiles: null,
        scanning: false,
        scanError: null,
        alreadyImported: alreadyImported(e.path),
        checked: true,
      }));
      setRows(initial);
      setLoadingRoot(false);
      void scanSequentially(gen, initial.map(r => r.path));
    } catch (err) {
      if (gen !== genRef.current) return;
      setRootError(err instanceof Error ? err.message : 'Failed to list directory');
      setLoadingRoot(false);
    }
  }, [alreadyImported, scanSequentially]);

  const addFolder = useCallback((path: string) => {
    setRows(prev => {
      if (prev.some(r => r.path === path)) return prev;
      return [...prev, {
        path,
        name: basename(path),
        audioFiles: null,
        scanning: false,
        scanError: null,
        alreadyImported: alreadyImported(path),
        checked: true,
      }];
    });
    void scanRow(genRef.current, path);
  }, [alreadyImported, scanRow]);

  if (!open) return null;

  const toggleRow = (path: string) => {
    setRows(prev => prev.map(r => (r.path === path && r.audioFiles !== 0 ? { ...r, checked: !r.checked } : r)));
  };

  const toggleStage = (stage: PipelineStage) => {
    setStages(prev => ({ ...prev, [stage]: !prev[stage] }));
  };

  const selectedRows = rows.filter(r => r.checked && r.audioFiles !== 0);
  const selectedStages = PIPELINE_STAGES.filter(s => stages[s]);
  const canSubmit = selectedRows.length > 0 && selectedStages.length > 0 && !submitting;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      await startPipeline({
        folders: selectedRows.map(r => ({ sourceDir: r.path })),
        stages: selectedStages,
      });
      setPhase('monitor');
      onClose();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Failed to start pipeline');
    } finally {
      setSubmitting(false);
    }
  };

  const field = 'w-full rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-amber-500';

  return createPortal(
    <>
      <div
        className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
        onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      >
        <div className="w-[680px] max-h-[85vh] flex flex-col overflow-hidden rounded-xl border border-zinc-200 dark:border-white/10 bg-white dark:bg-suno-card shadow-2xl">
          <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-200 dark:border-white/5">
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">{t('trainingStudio.batch.title')}</h3>
            <button onClick={onClose} className="text-zinc-500 hover:text-zinc-800 dark:hover:text-white transition-colors">
              <X size={16} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
            {/* Root folder */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.batch.rootFolder')}</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={rootDir}
                  onChange={(e) => setRootDir(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && rootDir.trim()) void loadRoot(rootDir.trim()); }}
                  placeholder={t('trainingStudio.wizard.pathPlaceholder')}
                  className={`${field} font-mono text-xs`}
                />
                <button
                  onClick={() => setRootPickerOpen(true)}
                  className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors whitespace-nowrap"
                >
                  <FolderOpen size={14} /> {t('trainingStudio.wizard.browse')}
                </button>
              </div>
              <div className="text-[11px] text-zinc-500">{t('trainingStudio.batch.rootHint')}</div>
            </div>

            {rootError && (
              <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">{rootError}</div>
            )}

            {loadingRoot && (
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <Loader2 size={14} className="animate-spin" /> {t('trainingStudio.wizard.scanning')}
              </div>
            )}

            {/* Folder rows */}
            {rows.length > 0 && (
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">
                    {t('trainingStudio.batch.folders', { count: rows.length })}
                  </label>
                  <button
                    onClick={() => setAddPickerOpen(true)}
                    className="flex items-center gap-1 text-[11px] font-semibold text-amber-600 dark:text-amber-400 hover:underline"
                  >
                    <FolderPlus size={12} /> {t('trainingStudio.batch.addFolder')}
                  </button>
                </div>
                <div className="rounded-xl border border-zinc-200 dark:border-white/5 divide-y divide-zinc-200 dark:divide-white/5 max-h-64 overflow-y-auto">
                  {rows.map(row => {
                    const disabled = row.audioFiles === 0;
                    return (
                      <label
                        key={row.path}
                        className={`flex items-center gap-2.5 px-3 py-2 text-xs ${disabled ? 'opacity-50' : 'cursor-pointer hover:bg-black/5 dark:hover:bg-white/5'}`}
                      >
                        <input
                          type="checkbox"
                          checked={row.checked && !disabled}
                          disabled={disabled}
                          onChange={() => toggleRow(row.path)}
                          className="accent-amber-500 flex-shrink-0"
                        />
                        <span className="flex-1 min-w-0 truncate text-zinc-700 dark:text-zinc-300 font-medium" title={row.path}>{row.name}</span>
                        {row.scanning ? (
                          <Loader2 size={12} className="animate-spin text-zinc-400 flex-shrink-0" />
                        ) : row.scanError ? (
                          <span className="text-red-500 flex-shrink-0 truncate max-w-[160px]" title={row.scanError}>{row.scanError}</span>
                        ) : (
                          <span className="text-zinc-500 flex-shrink-0 tabular-nums">
                            {t('trainingStudio.batch.audioCount', { count: row.audioFiles ?? 0 })}
                          </span>
                        )}
                        {row.alreadyImported && (
                          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-sky-500 bg-sky-500/10 border-sky-500/20 flex-shrink-0">
                            {t('trainingStudio.batch.alreadyImported')}
                          </span>
                        )}
                      </label>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Stages */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.batch.stages')}</label>
              <div className="flex flex-wrap gap-2">
                {PIPELINE_STAGES.map(stage => (
                  <label
                    key={stage}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-zinc-200 dark:border-white/10 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={stages[stage]}
                      onChange={() => toggleStage(stage)}
                      className="accent-amber-500"
                    />
                    {t(STAGE_LABEL_KEYS[stage])}
                  </label>
                ))}
              </div>
              <div className="flex items-start gap-2 text-[11px] text-zinc-500">
                <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
                {t('trainingStudio.batch.defaultsHint')}
              </div>
            </div>

            {submitError && (
              <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">{submitError}</div>
            )}
          </div>

          <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20">
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-zinc-200 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-300 dark:hover:bg-zinc-600 transition-colors"
            >
              {t('trainingStudio.wizard.cancel')}
            </button>
            <button
              onClick={handleSubmit}
              disabled={!canSubmit}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {submitting && <Loader2 size={13} className="animate-spin" />}
              {t('trainingStudio.batch.start')}
            </button>
          </div>
        </div>
      </div>

      <FolderPicker
        open={rootPickerOpen}
        onClose={() => setRootPickerOpen(false)}
        onSelect={(p) => { setRootPickerOpen(false); void loadRoot(p); }}
      />
      <FolderPicker
        open={addPickerOpen}
        onClose={() => setAddPickerOpen(false)}
        onSelect={(p) => { setAddPickerOpen(false); addFolder(p); }}
      />
    </>,
    document.body,
  );
};

export default BatchImportWizard;

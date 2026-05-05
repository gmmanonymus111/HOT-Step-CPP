// MasteringSection.tsx — Reference-based mastering controls
//
// Collapsible accordion section with:
// - Toggle: enable/disable reference mastering
// - Reference track selector with upload support

import React, { useState, useEffect, useCallback } from 'react';
import { ChevronDown, Upload, Trash2, Sparkles, Music2 } from 'lucide-react';
import { masteringApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';

interface MasteringSectionProps {
  masteringEnabled: boolean;
  onMasteringEnabledChange: (v: boolean) => void;
  masteringReference: string;
  onMasteringReferenceChange: (v: string) => void;
  timbreReference: boolean;
  onTimbreReferenceChange: (v: boolean) => void;
  timbreAudioPath?: string;
}

interface ReferenceTrack {
  name: string;
  size: number;
  url: string;
}

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

export const MasteringSection: React.FC<MasteringSectionProps> = ({
  masteringEnabled,
  onMasteringEnabledChange,
  masteringReference,
  onMasteringReferenceChange,
  timbreReference,
  onTimbreReferenceChange,
  timbreAudioPath = '',
}) => {
  const { token } = useAuth();
  const [open, setOpen] = useState(false);
  const [references, setReferences] = useState<ReferenceTrack[]>([]);
  const [uploading, setUploading] = useState(false);

  // Load references on mount
  useEffect(() => {
    masteringApi.listReferences()
      .then(data => setReferences(data.references))
      .catch(() => {});
  }, []);

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !token) return;

    try {
      setUploading(true);
      const result = await masteringApi.uploadReference(file, token);
      onMasteringReferenceChange(result.name);
      // Refresh list
      const data = await masteringApi.listReferences();
      setReferences(data.references);
    } catch (err) {
      console.error('[Mastering] Upload failed:', err);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  }, [token, onMasteringReferenceChange]);

  const handleDelete = useCallback(async (name: string) => {
    if (!token) return;
    try {
      await masteringApi.deleteReference(name, token);
      if (masteringReference === name) onMasteringReferenceChange('');
      const data = await masteringApi.listReferences();
      setReferences(data.references);
    } catch (err) {
      console.error('[Mastering] Delete failed:', err);
    }
  }, [token, masteringReference, onMasteringReferenceChange]);

  return (
    <div className="space-y-1 pt-3 border-t border-zinc-200 dark:border-white/5">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400 uppercase tracking-wider">Mastering</span>
          <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
            masteringEnabled ? 'bg-amber-500/20 text-amber-400' : 'bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-400'
          }`}>
            {masteringEnabled ? 'ON' : 'OFF'}
          </span>
        </div>
        <ChevronDown size={14} className={`text-zinc-500 transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="px-3 pb-3 space-y-3">
          {/* Enable toggle */}
          <label className="flex items-center gap-2.5 cursor-pointer">
            <input
              type="checkbox"
              checked={masteringEnabled}
              onChange={e => onMasteringEnabledChange(e.target.checked)}
              className="rounded border-zinc-600 bg-zinc-100 dark:bg-zinc-800 text-amber-500 focus:ring-amber-500/20"
            />
            <span className="text-sm text-zinc-600 dark:text-zinc-400">Apply Reference Mastering</span>
            <Sparkles size={14} className="text-amber-400 ml-auto" />
          </label>

          {masteringEnabled && (
            <>
              {/* Reference selector */}
              <div>
                <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
                  Reference Track
                </label>
                {references.length > 0 ? (
                  <select
                    className="w-full px-3 py-2 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 outline-none transition-colors cursor-pointer"
                    value={masteringReference}
                    onChange={e => onMasteringReferenceChange(e.target.value)}
                  >
                    <option value="">Select a reference...</option>
                    {references.map(r => (
                      <option key={r.name} value={r.name}>
                        {r.name} ({formatFileSize(r.size)})
                      </option>
                    ))}
                  </select>
                ) : (
                  <div className="text-xs text-zinc-500 italic px-1">
                    No reference tracks uploaded yet
                  </div>
                )}
              </div>

              {/* Selected reference info + delete */}
              {masteringReference && (
                <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-amber-500/5 border border-amber-500/10">
                  <Music2 size={14} className="text-amber-400 flex-shrink-0" />
                  <span className="text-xs text-amber-300 truncate flex-1">{masteringReference}</span>
                  <button
                    onClick={() => handleDelete(masteringReference)}
                    className="p-1 rounded hover:bg-red-500/10 text-zinc-500 hover:text-red-400 transition-colors flex-shrink-0"
                    title="Delete reference"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              )}

              {/* Upload button */}
              <div className="flex items-center gap-2">
                <input
                  type="file"
                  accept="audio/*"
                  id="mastering-ref-upload"
                  className="hidden"
                  onChange={handleUpload}
                />
                <label
                  htmlFor="mastering-ref-upload"
                  className={`flex items-center gap-2 px-3 py-2 text-xs font-semibold rounded-xl border cursor-pointer transition-all ${
                    uploading
                      ? 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 border-zinc-200 dark:border-white/5 cursor-wait'
                      : 'bg-white dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400 border-zinc-300 dark:border-white/10 hover:border-amber-500/30 hover:text-amber-400'
                  }`}
                >
                  {uploading ? (
                    <><span className="w-3 h-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" /> Uploading...</>
                  ) : (
                    <><Upload size={14} /> Upload Reference</>
                  )}
                </label>
              </div>

              {/* Timbre reference toggle */}
              {masteringReference && (
                timbreAudioPath ? (
                  <div className="flex items-center gap-1.5 mt-1 px-2 py-1.5 rounded-lg bg-teal-500/5 border border-teal-500/10">
                    <Music2 size={14} className="text-teal-400" />
                    <span className="text-[10px] text-teal-400">Timbre: using dedicated reference ({timbreAudioPath.split(/[\\/]/).pop()})</span>
                  </div>
                ) : (
                  <label className="flex items-center gap-2.5 cursor-pointer mt-1">
                    <input
                      type="checkbox"
                      checked={timbreReference}
                      onChange={e => onTimbreReferenceChange(e.target.checked)}
                      className="rounded border-zinc-600 bg-zinc-100 dark:bg-zinc-800 text-teal-500 focus:ring-teal-500/20"
                    />
                    <span className="text-sm text-zinc-600 dark:text-zinc-400">Also use as timbre reference</span>
                    <Music2 size={14} className="text-teal-400 ml-auto" />
                  </label>
                )
              )}
              {timbreReference && masteringReference && !timbreAudioPath && (
                <p className="text-[10px] text-zinc-600 leading-relaxed">
                  The reference track will be VAE-encoded and fed into the timbre conditioning pipeline,
                  guiding the generation&apos;s tone and texture to match the reference.
                </p>
              )}

              {/* Info */}
              <p className="text-[10px] text-zinc-600 leading-relaxed">
                The generated audio will be mastered to match the RMS level, frequency spectrum,
                and dynamic characteristics of the reference track.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
};

// useStreamGeneration.ts — React hook for streaming audio preview via SSE
//
// Connects to GET /api/generate/stream/:jobId once a generation job is running.
// Receives 'preview' events with WAV file paths, fetches them, decodes to
// AudioBuffers, and queues them for gapless sequential playback.
//
// Exposes: status, previews[], play(), pause(), stop()

import { useState, useEffect, useRef, useCallback } from 'react';

export interface StreamPreview {
  url: string;
  step: number;
  totalSteps: number;
  slot: number;
  /** AudioBuffer decoded from the preview WAV (null if not yet loaded) */
  buffer: AudioBuffer | null;
  /** Whether this preview has been played */
  played: boolean;
}

export interface StreamStatus {
  status: string;
  stage: string;
  progress: number;
}

export interface StreamGenerationState {
  /** Whether the SSE connection is active */
  connected: boolean;
  /** Latest status from the server */
  status: StreamStatus | null;
  /** All received previews in order */
  previews: StreamPreview[];
  /** Whether audio is currently playing */
  playing: boolean;
  /** Whether the generation is complete */
  done: boolean;
  /** Error message if generation failed */
  error: string | null;
  /** Final result (audio URLs etc.) when done */
  result: any | null;
}

export function useStreamGeneration(jobId: string | null) {
  const [state, setState] = useState<StreamGenerationState>({
    connected: false,
    status: null,
    previews: [],
    playing: false,
    done: false,
    error: null,
    result: null,
  });

  const esRef = useRef<EventSource | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const playQueueRef = useRef<AudioBuffer[]>([]);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const isPlayingRef = useRef(false);
  const nextPlayTimeRef = useRef(0);
  const previewsRef = useRef<StreamPreview[]>([]);

  // Fetch and decode a preview WAV file
  const fetchPreview = useCallback(async (url: string): Promise<AudioBuffer | null> => {
    try {
      // Server sends URLs relative to the audio static middleware (e.g. /audio/stream/preview.wav)
      const res = await fetch(url);
      if (!res.ok) return null;
      const arrayBuffer = await res.arrayBuffer();

      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      return await audioCtxRef.current.decodeAudioData(arrayBuffer);
    } catch (err) {
      console.warn('[StreamGen] Failed to decode preview:', url, err);
      return null;
    }
  }, []);

  // Play next buffer in queue
  const playNextBuffer = useCallback(() => {
    if (!isPlayingRef.current || playQueueRef.current.length === 0) return;
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext();
    }
    const ctx = audioCtxRef.current;
    const buffer = playQueueRef.current.shift()!;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    // Schedule at the next play time for gapless playback
    const startTime = Math.max(ctx.currentTime, nextPlayTimeRef.current);
    source.start(startTime);
    nextPlayTimeRef.current = startTime + buffer.duration;

    currentSourceRef.current = source;
    source.onended = () => {
      currentSourceRef.current = null;
      if (isPlayingRef.current && playQueueRef.current.length > 0) {
        playNextBuffer();
      } else if (playQueueRef.current.length === 0) {
        setState(prev => ({ ...prev, playing: false }));
        isPlayingRef.current = false;
      }
    };
  }, []);

  // Controls
  const play = useCallback(() => {
    if (isPlayingRef.current) return;
    isPlayingRef.current = true;
    setState(prev => ({ ...prev, playing: true }));

    // Resume AudioContext if suspended
    if (audioCtxRef.current?.state === 'suspended') {
      audioCtxRef.current.resume();
    }

    // Reset play time
    if (audioCtxRef.current) {
      nextPlayTimeRef.current = audioCtxRef.current.currentTime;
    }

    // Queue all unplayed previews
    for (const p of previewsRef.current) {
      if (p.buffer && !p.played) {
        playQueueRef.current.push(p.buffer);
        p.played = true;
      }
    }

    if (playQueueRef.current.length > 0) {
      playNextBuffer();
    }
  }, [playNextBuffer]);

  const pause = useCallback(() => {
    isPlayingRef.current = false;
    setState(prev => ({ ...prev, playing: false }));
    if (currentSourceRef.current) {
      try { currentSourceRef.current.stop(); } catch { /* already stopped */ }
      currentSourceRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    pause();
    playQueueRef.current = [];
    nextPlayTimeRef.current = 0;
    // Reset all previews to unplayed
    for (const p of previewsRef.current) {
      p.played = false;
    }
  }, [pause]);

  // SSE connection
  useEffect(() => {
    if (!jobId) return;

    const es = new EventSource(`/api/generate/stream/${jobId}`);
    esRef.current = es;

    es.addEventListener('open', () => {
      setState(prev => ({ ...prev, connected: true }));
    });

    es.addEventListener('status', (event) => {
      try {
        const data = JSON.parse(event.data);
        setState(prev => ({
          ...prev,
          status: {
            status: data.status,
            stage: data.stage || '',
            progress: data.progress || 0,
          },
        }));
      } catch { /* ignore */ }
    });

    es.addEventListener('preview', async (event) => {
      try {
        const data = JSON.parse(event.data);
        const preview: StreamPreview = {
          url: data.url,
          step: data.step,
          totalSteps: data.totalSteps,
          slot: data.slot,
          buffer: null,
          played: false,
        };

        // Add to list immediately (buffer loading is async)
        previewsRef.current = [...previewsRef.current, preview];
        setState(prev => ({
          ...prev,
          previews: [...previewsRef.current],
        }));

        // Fetch and decode the WAV
        const buffer = await fetchPreview(data.url);
        if (buffer) {
          preview.buffer = buffer;

          // If currently playing, auto-queue this new buffer
          if (isPlayingRef.current) {
            playQueueRef.current.push(buffer);
            preview.played = true;
            // If nothing is currently playing, start it
            if (!currentSourceRef.current) {
              playNextBuffer();
            }
          }

          setState(prev => ({
            ...prev,
            previews: [...previewsRef.current],
          }));
        }
      } catch { /* ignore */ }
    });

    es.addEventListener('done', (event) => {
      try {
        const data = JSON.parse(event.data);
        setState(prev => ({
          ...prev,
          done: true,
          result: data.result,
        }));
      } catch { /* ignore */ }
      es.close();
    });

    es.addEventListener('error', (event) => {
      // Check if it's a server-sent error event vs connection error
      const messageEvent = event as MessageEvent;
      if (messageEvent.data) {
        try {
          const data = JSON.parse(messageEvent.data);
          setState(prev => ({
            ...prev,
            error: data.error || 'Generation failed',
            done: true,
          }));
        } catch { /* ignore */ }
      }
      setState(prev => ({ ...prev, connected: false }));
      es.close();
    });

    return () => {
      es.close();
      esRef.current = null;
      setState(prev => ({ ...prev, connected: false }));
    };
  }, [jobId, fetchPreview, playNextBuffer]);

  // Cleanup AudioContext on unmount
  useEffect(() => {
    return () => {
      if (currentSourceRef.current) {
        try { currentSourceRef.current.stop(); } catch { /* */ }
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
      }
    };
  }, []);

  return {
    ...state,
    play,
    pause,
    stop,
  };
}

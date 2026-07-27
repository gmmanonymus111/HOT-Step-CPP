// useTrainingStream.ts — EventSource hook for training job progress
//
// The server replays the whole event buffer on connect, so every handler in
// the store must be idempotent (sample rows are replaced by id, never
// appended). EventSource reconnects on its own; we only close on unmount.

import { useEffect } from 'react';
import { useTrainingStore } from '../../stores/trainingStore';
import { jobStreamUrl, type TrainingStreamEvent } from '../../services/trainingApi';

export function useTrainingStream(jobId: string | null) {
  const applyStreamEvent = useTrainingStore(s => s.applyStreamEvent);

  useEffect(() => {
    if (!jobId) return;
    const es = new EventSource(jobStreamUrl(jobId));
    es.onmessage = (e) => {
      try { applyStreamEvent(JSON.parse(e.data) as TrainingStreamEvent); } catch { /* ignore malformed frame */ }
    };
    es.onerror = () => { /* EventSource auto-reconnects; buffer replay makes it idempotent */ };
    return () => es.close();
  }, [jobId, applyStreamEvent]);
}

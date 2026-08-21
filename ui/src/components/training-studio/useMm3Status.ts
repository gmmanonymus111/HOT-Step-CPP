// useMm3Status.ts — the MiniMax-Music3 dataset status both MM3 cards read.
//
// Its own file so neither card exports a non-component (fast refresh), and so
// the codes card and the train card can never disagree about what exists on
// disk: one fetch, one shape, two consumers.

import { useEffect, useState } from 'react';

import * as trainingApi from '../../services/trainingApi';
import type { Mm3Status } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';

/** Re-reads when one of OUR jobs reaches a terminal state — exactly when the
 *  codes cache on disk may have changed. */
export function useMm3Status(datasetId: string): {
  status: Mm3Status | null; error: string | null;
} {
  const activeJob = useTrainingStore(s => s.activeJob);
  const [status, setStatus] = useState<Mm3Status | null>(null);
  const [error, setError] = useState<string | null>(null);

  const kind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const running = jobStatus === 'queued' || jobStatus === 'running';
  const mine = kind === 'mm3-codes' || kind === 'mm3-train-lm';
  const finishedKey = mine && !running ? `${activeJob?.id ?? ''}:${jobStatus ?? ''}` : '';

  useEffect(() => {
    let cancelled = false;
    trainingApi.getMm3Status(datasetId)
      .then(s => { if (!cancelled) { setStatus(s); setError(null); } })
      .catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)); });
    return () => { cancelled = true; };
  }, [datasetId, finishedKey]);

  return { status, error };
}


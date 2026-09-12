// generation/residency.ts — queued demand accounting (step 5)
//
// This first slice is deliberately a leaf. It does not import the backend
// registry, GPU lane, engine client, or any mutating backend operation. The
// eviction and hold APIs described in §2.3 are added only when their real
// ownership and lane dependencies arrive in later steps.

type Family = string;

/** Jobs queued or running per backend family. The selector is intentionally
 * not consulted: demand belongs to the job's captured family. */
const demand = new Map<Family, number>();

export function noteEnqueued(family: Family): void {
  demand.set(family, (demand.get(family) ?? 0) + 1);
}

/** Decrement once from the outer runOnGpuLane promise's finally. Never allow a
 * bookkeeping mismatch to create negative demand. Later eviction logic will
 * use the zero transition to enqueue a stale-safe release. */
export function noteFinished(family: Family): void {
  const next = (demand.get(family) ?? 0) - 1;
  if (next > 0) demand.set(family, next);
  else demand.delete(family);
}

export function familyDemand(family: Family): number {
  return demand.get(family) ?? 0;
}

// Future compatible exports from §2.3 are intentionally not implemented in
// this step: activeFamily(), holdEngine(), engineHeld(), enqueueEviction(),
// and evictOtherFamilies() require the real GPU lane and registry injection.

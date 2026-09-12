export type WorkKind = 'note' | 'question' | 'handoff' | 'blocker' | 'direction' | 'context';
export type WorkState = 'idle' | 'doing' | 'blocked' | 'done';
export type ReservationInput = { resource: string; mode?: 'use' | 'exclusive'; reason: string };
export type JoinInput = { channel: string; name: string; role?: string; brief?: string };
export type UpdateInput = {
  channel: string; agent: string; request_id: string; text: string;
  kind?: WorkKind; to?: string; reply_to?: number; resolve?: number;
  state?: WorkState; activity?: string; detail?: string;
  refs?: string[];
  reserve?: ReservationInput[];
};
export type SyncInput = {
  channel: string; agent: string; ack?: string; snapshot?: boolean;
  page?: string; max_chars?: number;
};
export type ReleaseInput = { channel: string; agent: string; request_id: string; token: string; note: string };
export type RecoverInput = { channel: string; id: string; request_id: string; evidence: string };

// Public WorkStore API, shared by MCP, the viewer and the guarded command runner:
// constructor(dbPath: string, options?: { session?: string; now?: () => number })
// join(input: JoinInput, human?: boolean): { agent: string; channel: string; protocol: string }
// update(input: UpdateInput): { id: number; grants?: { id:string; resource:string; token:string }[] }
// sync(input: SyncInput): object (compact deltas; snapshot pages; opaque read receipt)
// acknowledge({channel,agent,receipt?,message?}): object (read != explicit acknowledgement)
// release(input: ReleaseInput): { released: string }
// recover(input: RecoverInput): object (human viewer only, explicit recovery evidence)
// detail(channel: string,id: number,offset?: number,maxChars?: number): object
// list(): object[]
// view(channel: string,after?: number): object (browser snapshot and events)
// archive(channel: string,archived: boolean): object (human viewer only)
// heartbeat(): void (only reservations owned by this connection/session)
// disconnect(): void (mark this session's outstanding reservations uncertain)
// close(): void (close DB; does not release reservations)

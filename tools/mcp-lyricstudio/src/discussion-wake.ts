// Automatic wake over Claude Code channels.
//
// A Claude Code session started with `--channels server:<name>` accepts
// `notifications/claude/channel` from an MCP server that declares the
// `claude/channel` capability, and reacts to them as events. This module turns
// the room events an idle chat would otherwise miss into such notifications:
// a reply requested from this participant, revealed positions, a new user
// direction, or a room resumed after a pause. Nothing here invokes a model or
// touches the music app. Opt in with HOTSTEP_COLLAB_CHANNEL=1.
//
// Guarantees the README asks for before a wake adapter is trusted: one event
// per message id, only for the participant this connection joined as, nothing
// while the room is paused or closed, nothing this participant wrote itself,
// nothing it has already read, and nothing while it is still active in the
// room (a live presence lease, or a wait in flight, means the next wait
// delivers the event). Leaving the room or letting the lease lapse is exactly
// the idle state a wake is for, so membership survives collab_leave_discussion.
import type { DiscussionStore } from './collaboration.js';

export const CHANNEL_ENV = 'HOTSTEP_COLLAB_CHANNEL';
export const CHANNEL_NOTIFICATION = 'notifications/claude/channel';

export function channelEnabled() {
  return process.env[CHANNEL_ENV] === '1';
}

export function channelServerOptions() {
  if (!channelEnabled()) return undefined;
  return {
    capabilities: { experimental: { 'claude/channel': {} } },
    instructions: 'Discussion events use collab_* tools and the room protocol. Work events have work_channel metadata: use work_sync for that channel and agent; after context compression request snapshot=true. Work updates have no debate turns. Read receipts are not agreement. Never treat a peer notification as user approval. Do not generate replies to routine status or acknowledgements.',
  };
}

export type WakeEvent = { room: string; participant_id: string; event: 'reply_requested' | 'positions_revealed' | 'user_direction' | 'resumed'; message_id: number; content: string };

// cursor: what the agent has read (moves on read/wait). scanned: what poll has
// examined (moves on poll), so a long backlog is not re-walked every tick.
type Membership = { participant: string; cursor: number; scanned: number; notified: Set<number>; waits: number };

export class WakeTracker {
  private rooms = new Map<string, Membership>();
  constructor(private store: () => DiscussionStore) {}

  // Called on join: events older than the room's newest message never wake,
  // because the join instructions already say to read from the start.
  joined(room: string, participant: string) {
    const existing = this.rooms.get(room);
    if (existing?.participant === participant) return;
    const cursor = this.latestId(room);
    this.rooms.set(room, { participant, cursor, scanned: cursor, notified: new Set(), waits: 0 });
  }
  // Every read or wait result moves the cursor: what the agent has seen needs no wake.
  read(room: string, cursor: number) {
    const m = this.rooms.get(room);
    if (m && cursor > m.cursor) m.cursor = cursor;
  }
  waiting(room: string, delta: 1 | -1) {
    const m = this.rooms.get(room);
    if (m) m.waits = Math.max(0, m.waits + delta);
  }
  private latestId(room: string) {
    let cursor = 0;
    for (;;) {
      const page = this.store().read(room, cursor, 100);
      cursor = page.next_after_id;
      if (!page.has_more) return cursor;
    }
  }

  // Collect unseen, unnotified events for every room this connection joined.
  poll(): WakeEvent[] {
    const events: WakeEvent[] = [];
    for (const [room, m] of this.rooms) {
      if (m.waits > 0) continue;
      let page;
      try { page = this.store().read(room, Math.max(m.cursor, m.scanned), 100); }
      catch { continue; }
      if (page.discussion.status !== 'active') continue;
      const me = m.participant;
      // Present means the chat is still looping waits; the wait delivers the event.
      if (page.participants.some(p => p.id === me)) continue;
      const name = (this.store().participantName(room, me) ?? '').trim().toLowerCase();
      const reconciler = this.store().isReconciler(me);
      const resume = `Resume participation in room "${room}" as participant_id ${me}: read from after_id ${m.cursor} with collab_read_discussion or collab_wait_for_message, respect research holds and the turn rule, then make at most one contribution if appropriate. Planning only.`;
      const push = (event: WakeEvent['event'], id: number, what: string) => {
        if (m.notified.has(id)) return;
        m.notified.add(id);
        events.push({ room, participant_id: me, event, message_id: id, content: `${what} ${resume}` });
      };
      for (;;) {
        for (const message of page.messages) {
          m.scanned = Math.max(m.scanned, message.id);
          if (message.participant_id === me || message.author.trim().toLowerCase() === name) continue;
          if (message.kind === 'user_direction' && message.author === 'You') {
            push('user_direction', message.id, `Discussion room "${room}": the user posted direction at message #${message.id}.`);
          } else if (message.kind === 'coordination' && message.body.startsWith('Positions revealed')) {
            push('positions_revealed', message.id, `Discussion room "${room}": sealed positions were revealed at message #${message.id}. ${reconciler
              ? 'Read every position, then wait for both pair critiques before drafting the merged plan; take no side and post no critique.'
              : 'Read every position, then critique from your role.'}`);
          } else if (message.kind === 'status') {
            try {
              const value = JSON.parse(message.body);
              if (value.status === 'active') push('resumed', message.id, `Discussion room "${room}": the room was resumed at message #${message.id}.`);
            } catch { /* Not a status event this module understands. */ }
          } else if (message.mentions.some(x => x.participant_id === me)) {
            // From the mention row itself: a cancelled wait clears the pending request but not the mention.
            push('reply_requested', message.id, `Discussion room "${room}": ${message.author} requested your reply at message #${message.id}.`);
          }
        }
        if (!page.has_more) break;
        try { page = this.store().read(room, page.next_after_id, 100); }
        catch { break; }
      }
      const request = page.coordination.requests.find(r => r.participant_id === me);
      if (request && request.message_id > m.cursor) {
        push('reply_requested', request.message_id, `Discussion room "${room}": a participant requested your reply at message #${request.message_id}.`);
      }
    }
    return events;
  }
}

import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import test from 'node:test';
import { DiscussionStore } from '../src/collaboration.js';
import { WORK_LEASE_MS, WorkStore } from '../src/work-store.js';

type Fixture = {
  directory: string;
  dbPath: string;
  clock: { value: number };
  a: WorkStore;
  b: WorkStore;
  alice: string;
  bob: string;
};

function makeFixture(channel = 'work'): Fixture {
  const directory = mkdtempSync(join(tmpdir(), 'hotstep-work-store-'));
  const dbPath = join(directory, 'work.db');
  const clock = { value: 1_000_000 };
  const now = () => clock.value;
  const a = new WorkStore(dbPath, { session: 'session-a', now });
  const b = new WorkStore(dbPath, { session: 'session-b', now });
  const alice = a.join({ channel, name: 'Alice', role: 'builder', brief: 'Coordinate the work.' }).agent;
  const bob = b.join({ channel, name: 'Bob', role: 'reviewer' }).agent;
  return { directory, dbPath, clock, a, b, alice, bob };
}

function closeFixture(fixture: Fixture): void {
  fixture.b.close();
  fixture.a.close();
  const target = resolve(fixture.directory);
  assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-work-store-'));
  rmSync(target, { recursive: true, force: true });
}

function asRecord(value: unknown): Record<string, any> {
  assert.equal(typeof value, 'object');
  assert.ok(value);
  return value as Record<string, any>;
}

function json(value: unknown): string { return JSON.stringify(value); }

test('independent stores share consecutive updates and have no formal turn gate', () => {
  const f = makeFixture();
  try {
    const first = f.a.update({ channel: 'work', agent: f.alice, request_id: 'one', text: 'First update.' });
    const second = f.a.update({ channel: 'work', agent: f.alice, request_id: 'two', text: 'Second update.' });
    assert.notEqual(first.id, second.id);
    const page = f.b.sync({ channel: 'work', agent: f.bob, max_chars: 800 });
    assert.deepEqual((page.changes as any[]).filter(item => item.text.endsWith('update.')).map(item => item.text), ['First update.', 'Second update.']);
  } finally { closeFixture(f); }
});

test('a new join gets current activity, pinned context, and old unread directions', () => {
  const f = makeFixture();
  try {
    f.a.update({ channel: 'work', agent: f.alice, request_id: 'activity', text: 'Building now.', state: 'doing', activity: 'compiling' });
    const direction = f.a.update({ channel: 'work', agent: f.alice, request_id: 'direction', text: 'Keep the GGML path selectable.', kind: 'direction', to: 'Bob' });
    f.a.update({ channel: 'work', agent: f.alice, request_id: 'context', text: 'This correction remains relevant.', kind: 'context' });
    const newcomer = new WorkStore(f.dbPath, { session: 'session-c', now: () => f.clock.value });
    try {
      const agent = newcomer.join({ channel: 'work', name: 'Carol', role: 'observer' }).agent;
      const snapshot = newcomer.sync({ channel: 'work', agent, snapshot: true, max_chars: 4000 });
      const items = snapshot.snapshot as any[];
      assert.ok(items.some(item => item.type === 'activity' && item.name === 'Alice' && item.state === 'doing' && item.activity === 'compiling'));
      assert.ok(items.some(item => item.type === 'pinned' && item.id === direction.id && item.kind === 'direction'));
      assert.ok(items.some(item => item.type === 'pinned' && item.kind === 'context'));
    } finally { newcomer.close(); }
  } finally { closeFixture(f); }
});

test('explicit read receipts can be acknowledged after reopening the same member', () => {
  const f = makeFixture();
  try {
    f.a.update({ channel: 'work', agent: f.alice, request_id: 'receipt-event', text: 'Persist this receipt.' });
    const first = asRecord(f.b.sync({ channel: 'work', agent: f.bob }));
    assert.ok(first.receipt);
    f.b.close();
    const reopened = new WorkStore(f.dbPath, { session: 'session-b-reopened', now: () => f.clock.value });
    try {
      const bobAgain = reopened.join({ channel: 'work', name: 'Bob' }).agent;
      assert.equal(bobAgain, f.bob);
      reopened.acknowledge({ channel: 'work', agent: bobAgain, receipt: first.receipt as string });
      const empty = reopened.sync({ channel: 'work', agent: bobAgain });
      assert.deepEqual(empty, { cursor: first.cursor, changed: false });
    } finally { reopened.close(); }
  } finally {
    // b was closed above; closeFixture is intentionally tolerant of a closed handle only by not calling it.
    f.a.close();
    const target = resolve(f.directory);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-work-store-'));
    rmSync(target, { recursive: true, force: true });
  }
});

test('empty deltas are tiny and a lost response does not advance the read cursor', () => {
  const f = makeFixture();
  try {
    const joined = asRecord(f.b.sync({ channel: 'work', agent: f.bob }));
    f.b.acknowledge({ channel: 'work', agent: f.bob, receipt: joined.receipt as string });
    const empty = f.b.sync({ channel: 'work', agent: f.bob });
    assert.equal((empty as any).changed, false);
    assert.ok(json(empty).length < 80);
    const event = f.a.update({ channel: 'work', agent: f.alice, request_id: 'lost-response', text: 'Read me twice until acknowledged.' });
    const first = asRecord(f.b.sync({ channel: 'work', agent: f.bob }));
    assert.equal((first.changes as any[])[0].id, event.id);
    const retry = asRecord(f.b.sync({ channel: 'work', agent: f.bob }));
    assert.deepEqual(retry.changes, first.changes);
    assert.equal(retry.cursor, first.cursor);
    f.b.acknowledge({ channel: 'work', agent: f.bob, receipt: first.receipt as string });
    assert.deepEqual(f.b.sync({ channel: 'work', agent: f.bob }), { cursor: first.cursor, changed: false });
  } finally { closeFixture(f); }
});

test('snapshot pagination retains every pinned item, retries the same opaque page, and acknowledges only the last page', () => {
  const f = makeFixture();
  try {
    const expected = new Set<number>();
    for (let i = 0; i < 16; i++) {
      const result = f.a.update({
        channel: 'work', agent: f.alice, request_id: `pin-${i}`, text: `Pinned item ${i}: ${'x'.repeat(50)}`,
        kind: i % 2 ? 'context' : 'direction',
      });
      expected.add(result.id);
    }
    const first = asRecord(f.b.sync({ channel: 'work', agent: f.bob, snapshot: true, max_chars: 800 }));
    assert.equal(first.more, true);
    assert.ok(first.page);
    const retry = asRecord(f.b.sync({ channel: 'work', agent: f.bob, page: first.page as string, max_chars: 800 }));
    const retryAgain = asRecord(f.b.sync({ channel: 'work', agent: f.bob, page: first.page as string, max_chars: 800 }));
    assert.deepEqual(retryAgain, retry);

    const seen = new Set<number>();
    let page = first;
    for (const item of page.snapshot as any[]) if (item.type === 'pinned') seen.add(item.id);
    while (page.more) {
      page = asRecord(f.b.sync({ channel: 'work', agent: f.bob, page: page.page as string, max_chars: 800 }));
      for (const item of page.snapshot as any[]) if (item.type === 'pinned') seen.add(item.id);
    }
    assert.deepEqual(seen, expected);
    assert.ok(page.receipt, 'the receipt is returned only on the final page');
  } finally { closeFixture(f); }
});

test('events arriving while a snapshot is paged are returned after its fixed boundary', () => {
  const f = makeFixture();
  try {
    for (let i = 0; i < 12; i++) f.a.update({ channel: 'work', agent: f.alice, request_id: `old-${i}`, text: `Old pinned ${i}`, kind: 'direction' });
    let page = asRecord(f.b.sync({ channel: 'work', agent: f.bob, snapshot: true, max_chars: 800 }));
    const late = f.a.update({ channel: 'work', agent: f.alice, request_id: 'late-event', text: 'Arrived after snapshot began.' });
    while (page.more) page = asRecord(f.b.sync({ channel: 'work', agent: f.bob, page: page.page as string, max_chars: 800 }));
    f.b.acknowledge({ channel: 'work', agent: f.bob, receipt: page.receipt as string });
    const delta = asRecord(f.b.sync({ channel: 'work', agent: f.bob }));
    assert.equal((delta.changes as any[]).find(item => item.id === late.id)?.text, 'Arrived after snapshot began.');
  } finally { closeFixture(f); }
});

test('idempotent updates return the same grants without duplicate events and reject changed bodies', () => {
  const f = makeFixture();
  try {
    const input = { channel: 'work', agent: f.alice, request_id: 'grant-once', text: 'Claim once.', reserve: [{ resource: 'gpu0', mode: 'exclusive' as const, reason: 'Build.' }] };
    const first = f.a.update(input);
    const before = f.a.view('work').events.length;
    const retry = f.a.update(input);
    assert.deepEqual(retry, first);
    assert.equal(f.a.view('work').events.length, before);
    assert.throws(() => f.a.update({ ...input, text: 'Changed body.' }), /different content/);
  } finally { closeFixture(f); }
});

test('multi-resource claims roll back atomically and conflict across channels', () => {
  const f = makeFixture();
  try {
    const other = new WorkStore(f.dbPath, { session: 'session-other', now: () => f.clock.value });
    try {
      const otherAgent = other.join({ channel: 'elsewhere', name: 'Other' }).agent;
      other.update({ channel: 'elsewhere', agent: otherAgent, request_id: 'hold-r2', text: 'Hold r2.', reserve: [{ resource: 'r2', reason: 'Other work.' }] });
      assert.throws(() => f.a.update({ channel: 'work', agent: f.alice, request_id: 'two-claims', text: 'Both or neither.', reserve: [{ resource: 'r1', reason: 'First.' }, { resource: 'r2', reason: 'Second.' }] }), /r2|reserved|conflict/i);
      assert.equal((f.a.view('work').reservations as any[]).some(item => item.resource === 'r1'), false);
      assert.equal((f.a.view('elsewhere').reservations as any[]).filter(item => item.resource === 'r2').length, 1);
    } finally { other.close(); }
  } finally { closeFixture(f); }
});

test('shared use reservations coexist while exclusive reservations are blocked', () => {
  const f = makeFixture();
  try {
    const one = f.a.update({ channel: 'work', agent: f.alice, request_id: 'use-a', text: 'Shared A.', reserve: [{ resource: 'cache', mode: 'use', reason: 'Read.' }] });
    const two = f.b.update({ channel: 'work', agent: f.bob, request_id: 'use-b', text: 'Shared B.', reserve: [{ resource: 'cache', mode: 'use', reason: 'Read.' }] });
    assert.equal(one.grants?.length, 1);
    assert.equal(two.grants?.length, 1);
    assert.throws(() => f.a.update({ channel: 'work', agent: f.alice, request_id: 'exclusive', text: 'Need exclusive.', reserve: [{ resource: 'cache', mode: 'exclusive', reason: 'Write.' }] }), /cache|reserved|conflict/i);
  } finally { closeFixture(f); }
});

test('an expired lease remains recovery-blocking even after a late heartbeat', () => {
  const f = makeFixture();
  try {
    const held = f.a.update({ channel: 'work', agent: f.alice, request_id: 'lease', text: 'Long work.', reserve: [{ resource: 'lease-resource', reason: 'Long work.' }] });
    f.clock.value += WORK_LEASE_MS + 1;
    f.a.heartbeat();
    assert.equal(f.b.view('work').reservations[0].status, 'recovery');
    assert.throws(() => f.b.update({ channel: 'work', agent: f.bob, request_id: 'blocked-by-expiry', text: 'Cannot steal.', reserve: [{ resource: 'lease-resource', reason: 'Attempt.' }] }), /recovery|reserved|lease-resource/i);
    assert.throws(() => f.b.release({ channel: 'work', agent: f.bob, request_id: 'foreign-release', token: held.grants![0].token, note: 'No.' }), /token|agent|belong/i);
  } finally { closeFixture(f); }
});

test('human recovery requires evidence and uncertainty, and foreign owners cannot release', () => {
  const f = makeFixture();
  try {
    const held = f.a.update({ channel: 'work', agent: f.alice, request_id: 'recover-me', text: 'Needs recovery.', reserve: [{ resource: 'recoverable', reason: 'Investigate.' }] });
    const id = held.grants![0].id;
    const human = new WorkStore(f.dbPath, { session: 'human-session', now: () => f.clock.value });
    try {
      const you = human.join({ channel: 'work', name: 'You' }, true).agent;
      assert.throws(() => human.recover({ channel: 'work', id, request_id: 'no-evidence', evidence: '' }), /evidence/i);
      assert.throws(() => human.recover({ channel: 'work', id, request_id: 'still-live', evidence: 'No uncertainty yet.' }), /connected|release|owner/i);
      f.a.disconnect();
      const recovered = human.recover({ channel: 'work', id, request_id: 'recover', evidence: 'Alice connection ended; verified no process holds the resource.' });
      assert.equal(recovered.recovered, id);
      assert.equal(human.view('work').reservations.length, 0);
      assert.ok(you);
    } finally { human.close(); }
  } finally { closeFixture(f); }
});

test('pinned context remains until its author or the human resolves it', () => {
  const f = makeFixture();
  try {
    const context = f.a.update({ channel: 'work', agent: f.alice, request_id: 'pin-context', text: 'Correct the prior assumption.', kind: 'context' });
    assert.throws(() => f.b.update({ channel: 'work', agent: f.bob, request_id: 'foreign-resolve', text: 'I read it.', resolve: context.id }), /author|human|resolve/i);
    assert.ok((f.b.sync({ channel: 'work', agent: f.bob, snapshot: true }).snapshot as any[]).some(item => item.type === 'pinned' && item.id === context.id));
    const human = new WorkStore(f.dbPath, { session: 'human', now: () => f.clock.value });
    try {
      const you = human.join({ channel: 'work', name: 'You' }, true).agent;
      human.update({ channel: 'work', agent: you, request_id: 'human-resolve', text: 'Resolved by the human.', resolve: context.id });
    } finally { human.close(); }
    assert.equal((f.a.view('work').pinned as any[]).some(item => item.id === context.id), false);
  } finally { closeFixture(f); }
});

test('directed message acknowledgements are separate from read receipts', () => {
  const f = makeFixture();
  try {
    const directed = f.a.update({ channel: 'work', agent: f.alice, request_id: 'directed', text: 'Please review this.', to: 'Bob' });
    const read = asRecord(f.b.sync({ channel: 'work', agent: f.bob }));
    const before = (f.b.view('work').readers as any[]).find((reader: any) => reader.name === 'Bob').cursor;
    const ack = f.b.acknowledge({ channel: 'work', agent: f.bob, message: directed.id });
    assert.equal(ack.acknowledged, directed.id);
    assert.equal((f.b.view('work').readers as any[]).find((reader: any) => reader.name === 'Bob').cursor, before);
    const delta = asRecord(f.b.sync({ channel: 'work', agent: f.bob }));
    assert.ok((delta.changes as any[]).some(item => item.kind === 'ack' && item.reply_to === directed.id));
    f.b.acknowledge({ channel: 'work', agent: f.bob, receipt: read.receipt as string });
  } finally { closeFixture(f); }
});

test('large details stay out of compact messages and are fetched by detail pages', () => {
  const f = makeFixture();
  try {
    const detail = 'DETAIL-' + 'z'.repeat(399);
    const result = f.a.update({ channel: 'work', agent: f.alice, request_id: 'large-detail', text: 'Compact summary.', detail });
    const delta = asRecord(f.b.sync({ channel: 'work', agent: f.bob }));
    const item = (delta.changes as any[]).find(row => row.id === result.id);
    assert.deepEqual(item.detail, { id: result.id, chars: detail.length, sha256: item.detail.sha256 });
    assert.equal('text' in item && item.text, 'Compact summary.');
    assert.doesNotMatch(json(item), new RegExp(detail));
    const parts: string[] = [];
    let offset = 0;
    while (true) {
      const page = f.b.detail('work', result.id, offset, 100);
      parts.push(page.text);
      offset = page.next_offset;
      if (!page.more) break;
    }
    assert.equal(parts.join(''), detail);
  } finally { closeFixture(f); }
});

test('public views, snapshots, and deltas never expose reservation grant tokens', () => {
  const f = makeFixture();
  try {
    const result = f.a.update({ channel: 'work', agent: f.alice, request_id: 'secret-grant', text: 'Private grant.', reserve: [{ resource: 'secret-resource', reason: 'Private.' }] });
    const grantToken = result.grants![0].token;
    assert.doesNotMatch(json(f.a.view('work')), new RegExp(grantToken));
    assert.doesNotMatch(json(f.b.sync({ channel: 'work', agent: f.bob, snapshot: true })), new RegExp(grantToken));
    assert.doesNotMatch(json(f.b.sync({ channel: 'work', agent: f.bob })), new RegExp(grantToken));
  } finally { closeFixture(f); }
});

test('work tables do not modify historical DiscussionStore state', () => {
  const directory = mkdtempSync(join(tmpdir(), 'hotstep-work-discussion-'));
  const dbPath = join(directory, 'shared.db');
  const discussion = new DiscussionStore(dbPath);
  const participant = discussion.join('history', 'Codex', 'Keep this transcript').participant_id;
  discussion.post('history', participant, 'old-message', 'reply', 'Historical evidence.');
  const before = discussion.read('history', 0, 100);
  const work = new WorkStore(dbPath, { session: 'work', now: () => 1_000_000 });
  try {
    const agent = work.join({ channel: 'work', name: 'Agent', brief: 'Separate work state.' }).agent;
    work.update({ channel: 'work', agent, request_id: 'new-work', text: 'Does not enter discussion.' });
    const after = discussion.read('history', 0, 100);
    assert.deepEqual(after.messages, before.messages);
    assert.deepEqual(after.discussion, before.discussion);
    assert.deepEqual(after.participants, before.participants);
  } finally {
    work.close();
    discussion.close();
    const target = resolve(directory);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-work-discussion-'));
    rmSync(target, { recursive: true, force: true });
  }
});

test('archiving preserves work history and refuses to discard outstanding holds', () => {
  const f = makeFixture('archive-room');
  try {
    const event = f.a.update({ channel: 'archive-room', agent: f.alice, request_id: 'archive-note', text: 'Keep this history.' });
    const held = f.a.update({ channel: 'archive-room', agent: f.alice, request_id: 'archive-hold', text: 'Hold before archive.', reserve: [{ resource: 'archive-resource', reason: 'Still active.' }] });
    assert.throws(() => f.a.archive('archive-room', true), /release|recover|reservation/i);
    f.a.release({ channel: 'archive-room', agent: f.alice, request_id: 'archive-release', token: held.grants![0].token, note: 'Work complete.' });
    f.a.archive('archive-room', true);
    const view = f.b.view('archive-room');
    assert.equal(view.channel.archived, 1);
    assert.ok((view.events as any[]).some(item => item.id === event.id && item.text === 'Keep this history.'));
    assert.throws(() => f.a.update({ channel: 'archive-room', agent: f.alice, request_id: 'archived-write', text: 'Must not write.' }), /archived/i);
  } finally { closeFixture(f); }
});

test('sync samples stay within their declared character budget unless explicitly over budget', () => {
  const f = makeFixture();
  try {
    const joined = asRecord(f.b.sync({ channel: 'work', agent: f.bob }));
    f.b.acknowledge({ channel: 'work', agent: f.bob, receipt: joined.receipt as string });
    assert.ok(json(f.b.sync({ channel: 'work', agent: f.bob })).length < 80);
    f.a.update({ channel: 'work', agent: f.alice, request_id: 'brief', text: 'A short normal update.' });
    const page = f.b.sync({ channel: 'work', agent: f.bob, max_chars: 800 });
    assert.ok(json(page).length <= 800 || page.over_budget === true);
  } finally { closeFixture(f); }
});

test('notifications deliver only unread peer pins for this member or broadcast, without acknowledging reads', () => {
  const f = makeFixture();
  try {
    // Start after the join events so the notification cursor describes only
    // the work that follows this read boundary.
    const baseline = (f.b.view('work').events as any[]).at(-1).id as number;
    const targeted = f.a.update({ channel: 'work', agent: f.alice, request_id: 'notify-target', text: 'Review this correction.', kind: 'context', to: 'Bob' });
    const broadcastQuestion = f.a.update({ channel: 'work', agent: f.alice, request_id: 'notify-question', text: 'Which path should we measure?', kind: 'question' });
    const broadcastDirection = f.a.update({ channel: 'work', agent: f.alice, request_id: 'notify-direction', text: 'Keep the CPU fallback available.', kind: 'direction' });
    f.a.update({ channel: 'work', agent: f.alice, request_id: 'notify-other', text: 'Only Carol should see this.', kind: 'direction', to: 'Carol' });
    f.a.update({ channel: 'work', agent: f.alice, request_id: 'notify-note', text: 'Routine progress note.' });
    f.a.update({ channel: 'work', agent: f.alice, request_id: 'notify-blocker', text: 'A blocker is recorded for the full team.', kind: 'blocker' });
    f.b.update({ channel: 'work', agent: f.bob, request_id: 'notify-self', text: 'My own context.', kind: 'context' });
    const resolved = f.a.update({ channel: 'work', agent: f.alice, request_id: 'notify-resolved', text: 'Already resolved context.', kind: 'context' });
    f.a.update({ channel: 'work', agent: f.alice, request_id: 'notify-resolve', text: 'Resolved by author.', resolve: resolved.id });
    f.b.acknowledge({ channel: 'work', agent: f.bob, message: targeted.id });

    const before = (f.b.view('work').readers as any[]).find(reader => reader.name === 'Bob').cursor as number;
    const first = f.b.notifications('work', f.bob, baseline) as any;
    assert.deepEqual(first.events.map((event: any) => event.id), [targeted.id, broadcastQuestion.id, broadcastDirection.id]);
    assert.deepEqual(first.events.map((event: any) => event.kind), ['context', 'question', 'direction']);
    assert.equal((f.b.view('work').readers as any[]).find(reader => reader.name === 'Bob').cursor, before);

    const after = f.b.notifications('work', f.bob, first.cursor) as any;
    assert.deepEqual(after.events, []);
    assert.equal(after.cursor, first.cursor);
    assert.equal((f.b.view('work').readers as any[]).find(reader => reader.name === 'Bob').cursor, before);

    f.a.archive('work', true);
    const archived = f.b.notifications('work', f.bob, baseline) as any;
    assert.deepEqual(archived.events, []);
  } finally { closeFixture(f); }
});

import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import test from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { getDefaultEnvironment, StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { DiscussionStore, registerCollaborationTools } from '../src/collaboration.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

const packageDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');

test('shared discussions over two independent MCP stdio processes', { timeout: 30000 }, async t => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-collab-test-'));
  const dbPath = join(temp, 'discussion.db');
  const clients: Client[] = [];
  async function connect(name: string) {
    const client = new Client({ name, version: '1.0.0' });
    clients.push(client);
    await client.connect(new StdioClientTransport({
      command: process.execPath,
      args: [...process.execArgv.filter(arg => arg.startsWith('--preserve-symlinks')), '--import', 'tsx', 'src/collaboration-server.ts'],
      cwd: packageDir,
      env: { ...getDefaultEnvironment(), HOTSTEP_COLLAB_DB: dbPath },
      stderr: 'inherit',
    }));
    return client;
  }
  async function call(client: Client, name: string, args: Record<string, unknown>) {
    const response = await client.callTool({ name, arguments: name === 'collab_read_discussion' || name === 'collab_wait_for_message' ? { compact: false, ...args } : args });
    const content = response.content as { type: string; text: string }[];
    if (response.isError) throw new Error(content[0].text);
    return JSON.parse(content[0].text);
  }
  try {
    const [codex, claude] = await Promise.all([connect('test-codex'), connect('test-claude')]);
    let codexId = '';
    let claudeId = '';
    const room = 'cache-design';
    const post = (client: Client, participant_id: string, request_id: string, body: string, extra = {}) => call(client, 'collab_post_message', { room, participant_id, request_id, body, ...extra });

    await t.test('tools load without music database; concurrent joins share one room', async () => {
      const tools = await codex.listTools();
      assert.equal(tools.tools.filter(tool => tool.name.startsWith('collab_')).length, 13);
      assert.equal(tools.tools.filter(tool => tool.name.startsWith('work_')).length, 6);
      await assert.rejects(call(codex, 'collab_join_discussion', { room: 'missing', name: 'Codex' }), /brief is required/);
      const joined = await Promise.all([
        call(codex, 'collab_join_discussion', { room, name: 'Codex', brief: 'Review cache design without touching running jobs.' }),
        call(claude, 'collab_join_discussion', { room, name: 'Claude', brief: 'Review cache design without touching running jobs.' }),
      ]);
      [codexId, claudeId] = joined.map(j => j.participant_id);
      assert.notEqual(codexId, claudeId);
      assert.match(joined[0].protocol, /There is no automatic idle-time or reply-count cutoff/);
      assert.doesNotMatch(joined[0].protocol, /3 consecutive timeouts|8 substantive replies/);
      const waitTool = tools.tools.find(tool => tool.name === 'collab_wait_for_message');
      assert.match(waitTool?.description ?? '', /repeat empty waits while active/);
      const page = await call(codex, 'collab_read_discussion', { room });
      assert.equal(page.participants.length, 2);
      assert.equal(page.discussion.revision, 0);
    });

    await t.test('a waiting client receives the other process message; own writes are readable', async () => {
      const waiting = call(codex, 'collab_wait_for_message', { room, after_id: 0, timeout_ms: 2000 });
      await delay(100);
      const proposal = await post(claude, claudeId, 'proposal-1', 'Use a content hash.', { kind: 'proposal' });
      const page = await waiting;
      assert.equal(page.timed_out, false);
      assert.equal(page.messages[0].author, 'Claude');
      assert.equal(page.next_after_id, proposal.id);
      const reply = await post(codex, codexId, 'critique-1', 'Include the base model identity.', { kind: 'critique', reply_to: proposal.id });
      const received = await call(claude, 'collab_wait_for_message', { room, after_id: proposal.id, timeout_ms: 500 });
      assert.equal(received.messages[0].id, reply.id);
      assert.equal(received.messages[0].reply_to, proposal.id);
    });

    await t.test('concurrent retry is stored once; changed retry and foreign identities are rejected', async () => {
      await post(claude, claudeId, 'retry-direction', 'Please expand on the identity.');
      const [a, b] = await Promise.all([
        post(codex, codexId, 'retry', 'One message'),
        post(claude, codexId, 'retry', 'One message'),
      ]);
      assert.equal(a.id, b.id);
      await assert.rejects(post(codex, codexId, 'retry', 'Changed content'), /different content/);
      const other = await call(claude, 'collab_join_discussion', { room: 'other', name: 'Claude', brief: 'Separate discussion' });
      await assert.rejects(post(claude, other.participant_id, 'wrong-room', 'Invalid'), /Unknown participant/);
      const foreignMessage = await call(claude, 'collab_post_message', { room: 'other', participant_id: other.participant_id, request_id: 'foreign', body: 'Other room message' });
      await assert.rejects(post(codex, codexId, 'foreign-reply', 'Invalid reply', { reply_to: foreignMessage.id }), /reply_to/);
    });

    await t.test('pagination preserves all messages despite global ID gaps', async () => {
      await post(claude, claudeId, 'after-gap', 'Message following other room write');
      const all = await call(codex, 'collab_read_discussion', { room });
      const ids: number[] = [];
      let cursor = 0;
      while (true) {
        const page = await call(codex, 'collab_read_discussion', { room, after_id: cursor, limit: 1 });
        ids.push(...page.messages.map((m: { id: number }) => m.id));
        cursor = page.next_after_id;
        if (!page.has_more) break;
      }
      assert.deepEqual(ids, all.messages.map((m: { id: number }) => m.id));
      const idle = await call(codex, 'collab_wait_for_message', { room, after_id: cursor, timeout_ms: 50 });
      assert.equal(idle.timed_out, true);
      assert.equal(idle.next_after_id, cursor);
      assert.deepEqual(idle.messages, []);
      await assert.rejects(call(codex, 'collab_wait_for_message', { room, timeout_ms: 26000 }), /25000|validation/i);
    });

    await t.test('pause wakes waiters, blocks writes, and join does not resume; resume and close are visible', async () => {
      const page = await call(codex, 'collab_read_discussion', { room });
      const waiting = call(codex, 'collab_wait_for_message', { room, after_id: page.next_after_id, timeout_ms: 2000 });
      await call(claude, 'collab_set_status', { room, participant_id: claudeId, request_id: 'pause', status: 'paused', reason: 'User asks us to pause.' });
      assert.equal((await waiting).discussion.status, 'paused');
      await assert.rejects(post(codex, codexId, 'paused-post', 'Must fail'), /paused/);
      const joined = await call(codex, 'collab_join_discussion', { room, name: 'Observer', brief: 'Do not overwrite' });
      assert.equal(joined.discussion.status, 'paused');
      assert.equal(joined.discussion.brief, page.discussion.brief);
      const paused = await call(codex, 'collab_read_discussion', { room });
      const noNewMessages = await call(codex, 'collab_wait_for_message', { room, after_id: paused.next_after_id, timeout_ms: 2000 });
      assert.equal(noNewMessages.timed_out, false);
      assert.equal(noNewMessages.discussion.status, 'paused');
      await call(claude, 'collab_set_status', { room, participant_id: claudeId, request_id: 'resume', status: 'active', reason: 'User asks us to resume.' });
      await post(codex, codexId, 'resumed-post', 'Back to the review');
    });

    await t.test('simultaneous decision revisions cannot overwrite each other', async () => {
      await post(claude, claudeId, 'decision-direction', 'Ready to record.');
      // A human steering message gives both agents an opportunity to propose.
      const setup = new DiscussionStore(dbPath);
      try {
        const humanId = '11111111-1111-4111-8111-111111111111';
        setup.joinViewer(room, humanId);
        setup.post(room, humanId, 'pick-plan', 'user_direction', 'Record the proposal.');
      } finally { setup.close(); }
      const decide = (client: Client, id: string, request: string, plan: string) => call(client, 'collab_record_decision', {
        room, participant_id: id, request_id: request, expected_revision: 0, plan, disagreements: 'Needs user judgment.',
      });
      const revisions = await Promise.allSettled([
        decide(codex, codexId, 'codex-decision', 'Hash model and adapter'),
        decide(claude, claudeId, 'claude-decision', 'Hash all inputs'),
      ]);
      assert.equal(revisions.filter(r => r.status === 'fulfilled').length, 1);
      const failed = revisions.find(r => r.status === 'rejected') as PromiseRejectedResult;
      assert.match(failed.reason.message, /Decision changed/);
      const winner = revisions[0].status === 'fulfilled' ? [codex, codexId, 'codex-decision', 'Hash model and adapter'] as const : [claude, claudeId, 'claude-decision', 'Hash all inputs'] as const;
      const retry = await decide(winner[0], winner[1], winner[2], winner[3]);
      assert.equal(retry.revision, 1);
      const page = await call(codex, 'collab_read_discussion', { room });
      assert.equal(page.discussion.revision, 1);
      assert.equal(page.decision.disagreements, 'Needs user judgment.');
    });

    await t.test('one contribution per speaker is enforced across processes, decisions and rejoining', async () => {
      const a = await call(codex, 'collab_join_discussion', { room: 'turns', name: 'Codex', brief: 'Take turns' });
      const b = await call(claude, 'collab_join_discussion', { room: 'turns', name: 'Claude' });
      const send = (id: string, request_id: string, kind = 'reply', body = 'One contribution') => call(codex, 'collab_post_message', { room: 'turns', participant_id: id, request_id, kind, body });
      const first = await send(a.participant_id, 'first');
      assert.deepEqual(Object.keys(first).sort(), ['id', 'kind']);
      await assert.rejects(send(a.participant_id, 'second'), /Wait for another/);
      await assert.rejects(send(a.participant_id, 'steering', 'user_direction'), /Wait for another/);
      const rejoined = await call(claude, 'collab_join_discussion', { room: 'turns', name: 'Codex' });
      await assert.rejects(send(rejoined.participant_id, 'rejoin'), /Wait for another/);
      await assert.rejects(call(codex, 'collab_join_discussion', { room: 'turns', name: 'You' }), /reserved/);
      await assert.rejects(call(codex, 'collab_record_decision', { room: 'turns', participant_id: a.participant_id, request_id: 'decision', expected_revision: 0, plan: 'No second contribution' }), /Wait for another/);
      await assert.rejects(send(b.participant_id, 'long', 'reply', 'x'.repeat(2401)), /2400/);
      await send(b.participant_id, 'reply');
      const decision = await call(codex, 'collab_record_decision', { room: 'turns', participant_id: a.participant_id, request_id: 'decision', expected_revision: 0, plan: 'The accepted proposal', disagreements: 'None' });
      assert.deepEqual(Object.keys(decision).sort(), ['message_id', 'open_items', 'paused', 'revision']);
      await assert.rejects(send(a.participant_id, 'after-decision'), /Wait for another/);
      // Another agent's control event cannot masquerade as its discussion reply.
      await call(claude, 'collab_set_status', { room: 'turns', participant_id: b.participant_id, request_id: 'active', status: 'active', reason: 'Continue' });
      await assert.rejects(send(a.participant_id, 'after-status'), /Wait for another/);
      assert.deepEqual(await send(a.participant_id, 'first'), first); // Retry still works after turn moves.
      await send(b.participant_id, 'unlock');
      const racing = await Promise.allSettled([
        send(a.participant_id, 'race-a'),
        call(claude, 'collab_post_message', { room: 'turns', participant_id: a.participant_id, request_id: 'race-b', body: 'Competing contribution' }),
      ]);
      assert.equal(racing.filter(r => r.status === 'fulfilled').length, 1);
      assert.match((racing.find(r => r.status === 'rejected') as PromiseRejectedResult).reason.message, /Wait for another/);
    });

    await t.test('compact reads retain text once, advance cursors, and omit the plan on empty waits', async () => {
      const full = await call(codex, 'collab_read_discussion', { room: 'turns' });
      const page = await call(codex, 'collab_read_discussion', { room: 'turns', compact: true });
      assert.equal(page.decision.plan, 'The accepted proposal');
      assert.equal(page.messages[0].body, 'One contribution');
      assert.equal(page.messages[0].request_id, undefined);
      assert.equal(page.messages.find((m: any) => m.kind === 'decision').body.includes('accepted proposal'), false);
      assert.equal(page.next_after_id, full.next_after_id);
      const idle = await call(codex, 'collab_wait_for_message', { room: 'turns', after_id: page.next_after_id, timeout_ms: 0, compact: true });
      assert.equal(idle.timed_out, true);
      assert.equal(idle.decision, undefined);
      assert.deepEqual(idle.participants, page.participants);
      assert.equal(idle.discussion.brief, undefined);
      assert.equal(idle.discussion.revision, 1);
      assert.equal(idle.next_after_id, page.next_after_id);
      const beforeDecision = full.messages.find((m: any) => m.kind === 'decision').id - 1;
      const changed = await call(codex, 'collab_read_discussion', { room: 'turns', after_id: beforeDecision, compact: true, limit: 1 });
      assert.equal(changed.decision.plan, 'The accepted proposal');
      assert.equal(changed.has_more, true);
    });

    await t.test('MCP cancellation leaves the client usable', async () => {
      const page = await call(codex, 'collab_read_discussion', { room });
      const controller = new AbortController();
      const pending = codex.callTool({ name: 'collab_wait_for_message', arguments: { room, after_id: page.next_after_id, timeout_ms: 20000 } }, undefined, { signal: controller.signal });
      const rejection = assert.rejects(pending, /abort|cancel/i);
      await delay(50);
      controller.abort();
      await rejection;
      assert.ok((await codex.listTools()).tools.length);
    });

    await t.test('MCP research activity is visible to the peer and the answer releases it', async () => {
      const a = await call(codex, 'collab_join_discussion', { room: 'research', name: 'Codex', brief: 'Investigate before replying.' });
      const b = await call(claude, 'collab_join_discussion', { room: 'research', name: 'Claude' });
      await call(codex, 'collab_set_activity', { room: 'research', participant_id: a.participant_id, activity: 'researching', reason: 'Checking actual graph allocation.' });
      const page = await call(claude, 'collab_read_discussion', { room: 'research' });
      assert.equal(page.coordination.research.name, 'Codex');
      assert.equal(page.messages.length, 0);
      await assert.rejects(call(claude, 'collab_post_message', { room: 'research', participant_id: b.participant_id, request_id: 'too-soon', body: 'My proposal' }), /researching/);
      await call(codex, 'collab_post_message', { room: 'research', participant_id: a.participant_id, request_id: 'answer', body: 'Graph allocation is cached.', read_after_id: page.next_after_id });
      assert.equal((await call(claude, 'collab_read_discussion', { room: 'research' })).coordination.research, null);
    });

    await t.test('MCP plan agreements are explicit and close the room for both clients', async () => {
      const page = await call(codex, 'collab_read_discussion', { room: 'research' });
      const a = page.participants.find((p: { name: string }) => p.name === 'Codex').id;
      const b = page.participants.find((p: { name: string }) => p.name === 'Claude').id;
      await call(claude, 'collab_record_decision', { room: 'research', participant_id: b, request_id: 'consensus-plan', expected_revision: 0, plan: 'Measure the graph allocation.' });
      const agree = async (client: Client, participant_id: string) => {
        const read = await call(client, 'collab_read_discussion', { room: 'research' });
        return call(client, 'collab_agree_plan', { room: 'research', participant_id, request_id: 'agree', revision: 1, read_after_id: read.next_after_id });
      };
      assert.equal((await agree(claude, b)).discussion.status, 'active');
      assert.equal((await agree(codex, a)).discussion.status, 'closed');
      const closed = await call(claude, 'collab_wait_for_message', { room: 'research', compact: true, after_id: 0 });
      assert.equal(closed.consensus.reached, true);
      assert.equal(closed.discussion.status, 'closed');
    });

    await t.test('MCP rejoin, leave and identified reads update the shared live roster', async () => {
      const a = await call(codex, 'collab_join_discussion', { room: 'presence', name: 'Codex', brief: 'Show live agents' });
      await call(claude, 'collab_join_discussion', { room: 'presence', name: 'Claude' });
      const again = await call(codex, 'collab_join_discussion', { room: 'presence', name: ' codex ' });
      assert.equal(again.participant_id, a.participant_id);
      const read = () => call(claude, 'collab_read_discussion', { room: 'presence', compact: true });
      assert.equal((await read()).participants.length, 2);
      await call(codex, 'collab_leave_discussion', { room: 'presence', participant_id: a.participant_id });
      assert.deepEqual((await read()).participants.map((p: { name: string }) => p.name), ['Claude']);
      const observer = await call(codex, 'collab_read_discussion', { room: 'presence' });
      assert.equal(observer.participants.length, 1);
      const monitoring = await call(codex, 'collab_read_discussion', { room: 'presence', participant_id: a.participant_id });
      assert.equal(monitoring.participants.length, 2);
      await assert.rejects(call(codex, 'collab_read_discussion', { room: 'research', participant_id: a.participant_id }), /Unknown participant/);
    });

    await t.test('restart preserves transcript, identities, decisions, and closed state', async () => {
      await call(codex, 'collab_set_status', { room, participant_id: codexId, request_id: 'close', status: 'closed', reason: 'Review completed.' });
      const before = await call(codex, 'collab_read_discussion', { room });
      await Promise.all([codex.close(), claude.close()]);
      const restarted = await connect('test-restarted');
      assert.deepEqual(await call(restarted, 'collab_read_discussion', { room }), before);
      const listed = await call(restarted, 'collab_list_discussions', {});
      assert.equal(listed.length, 5);
      await assert.rejects(post(restarted, codexId, 'closed-post', 'Must fail'), /closed/);
    });
  } finally {
    await Promise.allSettled(clients.map(c => c.close()));
    // Only remove the unique test directory allocated above, never app artifacts.
    const target = resolve(temp);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-collab-test-'));
    rmSync(target, { recursive: true, force: true });
  }
});

test('wait cancellation releases its timer; registration opens no database', async () => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-collab-test-'));
  const store = new DiscussionStore(join(temp, 'test.db'));
  try {
    store.join('room', 'Codex', 'Test cancellation');
    const controller = new AbortController();
    const pending = store.wait('room', 0, 25000, 50, controller.signal);
    const rejection = assert.rejects(pending, /abort/i);
    controller.abort();
    await rejection;
    const server = new McpServer({ name: 'lazy-test', version: '1.0.0' });
    const registration = registerCollaborationTools(server, join(temp, 'missing-parent', 'never-opened.db'));
    assert.equal(existsSync(join(temp, 'missing-parent')), false);
    registration.close();
    await server.close();
  } finally {
    store.close();
    const target = resolve(temp);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-collab-test-'));
    rmSync(target, { recursive: true, force: true });
  }
});

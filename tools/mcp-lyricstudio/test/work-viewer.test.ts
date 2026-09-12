import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { once } from 'node:events';
import test from 'node:test';
import { createDiscussionViewer } from '../src/discussion-viewer.js';
import { WorkStore } from '../src/work-store.js';

type Json = Record<string, any>;

test('Work viewer exposes a safe observer board and explicit human controls', { timeout: 30000 }, async () => {
  const directory = mkdtempSync(join(tmpdir(), 'hotstep-work-viewer-'));
  const dbPath = join(directory, 'collaboration.db');
  const server = createDiscussionViewer(dbPath);
  let agent: WorkStore | undefined;
  try {
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    const address = server.address();
    assert.ok(address && typeof address !== 'string');
    const base = `http://127.0.0.1:${address.port}`;
    const headers = { 'Content-Type': 'application/json', Origin: base };
    const post = async (path: string, value: Json, origin = base) => fetch(base + path, {
      method: 'POST', headers: { ...headers, Origin: origin }, body: JSON.stringify(value),
    });
    const read = async (path: string) => {
      const response = await fetch(base + path);
      let value: Json = {};
      try { value = await response.json(); } catch { /* Asset responses are not JSON. */ }
      return { response, value };
    };

    const page = await fetch(base + '/work');
    assert.equal(page.status, 200);
    const html = await page.text();
    assert.match(html, /href="\/"[^>]*>Discussion/);
    assert.match(html, /class="active" href="\/work"/);
    assert.equal((await fetch(base + '/work.js')).status, 200);
    assert.equal((await fetch(base + '/work.css')).status, 200);

    const created = await post('/api/work/channels', { channel: 'viewer-work', brief: 'Check shared work safely.' });
    assert.equal(created.status, 200);
    const createdBody = await created.json() as Json;
    assert.equal(createdBody.joined.channel, 'viewer-work');
    assert.equal(createdBody.view.channel.archived, false);

    agent = new WorkStore(dbPath, { session: 'viewer-test-agent' });
    const agentId = agent.join({ channel: 'viewer-work', name: 'Agent', role: 'builder' }).agent;

    const detail = 'DETAIL-' + 'z'.repeat(6999);
    const context = agent.update({
      channel: 'viewer-work', agent: agentId, request_id: 'context-with-refs', kind: 'context',
      text: 'The current context needs correction.', refs: ['engine/src/main.cpp:42', 'logs/session/engine.log'], detail,
    });
    const firstDetail = await read(`/api/work/viewer-work/detail/${context.id}?offset=0&max_chars=4000`);
    assert.equal(firstDetail.response.status, 200);
    assert.equal(firstDetail.value.text.length, 4000);
    assert.equal(firstDetail.value.more, true);
    const secondDetail = await read(`/api/work/viewer-work/detail/${context.id}?offset=${firstDetail.value.next_offset}&max_chars=4000`);
    assert.equal(secondDetail.response.status, 200);
    assert.equal(firstDetail.value.text + secondDetail.value.text, detail);
    assert.equal(secondDetail.value.more, false);

    const viewBeforeObserve = agent.view('viewer-work');
    const agentReaderBefore = (viewBeforeObserve.readers as Json[]).find(reader => reader.name === 'Agent');
    assert.ok(agentReaderBefore);
    const observed = await read('/api/work/viewer-work?after=0');
    assert.equal(observed.response.status, 200);
    const agentReaderAfter = (agent.view('viewer-work').readers as Json[]).find(reader => reader.name === 'Agent');
    assert.ok(agentReaderAfter);
    assert.equal(agentReaderAfter.cursor, agentReaderBefore.cursor, 'browser view must not advance agent read cursors');
    assert.ok(observed.value.pinned.some((item: Json) => item.id === context.id && item.kind === 'context'));
    assert.ok(observed.value.events.some((item: Json) => item.id === context.id && item.refs?.length === 2));

    assert.equal((await post('/api/work/viewer-work/messages', { text: 'x'.repeat(601), request_id: 'too-long' })).status, 400);
    assert.equal((await post('/api/work/viewer-work/messages', { text: 'short', detail: 'x'.repeat(24001), request_id: 'too-long-detail' })).status, 400);
    assert.equal((await post('/api/work/viewer-work/messages', { text: 'short', refs: ['x'.repeat(241)], request_id: 'too-long-ref' })).status, 400);

    const humanContextResponse = await post('/api/work/viewer-work/messages', {
      kind: 'context', text: 'Human correction is pinned until resolved.', request_id: 'human-context', refs: ['user-notes.md:8'],
    });
    assert.equal(humanContextResponse.status, 200);
    const humanContext = await humanContextResponse.json() as Json;
    let current = (await read('/api/work/viewer-work?after=0')).value;
    assert.ok(current.pinned.some((item: Json) => item.id === humanContext.id));
    const resolvedResponse = await post('/api/work/viewer-work/messages', {
      text: 'Checked and resolved explicitly.', kind: 'note', resolve: humanContext.id, request_id: 'resolve-human-context',
    });
    assert.equal(resolvedResponse.status, 200);
    current = (await read('/api/work/viewer-work?after=0')).value;
    assert.equal(current.pinned.some((item: Json) => item.id === humanContext.id), false);

    const directed = agent.update({ channel: 'viewer-work', agent: agentId, request_id: 'directed-human', text: 'Please acknowledge this.', kind: 'question', to: 'You' });
    const humanReaderBeforeAck = (current.readers as Json[]).find(reader => reader.name === 'You');
    assert.ok(humanReaderBeforeAck);
    const ackResponse = await post('/api/work/viewer-work/acknowledge', { message: directed.id });
    assert.equal(ackResponse.status, 200);
    assert.equal((await ackResponse.json()).acknowledged, directed.id);
    current = (await read('/api/work/viewer-work?after=0')).value;
    assert.ok(current.acknowledgements.some((ack: Json) => ack.agent === 'You' && ack.message === directed.id));
    assert.equal((current.readers as Json[]).find(reader => reader.name === 'You')!.cursor, humanReaderBeforeAck.cursor, 'message acknowledgement is separate from read acknowledgement');

    for (let i = 0; i < 105; i++) agent.update({ channel: 'viewer-work', agent: agentId, request_id: `feed-${i}`, text: `Feed event ${i}.` });
    let after = 0;
    let pageCount = 0;
    let eventCount = 0;
    let more = true;
    while (more) {
      const result = (await read(`/api/work/viewer-work?after=${after}`)).value;
      pageCount++;
      eventCount += result.events.length;
      more = result.more;
      after = result.cursor;
      assert.ok(pageCount < 10, 'fresh event pagination should make progress');
    }
    assert.ok(pageCount >= 2);
    assert.ok(eventCount > 100);

    const held = agent.update({ channel: 'viewer-work', agent: agentId, request_id: 'live-hold', text: 'Hold a resource.', reserve: [{ resource: 'viewer-resource', reason: 'Testing recovery.' }] });
    const holdId = held.grants![0].id;
    const liveRecovery = await post('/api/work/viewer-work/recover', { id: holdId, request_id: 'live-recovery', evidence: 'No evidence yet.' });
    assert.equal(liveRecovery.status, 409);
    assert.match((await liveRecovery.json()).error, /connected|release|owner/i);
    assert.doesNotMatch(JSON.stringify((await read('/api/work/viewer-work?after=0')).value), new RegExp(held.grants![0].token));
    agent.disconnect();
    const recovered = await post('/api/work/viewer-work/recover', { id: holdId, request_id: 'disconnected-recovery', evidence: 'Agent connection ended; verified the process is no longer using viewer-resource.' });
    assert.equal(recovered.status, 200);
    assert.equal((await recovered.json()).recovered, holdId);

    const foreignWrite = await post('/api/work/viewer-work/messages', { text: 'Cross-origin write', request_id: 'foreign' }, 'http://example.test');
    assert.equal(foreignWrite.status, 403);
  } finally {
    agent?.close();
    server.closeAllConnections();
    await new Promise<void>((resolveClose, rejectClose) => server.close(error => error ? rejectClose(error) : resolveClose()));
    const target = resolve(directory);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-work-viewer-'));
    rmSync(target, { recursive: true, force: true });
  }
});

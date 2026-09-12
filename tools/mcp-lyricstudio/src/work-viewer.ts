import type { IncomingMessage, ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';
import { WorkStore } from './work-store.js';
import type { JoinInput, RecoverInput, UpdateInput } from './work-contract.js';

const channelPattern = '[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}';
const assets = new Map<string, { type: string; body: Buffer }>([
  ['/work', { type: 'text/html', body: readFileSync(new URL('../viewer/work.html', import.meta.url)) }],
  ['/work.js', { type: 'text/javascript', body: readFileSync(new URL('../viewer/work.js', import.meta.url)) }],
  ['/work.css', { type: 'text/css', body: readFileSync(new URL('../viewer/work.css', import.meta.url)) }],
]);

type JsonRecord = Record<string, unknown>;

function send(res: ServerResponse, status: number, value: unknown) {
  if (res.headersSent) return;
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(value));
}

function text(value: unknown, max: number, required = false, label = 'text'): string | undefined {
  if (value === undefined || value === null) {
    if (required) throw new Error(`${label} is required.`);
    return undefined;
  }
  if (typeof value !== 'string') throw new Error(`${label} must be text.`);
  const result = value.trim();
  if (required && !result) throw new Error(`${label} is required.`);
  if (result.length > max) throw new Error(`${label} must be at most ${max} characters.`);
  return result || undefined;
}

function integer(value: string | null, fallback: number, max: number) {
  if (value === null || !/^\d+$/.test(value)) return fallback;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed <= max ? parsed : fallback;
}

function requestId(value: unknown) {
  const id = text(value, 100);
  return id || `viewer-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function channelFromUrl(pathname: string): string | undefined {
  const match = new RegExp(`^/api/work/(${channelPattern})(?:/.*)?$`).exec(pathname);
  return match?.[1];
}

function readJson(request: IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    request.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 128000) chunks.push(chunk);
    });
    request.on('error', reject);
    request.on('aborted', () => reject(new Error('Request cancelled.')));
    request.on('end', () => {
      if (size > 128000) { reject(new Error('Message is too large.')); return; }
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
      catch { reject(new Error('Invalid JSON.')); }
    });
  });
}

function record(value: unknown): JsonRecord {
  return value && typeof value === 'object' ? value as JsonRecord : {};
}

function humanJoin(channel: string, brief?: string): JoinInput {
  return { channel, name: 'You', role: 'human', ...(brief ? { brief } : {}) };
}

function references(value: unknown): string[] | undefined {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value) || value.length > 8) throw new Error('refs must contain at most 8 references.');
  return value.map((item, index) => text(item, 240, true, `refs[${index}]`)!);
}

function publicChannels(value: unknown) {
  return Array.isArray(value) ? value.map(item => {
    const channel = record(item);
    return { ...channel, ...(typeof channel.archived === 'number' ? { archived: channel.archived !== 0 } : {}) };
  }) : [];
}

function publicView(value: unknown) {
  const view = record(value);
  const channel = record(view.channel);
  return { ...view, channel: { ...channel, ...(typeof channel.archived === 'number' ? { archived: channel.archived !== 0 } : {}) } };
}

async function body(request: IncomingMessage): Promise<JsonRecord> {
  const value = record(await readJson(request));
  return value;
}

/**
 * Handles the Work observer page and its small human control API.
 * The parent server performs host and authentication checks before calling this.
 */
export async function handleWorkRequest(
  request: IncomingMessage,
  response: ServerResponse,
  dbPath: string,
): Promise<boolean> {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1');
  const asset = assets.get(url.pathname);
  const isApi = url.pathname === '/api/work' || url.pathname.startsWith('/api/work/');
  if (!asset && !isApi) return false;

  response.setHeader('Cache-Control', 'no-store');
  response.setHeader('X-Content-Type-Options', 'nosniff');
  if (asset) {
    if (request.method !== 'GET') {
      response.setHeader('Allow', 'GET');
      send(response, 405, { error: 'Method not allowed.' });
    } else {
      response.writeHead(200, { 'Content-Type': `${asset.type}; charset=utf-8` });
      response.end(asset.body);
    }
    return true;
  }

  const method = request.method ?? 'GET';
  if (method !== 'GET' && method !== 'POST') {
    response.setHeader('Allow', 'GET, POST');
    send(response, 405, { error: 'Method not allowed.' });
    return true;
  }
  if (method === 'POST' && !request.headers['content-type']?.startsWith('application/json')) {
    send(response, 415, { error: 'Use application/json.' });
    return true;
  }

  let store: WorkStore | undefined;
  try {
    if (method === 'GET' && (url.pathname === '/api/work' || url.pathname === '/api/work/channels')) {
      store = new WorkStore(dbPath, { session: 'viewer' });
      send(response, 200, { channels: publicChannels(store.list()) });
      return true;
    }

    if (url.pathname === '/api/work/channels' && method === 'POST') {
      const input = await body(request);
      const channel = text(input.channel, 100, true);
      const brief = text(input.brief, 1200);
      if (!channel || !new RegExp(`^${channelPattern}$`).test(channel)) {
        send(response, 400, { error: 'Choose a channel name using letters, numbers, dots, underscores or hyphens.' });
        return true;
      }
      store = new WorkStore(dbPath, { session: 'viewer' });
      const joined = store.join(humanJoin(channel, brief), true);
      send(response, 200, { joined, view: publicView(store.view(channel)) });
      return true;
    }

    const channel = channelFromUrl(url.pathname);
    if (!channel) { send(response, 404, { error: 'Work endpoint not found.' }); return true; }

    const detailMatch = new RegExp(`^/api/work/${channelPattern}/detail/(\\d+)$`).exec(url.pathname);
    if (method === 'GET' && detailMatch) {
      const id = Number(detailMatch[1]);
      if (!Number.isSafeInteger(id)) { send(response, 400, { error: 'Invalid detail id.' }); return true; }
      store = new WorkStore(dbPath, { session: 'viewer' });
      send(response, 200, store.detail(channel, id, integer(url.searchParams.get('offset'), 0, 4_000_000), integer(url.searchParams.get('max_chars'), 4_000, 4_000)));
      return true;
    }

    if (method === 'GET' && url.pathname === `/api/work/${channel}`) {
      store = new WorkStore(dbPath, { session: 'viewer' });
      send(response, 200, publicView(store.view(channel, integer(url.searchParams.get('after'), 0, Number.MAX_SAFE_INTEGER))));
      return true;
    }

    if (method !== 'POST') { send(response, 404, { error: 'Work endpoint not found.' }); return true; }
    const suffix = url.pathname.slice(`/api/work/${channel}`.length);
    const input = await body(request);
    store = new WorkStore(dbPath, { session: 'viewer' });

    if (suffix === '/join') {
      const joined = store.join(humanJoin(channel, text(input.brief, 1200)), true);
      send(response, 200, { joined, view: publicView(store.view(channel)) });
      return true;
    }

    // Every human write reuses the stable Work identity. The browser never
    // receives reservation grant tokens, and view() does not advance cursors.
    const human = store.join(humanJoin(channel), true);
    if (suffix === '/messages') {
      const message = text(input.text, 600, true);
      const kind = text(input.kind, 30) || 'note';
      const allowed = new Set(['note', 'question', 'direction', 'blocker', 'context', 'handoff']);
      if (!message || !allowed.has(kind)) { send(response, 400, { error: 'Choose a message kind and a message of 1 to 600 characters.' }); return true; }
      const update: UpdateInput = {
        channel, agent: human.agent, request_id: requestId(input.request_id), text: message,
        kind: kind as UpdateInput['kind'], to: text(input.to, 100) || undefined,
        reply_to: typeof input.reply_to === 'number' ? input.reply_to : undefined,
        resolve: typeof input.resolve === 'number' ? input.resolve : undefined,
        detail: text(input.detail, 24_000, false, 'detail'),
        refs: references(input.refs),
      };
      const updated = store.update(update);
      // Reservation tokens belong to the agent runtime. The human board may
      // see that a resource was reserved, but never receives a grant token.
      send(response, 200, { id: updated.id });
      return true;
    }
    if (suffix === '/acknowledge') {
      send(response, 200, store.acknowledge({ channel, agent: human.agent, receipt: text(input.receipt, 200) || undefined, message: typeof input.message === 'number' ? input.message : undefined }));
      return true;
    }
    if (suffix === '/archive') {
      if (typeof input.archived !== 'boolean') { send(response, 400, { error: 'archived must be true or false.' }); return true; }
      send(response, 200, store.archive(channel, input.archived));
      return true;
    }
    if (suffix === '/recover') {
      const id = text(input.id, 200, true);
      const evidence = text(input.evidence, 1_200, true, 'recovery evidence');
      if (!id || !evidence) { send(response, 400, { error: 'Explicit recovery evidence is required.' }); return true; }
      const recover: RecoverInput = { channel, id, request_id: requestId(input.request_id), evidence };
      send(response, 200, store.recover(recover));
      return true;
    }
    send(response, 404, { error: 'Work endpoint not found.' });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Work request failed.';
    const status = /invalid json|too large|cancelled|must be|required|choose |invalid work message|refs\[/i.test(message) ? 400 : /unknown|not found/i.test(message) ? 404 : 409;
    send(response, status, { error: message });
  } finally {
    store?.close();
  }
  return true;
}

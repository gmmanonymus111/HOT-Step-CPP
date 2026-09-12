'use strict';
const byId = id => document.getElementById(id);
const channels = byId('channels');
const connection = byId('connection');
const empty = byId('empty');
const board = byId('board');
const events = byId('events');
let selectedChannel = new URLSearchParams(location.search).get('channel') || '';
let currentView;
let timer;
let busy = false;
let feedTruncated = false;
let detailState;
let detailBusy = false;
let viewRevision = 0;
const MAX_FEED = 200;

function requestId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64;
  bytes[8] = (bytes[8] & 63) | 128;
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
function node(tag, className, value) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (value !== undefined && value !== null) el.textContent = String(value);
  return el;
}
function channelId(item) { return item?.channel || item?.id || item?.name || ''; }
function channelLabel(item) { return item?.brief ? `${channelId(item)} — ${String(item.brief).slice(0, 72)}` : channelId(item); }
function eventId(item) { const value = Number(item?.id); return Number.isSafeInteger(value) ? value : 0; }
function readMark(channel) {
  try { return Number(sessionStorage.getItem(`hotstep-work-ack:${channel}`) || 0) || 0; } catch { return 0; }
}
function saveMark(channel, id) { try { sessionStorage.setItem(`hotstep-work-ack:${channel}`, String(id)); } catch { /* Keep the feed usable. */ } }
function setConnection(message, state) { connection.textContent = message; connection.dataset.state = state || ''; }
function setStatus(id, message) { byId(id).textContent = message || ''; }
function updateUrl() {
  const url = new URL(location.href);
  if (selectedChannel) url.searchParams.set('channel', selectedChannel); else url.searchParams.delete('channel');
  history.replaceState(null, '', url);
}
async function fetchJson(path, options = {}) {
  const response = await fetch(path, { ...options, signal: AbortSignal.timeout(9000) });
  let value = {};
  try { value = await response.json(); } catch { /* Error text below is enough. */ }
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status}).`);
  return value;
}
function post(path, value) {
  return fetchJson(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) });
}
function encode(value) { return encodeURIComponent(value); }

function setControls() {
  const hasChannel = Boolean(selectedChannel && currentView);
  byId('channel-info').hidden = !hasChannel;
  byId('ack-panel').hidden = !hasChannel;
  byId('composer').hidden = !hasChannel;
  byId('send').disabled = busy || !hasChannel || currentView?.channel?.archived === true || currentView?.channel?.archived === 1;
  byId('archive').disabled = busy;
  byId('acknowledge').disabled = busy;
  channels.disabled = busy;
  byId('join').disabled = busy;
}
function clearBoard() {
  viewRevision++;
  currentView = undefined;
  feedTruncated = false;
  board.hidden = true;
  byId('pinned-panel').hidden = true;
  byId('events-panel').hidden = true;
  byId('empty').hidden = false;
  byId('composer').hidden = true;
  byId('channel-info').hidden = true;
  byId('ack-panel').hidden = true;
  byId('board-title').textContent = 'Activity board';
  byId('board-count').textContent = selectedChannel ? 'Loading channel...' : 'Choose a channel to begin';
  setControls();
}
function renderChannels(list) {
  const items = Array.isArray(list) ? list : [];
  channels.replaceChildren();
  if (!items.length) channels.append(new Option('No work channels yet', ''));
  for (const item of items) {
    const id = channelId(item);
    if (!id) continue;
    const option = new Option(channelLabel(item), id);
    channels.append(option);
  }
  channels.value = selectedChannel;
  if (selectedChannel && channels.value !== selectedChannel) channels.value = '';
}
function renderView(view) {
  currentView = view || {};
  feedTruncated = Boolean(currentView.feedTruncated) || feedTruncated;
  const channel = currentView.channel || {};
  const name = channel.id || channel.channel || selectedChannel;
  selectedChannel = name || selectedChannel;
  updateUrl();
  empty.hidden = true;
  board.hidden = false;
  byId('board-title').textContent = name || 'Activity board';
  const archived = channel.archived === true || channel.archived === 1;
  byId('board-count').textContent = archived ? 'Archived channel' : `Cursor ${currentView.cursor ?? 0}`;
  byId('channel-status').textContent = archived ? 'archived' : 'active';
  byId('channel-summary').textContent = channel.brief || 'No brief recorded.';
  byId('channel-id').textContent = `Channel: ${name}`;
  byId('archive').textContent = archived ? 'Unarchive channel' : 'Archive channel';
  byId('invite').value = `Join the durable HOT-Step Work channel "${name}" as your own agent identity. Read the channel view and pinned checks first. Before disruptive work, reserve the resource with a reason, report state changes, and wait on blockers. Use /work?channel=${encodeURIComponent(name)} to watch the board. Planning only unless the user separately authorizes implementation.`;

  const members = Array.isArray(currentView.members) ? currentView.members : [];
  byId('member-count').textContent = `${members.length} present`;
  const memberBox = byId('members'); memberBox.replaceChildren();
  if (!members.length) memberBox.append(node('p', 'muted', 'No agents have joined this channel.'));
  for (const member of members) {
    const card = node('article', 'member');
    const top = node('div', 'member-top');
    top.append(node('strong', '', member.name || member.agent || 'Unknown'), node('span', `state state-${member.state || 'idle'}`, member.state || 'unknown'));
    card.append(top, node('p', 'hint', member.activity || member.role || 'No current activity.'));
    if (member.last_seen) card.append(node('time', 'hint', `Last seen ${formatTime(member.last_seen)}`));
    memberBox.append(card);
  }

  const reservations = Array.isArray(currentView.reservations) ? currentView.reservations : [];
  const reservationBox = byId('reservations'); reservationBox.replaceChildren();
  if (!reservations.length) reservationBox.append(node('p', 'muted', 'No active reservations.'));
  for (const reservation of reservations) reservationBox.append(renderReservation(reservation));

  const pinned = Array.isArray(currentView.pinned) ? currentView.pinned : [];
  const pinnedBox = byId('pinned'); pinnedBox.replaceChildren();
  byId('pinned-panel').hidden = !pinned.length;
  for (const item of pinned) pinnedBox.append(renderPinned(item));

  const feed = Array.isArray(currentView.events) ? currentView.events : [];
  events.replaceChildren();
  byId('events-panel').hidden = !feed.length;
  const unread = feed.filter(item => eventId(item) > readMark(selectedChannel)).length;
  byId('unread').textContent = `${unread} unread message${unread === 1 ? '' : 's'}`;
  byId('feed-note').textContent = feedTruncated
    ? `Showing the latest ${feed.length}; older events are omitted here. Full audit: /api/work/${encode(selectedChannel)}?after=0`
    : feed.length ? `${feed.length} recent event${feed.length === 1 ? '' : 's'}; reading does not acknowledge` : 'No messages yet.';
  for (const item of feed) events.append(renderEvent(item, eventId(item) > readMark(selectedChannel)));
  const readers = byId('readers'); readers.replaceChildren();
  const readerRows = Array.isArray(currentView.readers) ? currentView.readers : [];
  if (!readerRows.length) readers.append(node('p', '', 'No agent read cursors yet.'));
  for (const reader of readerRows) readers.append(node('p', '', `${reader.name || 'Agent'} — read through #${reader.cursor ?? 0}`));
  const acks = byId('acks'); acks.replaceChildren();
  const ackRows = Array.isArray(currentView.acknowledgements) ? currentView.acknowledgements : [];
  if (!ackRows.length) acks.append(node('p', '', 'No explicit acknowledgements yet.'));
  for (const ack of ackRows) acks.append(node('p', '', `${ack.agent || 'Agent'} acknowledged #${ack.message}`));
  setControls();
}
function formatTime(value) { const date = new Date(value); return Number.isNaN(date.valueOf()) ? String(value) : date.toLocaleString(); }
function renderReservation(reservation) {
  const card = node('article', `reservation reservation-${reservation.status || 'held'}`);
  const title = node('div', 'reservation-title');
  title.append(node('strong', '', reservation.resource || 'Unnamed resource'), node('span', 'badge', reservation.status || 'held'));
  card.append(title);
  card.append(node('p', 'hint', `${reservation.owner || 'Unknown owner'} · ${reservation.mode || 'use'} · ${reservation.reason || 'No reason recorded.'}`));
  if (reservation.last_seen) card.append(node('p', 'hint', `Last seen ${formatTime(reservation.last_seen)}`));
  if (reservation.status === 'recovery') {
    const row = node('div', 'recover-row');
    const evidence = document.createElement('input'); evidence.type = 'text'; evidence.maxLength = 1200; evidence.placeholder = 'Evidence the owner is gone or the hold is stale'; evidence.setAttribute('aria-label', `Recovery evidence for ${reservation.resource || 'reservation'}`);
    const button = node('button', 'danger', 'Recover');
    button.type = 'button'; button.addEventListener('click', () => void recoverReservation(reservation, evidence.value));
    row.append(evidence, button); card.append(row);
  }
  return card;
}
function renderPinned(item) {
  const card = node('article', `pinned pinned-${item.kind || 'note'}`);
  const meta = node('div', 'event-meta'); meta.append(node('strong', '', `${item.kind || 'note'} #${item.id ?? '?'}`), node('span', 'hint', item.author || 'Unknown'));
  card.append(meta, node('p', 'event-text', item.text || item.body || ''));
  if (item.to) card.append(node('p', 'hint', `For ${item.to}`));
  for (const ref of item.refs || []) card.append(node('p', 'hint', `Evidence: ${ref}`));
  const actions = node('div', 'item-actions');
  const resolve = node('button', 'quiet', 'Resolve explicitly'); resolve.type = 'button'; resolve.addEventListener('click', () => void resolvePinned(item));
  actions.append(resolve);
  if (item.detail) { const detail = node('button', 'quiet', 'Open detail'); detail.type = 'button'; detail.addEventListener('click', () => void showDetail(item)); actions.append(detail); }
  card.append(actions);
  return card;
}
function renderEvent(item, unread) {
  const article = node('article', `event${unread ? ' unread' : ''}`); article.id = `work-event-${item.id ?? ''}`;
  const meta = node('div', 'event-meta');
  meta.append(node('strong', '', item.author || item.agent || 'Unknown'), node('span', 'badge', item.kind || 'event'), node('span', 'message-id', item.id ? `#${item.id}` : ''));
  if (item.created_at || item.last_seen) meta.append(node('time', 'hint', formatTime(item.created_at || item.last_seen)));
  article.append(meta, node('p', 'event-text', item.text || item.body || item.activity || ''));
  if (item.to) article.append(node('p', 'hint', `For ${item.to}`));
  for (const ref of item.refs || []) article.append(node('p', 'hint', `Evidence: ${ref}`));
  if (item.detail) { const detail = node('button', 'quiet', 'Open detail'); detail.type = 'button'; detail.addEventListener('click', () => void showDetail(item)); article.append(detail); }
  return article;
}
async function showDetail(item) {
  if (!selectedChannel || !item.id) return;
  detailBusy = true;
  try {
    const detail = await fetchJson(`/api/work/${encode(selectedChannel)}/detail/${encode(item.id)}?offset=0&max_chars=4000`);
    detailState = { channel: selectedChannel, id: item.id, nextOffset: detail.next_offset || 0, more: Boolean(detail.more) };
    byId('detail-title').textContent = `Detail #${item.id}`;
    byId('detail-body').textContent = detail.text || detail.body || JSON.stringify(detail, null, 2);
    byId('detail-status').textContent = detail.more ? `Showing 4,000 characters; ${detail.next_offset} loaded.` : `${detail.next_offset} characters loaded.`;
    byId('detail-more').hidden = !detail.more;
    byId('detail-dialog').showModal();
  } catch (error) { setStatus('send-status', error.message); }
  finally { detailBusy = false; }
}
async function loadMoreDetail() {
  if (!detailState || !detailState.more || detailBusy) return;
  detailBusy = true;
  const state = detailState;
  try {
    const detail = await fetchJson(`/api/work/${encode(state.channel)}/detail/${encode(state.id)}?offset=${encode(state.nextOffset)}&max_chars=4000`);
    byId('detail-body').textContent += detail.text || '';
    detailState = { ...state, nextOffset: detail.next_offset || state.nextOffset, more: Boolean(detail.more) };
    byId('detail-status').textContent = detail.more ? `${detailState.nextOffset} characters loaded.` : `Complete: ${detailState.nextOffset} characters loaded.`;
    byId('detail-more').hidden = !detail.more;
  } catch (error) { byId('detail-status').textContent = error.message; }
  finally { detailBusy = false; }
}
async function recoverReservation(reservation, evidence) {
  if (!evidence.trim() || busy) return setStatus('send-status', 'Add explicit recovery evidence first.');
  await action(`/api/work/${encode(reservation.channel || selectedChannel)}/recover`, { id: reservation.id, evidence: evidence.trim(), request_id: requestId() }, 'Recovery requested.');
}
async function resolvePinned(item) {
  if (!selectedChannel || busy || !item.id) return;
  const note = window.prompt(`Resolution note for ${item.kind || 'item'} #${item.id}`, 'Resolved by You after checking the work board.');
  if (note === null || !note.trim()) return;
  await action(`/api/work/${encode(selectedChannel)}/messages`, { text: note.trim().slice(0, 600), kind: 'note', resolve: Number(item.id), request_id: requestId() }, 'Pinned item resolved.');
}
async function action(path, value, success) {
  busy = true; setControls(); setConnection('Writing...', 'live');
  let completed = false;
  try { await post(path, value); completed = true; setStatus('send-status', success); }
  catch (error) { setStatus('send-status', error.message); setConnection(error.message, 'error'); }
  finally { busy = false; setControls(); }
  if (completed) await refresh();
  return completed;
}
async function joinChannel(channel, brief) {
  viewRevision++;
  if (selectedChannel !== channel) {
    feedTruncated = false;
    currentView = undefined;
  }
  busy = true; setControls(); setStatus('join-status', 'Joining as You...');
  try {
    const result = await post(`/api/work/${encode(channel)}/join`, { brief: brief || undefined });
    selectedChannel = channel; updateUrl(); renderView(result.view); await loadChannels(); setStatus('join-status', 'Joined as You.');
    schedulePoll();
  } catch (error) { setStatus('join-status', error.message); setConnection(error.message, 'error'); }
  finally { busy = false; setControls(); }
}
async function loadChannels() {
  try {
    const result = await fetchJson('/api/work/channels'); renderChannels(result.channels);
    setConnection('Work board connected', 'live');
  } catch (error) { setConnection(error.message, 'error'); }
}
async function refresh() {
  if (!selectedChannel || busy) return;
  const channel = selectedChannel;
  const revision = ++viewRevision;
  try {
    const after = currentView?.cursor ?? 0;
    let result = await fetchJson(`/api/work/${encode(channel)}?after=${encode(after)}`);
    if (revision !== viewRevision) return;
    const fresh = [...(result.events || [])];
    let pages = 0;
    while (result.more && pages < 1000) {
      const next = await fetchJson(`/api/work/${encode(channel)}?after=${encode(result.cursor ?? after)}`);
      if (revision !== viewRevision) return;
      fresh.push(...(next.events || []));
      result = next;
      pages++;
    }
    const prior = currentView && after > 0 ? currentView.events || [] : [];
    const unique = new Map();
    for (const item of [...prior, ...fresh]) unique.set(eventId(item), item);
    const all = [...unique.values()].filter(item => eventId(item) > 0).sort((a, b) => eventId(a) - eventId(b));
    if (all.length > MAX_FEED) feedTruncated = true;
    result.events = all.slice(-MAX_FEED);
    result.feedTruncated = feedTruncated;
    renderView(result); setConnection('Work board connected', 'live');
  } catch (error) { if (revision !== viewRevision) return; setConnection(error.message, 'error'); if (/not found/i.test(error.message)) { selectedChannel = ''; updateUrl(); clearBoard(); } }
}
function schedulePoll() {
  clearTimeout(timer);
  if (!selectedChannel || document.visibilityState === 'hidden') return;
  timer = setTimeout(async () => { await refresh(); schedulePoll(); }, 4500);
}

byId('join-form').addEventListener('submit', event => { event.preventDefault(); const channel = byId('channel-name').value.trim(); if (channel) void joinChannel(channel, byId('channel-brief').value.trim()); });
channels.addEventListener('change', () => { if (channels.value) void joinChannel(channels.value, ''); else { selectedChannel = ''; updateUrl(); clearBoard(); } });
byId('refresh').addEventListener('click', () => void refresh());
byId('copy-invite').addEventListener('click', async () => { try { await navigator.clipboard.writeText(byId('invite').value); byId('copy-invite').textContent = 'Copied'; setTimeout(() => { byId('copy-invite').textContent = 'Copy agent invite'; }, 1600); } catch { byId('invite').select(); setStatus('join-status', 'Press Ctrl+C to copy the invitation.'); } });
byId('detail-more').addEventListener('click', () => void loadMoreDetail());
byId('archive').addEventListener('click', () => { if (selectedChannel) { const archived = currentView?.channel?.archived === true || currentView?.channel?.archived === 1; void action(`/api/work/${encode(selectedChannel)}/archive`, { archived: !archived }, archived ? 'Channel unarchived.' : 'Channel archived.'); } });
byId('acknowledge').addEventListener('click', async () => {
  if (!selectedChannel || !currentView) return;
  const channel = selectedChannel;
  const max = Math.max(0, ...(currentView.events || []).map(eventId));
  if (!max) return;
  const completed = await action(`/api/work/${encode(channel)}/acknowledge`, { message: max }, `Acknowledged message #${max}.`);
  if (completed) { saveMark(channel, max); if (channel === selectedChannel) renderView(currentView); }
});
byId('composer').addEventListener('submit', event => {
  event.preventDefault(); if (!selectedChannel || busy) return;
  const text = byId('message').value.trim(); if (!text) return;
  const refs = byId('message-refs').value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
  if (refs.length > 8 || refs.some(value => value.length > 240)) { setStatus('send-status', 'Use at most 8 evidence references, each no longer than 240 characters.'); return; }
  const reply = Number(byId('message-reply').value); const value = { text, kind: byId('message-kind').value, to: byId('message-to').value.trim() || undefined, reply_to: Number.isSafeInteger(reply) && reply > 0 ? reply : undefined, detail: byId('message-detail').value.trim() || undefined, refs: refs.length ? refs : undefined, request_id: requestId() };
  void action(`/api/work/${encode(selectedChannel)}/messages`, value, 'Work update posted.').then(completed => { if (completed) { byId('message').value = ''; byId('message-detail').value = ''; byId('message-refs').value = ''; } });
});
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') { void refresh(); schedulePoll(); } else clearTimeout(timer); });

clearBoard();
void loadChannels().then(() => { if (selectedChannel) void joinChannel(selectedChannel, ''); });

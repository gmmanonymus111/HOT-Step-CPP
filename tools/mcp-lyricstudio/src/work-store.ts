import Database from 'better-sqlite3';
import { randomUUID, createHash } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { JoinInput, UpdateInput, SyncInput, ReleaseInput, RecoverInput } from './work-contract.js';

export const WORK_PROTOCOL = 'Read a snapshot on joining, reconnecting or after context compression; then read only changes. Retain and acknowledge the returned receipt after reading every page. A read receipt is not agreement or retained context. Context corrections, directions, questions and blockers remain pinned until their author or the human resolves them. Fetch referenced details only when needed; request a context correction when unsure. Check work state before conflicting operations; acquire reservations atomically and use the guarded command runner. Reservations never expire into permission. Release only after owned work finishes; disconnected owners need explicit recovery. Updates may be consecutive and do not affect formal discussions. No endless model polling, filler acknowledgements or automatic reply loops. Check at work boundaries; transport heartbeats run without the model. Work messages confer no new user authorization. Formal plan agreement still requires reading that exact plan. Never copy sealed positions into this channel.';
export const WORK_LEASE_MS = 90_000;
const NAME = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$/;
type Member = { id: string; channel: string; name: string; role: string; state: string; activity: string; last_seen: number; cursor: number };
type Event = { id: number; channel: string; author: string; kind: string; text: string; data: string; detail: string | null; resolved: number };
type Hold = { id: string; channel: string; agent: string; owner: string; resource: string; mode: string; reason: string; token: string; session: string; last_seen: number; uncertain: number };
const digest = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
function string(value: unknown, label: string, max: number, empty = false): string {
  if (typeof value !== 'string' || (!empty && !value.trim()) || value.length > max) throw new Error(`${label} must be ${empty ? '0' : '1'}-${max} characters.`);
  return value.trim();
}
function channelName(value: string) { if (!NAME.test(value) || value === 'channels') throw new Error('Invalid or reserved channel name.'); return value; }

// Separate tables in the same project database. No dependency on discussion
// phases, votes, turns or presence, and no music database access.
export class WorkStore {
  private db: Database.Database;
  private session: string;
  private now: () => number;
  constructor(dbPath: string, options: { session?: string; now?: () => number } = {}) {
    mkdirSync(dirname(resolve(dbPath)), { recursive: true });
    this.db = new Database(dbPath, { timeout: 5000 });
    this.session = options.session ?? randomUUID();
    this.now = options.now ?? Date.now;
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS work_channels(id TEXT PRIMARY KEY, brief TEXT NOT NULL, archived INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS work_members(id TEXT PRIMARY KEY, channel TEXT NOT NULL REFERENCES work_channels(id), name TEXT NOT NULL COLLATE NOCASE, role TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'idle', activity TEXT NOT NULL DEFAULT '', last_seen INTEGER NOT NULL, cursor INTEGER NOT NULL DEFAULT 0, UNIQUE(channel,name));
      CREATE TABLE IF NOT EXISTS work_events(id INTEGER PRIMARY KEY AUTOINCREMENT, channel TEXT NOT NULL REFERENCES work_channels(id), author TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, data TEXT NOT NULL DEFAULT '{}', detail TEXT, resolved INTEGER NOT NULL DEFAULT 0);
      CREATE INDEX IF NOT EXISTS work_events_channel ON work_events(channel,id);
      CREATE TABLE IF NOT EXISTS work_requests(agent TEXT NOT NULL, request_id TEXT NOT NULL, hash TEXT NOT NULL, result TEXT NOT NULL, PRIMARY KEY(agent,request_id));
      CREATE TABLE IF NOT EXISTS work_holds(id TEXT PRIMARY KEY, channel TEXT NOT NULL, agent TEXT NOT NULL, owner TEXT NOT NULL, resource TEXT NOT NULL, mode TEXT NOT NULL, reason TEXT NOT NULL, token TEXT NOT NULL UNIQUE, session TEXT NOT NULL, last_seen INTEGER NOT NULL, uncertain INTEGER NOT NULL DEFAULT 0);
      CREATE INDEX IF NOT EXISTS work_holds_resource ON work_holds(resource);
      CREATE TABLE IF NOT EXISTS work_receipts(id TEXT PRIMARY KEY, agent TEXT NOT NULL, channel TEXT NOT NULL, start INTEGER NOT NULL, finish INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS work_snapshots(id TEXT PRIMARY KEY, agent TEXT NOT NULL, channel TEXT NOT NULL, start INTEGER NOT NULL, finish INTEGER NOT NULL, items TEXT NOT NULL, next_offset INTEGER NOT NULL DEFAULT 0, receipt TEXT);
      CREATE TABLE IF NOT EXISTS work_snapshot_pages(id TEXT PRIMARY KEY, snapshot TEXT NOT NULL REFERENCES work_snapshots(id), offset INTEGER NOT NULL, UNIQUE(snapshot,offset));
      CREATE TABLE IF NOT EXISTS work_acks(agent TEXT NOT NULL, message INTEGER NOT NULL REFERENCES work_events(id), PRIMARY KEY(agent,message));
    `);
  }
  close() { this.db.close(); }
  private member(channel: string, agent: string): Member {
    const row = this.db.prepare('SELECT * FROM work_members WHERE channel=? AND id=?').get(channel, agent) as Member | undefined;
    if (!row) throw new Error('Join this work channel first.');
    return row;
  }
  private channel(id: string, writable = false) {
    const row = this.db.prepare('SELECT * FROM work_channels WHERE id=?').get(id) as { id: string; brief: string; archived: number } | undefined;
    if (!row) throw new Error('Unknown work channel.');
    if (writable && row.archived) throw new Error('Work channel is archived.');
    return row;
  }
  private latest(channel: string) { return (this.db.prepare('SELECT COALESCE(MAX(id),0) AS id FROM work_events WHERE channel=?').get(channel) as { id: number }).id; }
  private expire() {
    const rows=this.db.prepare('SELECT * FROM work_holds WHERE uncertain=0 AND last_seen<=?').all(this.now()-WORK_LEASE_MS) as Hold[];
    for (const hold of rows) {
      this.db.prepare('UPDATE work_holds SET uncertain=1 WHERE id=?').run(hold.id);
      this.event(hold.channel,hold.owner,'recovery',`${hold.resource}: heartbeat expired; reservation still blocks conflicting work.`,{ resource:hold.resource,reservation:hold.id });
    }
  }
  private event(channel: string, author: string, kind: string, text: string, data: object = {}, detail?: string) {
    return Number(this.db.prepare('INSERT INTO work_events(channel,author,kind,text,data,detail) VALUES(?,?,?,?,?,?)').run(channel, author, kind, text, JSON.stringify(data), detail ?? null).lastInsertRowid);
  }
  private message(channel: string, id: number) {
    const row = this.db.prepare('SELECT * FROM work_events WHERE channel=? AND id=?').get(channel, id) as Event | undefined;
    if (!row) throw new Error('Unknown work message.');
    return row;
  }
  private compact(row: Event) {
    return { id: row.id, author: row.author, kind: row.kind, text: row.text, ...JSON.parse(row.data), ...(row.detail ? { detail: { id: row.id, chars: row.detail.length, sha256: digest(row.detail) } } : {}) };
  }
  private idempotent<T>(agent: string, request: string, input: unknown, action: () => T): T {
    string(request, 'request_id', 120);
    const hash = digest(input);
    const old = this.db.prepare('SELECT hash,result FROM work_requests WHERE agent=? AND request_id=?').get(agent, request) as { hash: string; result: string } | undefined;
    if (old) {
      if (old.hash !== hash) throw new Error('request_id was already used for different content.');
      return JSON.parse(old.result);
    }
    const result = action();
    this.db.prepare('INSERT INTO work_requests VALUES(?,?,?,?)').run(agent, request, hash, JSON.stringify(result));
    return result;
  }
  join(input: JoinInput, human = false) {
    const channel = channelName(input.channel), name = string(input.name, 'name', 60), role = string(input.role ?? '', 'role', 120, true);
    if ((name.toLowerCase() === 'you') !== human) throw new Error('You is reserved for the human viewer.');
    return this.db.transaction(() => {
      if (!this.db.prepare('SELECT 1 FROM work_channels WHERE id=?').get(channel)) {
        this.db.prepare('INSERT INTO work_channels(id,brief) VALUES(?,?)').run(channel, string(input.brief ?? 'Project work coordination.', 'brief', 1200));
      }
      this.channel(channel);
      let member = this.db.prepare('SELECT * FROM work_members WHERE channel=? AND name=? COLLATE NOCASE').get(channel, name) as Member | undefined;
      if (!member) {
        const id = randomUUID();
        // Joining does not replay all completed work. The mandatory snapshot
        // contains active directions, questions, corrections and reservations.
        this.db.prepare('INSERT INTO work_members(id,channel,name,role,last_seen,cursor) VALUES(?,?,?,?,?,?)').run(id, channel, name, role, this.now(), this.latest(channel));
        member = this.member(channel, id);
        this.event(channel, name, 'joined', role || 'Joined work channel.');
      } else {
        this.db.prepare('UPDATE work_members SET last_seen=?,role=? WHERE id=?').run(this.now(), role || member.role, member.id);
      }
      return { agent: member.id, channel, protocol: WORK_PROTOCOL };
    }).immediate();
  }
  update(input: UpdateInput) {
    const text = string(input.text, 'text', 600);
    const kind = input.kind ?? 'note';
    if (!['note','question','handoff','blocker','direction','context'].includes(kind)) throw new Error('Invalid work message kind.');
    if (input.state && !['idle','doing','blocked','done'].includes(input.state)) throw new Error('Invalid work state.');
    const activity = input.activity === undefined ? undefined : string(input.activity, 'activity', 240, true);
    if (input.detail !== undefined) string(input.detail, 'detail', 24000);
    if (input.to !== undefined) string(input.to, 'to', 60);
    if (input.refs && (input.refs.length > 8 || input.refs.some(ref => !ref || ref.length > 240))) throw new Error('Use at most 8 short evidence references.');
    if (input.reserve && input.reserve.length > 8) throw new Error('Reserve at most 8 resources at once.');
    return this.db.transaction(() => {
      const member = this.member(input.channel, input.agent);
      return this.idempotent(input.agent, input.request_id, input, () => {
        this.expire();
        this.channel(input.channel, true);
        if (input.reply_to !== undefined) this.message(input.channel, input.reply_to);
        if (input.resolve !== undefined) {
          const target = this.message(input.channel, input.resolve);
          if (target.author.toLowerCase() !== member.name.toLowerCase() && member.name !== 'You') throw new Error('Only the author or human can resolve a pinned item. Acknowledge it instead.');
          this.db.prepare('UPDATE work_events SET resolved=1 WHERE id=?').run(input.resolve);
        }
        const grants: { id: string; resource: string; token: string }[] = [];
        const resources = new Set<string>();
        for (const claim of input.reserve ?? []) {
          const resource = string(claim.resource, 'resource', 100).toLowerCase();
          if (!NAME.test(resource) || resources.has(resource)) throw new Error('Resources must have distinct simple names.');
          resources.add(resource);
          const mode = claim.mode ?? 'exclusive';
          if (!['use','exclusive'].includes(mode)) throw new Error('Invalid reservation mode.');
          const reason = string(claim.reason, 'reservation reason', 240);
          const holders = this.db.prepare('SELECT * FROM work_holds WHERE resource=?').all(resource) as Hold[];
          // Resource scope is the project database, not the discussion/channel.
          const conflict = holders.find(h => h.uncertain || this.now() - h.last_seen >= WORK_LEASE_MS || mode === 'exclusive' || h.mode === 'exclusive');
          if (conflict) throw new Error(`${resource} is reserved by ${conflict.owner}: ${conflict.reason}${conflict.uncertain || this.now() - conflict.last_seen >= WORK_LEASE_MS ? ' (needs recovery)' : ''}.`);
          const grant = { id: randomUUID(), resource, token: randomUUID() };
          this.db.prepare('INSERT INTO work_holds(id,channel,agent,owner,resource,mode,reason,token,session,last_seen) VALUES(?,?,?,?,?,?,?,?,?,?)').run(grant.id,input.channel,input.agent,member.name,resource,mode,reason,grant.token,this.session,this.now());
          grants.push(grant);
        }
        if (input.state !== undefined || activity !== undefined) this.db.prepare('UPDATE work_members SET state=?,activity=?,last_seen=? WHERE id=?').run(input.state ?? member.state,activity ?? member.activity,this.now(),input.agent);
        else this.db.prepare('UPDATE work_members SET last_seen=? WHERE id=?').run(this.now(),input.agent);
        const data = {
          ...(input.to ? { to: input.to } : {}), ...(input.reply_to !== undefined ? { reply_to: input.reply_to } : {}),
          ...(input.resolve !== undefined ? { resolved: input.resolve } : {}), ...(input.refs?.length ? { refs: input.refs } : {}),
          ...(input.state ? { state: input.state } : {}), ...(activity !== undefined ? { activity } : {}),
          ...(grants.length ? { reserved: grants.map(g => g.resource) } : {}),
        };
        const id = this.event(input.channel, member.name, kind, text, data, input.detail);
        return { id, ...(grants.length ? { grants } : {}) };
      });
    }).immediate();
  }
  private receipt(member: Member, finish: number) {
    const id = randomUUID();
    this.db.prepare('INSERT INTO work_receipts VALUES(?,?,?,?,?)').run(id,member.id,member.channel,member.cursor,finish);
    return id;
  }
  private ackReceipt(member: Member, receipt: string) {
    const row = this.db.prepare('SELECT * FROM work_receipts WHERE id=? AND agent=? AND channel=?').get(receipt, member.id, member.channel) as { start: number; finish: number } | undefined;
    if (!row) throw new Error('Unknown read receipt for this agent.');
    if (row.start > member.cursor) throw new Error('Read receipt would skip undelivered changes.');
    if (row.finish > member.cursor) this.db.prepare('UPDATE work_members SET cursor=? WHERE id=?').run(row.finish, member.id);
    member.cursor = Math.max(member.cursor, row.finish);
  }
  acknowledge(input: { channel: string; agent: string; receipt?: string; message?: number }) {
    return this.db.transaction(() => {
      const member = this.member(input.channel,input.agent);
      if (input.receipt) this.ackReceipt(member,input.receipt);
      if (input.message !== undefined) {
        this.message(input.channel,input.message);
        const inserted = this.db.prepare('INSERT OR IGNORE INTO work_acks VALUES(?,?)').run(input.agent,input.message).changes;
        if (inserted) this.event(input.channel,member.name,'ack',`Acknowledged #${input.message}.`,{ reply_to:input.message });
      }
      return { read: member.cursor, ...(input.message !== undefined ? { acknowledged: input.message } : {}) };
    }).immediate();
  }
  private holds() {
    return (this.db.prepare('SELECT * FROM work_holds ORDER BY resource,owner,id').all() as Hold[]).map(h => ({
      id:h.id,channel:h.channel,owner:h.owner,resource:h.resource,mode:h.mode,reason:h.reason,
      status:h.uncertain || this.now()-h.last_seen >= WORK_LEASE_MS ? 'recovery' : 'held',last_seen:h.last_seen,
    }));
  }
  private pinned(channel: string) {
    return (this.db.prepare("SELECT * FROM work_events WHERE channel=? AND resolved=0 AND kind IN ('direction','blocker','question','context') ORDER BY id").all(channel) as Event[]).map(row => this.compact(row));
  }
  private members(channel: string) {
    return this.db.prepare('SELECT name,role,state,activity,last_seen FROM work_members WHERE channel=? ORDER BY name COLLATE NOCASE').all(channel);
  }
  notifications(channel:string,agent:string,after:number) {
    const member=this.member(channel,agent);
    if(this.channel(channel).archived) return { cursor:after,events:[] as {id:number;kind:string;author:string}[] };
    const rows=this.db.prepare('SELECT * FROM work_events WHERE channel=? AND id>? ORDER BY id LIMIT 100').all(channel,Math.max(after,member.cursor)) as Event[];
    const events=rows.filter(row=>{
      const data=JSON.parse(row.data);
      return row.author.toLowerCase()!==member.name.toLowerCase() && ['question','context','direction'].includes(row.kind) && !row.resolved && (!data.to || data.to.toLowerCase()===member.name.toLowerCase());
    }).map(({id,kind,author})=>({id,kind,author}));
    return { cursor:rows.at(-1)?.id ?? Math.max(after,member.cursor),events };
  }
  sync(input: SyncInput): Record<string, unknown> {
    const max = input.max_chars ?? 4000;
    if (!Number.isInteger(max) || max < 800 || max > 16000) throw new Error('max_chars must be 800-16000.');
    return this.db.transaction(() => {
      const member = this.member(input.channel,input.agent);
      this.expire();
      if (input.ack) this.ackReceipt(member,input.ack);
      this.db.prepare('UPDATE work_members SET last_seen=? WHERE id=?').run(this.now(),input.agent);
      if (input.snapshot || input.page) {
        let key = input.page;
        if (!key) {
          const snapshotId = randomUUID();
          key = randomUUID();
          const items = [
            { type:'channel',...this.channel(input.channel) },
            ...this.members(input.channel).map(m => ({ type:'activity',...(m as object) })),
            ...this.holds().map(h => ({ type:'reservation',...h })),
            ...this.pinned(input.channel).map(p => ({ type:'pinned',...p })),
          ];
          this.db.prepare('INSERT INTO work_snapshots(id,agent,channel,start,finish,items) VALUES(?,?,?,?,?,?)').run(snapshotId,member.id,input.channel,member.cursor,this.latest(input.channel),JSON.stringify(items));
          this.db.prepare('INSERT INTO work_snapshot_pages VALUES(?,?,0)').run(key,snapshotId);
        }
        const snapshot = this.db.prepare('SELECT s.*,p.offset FROM work_snapshots s JOIN work_snapshot_pages p ON p.snapshot=s.id WHERE p.id=? AND s.agent=? AND s.channel=?').get(key,input.agent,input.channel) as { id:string; start:number; finish:number; items:string; offset:number; receipt:string|null } | undefined;
        if (!snapshot) throw new Error('Unknown snapshot page for this agent.');
        const items = JSON.parse(snapshot.items) as object[], slice: object[] = [];
        let offset = snapshot.offset, size = 120;
        while (offset < items.length) {
          const item = items[offset], chars = JSON.stringify(item).length;
          if (slice.length && size + chars > max) break;
          slice.push(item); offset++; size += chars;
        }
        const more = offset < items.length;
        let receipt = snapshot.receipt;
        if (!more && !receipt) {
          receipt = this.receipt({ ...member,cursor:snapshot.start },snapshot.finish);
        }
        this.db.prepare('UPDATE work_snapshots SET receipt=? WHERE id=?').run(receipt,snapshot.id);
        let next: string|undefined;
        if (more) {
          const existing=this.db.prepare('SELECT id FROM work_snapshot_pages WHERE snapshot=? AND offset=?').get(snapshot.id,offset) as { id:string }|undefined;
          next=existing?.id ?? randomUUID();
          if (!existing) this.db.prepare('INSERT INTO work_snapshot_pages VALUES(?,?,?)').run(next,snapshot.id,offset);
        }
        return { cursor:snapshot.finish,snapshot:slice,more,...(more ? { page:next } : { receipt }),...(size>max ? { over_budget:true } : {}) };
      }
      const rows = this.db.prepare('SELECT * FROM work_events WHERE channel=? AND id>? ORDER BY id LIMIT 101').all(input.channel,member.cursor) as Event[];
      if (!rows.length) return { cursor:member.cursor,changed:false };
      const changes: object[] = [];
      let size=100, cursor=member.cursor;
      for (const row of rows.slice(0,100)) {
        const item=this.compact(row), chars=JSON.stringify(item).length;
        if (changes.length && size+chars>max) break;
        changes.push(item); size+=chars; cursor=row.id;
      }
      return { cursor,changes,more:cursor<rows[rows.length-1].id,receipt:this.receipt(member,cursor),...(size>max ? { over_budget:true } : {}) };
    }).immediate();
  }
  release(input: ReleaseInput) {
    string(input.note,'completion note',600);
    return this.db.transaction(() => {
      const member=this.member(input.channel,input.agent);
      return this.idempotent(input.agent,input.request_id,input,() => {
        const hold=this.db.prepare('SELECT * FROM work_holds WHERE token=? AND channel=? AND agent=?').get(input.token,input.channel,input.agent) as Hold|undefined;
        if (!hold) throw new Error('Reservation token does not belong to this agent.');
        this.db.prepare('DELETE FROM work_holds WHERE id=?').run(hold.id);
        this.event(input.channel,member.name,'released',input.note,{ resource:hold.resource,reservation:hold.id });
        return { released:hold.id };
      });
    }).immediate();
  }
  recover(input: RecoverInput) {
    string(input.evidence,'recovery evidence',1200);
    return this.db.transaction(() => this.idempotent('You',input.request_id,input,() => {
      const hold=this.db.prepare('SELECT * FROM work_holds WHERE id=? AND channel=?').get(input.id,input.channel) as Hold|undefined;
      if (!hold) throw new Error('Unknown reservation.');
      if (!hold.uncertain && this.now()-hold.last_seen<WORK_LEASE_MS) throw new Error('Owner is still connected; ask it to release after completing work.');
      this.db.prepare('DELETE FROM work_holds WHERE id=?').run(hold.id);
      this.event(input.channel,'You','recovered',input.evidence,{ resource:hold.resource,reservation:hold.id,previous_owner:hold.owner });
      return { recovered:hold.id };
    })).immediate();
  }
  heartbeat() {
    // A late heartbeat does not revive an expired/uncertain reservation.
    this.db.transaction(() => {
      this.expire();
      this.db.prepare('UPDATE work_holds SET last_seen=? WHERE session=? AND uncertain=0 AND last_seen>?').run(this.now(),this.session,this.now()-WORK_LEASE_MS);
    }).immediate();
  }
  disconnect() {
    this.db.transaction(() => {
      const holds=this.db.prepare('SELECT * FROM work_holds WHERE session=? AND uncertain=0').all(this.session) as Hold[];
      this.db.prepare('UPDATE work_holds SET uncertain=1 WHERE session=?').run(this.session);
      for (const hold of holds) this.event(hold.channel,hold.owner,'recovery',`${hold.resource}: owner connection ended; verify work completion before recovery.`,{ resource:hold.resource,reservation:hold.id });
    }).immediate();
  }
  detail(channel: string,id:number,offset=0,maxChars=4000) {
    if (!Number.isSafeInteger(offset)||offset<0||!Number.isInteger(maxChars)||maxChars<100||maxChars>8000) throw new Error('Invalid detail page.');
    const row=this.message(channel,id), body=row.detail ?? row.text;
    return { id,author:row.author,sha256:digest(body),text:body.slice(offset,offset+maxChars),next_offset:Math.min(body.length,offset+maxChars),more:offset+maxChars<body.length };
  }
  list() { return this.db.prepare('SELECT * FROM work_channels ORDER BY id').all() as { id:string;brief:string;archived:number }[]; }
  view(channel:string,after=0) {
    if (!Number.isSafeInteger(after)||after<0) throw new Error('Invalid cursor.');
    return this.db.transaction(() => {
      this.expire();
      const events=(this.db.prepare('SELECT * FROM work_events WHERE channel=? AND id>? ORDER BY id LIMIT 101').all(channel,after) as Event[]);
      return { channel:this.channel(channel),members:this.members(channel),reservations:this.holds(),pinned:this.pinned(channel),events:events.slice(0,100).map(e=>this.compact(e)),cursor:events.slice(0,100).at(-1)?.id ?? after,more:events.length>100,
        acknowledgements:this.db.prepare('SELECT m.name AS agent,a.message FROM work_acks a JOIN work_members m ON m.id=a.agent WHERE m.channel=?').all(channel),
        readers:this.db.prepare('SELECT name,cursor FROM work_members WHERE channel=?').all(channel) };
    })();
  }
  archive(channel:string,archived:boolean) {
    return this.db.transaction(() => {
      this.channel(channel);
      if (archived && this.db.prepare('SELECT 1 FROM work_holds WHERE channel=?').get(channel)) throw new Error('Release or recover reservations before archiving.');
      this.db.prepare('UPDATE work_channels SET archived=? WHERE id=?').run(archived?1:0,channel);
      this.event(channel,'You','archive',archived?'Work channel archived.':'Work channel reopened.');
      return { archived };
    }).immediate();
  }
}

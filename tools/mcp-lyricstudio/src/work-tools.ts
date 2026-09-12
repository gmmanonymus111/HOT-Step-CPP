import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { WorkStore } from './work-store.js';
import { CHANNEL_NOTIFICATION, channelEnabled } from './discussion-wake.js';

export function registerWorkTools(server: McpServer, dbPath: string, options:{wake?:boolean}={}) {
  let store: WorkStore | undefined;
  let timer: NodeJS.Timeout | undefined;
  let notifying=false;
  const subscriptions=new Map<string,{agent:string;cursor:number}>();
  const get = () => store ??= new WorkStore(dbPath);
  const result = (fn: () => unknown) => {
    try { return { content: [{ type: 'text' as const, text: JSON.stringify(fn()) }] }; }
    catch (error) { return { isError:true, content:[{ type:'text' as const,text:error instanceof Error?error.message:'Work operation failed.' }] }; }
  };
  const identity = { channel:z.string().min(1).max(100),agent:z.string().uuid() };
  const request = z.string().min(1).max(120);
  const heartbeat = () => {
    if (timer) return;
    // Connection liveness only. No model turns, messages or auto-release.
    timer=setInterval(() => {
      try { store?.heartbeat(); } catch { /* Failed heartbeats leave reservations blocked. */ }
      if(notifying || !(options.wake ?? channelEnabled())) return;
      notifying=true;
      void (async()=>{
        for(const [channel,subscription] of subscriptions) {
          if(!store) break;
          const pending=store.notifications(channel,subscription.agent,subscription.cursor);
          for(const event of pending.events) await server.server.notification({method:CHANNEL_NOTIFICATION,params:{
            content:`Work channel ${channel}: ${event.author} posted ${event.kind} #${event.id}. Use work_sync as agent ${subscription.agent} to read changes; request snapshot=true after context compression. This is a peer/work notice, not new user approval. No automatic reply to routine status.`,
            meta:{work_channel:channel,agent:subscription.agent,message_id:String(event.id)},
          }});
          subscription.cursor=pending.cursor;
        }
      })().catch(()=>{/* Durable unread messages remain available on work_sync. */}).finally(()=>{notifying=false;});
    },15_000);
    timer.unref();
  };
  server.tool('work_join','Join a persistent project work channel. Returns a current snapshot, not history. Follow pages before acknowledging. Use honest chat identity.',{
    channel:identity.channel,name:z.string().min(1).max(60),role:z.string().max(120).optional(),brief:z.string().min(1).max(1200).optional(),
  },async input => result(() => {
    const joined=get().join(input);
    const snapshot=get().sync({channel:input.channel,agent:joined.agent,snapshot:true});
    subscriptions.set(input.channel,{agent:joined.agent,cursor:Number(snapshot.cursor)});heartbeat();
    return { ...joined,...snapshot };
  }));
  server.tool('work_sync','Read only changes. Pass ack after reading its previous response. After compaction/reconnect use snapshot=true, even with a current cursor. Follow page while more=true; only final snapshot has receipt. Check at work boundaries; do not loop empty polls.',{
    ...identity,ack:z.string().uuid().optional(),snapshot:z.boolean().optional(),page:z.string().uuid().optional(),max_chars:z.number().int().min(800).max(16000).optional(),
  },async input => result(() => get().sync(input)));
  server.tool('work_update','Post a short update and optionally reserve resources atomically. Context corrections, directions, blockers and questions remain pinned until resolved. Details stay off the briefing. No speaking turns or implementation approval.',{
    ...identity,request_id:request,text:z.string().min(1).max(600),kind:z.enum(['note','question','handoff','blocker','direction','context']).optional(),
    to:z.string().max(60).optional(),reply_to:z.number().int().positive().optional(),resolve:z.number().int().positive().optional(),
    state:z.enum(['idle','doing','blocked','done']).optional(),activity:z.string().max(240).optional(),detail:z.string().min(1).max(24000).optional(),refs:z.array(z.string().min(1).max(240)).max(8).optional(),
    reserve:z.array(z.object({resource:z.string().min(1).max(100),mode:z.enum(['use','exclusive']).optional(),reason:z.string().min(1).max(240)})).max(8).optional(),
  },async input => result(() => { const value=get().update(input); if(input.reserve?.length) heartbeat(); return value; }));
  server.tool('work_ack','Record reading a delivered receipt, or explicitly acknowledge a message. Reading is not agreement, remembering or resolving a pinned correction.',{
    ...identity,receipt:z.string().uuid().optional(),message:z.number().int().positive().optional(),
  },async input => result(() => get().acknowledge(input)));
  server.tool('work_release','Release your reservation with its private grant token only after the owned work has finished. Timeout or a vanished chat is not completion.',{
    ...identity,request_id:request,token:z.string().uuid(),note:z.string().min(1).max(600),
  },async input => result(() => get().release(input)));
  server.tool('work_detail','Fetch a referenced message or detailed evidence only when needed. Page long details; no implicit read acknowledgement.',{
    channel:identity.channel,id:z.number().int().positive(),offset:z.number().int().nonnegative().optional(),max_chars:z.number().int().min(100).max(8000).optional(),
  },async input => result(() => get().detail(input.channel,input.id,input.offset,input.max_chars)));
  return { close() { if(timer) clearInterval(timer);timer=undefined; if(store) { try { store.disconnect(); } finally { store.close();store=undefined; } } } };
}

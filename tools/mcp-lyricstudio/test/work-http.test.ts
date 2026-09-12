import test from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { createDiscussionHttp } from '../src/discussion-http.js';
import { WorkStore } from '../src/work-store.js';

test('two MCP clients exchange compact work state, reconnect and protect reservations', {timeout:30000}, async () => {
  const dir=mkdtempSync(join(tmpdir(),'hotstep-work-http-')),dbPath=join(dir,'work.db'),token=randomUUID();
  const {server,closeSessions}=createDiscussionHttp({dbPath,token,allowedHostnames:['127.0.0.1']});
  const clients:Client[]=[],transports:StreamableHTTPClientTransport[]=[];
  const direct=new WorkStore(dbPath);
  try {
    server.listen(0,'127.0.0.1');await once(server,'listening');
    const address=server.address();assert.ok(address&&typeof address!=='string');
    const base=`http://127.0.0.1:${address.port}`;
    const connect=async()=>{
      const transport=new StreamableHTTPClientTransport(new URL(base+'/mcp'),{requestInit:{headers:{Authorization:`Bearer ${token}`}}});
      const client=new Client({name:'work-test',version:'1'});clients.push(client);transports.push(transport);await client.connect(transport);return client;
    };
    const call=async(client:Client,name:string,args:Record<string,unknown>)=>{
      const result=await client.callTool({name,arguments:args});assert.ok(!result.isError,JSON.stringify(result));return JSON.parse((result.content as {text:string}[])[0].text);
    };
    const a=await connect(),b=await connect();
    const aj=await call(a,'work_join',{channel:'HOT-Step',name:'Codex'}),bj=await call(b,'work_join',{channel:'HOT-Step',name:'Claude'});
    assert.ok(aj.snapshot&&aj.receipt);assert.ok(bj.snapshot&&bj.receipt);
    await call(a,'work_ack',{channel:'HOT-Step',agent:aj.agent,receipt:aj.receipt});
    await call(b,'work_ack',{channel:'HOT-Step',agent:bj.agent,receipt:bj.receipt});
    const update=await call(a,'work_update',{channel:'HOT-Step',agent:aj.agent,request_id:randomUUID(),text:'Comparison running; keep app alive.',state:'doing',activity:'Baseline comparisons',reserve:[{resource:'app-server',mode:'use',reason:'Comparison owns engine jobs.'}]});
    const correction=await call(a,'work_update',{channel:'HOT-Step',agent:aj.agent,request_id:randomUUID(),kind:'context',to:'Claude',text:'The engine binary is still frozen.',refs:['discussion:YuE2_Integration_2/revision/2'],detail:'Full evidence. '.repeat(500)});
    const delta=await call(b,'work_sync',{channel:'HOT-Step',agent:bj.agent});
    assert.ok(delta.changes.some((e:{id:number})=>e.id===correction.id));
    assert.ok(JSON.stringify(delta).length<2200);
    assert.ok(!JSON.stringify(delta).includes(update.grants[0].token));
    const empty=await call(b,'work_sync',{channel:'HOT-Step',agent:bj.agent,ack:delta.receipt});
    assert.equal(empty.changed,false);assert.ok(JSON.stringify(empty).length<80);
    const rejected=await b.callTool({name:'work_update',arguments:{channel:'HOT-Step',agent:bj.agent,request_id:randomUUID(),text:'Restarting.',reserve:[{resource:'app-server',reason:'Restart',mode:'exclusive'}]}});
    assert.equal(rejected.isError,true);
    const full=await call(b,'work_detail',{channel:'HOT-Step',id:correction.id,max_chars:1000});assert.equal(full.more,true);assert.equal(full.text.length,1000);
    await transports[1].terminateSession();
    const resumed=await connect();const fresh=await call(resumed,'work_join',{channel:'HOT-Step',name:'Claude'});
    assert.equal(fresh.agent,bj.agent);assert.ok(fresh.snapshot.some((i:{type:string;id:number})=>i.type==='pinned'&&i.id===correction.id));
    await transports[0].terminateSession();
    assert.equal(direct.view('HOT-Step').reservations[0].status,'recovery');
    assert.ok(!JSON.stringify(direct.view('HOT-Step')).includes(update.grants[0].token));
    assert.equal((await fetch(base+'/work')).status,401);
  } finally {
    await closeSessions();await Promise.all(clients.map(c=>c.close().catch(()=>{})));
    server.close();server.closeAllConnections();direct.close();rmSync(dir,{recursive:true,force:true});
  }
});

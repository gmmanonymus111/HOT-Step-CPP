import { normalizeKeyscale, normalizeTimeSignature } from '../src/services/training/metadataVocab.js';
// Rebuild the FSM's exact 70-value keyscale set from metadata-fsm.h
const notes=['A','B','C','D','E','F','G'], accs=['','#','b','\u266F','\u266D'], modes=['major','minor'];
const FSM = new Set<string>();
for (const n of notes) for (const a of accs) for (const m of modes) FSM.add(n+a+' '+m);
let bad=0; const t=(n:string,g:string,w:string)=>{const ok=g===w; if(!ok)bad++;
  console.log((ok?'OK  ':'FAIL')+'  '+n.padEnd(38)+JSON.stringify(g)+(ok?'':'  want '+JSON.stringify(w)));};

t('F# Major -> lowercase', normalizeKeyscale('F# Major'), 'F# major');
t('C Minor', normalizeKeyscale('C Minor'), 'C minor');
t('Bb Major', normalizeKeyscale('Bb Major'), 'Bb major');
t('bb minor (lc note)', normalizeKeyscale('bb minor'), 'Bb minor');
t('unicode sharp passthrough', normalizeKeyscale('F\u266F minor'), 'F\u266F minor');
t('bare note -> reject', normalizeKeyscale('C'), '');
t('garbage -> reject', normalizeKeyscale('weird'), '');
t('empty -> empty', normalizeKeyscale(''), '');
t('4/4 -> 4', normalizeTimeSignature('4/4'), '4');
t('3/4 -> 3', normalizeTimeSignature('3/4'), '3');
t('6/8 -> 6', normalizeTimeSignature('6/8'), '6');
t('already 4', normalizeTimeSignature('4'), '4');
t('5/4 unsupported -> reject', normalizeTimeSignature('5/4'), '');
t('7/8 unsupported -> reject', normalizeTimeSignature('7/8'), '');
t('empty -> empty', normalizeTimeSignature(''), '');

// every output must be IN the FSM set (or empty)
const probes=['F# Major','C Minor','Bb Major','bb minor','A\u266D major','G# Minor','D Major'];
const outs=probes.map(normalizeKeyscale).filter(Boolean);
const notInFsm=outs.filter(o=>!FSM.has(o));
t('all outputs inside FSM 70-set', notInFsm.length?notInFsm.join(','):'yes', 'yes');
console.log(bad? '\n'+bad+' FAILED' : '\nall passed  (FSM set size '+FSM.size+')');


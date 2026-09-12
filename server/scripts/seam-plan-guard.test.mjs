// seam-plan-guard.test.mjs — execute the production ACE LM retry path in a
// small, dependency-mocked VM. This intentionally extracts runGeneration from
// source so the assertions exercise the route's current control flow rather
// than a copied version of it.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import ts from 'typescript';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

function readSource(relativePath) {
  return fs.readFileSync(path.join(repoRoot, relativePath), 'utf8');
}

function parseTypeScript(source, fileName) {
  return ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
}

function transpileExpression(source, fileName) {
  return ts.transpileModule(source, {
    fileName,
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
      sourceMap: false,
    },
  }).outputText;
}

function findFunction(source, fileName, names) {
  const sourceFile = parseTypeScript(source, fileName);
  for (const statement of sourceFile.statements) {
    if (ts.isFunctionDeclaration(statement) && statement.name && names.includes(statement.name.text)) {
      return source.slice(statement.pos, statement.end).replace(/\bexport\s+(?=(?:async\s+)?function\b)/, '');
    }
    if (!ts.isVariableStatement(statement)) continue;
    for (const declaration of statement.declarationList.declarations) {
      if (!ts.isIdentifier(declaration.name) || !names.includes(declaration.name.text)) continue;
      if (declaration.initializer && (ts.isArrowFunction(declaration.initializer) || ts.isFunctionExpression(declaration.initializer))) {
        return source.slice(declaration.initializer.pos, declaration.initializer.end);
      }
    }
  }
  return undefined;
}

function extractGenerationFunction() {
  // Keep the service location first so this fixture follows a future
  // extraction of runGeneration without needing production edits.
  const candidates = [
    ['server/src/services/backends/ace/generate.ts', ['runAceGeneration', 'runGeneration']],
    ['server/src/services/aceGeneration.ts', ['runAceGeneration', 'runGeneration']],
    ['server/src/routes/generate.ts', ['runGeneration', 'runAceGeneration']],
  ];
  for (const [relativePath, names] of candidates) {
    const absolutePath = path.join(repoRoot, relativePath);
    if (!fs.existsSync(absolutePath)) continue;
    const source = fs.readFileSync(absolutePath, 'utf8');
    const functionSource = findFunction(source, relativePath, names);
    if (!functionSource) continue;
    return transpileExpression(`(${functionSource})`, relativePath);
  }
  throw new Error('Could not find runGeneration or runAceGeneration in the known source locations');
}

function extractPlanGuard() {
  const relativePath = 'server/src/services/generation/planGuard.ts';
  const source = readSource(relativePath);
  const sourceFile = parseTypeScript(source, relativePath);
  const names = ['planLoopStats', 'planWorstWindow', 'degeneratePlanReason'];
  const declarations = sourceFile.statements
    .filter(statement => ts.isFunctionDeclaration(statement)
      && statement.name
      && names.includes(statement.name.text))
    .map(statement => source.slice(statement.pos, statement.end).replace(/\bexport\s+(?=(?:async\s+)?function\b)/, ''));
  assert.equal(declarations.length, names.length, 'plan guard helper declarations must be present');
  const code = transpileExpression(
    `(() => { ${declarations.join('\n')} return { degeneratePlanReason }; })()`,
    relativePath,
  );
  return vm.runInNewContext(code, { Math, Number, String, parseInt, isFinite }, { filename: relativePath });
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function healthyCodes(count = 50) {
  return Array.from({ length: count }, (_, i) => String(i + 1)).join(',');
}

test('production plan guard retries a degenerate LM result and preserves the live request', async () => {
  const generationSource = extractGenerationFunction();
  const { degeneratePlanReason } = extractPlanGuard();
  const badCodes = Array.from({ length: 100 }, () => '7').join(',');
  const goodCodes = healthyCodes();
  assert.ok(degeneratePlanReason(badCodes, 10), 'fixture must be degenerate under the production helper');
  assert.equal(degeneratePlanReason(goodCodes, 10), null, 'fixture must be healthy under the production helper');

  const lmRequests = [];
  const synthRequests = [];
  const cacheWrites = [];
  const lmResults = [
    [{ audio_codes: badCodes, caption: 'planner loop', lyrics: 'words', bpm: 92, duration: 10, keyscale: 'G major', timesignature: '4', lm_seed: 7 }],
    [{ audio_codes: goodCodes, caption: 'healthy planner caption', lyrics: 'words', bpm: 92, duration: 10, keyscale: 'G major', timesignature: '4', lm_seed: 1016 }],
  ];
  let lmCall = 0;
  const logs = [];
  const sentinel = new Error('SYNTH_SENTINEL');

  const context = {
    console: { log() {}, warn() {}, error() {} },
    performance,
    AbortController,
    setTimeout,
    clearTimeout,
    Map,
    Math,
    Date,
    Error,
    Buffer,
    URL,
    fs,
    path,
    translateParams: () => ({
      caption: 'requested style',
      lyrics: 'requested lyrics',
      task_type: 'text2music',
      seed: 42,
      lm_seed: 7,
      lm_rep_penalty: 1.1,
      lm_adapter: 'artist-lm-adapter',
      adapter: 'dit-style-adapter',
      duration: 10,
      inference_steps: 4,
      scheduler: 'euler',
      guidance_mode: 'apg',
      plugin_params: { marker: 'preserve-me' },
      adapter_runtime_quant: 'q8_0',
    }),
    getActiveBackendId: () => 'ace',
    releaseMinimaxVramForAce: async () => {},
    startGenerationLog() {},
    logGenerationParams() {},
    logGeneration(_id, level, message) { logs.push({ level, message }); },
    finishGenerationLog() {},
    failGenerationLog() {},
    computeLmCacheKey: () => 'fixture-cache-key',
    getLmCache: () => undefined,
    setLmCache(key, outputs) { cacheWrites.push({ key, outputs: clone(outputs) }); },
    getLmCacheSize: () => cacheWrites.length,
    subscribeLines: () => () => {},
    resolveTriggerSpecs: () => [],
    resolveAdapterTriggers: paths => paths.length ? [{ word: 'artist-trigger', position: 'prepend' }] : [],
    readAdapterTrigger: () => ({ trigger: '', position: '' }),
    applyTriggers: (caption, specs) => ({
      caption: `${specs[0].word}, ${caption}`,
      applied: [specs[0].word],
      skipped: [],
    }),
    degeneratePlanReason,
    pollUntilDone: async () => {},
    loadSourceAudio: () => undefined,
    loadSourceLatent: () => undefined,
    applyTempoAndPitch: value => value,
    loadTimbreReference: async () => undefined,
    getCachedLatent: () => undefined,
    saveCachedLatent: () => {},
    runPostProcessingChain: async () => {},
    normalizePpParams: value => value,
    aceClient: {
      async submitLm(request) {
        lmRequests.push(clone(request));
        return `lm-${++lmCall}`;
      },
      async getJobResult() {
        const result = lmResults.shift();
        assert.ok(result, 'unexpected LM result request');
        return { json: async () => clone(result) };
      },
      async submitSynth(request) {
        synthRequests.push(clone(request));
        throw sentinel;
      },
    },
  };
  const runGeneration = vm.runInNewContext(generationSource, context, {
    filename: 'production-runGeneration.ts',
  });

  const job = {
    id: 'plan-guard-fixture',
    userId: 'fixture-user',
    status: 'pending',
    params: {
      caption: 'requested style',
      lyrics: 'requested lyrics',
      lmAdapter: 'artist-lm-adapter',
      cacheLmCodes: true,
      batchSize: 1,
      randomSeed: false,
      coResident: false,
    },
  };

  await runGeneration(job);

  assert.equal(lmRequests.length, 2, 'initial LM call plus one plan-guard retry');
  assert.equal(lmRequests[0].lm_seed, 7);
  assert.equal(lmRequests[0].lm_rep_penalty, 1.1);
  assert.equal(lmRequests[1].lm_seed, 1016, 'retry seed is shifted by 1009');
  assert.ok(Math.abs(lmRequests[1].lm_rep_penalty - 1.15) < 1e-9, 'retry penalty increases by .05');
  for (const request of lmRequests) {
    assert.equal(request.scheduler, 'euler');
    assert.equal(request.guidance_mode, 'apg');
    assert.deepEqual(request.plugin_params, { marker: 'preserve-me' });
    assert.equal(request.adapter_runtime_quant, 'q8_0');
  }

  assert.equal(cacheWrites.length, 1, 'only the final LM result is cached');
  assert.equal(cacheWrites[0].outputs[0].audio_codes, goodCodes);
  assert.notEqual(cacheWrites[0].outputs[0].audio_codes, badCodes);

  assert.equal(synthRequests.length, 1, 'sentinel stops immediately at the first synth call');
  assert.equal(synthRequests[0].audio_codes, goodCodes);
  assert.match(synthRequests[0].caption, /artist-trigger/);
  assert.equal(synthRequests[0].scheduler, 'euler');
  assert.equal(synthRequests[0].guidance_mode, 'apg');
  assert.deepEqual(synthRequests[0].plugin_params, { marker: 'preserve-me' });
  assert.equal(job.status, 'failed', 'the synth sentinel is consumed by the real generation error path');
  assert.ok(logs.some(entry => entry.message.includes('Plan Guard') && entry.message.includes('retry 1/2')));
});

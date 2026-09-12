// Execute the registered backend wrappers from source, without importing the
// server or native dependencies. The heavy one-attempt runners are mocked;
// the wrapper and outcome mapper are production code extracted by the TS AST.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import ts from 'typescript';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext };

function sourceFile(relativePath) {
  const text = fs.readFileSync(path.join(root, relativePath), 'utf8');
  return { text, ast: ts.createSourceFile(relativePath, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS) };
}

function transpile(code, fileName) {
  return ts.transpileModule(code, { fileName, compilerOptions }).outputText;
}

function extract(relativePath, backendName) {
  const { text, ast } = sourceFile(relativePath);
  const declaration = ast.statements.flatMap(statement => ts.isVariableStatement(statement)
    ? [...statement.declarationList.declarations] : []).find(d => d.name.getText(ast) === backendName);
  assert.ok(declaration?.initializer && ts.isObjectLiteralExpression(declaration.initializer), `${backendName} object not found`);
  const method = declaration.initializer.properties.find(property => property.name?.getText(ast) === 'generate');
  assert.ok(method, `${backendName}.generate method not found`);
  const outcome = ast.statements.find(statement => ts.isFunctionDeclaration(statement)
    && statement.name?.text === 'outcomeFromJob');
  assert.ok(outcome, `${backendName} outcomeFromJob not found`);
  return {
    method: transpile(`({${text.slice(method.pos, method.end)}}).generate`, relativePath),
    outcome: transpile(`(${text.slice(outcome.pos, outcome.end)})`, relativePath),
  };
}

const backends = [
  { name: 'ace', file: 'server/src/services/backends/ace/index.ts', object: 'aceBackend', runner: 'runAceGeneration' },
  { name: 'minimax-m3', file: 'server/src/services/backends/minimax/index.ts', object: 'minimaxBackend', runner: 'runMinimaxGeneration' },
];

for (const backend of backends) {
  test(`${backend.name} production generate wrapper forwards context and maps outcomes`, async () => {
    const extracted = extract(backend.file, backend.object);
    const calls = [];
    const modes = { current: 'success' };
    const mapperError = new Error('request mapper sentinel');
    const runner = async (job, deps) => {
      calls.push({ job, deps });
      if (modes.current === 'mapper') throw mapperError;
      if (modes.current === 'success') {
        job.status = 'succeeded';
        job.result = {
          audioUrls: ['/audio/raw.wav'], masteredAudioUrl: '/audio/mastered.wav',
          noAdapterAudioUrl: '/audio/noadapter.wav', songIds: ['song-1'],
          timing: [{ name: 'LM', ms: 12 }],
        };
      } else if (modes.current === 'failure') {
        job.status = 'failed';
        job.error = 'engine failure consumed by runner';
      }
    };
    const context = { [backend.runner]: runner, outcomeFromJob: vm.runInNewContext(extracted.outcome) };
    const generate = vm.runInNewContext(extracted.method, context);
    const poller = () => 'poller';
    const signal = { aborted: false };

    const successJob = { status: 'pending' };
    const success = await generate(successJob, { pollUntilDone: poller, signal });
    assert.equal(calls[0].job, successJob);
    assert.equal(calls[0].deps.pollUntilDone, poller);
    assert.equal(calls[0].deps.signal, signal);
    assert.equal(success.endReason, 'completed');
    assert.deepEqual(success.songIds, ['song-1']);
    assert.equal(JSON.stringify(success.artifacts), JSON.stringify([
      { kind: 'audio', trackIndex: 0, url: '/audio/raw.wav' },
      { kind: 'mastered', trackIndex: 0, url: '/audio/mastered.wav' },
      { kind: 'noadapter', trackIndex: 0, url: '/audio/noadapter.wav' },
    ]));
    assert.deepEqual(success.result, successJob.result);

    modes.current = 'failure';
    const failed = await generate({ status: 'pending' }, { pollUntilDone: poller, signal });
    assert.equal(failed.endReason, 'failed');
    assert.equal(failed.error, 'engine failure consumed by runner');

    modes.current = 'mapper';
    await assert.rejects(generate({ status: 'pending' }, { pollUntilDone: poller, signal }), error => error === mapperError);
  });
}

// llm/types.ts — Shared types for the LLM provider system

export interface ProviderInfo {
  id: string;
  name: string;
  available: boolean;
  models: string[];
  default_model: string;
}

export interface GenerationResponse {
  lyrics: string;
  provider: string;
  model: string;
  title: string;
  subject: string;
  bpm: number;
  key: string;
  caption: string;
  duration: number;
  system_prompt: string;
  user_prompt: string;
}

export type ChunkCallback = (chunk: string) => void;

export interface CallOptions {
  temperature?: number;
  top_p?: number;
  /**
   * Best-effort "answer without reasoning" for local thinking models.
   * There is NO universal off-switch across runtimes, so providers layer
   * every mechanism that is harmless where unsupported:
   *  - `reasoning_effort: 'none'` (LM Studio — EMPIRICALLY VERIFIED 2026-07-09
   *    on Qwen3.6: reasoning drops to zero; gemma accepts it harmlessly)
   *  - `/no_think` soft switch appended to the system prompt (older Qwen3;
   *    Qwen3.5+ dropped it — verified ignored on Qwen3.6)
   *  - `chat_template_kwargs: { enable_thinking: false }` (llama.cpp server;
   *    LM Studio ignores it — verified)
   *  - `think: false` (Ollama native)
   * Models with no supported mechanism still think; stripThinkingBlocks()
   * downstream keeps the output clean either way.
   */
  noThink?: boolean;
  [key: string]: any;
}

// Global skip thinking signal
export let skipThinkingSignal = false;
export function setSkipThinking() {
  skipThinkingSignal = true;
  console.log('[LLM] Skip-thinking signal received');
}
export function resetSkipThinking() {
  skipThinkingSignal = false;
}

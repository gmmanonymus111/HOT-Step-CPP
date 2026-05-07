// assistantApi.ts — Frontend API client for the AI Assistant
//
// Provides streaming chat via SSE (fetch + ReadableStream) and
// provider listing. Same pattern as Lyric Studio's streaming calls.

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AssistantAction {
  set: string;
  value: any;
}

export interface ChatStreamParams {
  message: string;
  history: ChatMessage[];
  currentSettings: Record<string, any>;
  provider: string;
  model?: string;
}

export interface AssistantProvider {
  id: string;
  name: string;
  available: boolean;
  models: string[];
  default_model: string;
}

/**
 * Stream a chat response from the assistant.
 * Returns an AbortController so the caller can cancel mid-stream.
 */
export function chatStream(
  params: ChatStreamParams,
  onChunk: (text: string) => void,
  onComplete: (fullText: string) => void,
  onError: (error: string) => void,
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch('/api/assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: controller.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        onError(err.error || `HTTP ${res.status}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        onError('No response body');
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            // Store event type for the next data line
            (reader as any).__lastEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            const eventType = (reader as any).__lastEvent || 'chunk';
            const dataStr = line.slice(6).trim();
            try {
              const data = JSON.parse(dataStr);
              if (eventType === 'chunk') {
                onChunk(data.text || '');
              } else if (eventType === 'complete') {
                onComplete(data.text || '');
              } else if (eventType === 'error') {
                onError(data.error || 'Unknown error');
              }
            } catch {
              // Ignore malformed JSON
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        onError(err.message || 'Connection failed');
      }
    }
  })();

  return controller;
}

/**
 * Parse action blocks from assistant response text.
 * Actions are embedded in fenced code blocks with the "actions" language tag.
 */
export function parseActions(text: string): AssistantAction[] {
  const regex = /```actions\s*\n([\s\S]*?)```/g;
  const actions: AssistantAction[] = [];
  let match;
  while ((match = regex.exec(text)) !== null) {
    try {
      const parsed = JSON.parse(match[1].trim());
      if (Array.isArray(parsed)) {
        actions.push(...parsed.filter(a => a && typeof a.set === 'string'));
      }
    } catch {
      // Ignore malformed action blocks
    }
  }
  return actions;
}

/**
 * Strip action blocks from text to get the display-only content.
 */
export function stripActionBlocks(text: string): string {
  return text.replace(/```actions\s*\n[\s\S]*?```/g, '').trim();
}

/**
 * Fetch available LLM providers for the assistant.
 */
export async function getProviders(): Promise<AssistantProvider[]> {
  const res = await fetch('/api/assistant/providers');
  if (!res.ok) throw new Error(`Failed to fetch providers: ${res.statusText}`);
  return res.json();
}

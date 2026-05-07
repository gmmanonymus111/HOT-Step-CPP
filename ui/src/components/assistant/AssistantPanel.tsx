// AssistantPanel.tsx — AI Assistant sidebar with streaming chat
//
// Follows the same sidebar pattern as TerminalPanel.tsx. Streams LLM
// responses via SSE, parses action blocks, and applies settings changes
// through GlobalParamsContext setters.

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { X, Send, Trash2, Sparkles, ArrowRight, Check } from 'lucide-react';
import { usePersistedState } from '../../hooks/usePersistedState';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { useAssistantActions, type ActionDiff } from '../../hooks/useAssistantActions';
import { SimpleMarkdown } from './SimpleMarkdown';
import {
  chatStream,
  parseActions,
  stripActionBlocks,
  stripThinkingBlocks,
  extractThinkingAndResponse,
  getProviders,
  type ChatMessage,
  type AssistantAction,
  type AssistantProvider,
} from '../../services/assistantApi';
import './AssistantPanel.css';

interface AssistantPanelProps {
  onClose: () => void;
}

interface DisplayMessage {
  id: number;
  role: 'user' | 'assistant' | 'error';
  content: string;
  rawContent?: string;  // original LLM output (with thinking tags) for re-parsing
  actions?: AssistantAction[];
  actionsApplied?: boolean;
}

let messageIdCounter = 0;

export const AssistantPanel: React.FC<AssistantPanelProps> = ({ onClose }) => {
  // ── Provider state ──
  const [providers, setProviders] = useState<AssistantProvider[]>([]);
  const [selectedProvider, setSelectedProvider] = usePersistedState('hs-assistant-provider', '');
  const [selectedModel, setSelectedModel] = usePersistedState('hs-assistant-model', '');

  // ── Chat state ──
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState('');

  // ── Refs ──
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // ── Hooks ──
  const globalParams = useGlobalParams();
  const { applyActions, previewActions } = useAssistantActions();

  // ── Load providers on mount ──
  useEffect(() => {
    getProviders()
      .then((p) => {
        setProviders(p);
        // Auto-select first available provider if none persisted
        if (!selectedProvider) {
          const first = p.find(pr => pr.available);
          if (first) {
            setSelectedProvider(first.id);
            setSelectedModel(first.default_model);
          }
        }
      })
      .catch(err => console.error('[Assistant] Failed to load providers:', err));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Auto-scroll ──
  useEffect(() => {
    requestAnimationFrame(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    });
  }, [messages, streamingText]);

  // ── Get current provider's models ──
  const currentProvider = providers.find(p => p.id === selectedProvider);
  const availableModels = currentProvider?.models || [];

  // When provider changes, auto-select its default model
  const handleProviderChange = useCallback((providerId: string) => {
    setSelectedProvider(providerId);
    const p = providers.find(pr => pr.id === providerId);
    if (p) setSelectedModel(p.default_model);
  }, [providers, setSelectedProvider, setSelectedModel]);

  // ── Build history for context ──
  const buildHistory = useCallback((): ChatMessage[] => {
    return messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      }));
  }, [messages]);

  // ── Send message ──
  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || isStreaming || !selectedProvider) return;

    // Add user message
    const userMsg: DisplayMessage = {
      id: ++messageIdCounter,
      role: 'user',
      content: text,
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsStreaming(true);
    setStreamingText('');

    // Get current settings snapshot
    const currentSettings = globalParams.getGlobalParams();

    // Start streaming
    const abort = chatStream(
      {
        message: text,
        history: buildHistory(),
        currentSettings,
        provider: selectedProvider,
        model: selectedModel || undefined,
      },
      // onChunk
      (chunk) => {
        setStreamingText(prev => prev + chunk);
      },
      // onComplete
      (fullText) => {
        // Parse actions from the clean (non-thinking) text
        const cleanText = stripThinkingBlocks(fullText);
        const actions = parseActions(cleanText);
        const displayContent = stripActionBlocks(cleanText);

        const assistantMsg: DisplayMessage = {
          id: ++messageIdCounter,
          role: 'assistant',
          content: displayContent,
          rawContent: fullText,  // keep raw for thinking extraction
          actions: actions.length > 0 ? actions : undefined,
        };
        setMessages(prev => [...prev, assistantMsg]);
        setStreamingText('');
        setIsStreaming(false);
      },
      // onError
      (error) => {
        const errorMsg: DisplayMessage = {
          id: ++messageIdCounter,
          role: 'error',
          content: error,
        };
        setMessages(prev => [...prev, errorMsg]);
        setStreamingText('');
        setIsStreaming(false);
      },
    );

    abortRef.current = abort;
  }, [input, isStreaming, selectedProvider, selectedModel, globalParams, buildHistory]);

  // ── Handle Enter key ──
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  // ── Apply actions from a message ──
  const handleApply = useCallback((msgId: number, actions: AssistantAction[]) => {
    const count = applyActions(actions);
    console.log(`[Assistant] Applied ${count} setting changes`);
    setMessages(prev => prev.map(m =>
      m.id === msgId ? { ...m, actionsApplied: true } : m
    ));
  }, [applyActions]);

  // ── Clear chat ──
  const handleClear = useCallback(() => {
    if (isStreaming && abortRef.current) {
      abortRef.current.abort();
    }
    setMessages([]);
    setStreamingText('');
    setIsStreaming(false);
  }, [isStreaming]);

  // ── Quick suggestion ──
  const handleSuggestion = useCallback((text: string) => {
    setInput(text);
    // Focus the input after setting
    requestAnimationFrame(() => inputRef.current?.focus());
  }, []);

  // ── Auto-resize textarea ──
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, []);

  // ── Render ──
  return (
    <div className="assistant-panel">
      {/* Header */}
      <div className="assistant-header">
        <div className="assistant-header-left">
          <Sparkles size={14} style={{ color: 'rgba(139, 92, 246, 0.7)' }} />
          <span className="assistant-title">Assistant</span>
        </div>
        <div className="assistant-header-actions">
          <button onClick={handleClear} className="assistant-header-btn" title="Clear chat">
            <Trash2 size={12} />
          </button>
          <button onClick={onClose} className="assistant-header-btn" title="Close assistant">
            <X size={12} />
          </button>
        </div>
      </div>

      {/* Provider & model selectors */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-zinc-200 dark:border-white/5 bg-zinc-50/80 dark:bg-zinc-900/50">
        <div className="flex-1 min-w-0">
          <label className="block text-[10px] font-medium text-zinc-500 uppercase tracking-wider mb-1">Provider</label>
          <select
            className="w-full px-3 py-1.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/20 outline-none transition-colors cursor-pointer"
            value={selectedProvider}
            onChange={(e) => handleProviderChange(e.target.value)}
          >
            {providers.length === 0 && <option value="">Loading...</option>}
            {providers.map(p => (
              <option key={p.id} value={p.id} disabled={!p.available}>
                {p.name}{!p.available ? ' (unavailable)' : ''}
              </option>
            ))}
          </select>
        </div>
        <div className="flex-1 min-w-0">
          <label className="block text-[10px] font-medium text-zinc-500 uppercase tracking-wider mb-1">Model</label>
          <select
            className="w-full px-3 py-1.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/20 outline-none transition-colors cursor-pointer"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {availableModels.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
            {availableModels.length === 0 && (
              <option value="">Default</option>
            )}
          </select>
        </div>
      </div>

      {/* Messages or welcome screen */}
      {messages.length === 0 && !isStreaming ? (
        <div className="assistant-welcome">
          <div className="assistant-welcome-icon">✨</div>
          <div className="assistant-welcome-title">HOT-Step Assistant</div>
          <div className="assistant-welcome-subtitle">
            I can help you configure settings, recommend presets, troubleshoot issues, and adjust parameters directly.
          </div>
          <div className="assistant-welcome-suggestions">
            <button className="assistant-suggestion-btn" onClick={() => handleSuggestion('Set me up for lo-fi hip hop')}>
              🎵 Set me up for lo-fi hip hop
            </button>
            <button className="assistant-suggestion-btn" onClick={() => handleSuggestion('What solver should I use for clean vocals?')}>
              🎤 Best solver for clean vocals?
            </button>
            <button className="assistant-suggestion-btn" onClick={() => handleSuggestion('Review my current settings and suggest improvements')}>
              🔧 Review my current settings
            </button>
            <button className="assistant-suggestion-btn" onClick={() => handleSuggestion('How do I reduce metallic artifacts?')}>
              🔇 Fix metallic artifacts
            </button>
          </div>
        </div>
      ) : (
        <div className="assistant-messages" ref={scrollRef}>
          {messages.map((msg) => {
            // Extract thinking for assistant messages
            const parsed = msg.role === 'assistant' && msg.rawContent
              ? extractThinkingAndResponse(msg.rawContent)
              : null;

            return (
              <React.Fragment key={msg.id}>
                {/* Thinking block (collapsible) */}
                {parsed?.thinking && (
                  <ThinkingBlock thinking={parsed.thinking} isStreaming={false} />
                )}

                {/* Message bubble */}
                <div className={`assistant-msg assistant-msg--${msg.role}`}>
                  {msg.role === 'assistant'
                    ? <SimpleMarkdown content={msg.content} />
                    : msg.content
                  }
                </div>

                {/* Action card */}
                {msg.role === 'assistant' && msg.actions && msg.actions.length > 0 && (
                  <ActionCard
                    actions={msg.actions}
                    applied={msg.actionsApplied || false}
                    onApply={() => handleApply(msg.id, msg.actions!)}
                    previewActions={previewActions}
                  />
                )}
              </React.Fragment>
            );
          })}

          {/* Streaming message */}
          {isStreaming && streamingText && (() => {
            const parsed = extractThinkingAndResponse(streamingText);
            const isStillThinking = parsed.thinking && !parsed.response;
            return (
              <>
                {parsed.thinking && (
                  <ThinkingBlock thinking={parsed.thinking} isStreaming={isStillThinking || false} />
                )}
                {parsed.response && (
                  <div className="assistant-msg assistant-msg--assistant assistant-msg--streaming">
                    <SimpleMarkdown content={parsed.response} />
                    <span className="assistant-cursor" />
                  </div>
                )}
                {!parsed.response && !parsed.thinking && (
                  <div className="assistant-msg assistant-msg--assistant assistant-msg--streaming">
                    <span className="assistant-cursor" />
                  </div>
                )}
              </>
            );
          })()}

          {/* Streaming but no text yet */}
          {isStreaming && !streamingText && (
            <div className="assistant-msg assistant-msg--assistant assistant-msg--streaming">
              <span className="assistant-cursor" />
            </div>
          )}
        </div>
      )}

      {/* Input bar */}
      <div className="assistant-input-bar">
        <textarea
          ref={inputRef}
          className="assistant-input"
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder={isStreaming ? 'Waiting for response...' : 'Ask about settings, request a preset...'}
          disabled={isStreaming}
          rows={1}
        />
        <button
          className="assistant-send-btn"
          onClick={handleSend}
          disabled={isStreaming || !input.trim() || !selectedProvider}
          title="Send message"
        >
          <Send size={16} />
        </button>
      </div>
    </div>
  );
};

// ── Action Card sub-component ─────────────────────────────────────────────────

interface ActionCardProps {
  actions: AssistantAction[];
  applied: boolean;
  onApply: () => void;
  previewActions: (actions: AssistantAction[]) => ActionDiff[];
}

const ActionCard: React.FC<ActionCardProps> = ({ actions, applied, onApply, previewActions }) => {
  const diffs = previewActions(actions);

  // Format a value for display
  const fmt = (v: any): string => {
    if (v === undefined || v === null) return '—';
    if (typeof v === 'boolean') return v ? 'ON' : 'OFF';
    if (typeof v === 'number') return String(v);
    return String(v);
  };

  return (
    <div className="assistant-action-card">
      <div className="assistant-action-card-title">
        <Sparkles size={11} />
        Suggested Changes
      </div>
      <div className="assistant-action-diff">
        {diffs.map((d) => (
          <div key={d.key} className="assistant-action-row">
            <span className="assistant-action-label">{d.label}</span>
            <span className="assistant-action-from">{fmt(d.from)}</span>
            <ArrowRight size={10} className="assistant-action-arrow" />
            <span className="assistant-action-to">{fmt(d.to)}</span>
          </div>
        ))}
      </div>
      <div className="assistant-action-buttons">
        {applied ? (
          <button className="assistant-apply-btn assistant-apply-btn--applied" disabled>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px', justifyContent: 'center' }}>
              <Check size={12} /> Applied
            </span>
          </button>
        ) : (
          <button className="assistant-apply-btn" onClick={onApply}>
            Apply All
          </button>
        )}
      </div>
    </div>
  );
};

// ── Thinking Block sub-component ──────────────────────────────────────────────

interface ThinkingBlockProps {
  thinking: string;
  isStreaming: boolean;
}

const ThinkingBlock: React.FC<ThinkingBlockProps> = ({ thinking, isStreaming }) => (
  <details className="assistant-thinking" open={isStreaming}>
    <summary className="assistant-thinking-summary">
      <svg className="thinking-chevron" width="10" height="10" viewBox="0 0 10 10" fill="currentColor">
        <path d="M3 1.5L7 5L3 8.5" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" />
      </svg>
      <span className={isStreaming ? 'assistant-thinking-label' : ''}>
        💭 {isStreaming ? 'Thinking...' : 'Thought process'}
      </span>
    </summary>
    <div className="assistant-thinking-content">
      {thinking}
      {isStreaming && <span className="assistant-cursor" />}
    </div>
  </details>
);

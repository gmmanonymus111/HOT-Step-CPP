// SimpleMarkdown.tsx — Lightweight markdown renderer for assistant chat
//
// Handles the subset of markdown that LLMs commonly output:
// - Headings (###, ##, #)
// - Bold (**text**)
// - Italic (*text*)
// - Inline code (`code`)
// - Fenced code blocks (```...```)
// - Unordered lists (- item, * item)
// - Ordered lists (1. item)
// - Horizontal rules (---, ***)
// - Line breaks
//
// No external dependencies. Intentionally simple — this isn't a full
// markdown parser, just enough to make LLM responses readable.

import React from 'react';

/** Parse inline markdown (bold, italic, code) within a text string */
function parseInline(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  // Regex: inline code, bold, italic (in priority order)
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)/g;
  let lastIndex = 0;
  let match;

  while ((match = re.exec(text)) !== null) {
    // Text before the match
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }

    if (match[1]) {
      // Inline code
      const code = match[1].slice(1, -1);
      nodes.push(
        <code key={match.index} className="smd-inline-code">{code}</code>
      );
    } else if (match[2]) {
      // Bold
      const bold = match[2].slice(2, -2);
      nodes.push(
        <strong key={match.index} className="smd-bold">{bold}</strong>
      );
    } else if (match[3]) {
      // Italic
      const italic = match[3].slice(1, -1);
      nodes.push(
        <em key={match.index} className="smd-italic">{italic}</em>
      );
    }

    lastIndex = match.index + match[0].length;
  }

  // Remaining text
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes.length > 0 ? nodes : [text];
}

interface SimpleMarkdownProps {
  content: string;
}

export const SimpleMarkdown: React.FC<SimpleMarkdownProps> = ({ content }) => {
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // Empty line → spacer
    if (!trimmed) {
      elements.push(<div key={i} className="smd-spacer" />);
      i++;
      continue;
    }

    // Fenced code block
    if (trimmed.startsWith('```')) {
      const lang = trimmed.slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      elements.push(
        <pre key={`code-${i}`} className="smd-code-block">
          {lang && <span className="smd-code-lang">{lang}</span>}
          <code>{codeLines.join('\n')}</code>
        </pre>
      );
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}$/.test(trimmed)) {
      elements.push(<hr key={i} className="smd-hr" />);
      i++;
      continue;
    }

    // Headings
    if (trimmed.startsWith('### ')) {
      elements.push(
        <h4 key={i} className="smd-h3">{parseInline(trimmed.slice(4))}</h4>
      );
      i++;
      continue;
    }
    if (trimmed.startsWith('## ')) {
      elements.push(
        <h3 key={i} className="smd-h2">{parseInline(trimmed.slice(3))}</h3>
      );
      i++;
      continue;
    }
    if (trimmed.startsWith('# ')) {
      elements.push(
        <h2 key={i} className="smd-h1">{parseInline(trimmed.slice(2))}</h2>
      );
      i++;
      continue;
    }

    // Unordered list items (collect consecutive)
    if (/^[-*•]\s/.test(trimmed)) {
      const items: { key: number; content: React.ReactNode[] }[] = [];
      while (i < lines.length && /^[-*•]\s/.test(lines[i].trim())) {
        items.push({ key: i, content: parseInline(lines[i].trim().slice(2)) });
        i++;
      }
      elements.push(
        <ul key={`ul-${items[0].key}`} className="smd-ul">
          {items.map(it => <li key={it.key} className="smd-li">{it.content}</li>)}
        </ul>
      );
      continue;
    }

    // Ordered list items (collect consecutive)
    if (/^\d+[.)]\s/.test(trimmed)) {
      const items: { key: number; content: React.ReactNode[] }[] = [];
      while (i < lines.length && /^\d+[.)]\s/.test(lines[i].trim())) {
        const text = lines[i].trim().replace(/^\d+[.)]\s/, '');
        items.push({ key: i, content: parseInline(text) });
        i++;
      }
      elements.push(
        <ol key={`ol-${items[0].key}`} className="smd-ol">
          {items.map(it => <li key={it.key} className="smd-li">{it.content}</li>)}
        </ol>
      );
      continue;
    }

    // Regular paragraph
    elements.push(
      <p key={i} className="smd-p">{parseInline(trimmed)}</p>
    );
    i++;
  }

  return <div className="smd-root">{elements}</div>;
};

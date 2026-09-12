import { useMemo } from "react";
import { MarkdownMessage } from "./MarkdownMessage";
import type { Citation } from "../api/client";
import { AgentTrace, TurnMetricsBar } from "./AgentTrace";
import type { TurnTrace } from "./AgentTrace";

export type Message = {
  role: "user" | "assistant" | "error";
  text: string;
  trace?: TurnTrace;
};

export function MessageList({
  messages,
  citations,
}: {
  messages: Message[];
  citations: Citation[];
}) {
  // The same cid keeps the same number across every message, matching the source
  // sidebar's numbering, so a `[n]` link always lands on the right card.
  const citationOrder = useMemo(
    () => new Map(citations.map((item, index) => [item.cid, index + 1])),
    [citations],
  );

  return (
    <div className="message-list" aria-live="polite">
      {messages.map((message, index) => (
        <article className={`message message-${message.role}`} key={`${message.role}-${index}`}>
          <span className="message-role">{message.role === "user" ? "你" : message.role === "error" ? "错误" : "FinHarness"}</span>
          {message.role === "assistant" && message.trace && <AgentTrace trace={message.trace} />}
          {message.role === "assistant" ? (
            <MarkdownMessage text={message.text} citationOrder={citationOrder} />
          ) : (
            <p className="message-plain">{message.text}</p>
          )}
          {message.role === "assistant" && message.trace?.metrics && (
            <TurnMetricsBar metrics={message.trace.metrics} />
          )}
        </article>
      ))}
    </div>
  );
}

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
  // 同一个 cid 在每条消息中保持相同编号，与数据来源
  // 侧栏的编号一致，因此 `[n]` 链接总能落到正确的卡片上。
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

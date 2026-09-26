import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MarkdownMessage } from "./MarkdownMessage";
import type { Citation } from "../api/client";
import { TurnMetricsBar } from "./AgentTrace";
import type { TurnTrace } from "./AgentTrace";
import { LiveStatusOverlay } from "./LiveStatusOverlay";
import { isNearBottom } from "../lib/autoScroll";

export type Message = {
  role: "user" | "assistant" | "error";
  text: string;
  trace?: TurnTrace;
  /** 触发这轮失败的用户提问。带上它，错误气泡才能提供一键重试，
   * 用户不必凭记忆重打一遍长问题。 */
  retryPrompt?: string;
};

/** 单条消息行。流式期间 ChatPanel 只替换最后一条 assistant 消息的对象
 * 引用（其余消息 identity 不变），memo 让历史行跳过重渲染——否则每个
 * delta 到达时，全部历史消息的 markdown 都会重新解析一遍，长会话后期
 * 表现为打字卡顿。citationOrder 已在父级 useMemo，prop 引用稳定；
 * onRetry 由父组件用 ref 兜底，同样引用稳定。 */
const MessageRow = memo(function MessageRow({
  message,
  citationOrder,
  onRetry,
  onOpenProcess,
}: {
  message: Message;
  citationOrder: Map<string, number>;
  onRetry?: (prompt: string) => void;
  onOpenProcess?: (toolName?: string) => void;
}) {
  return (
    <article className={`message message-${message.role}`}>
      <span className="message-role">{message.role === "user" ? "你" : message.role === "error" ? "错误" : "FinHarness"}</span>
      {message.role === "assistant" && message.trace && (
        <LiveStatusOverlay trace={message.trace} onOpenProcess={onOpenProcess} />
      )}
      {message.role === "assistant" ? (
        <MarkdownMessage text={message.text} citationOrder={citationOrder} />
      ) : (
        <p className="message-plain">{message.text}</p>
      )}
      {message.role === "error" && message.retryPrompt && onRetry && (
        <div className="message-error-actions">
          <button type="button" className="error-retry-button" onClick={() => onRetry(message.retryPrompt ?? "")}>
            重试这轮
          </button>
        </div>
      )}
      {message.role === "assistant" && message.trace?.metrics && (
        <TurnMetricsBar metrics={message.trace.metrics} />
      )}
    </article>
  );
});

export function MessageList({
  messages,
  citations,
  onRetry,
  onOpenProcess,
}: {
  messages: Message[];
  citations: Citation[];
  onRetry?: (prompt: string) => void;
  onOpenProcess?: (toolName?: string) => void;
}) {
  // 同一个 cid 在每条消息中保持相同编号，与数据来源
  // 侧栏的编号一致，因此 `[n]` 链接总能落到正确的卡片上。
  const citationOrder = useMemo(
    () => new Map(citations.map((item, index) => [item.cid, index + 1])),
    [citations],
  );
  // 保持回调引用稳定，memo 的历史行才不会因父组件重渲染而全部失效。
  const onRetryRef = useRef(onRetry);
  onRetryRef.current = onRetry;
  const handleRetry = useCallback(
    (prompt: string) => onRetryRef.current?.(prompt),
    [],
  );
  const onOpenProcessRef = useRef(onOpenProcess);
  onOpenProcessRef.current = onOpenProcess;
  const handleOpenProcess = useCallback(
    (toolName?: string) => onOpenProcessRef.current?.(toolName),
    [],
  );
  // 跟随开关（ref 而非 state）：滚动决策在每个流事件后发生，不值得
  // 为它触发一次渲染。
  const followRef = useRef(true);
  // 上一条消息的 role：用户新发送 / 错误气泡到达时强制回到底部，
  // 即使用户此前已上翻阅读。
  const lastRoleRef = useRef<string | null>(null);

  useEffect(() => {
    // 保存历史后重进一个旧对话：直接落在末尾，与实时对话一致。
    if (messages.length > 0 && lastRoleRef.current === null) {
      lastRoleRef.current = messages[messages.length - 1].role;
      window.scrollTo({ top: document.documentElement.scrollHeight });
      return;
    }
    const last = messages[messages.length - 1];
    const lastRole = last?.role ?? null;
    const lastMessageChanged = lastRole !== lastRoleRef.current;
    lastRoleRef.current = lastRole;
    if (!last) return;
    // 用户新发送或错误气泡到达：无条件回到底部（用户主动触发的事件，
    // 视线应该跟过去）。assistant 流式更新则尊重当前跟随状态。
    if (lastMessageChanged && lastRole !== "assistant") {
      followRef.current = true;
    }
    if (followRef.current) {
      window.scrollTo({ top: document.documentElement.scrollHeight });
    }
  }, [messages]);

  useEffect(() => {
    // 页面本身是滚动容器（message-list 无独立 overflow），因此监听 window。
    const onScroll = () => {
      const doc = document.documentElement;
      if (isNearBottom(window.scrollY, window.innerHeight, doc.scrollHeight)) {
        followRef.current = true;
      }
    };
    const onWheel = (event: WheelEvent) => {
      // 上翻即退出跟随；下翻只是靠近底部，由 scroll 事件按阈值恢复。
      if (event.deltaY < 0) followRef.current = false;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("wheel", onWheel, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("wheel", onWheel);
    };
  }, []);

  // 屏幕阅读器播报：aria-live 放在整个消息列表上，流式的每个 delta 都
  // 会被当成新内容反复尝试播报，反而无法收听。改为独立的 sr-only 区域，
  // 在轮次开始/结束时各播报一次摘要；列表本身不再带 aria-live。
  const [announcement, setAnnouncement] = useState("");
  useEffect(() => {
    const first = messages[0];
    const last = messages[messages.length - 1];
    if (!first) return;
    if (last?.role === "assistant" && last.trace?.status === "running") {
      setAnnouncement(`FinHarness 正在研究：${first.text}`);
    } else if (last?.role === "assistant") {
      setAnnouncement(`FinHarness 的研究结论已生成：${first.text}`);
    } else if (last?.role === "error") {
      setAnnouncement("本轮执行遇到错误，可在错误气泡上重试。");
    } else if (last?.role === "user") {
      setAnnouncement(`已提交研究问题：${last.text}`);
    }
  }, [messages]);

  return (
    <>
      <div className="message-list">
        {messages.map((message, index) => (
          <MessageRow
            key={`${message.role}-${index}`}
            message={message}
            citationOrder={citationOrder}
            onRetry={handleRetry}
            onOpenProcess={handleOpenProcess}
          />
        ))}
      </div>
      <div className="sr-only" aria-live="polite">{announcement}</div>
    </>
  );
}

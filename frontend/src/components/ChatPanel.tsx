import { FormEvent, useRef, useState } from "react";
import { Button, Empty } from "antd";
import { artifactUrl, respondChat, streamChat } from "../api/client";
import { Interaction, InteractionPrompt } from "./InteractionPrompt";
import { Message, MessageList } from "./MessageList";
import { ToolStatus } from "./ToolStatus";

type Props = {
  sessionId: string | null;
  onSession: (id: string) => void;
  configured: boolean;
  onOpenSettings: () => void;
};

/** A file the engine produced (chart, report) that the user can download. */
type Artifact = { path: string; name: string };

function fileName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

export function ChatPanel({ sessionId, onSession, configured, onOpenSettings }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [interaction, setInteraction] = useState<Interaction | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  function rememberArtifacts(raw: unknown) {
    const paths = (raw as string[] | undefined) ?? [];
    if (paths.length === 0) return;
    setArtifacts((current) => {
      const seen = new Set(current.map((item) => item.path));
      const added = paths
        .filter((path) => !seen.has(path))
        .map((path) => ({ path, name: fileName(path) }));
      return added.length ? [...current, ...added] : current;
    });
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const message = input.trim();
    if (!message || busy) return;
    setInput("");
    setMessages((current) => [...current, { role: "user", text: message }, { role: "assistant", text: "" }]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    setNotice(null);
    try {
      await streamChat(message, sessionId, (event) => {
        if (event.event === "session") onSession(String(event.data.session_id));
        if (event.event === "tool_status") {
          setStatus(`${String(event.data.name ?? "tool")} ${String(event.data.status ?? "")}`);
          rememberArtifacts(event.data.attachments);
        }
        // Engine notices the UI should not swallow: window compaction and the
        // loop guard both change what the user is looking at.
        if (event.event === "context_compacted") {
          const before = Number(event.data.before_tokens ?? 0);
          const after = Number(event.data.after_tokens ?? 0);
          const degraded = Boolean(event.data.degraded);
          setNotice(
            `上下文已达上限，已压缩历史（${before} → ${after} tokens）${degraded ? "，摘要降级" : ""}`,
          );
        }
        if (event.event === "loop_guard") {
          const action = String(event.data.action ?? "refused");
          setNotice(
            action === "would_abort"
              ? `检测到重复调用，已终止本轮：${String(event.data.name ?? "")}`
              : `检测到重复调用，已跳过并提示模型改用已有结果：${String(event.data.name ?? "")}`,
          );
        }
        if (event.event === "interactive_request") {
          // The engine is paused awaiting an answer; surface the dialog.
          setInteraction({
            requestId: String(event.data.request_id ?? ""),
            kind: String(event.data.kind ?? "question"),
            prompt: String(event.data.prompt ?? ""),
            options: (event.data.options as string[] | undefined) ?? [],
          });
        }
        if (event.event === "delta") {
          setMessages((current) => {
            const next = [...current];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, text: last.text + String(event.data.text ?? "") };
            return next;
          });
        }
        if (event.event === "error") {
          setInteraction(null);
          setMessages((current) => [...current, { role: "error", text: String(event.data.message ?? "Unknown error") }]);
        }
        if (event.event === "done") {
          setStatus(null);
          setInteraction(null);
        }
      }, controller.signal);
    } catch (error) {
      if ((error as Error).name !== "AbortError") setMessages((current) => [...current, { role: "error", text: (error as Error).message }]);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  async function handleRespond(requestId: string, response: string) {
    setInteraction(null);
    try {
      await respondChat(requestId, response);
    } catch (error) {
      setMessages((current) => [...current, { role: "error", text: `回答提交失败：${(error as Error).message}` }]);
    }
  }

  return (
    <section className="chat-panel">
      {!configured && messages.length === 0 && (
        <Empty
          className="config-empty-state"
          description="尚未配置模型供应商，配置后即可开始对话"
        >
          <Button type="primary" onClick={onOpenSettings}>
            去配置供应商
          </Button>
        </Empty>
      )}
      <MessageList messages={messages} />
      {notice && <div className="engine-notice">{notice}</div>}
      {artifacts.length > 0 && (
        <div className="artifact-list" aria-label="产出文件">
          <span className="artifact-title">产出文件</span>
          {artifacts.map((item) => (
            <a key={item.path} className="artifact-link" href={artifactUrl(item.path)} download>
              {item.name}
            </a>
          ))}
        </div>
      )}
      <ToolStatus status={status} />
      <form className="composer" onSubmit={submit}>
        <textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="输入研究问题" rows={3} disabled={busy} />
        <button type="submit" disabled={busy || !input.trim() || !configured}>{busy ? "处理中" : "发送"}</button>
      </form>
      <InteractionPrompt interaction={interaction} onRespond={handleRespond} busy={false} />
    </section>
  );
}

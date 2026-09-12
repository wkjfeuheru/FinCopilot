import { FormEvent, useRef, useState } from "react";
import { Button, Empty } from "antd";
import { respondChat, streamChat } from "../api/client";
import { Interaction, InteractionPrompt } from "./InteractionPrompt";
import { Message, MessageList } from "./MessageList";
import { ToolStatus } from "./ToolStatus";

type Props = {
  sessionId: string | null;
  onSession: (id: string) => void;
  configured: boolean;
  onOpenSettings: () => void;
};

export function ChatPanel({ sessionId, onSession, configured, onOpenSettings }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [interaction, setInteraction] = useState<Interaction | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const message = input.trim();
    if (!message || busy) return;
    setInput("");
    setMessages((current) => [...current, { role: "user", text: message }, { role: "assistant", text: "" }]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(message, sessionId, (event) => {
        if (event.event === "session") onSession(String(event.data.session_id));
        if (event.event === "tool_status") setStatus(`${String(event.data.name ?? "tool")} ${String(event.data.status ?? "")}`);
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
      <ToolStatus status={status} />
      <form className="composer" onSubmit={submit}>
        <textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="输入研究问题" rows={3} disabled={busy} />
        <button type="submit" disabled={busy || !input.trim() || !configured}>{busy ? "处理中" : "发送"}</button>
      </form>
      <InteractionPrompt interaction={interaction} onRespond={handleRespond} busy={false} />
    </section>
  );
}

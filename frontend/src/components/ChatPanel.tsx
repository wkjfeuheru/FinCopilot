import { FormEvent, useRef, useState } from "react";
import { streamChat } from "../api/client";
import { Message, MessageList } from "./MessageList";
import { ToolStatus } from "./ToolStatus";

type Props = { sessionId: string | null; onSession: (id: string) => void };

export function ChatPanel({ sessionId, onSession }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
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
        if (event.event === "delta") {
          setMessages((current) => {
            const next = [...current];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, text: last.text + String(event.data.text ?? "") };
            return next;
          });
        }
        if (event.event === "error") setMessages((current) => [...current, { role: "error", text: String(event.data.message ?? "Unknown error") }]);
        if (event.event === "done") setStatus(null);
      }, controller.signal);
    } catch (error) {
      if ((error as Error).name !== "AbortError") setMessages((current) => [...current, { role: "error", text: (error as Error).message }]);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  return (
    <section className="chat-panel">
      <MessageList messages={messages} />
      <ToolStatus status={status} />
      <form className="composer" onSubmit={submit}>
        <textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="输入研究问题" rows={3} disabled={busy} />
        <button type="submit" disabled={busy || !input.trim()}>{busy ? "处理中" : "发送"}</button>
      </form>
    </section>
  );
}

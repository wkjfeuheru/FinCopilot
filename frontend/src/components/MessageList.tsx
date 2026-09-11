export type Message = { role: "user" | "assistant" | "error"; text: string };

export function MessageList({ messages }: { messages: Message[] }) {
  return (
    <div className="message-list" aria-live="polite">
      {messages.map((message, index) => (
        <article className={`message message-${message.role}`} key={`${message.role}-${index}`}>
          <span className="message-role">{message.role === "user" ? "你" : message.role === "error" ? "错误" : "FinHarness"}</span>
          <p>{message.text}</p>
        </article>
      ))}
    </div>
  );
}

type Props = { sessionId: string | null; onNewSession: () => void };

export function SessionBar({ sessionId, onNewSession }: Props) {
  return (
    <div className="session-bar">
      <span>{sessionId ? `会话 ${sessionId}` : "新会话"}</span>
      <button type="button" onClick={onNewSession}>新会话</button>
    </div>
  );
}

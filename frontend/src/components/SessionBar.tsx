type Props = {
  sessionId: string | null;
  conversationId: string | null;
  onNewSession: () => void;
};

export function SessionBar({ sessionId, conversationId, onNewSession }: Props) {
  const label = conversationId ? `对话 ${conversationId}` : "新对话";
  return (
    <div className="session-bar">
      <span>{label}</span>
      <button type="button" onClick={onNewSession}>新建对话</button>
    </div>
  );
}

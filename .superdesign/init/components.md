# Shared UI components

## ConversationList

- Source: `frontend/src/components/ConversationList.tsx`
- Purpose: conversation history, refresh, create and delete actions.

```tsx
export function ConversationList({ conversations, activeId, loading, onSelect, onNew, onRefresh, onDelete }: Props) {
  return <aside className="conversation-list">{/* Ant Design history list */}</aside>;
}
```

## SessionBar

- Source: `frontend/src/components/SessionBar.tsx`
- Purpose: current conversation identity and new-conversation action.

```tsx
export function SessionBar({ sessionId, conversationId, onNewSession }: Props) {
  const label = conversationId ? `对话 ${conversationId}` : "新对话";
  return <div className="session-bar"><span>{label}</span><button type="button" onClick={onNewSession}>新建对话</button></div>;
}
```

## ToolStatus

- Source: `frontend/src/components/ToolStatus.tsx`
- Purpose: compact real-time action status.

```tsx
export function ToolStatus({ status }: { status: string | null }) {
  if (!status) return null;
  return <div className="tool-status">{status}</div>;
}
```

## AgentTrace

- Source: `frontend/src/components/AgentTrace.tsx`
- Purpose: an observable, expandable per-turn execution ledger.

```tsx
export function AgentTrace({ trace }: { trace: TurnTrace }) {
  return <details className={`turn-trace turn-trace-${trace.status}`}>{/* status steps and metrics */}</details>;
}
```

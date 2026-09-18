# Page dependency trees

## `/` — Financial research workspace

Entry: `frontend/src/App.tsx`

Dependencies:

- `frontend/src/components/ConversationList.tsx`
- `frontend/src/components/SessionBar.tsx`
- `frontend/src/components/ChatPanel.tsx`
  - `frontend/src/components/MessageList.tsx`
    - `frontend/src/components/AgentTrace.tsx`
    - `frontend/src/components/MarkdownMessage.tsx`
  - `frontend/src/components/InteractionPrompt.tsx`
  - `frontend/src/components/ToolStatus.tsx`
- `frontend/src/components/SourceSidebar.tsx`
- `frontend/src/components/SettingsModal.tsx`
- `frontend/src/styles/app.css`

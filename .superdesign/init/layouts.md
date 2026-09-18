# Layouts

## App shell

- Source: `frontend/src/App.tsx`
- Description: a single authenticated research workspace with an application header and three-column body.

```tsx
<main className="app-shell">
  <header className="app-header">{/* product identity, model setting, user */}</header>
  <section className="workspace">
    <div className="workspace-body">
      <ConversationList />
      <div className="conversation-panel"><SessionBar /><ChatPanel /></div>
      <SourceSidebar />
    </div>
  </section>
  <SettingsModal />
</main>
```

## Research conversation panel

- Source: `frontend/src/components/ChatPanel.tsx`
- Description: chat messages, execution trace, source-ready output, composer, and interactive prompts.

```tsx
<section className="chat-panel">
  <MessageList />
  <div className="engine-notice" />
  <div className="artifact-list" />
  <ToolStatus />
  <form className="composer">{/* textarea + submit */}</form>
</section>
```

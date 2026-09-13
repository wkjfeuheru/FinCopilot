import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ConfigProvider, Tag } from "antd";
import zhCN from "antd/locale/zh_CN";
import { ConversationList } from "./components/ConversationList";
import { ChatPanel } from "./components/ChatPanel";
import type { ChatPanelHandle, ChatView } from "./components/ChatPanel";
import { SessionBar } from "./components/SessionBar";
import { SettingsModal } from "./components/SettingsModal";
import { SourceSidebar } from "./components/SourceSidebar";
import type { Activity } from "./components/SourceSidebar";
import { fetchConfig } from "./api/config";
import type { ConfigSnapshot } from "./api/config";
import {
  deleteConversation,
  fetchCitations,
  listConversations,
  loadConversationMessages,
} from "./api/client";
import type { Citation, ConversationSummary, HistoryMessage } from "./api/client";

const EMPTY_CONFIG: ConfigSnapshot = { configured: false, active_id: null, configs: [] };
// Remembering the conversation locally is what lets a reload resume it: the
// server's conversation store outlives the execution session, but only the
// client knows which conversation the user was reading.
const STORAGE_KEY = "finharness.conversation_id";

/** Rebuild the activity feed from a saved trace, for a conversation's history. */
function activitiesFromView(view: ChatView | undefined): Activity[] {
  if (!view) return [];
  const activities: Activity[] = [];
  for (const message of view.messages) {
    for (const step of message.trace?.steps ?? []) {
      if (step.kind !== "tool" && step.kind !== "skill" && step.kind !== "plan") continue;
      activities.push({
        key: `${activities.length}-${step.key}`,
        label: step.label,
        status:
          step.status === "error" ? "error" : step.status === "running" ? "running" : "done",
        detail: step.detail ?? (step.durationMs ? `耗时 ${step.durationMs} ms` : undefined),
      });
    }
  }
  return activities;
}

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(() =>
    window.localStorage.getItem(STORAGE_KEY),
  );
  const [sessionVersion, setSessionVersion] = useState(0);
  const [snapshot, setSnapshot] = useState<ConfigSnapshot | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [history, setHistory] = useState<HistoryMessage[]>([]);
  // Which conversation the loaded `history` belongs to. Without this the
  // transcript from the previously open conversation would be seeded into the
  // newly selected one while its own history is still loading.
  const [historyConversationId, setHistoryConversationId] = useState<string | null>(null);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [activities, setActivities] = useState<Activity[]>([]);
  // Conversations this client has already rendered, keyed by conversation id.
  // The server persists only readable turns, so the execution trace and produced
  // files must be kept here or they vanish when the user switches away and back.
  const viewCacheRef = useRef(new Map<string, ChatView>());
  const chatRef = useRef<ChatPanelHandle>(null);

  const refreshConfig = useCallback(async (): Promise<ConfigSnapshot> => {
    try {
      const data = await fetchConfig();
      setSnapshot(data);
      return data;
    } catch {
      setSnapshot(EMPTY_CONFIG);
      return EMPTY_CONFIG;
    }
  }, []);

  const refreshConversations = useCallback(async () => {
    setLoadingConversations(true);
    try {
      setConversations(await listConversations());
    } catch {
      setConversations([]);
    } finally {
      setLoadingConversations(false);
    }
  }, []);

  useEffect(() => {
    void refreshConfig().then((data) => {
      if (!data.configured) setSettingsOpen(true);
    });
    void refreshConversations();
  }, [refreshConfig, refreshConversations]);

  // Restore the stored conversation's transcript so the reader sees where they
  // left off, and continue the same memory scope.
  useEffect(() => {
    if (!conversationId) {
      setHistory([]);
      setHistoryConversationId(null);
      setCitations([]);
      return;
    }
    // Ignore a late response if the user has already switched to another
    // conversation: only the effect for the currently selected id may apply.
    let cancelled = false;
    void loadConversationMessages(conversationId)
      .then((messages) => {
        if (cancelled) return;
        setHistory(messages);
        setHistoryConversationId(conversationId);
      })
      .catch(() => {
        if (cancelled) return;
        setHistory([]);
        setHistoryConversationId(conversationId);
      });
    // Sources are recovered from the store too, so a resumed (or restarted)
    // conversation still shows the data it was built on.
    void fetchCitations(conversationId, null).then((data) => {
      if (!cancelled) setCitations(data);
    });
    return () => {
      cancelled = true;
    };
  }, [conversationId]);
  function rememberConversation(id: string | null) {
    setConversationId(id);
    if (id) window.localStorage.setItem(STORAGE_KEY, id);
    else window.localStorage.removeItem(STORAGE_KEY);
  }

  // Save what the panel is currently showing against the open conversation
  // before navigating away from it.
  function cacheCurrentView() {
    const view = chatRef.current?.snapshot();
    if (conversationId && view && view.messages.length > 0) {
      viewCacheRef.current.set(conversationId, view);
    }
  }

  function startNewConversation() {
    // Drop only the client handle: history for the old conversation stays in the
    // store and remains reachable from the list.
    cacheCurrentView();
    rememberConversation(null);
    setSessionId(null);
    setHistory([]);
    setCitations([]);
    setActivities([]);
    setSessionVersion((version) => version + 1);
  }

  async function handleDeleteConversation(id: string) {
    try {
      await deleteConversation(id);
    } catch {
      // The list refresh below is the source of truth; a failure here just
      // leaves the row in place.
    }
    // Deleting the conversation being viewed returns the UI to a fresh state,
    // and drops whatever this client had cached for it.
    if (id === conversationId) startNewConversation();
    viewCacheRef.current.delete(id);
    await refreshConversations();
  }

  function selectConversation(id: string) {
    if (id === conversationId) return;
    cacheCurrentView();
    // Restore the trace/activity feed we kept for this conversation; the server
    // transcript alone does not carry it.
    setActivities(activitiesFromView(viewCacheRef.current.get(id)));
    rememberConversation(id);
    setSessionId(null);
    setSessionVersion((version) => version + 1);
  }

  // The server allocates the conversation id on the first turn of a new
  // conversation; adopt it so the next turn (and the list) can reference it.
  function handleSession(session: string, conversation: string | null) {
    setSessionId(session);
    if (conversation && conversation !== conversationId) {
      rememberConversation(conversation);
      void refreshConversations();
    }
  }

  // Stable identity: a fresh array every render would re-trigger the child's
  // history-sync effect on each pass. Only surface history once it belongs to
  // the selected conversation, so a switch never seeds the old transcript.
  const restoredMessages = useMemo(
    () =>
      historyConversationId === conversationId
        ? history.map((item) => ({ role: item.role, text: item.text }))
        : [],
    [history, historyConversationId, conversationId],
  );

  const active = snapshot?.configs.find((config) => config.is_active) ?? null;

  return (
    <ConfigProvider locale={zhCN} theme={{ token: {
      motion: false,
      colorPrimary: "#176b63",
      colorInfo: "#176b63",
      colorSuccess: "#247552",
      colorWarning: "#a97620",
      colorError: "#b64444",
      colorText: "#243649",
      colorTextSecondary: "#637487",
      colorBorder: "#d8e0e7",
      borderRadius: 6,
      fontFamily: '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
      fontSize: 13,
      controlHeight: 34,
    } }}>
      <main className="app-shell">
        <header className="app-header">
          <div>
            <p className="eyebrow">FINANCIAL RESEARCH COPILOT</p>
            <h1>FinHarness</h1>
          </div>
          <button
            type="button"
            className="status-pill provider-chip"
            onClick={() => setSettingsOpen(true)}
            title="点击配置模型供应商"
          >
            {active ? (
              <>
                <Tag color="blue" style={{ marginInlineEnd: 6 }}>
                  {active.name}
                </Tag>
                {active.model}
              </>
            ) : (
              "未配置供应商"
            )}
          </button>
        </header>
        <section className="workspace">
          <div className="workspace-body">
            <ConversationList
              conversations={conversations}
              activeId={conversationId}
              loading={loadingConversations}
              onSelect={selectConversation}
              onNew={startNewConversation}
              onRefresh={() => void refreshConversations()}
              onDelete={(id) => void handleDeleteConversation(id)}
            />
            <div className="conversation-panel">
              <SessionBar
                sessionId={sessionId}
                conversationId={conversationId}
                onNewSession={startNewConversation}
              />
              <ChatPanel
                key={sessionVersion}
                ref={chatRef}
                sessionId={sessionId}
                conversationId={conversationId}
                initialMessages={restoredMessages}
                cachedView={conversationId ? viewCacheRef.current.get(conversationId) : undefined}
                onSession={handleSession}
                configured={snapshot?.configured ?? true}
                onOpenSettings={() => setSettingsOpen(true)}
                citations={citations}
                onCitations={setCitations}
                activities={activities}
                onActivities={setActivities}
              />
            </div>
            <SourceSidebar activities={activities} citations={citations} />
          </div>
        </section>
        <SettingsModal
          open={settingsOpen}
          onClose={(savedAndActivated) => {
            setSettingsOpen(false);
            void refreshConfig().then((data) => {
              if (savedAndActivated && data.active_id !== null) startNewConversation();
            });
          }}
          refreshConfig={refreshConfig}
        />
      </main>
    </ConfigProvider>
  );
}

export default App;

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, ConfigProvider, Tag } from "antd";
import zhCN from "antd/locale/zh_CN";
import { ConversationList } from "./components/ConversationList";
import { ChatPanel } from "./components/ChatPanel";
import type { ChatPanelHandle, ChatView } from "./components/ChatPanel";
import { LoginScreen } from "./components/LoginScreen";
import { MonitorView } from "./components/MonitorView";
import { AdminView } from "./components/AdminView";
import { SessionBar } from "./components/SessionBar";
import { SettingsModal } from "./components/SettingsModal";
import { traceFromStoredTurn } from "./components/AgentTrace";
import { SourceSidebar } from "./components/SourceSidebar";
import type { Activity } from "./components/SourceSidebar";
import { fetchConfig } from "./api/config";
import type { ConfigSnapshot } from "./api/config";
import { fetchMe, logout } from "./api/auth";
import type { AuthUser } from "./api/auth";
import { AUTH_EXPIRED_EVENT } from "./api/http";
import {
  deleteConversation,
  fetchCitations,
  listConversations,
  loadConversationMessages,
} from "./api/client";
import type {
  Citation,
  ConversationSummary,
  HistoryMessage,
  ResumableTurn,
} from "./api/client";

const EMPTY_CONFIG: ConfigSnapshot = { configured: false, active_id: null, configs: [] };
// 在本地记住对话，才能让页面刷新后恢复它：服务端的对话存储
// 比执行会话存活更久，但只有客户端知道用户当时
// 正在阅读哪个对话。按用户分键，换账号登录不会串到
// 另一个用户上次打开的对话。
const STORAGE_KEY_PREFIX = "finharness.conversation_id";

function storageKeyFor(userId: string): string {
  return `${STORAGE_KEY_PREFIX}.${userId}`;
}

/** 从已保存的 trace 重建活动流，用于某个对话的历史记录。 */
function activitiesFromView(view: ChatView | undefined): Activity[] {
  if (!view) return [];
  const activities: Activity[] = [];
  for (const message of view.messages) {
    for (const step of message.trace?.steps ?? []) {
      if (step.kind !== "tool" && step.kind !== "skill" && step.kind !== "plan" && step.kind !== "agent") continue;
      activities.push({
        key: `${activities.length}-${step.key}`,
        tool: step.toolName ?? (step.kind === "skill" ? "skill" : step.kind === "agent" ? "spawn_agent" : "tool"),
        label: step.label,
        status:
          step.status === "error" ? "error" : step.status === "running" ? "running" : "done",
        detail: step.detail ?? (step.durationMs ? `耗时 ${step.durationMs} ms` : undefined),
        attachments: step.attachments,
      });
    }
  }
  return activities;
}

function App() {
  // 认证门：null 表示还在探测会话，undefined 表示未登录。
  const [user, setUser] = useState<AuthUser | null | undefined>(undefined);
  // 顶层视图：工作台、管理（仅管理员）。不引入路由库，沿用条件渲染（管理是
  // 独立只读页，不需要深链）。
  const [view, setView] = useState<"chat" | "monitor" | "admin">("chat");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [sessionVersion, setSessionVersion] = useState(0);
  const [snapshot, setSnapshot] = useState<ConfigSnapshot | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [history, setHistory] = useState<HistoryMessage[]>([]);
  // 已加载的 `history` 属于哪个对话。没有它，之前打开的对话
  // 的记录会在新选中的对话自身历史仍在加载时
  // 被误填入其中。
  const [historyConversationId, setHistoryConversationId] = useState<string | null>(null);
  // 上一轮被停止后服务端留下的可继续断点（键为对话 id，与 history 同源）。
  // 刷新页面后它让"继续研究"入口仍在，而实时流结束时由 ChatPanel 请求刷新。
  const [resumable, setResumable] = useState<ResumableTurn | null>(null);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [activities, setActivities] = useState<Activity[]>([]);
  const [focusedTool, setFocusedTool] = useState<string | null>(null);

  function focusActivity(tool?: string) {
    setFocusedTool(tool ?? null);
    window.requestAnimationFrame(() => {
      document.getElementById("execution-activity")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
      if (tool) {
        document.getElementById(`activity-group-${tool}`)?.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
        });
      }
    });
  }
  // 本客户端已渲染过的对话，以 conversation id 为键。
  // 在切换时保留完整渲染的视图。历史回答可以从服务端
  // 重建其 trace；产出文件保存在各步的 attachments 中，
  // 因此无论是这个内存快照还是服务端还原的轮次，
  // 都能还原“产出文件”一栏。
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

  // 挂载时探测会话；任何 API 返回 401（会话过期/被撤销）时回到登录页。
  useEffect(() => {
    void fetchMe().then((me) => setUser(me));
    const onExpired = () => {
      setUser(undefined);
      setConversationId(null);
      setConversations([]);
    };
    window.addEventListener(AUTH_EXPIRED_EVENT, onExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, onExpired);
  }, []);

  // 登录后恢复该用户上次打开的对话；每个用户的对话句柄分键存放。
  useEffect(() => {
    if (!user) return;
    setConversationId(window.localStorage.getItem(storageKeyFor(user.id)));
  }, [user]);

  // 管理员登出/会话过期后换普通用户登录时，view 可能残留为
  // monitor/admin；回落到工作台，避免渲染无权访问的页面。
  useEffect(() => {
    if (user && user.role !== "admin" && view !== "chat") setView("chat");
  }, [user, view]);

  useEffect(() => {
    if (!user) return;
    void refreshConfig().then((data) => {
      if (!data.configured) setSettingsOpen(true);
    });
    void refreshConversations();
  }, [user, refreshConfig, refreshConversations]);

  async function handleLogout() {
    await logout();
    setUser(undefined);
    setConversationId(null);
    setConversations([]);
    setSessionId(null);
    viewCacheRef.current.clear();
  }

  // 只读取指定对话的可继续状态；用于一轮结束后刷新（服务端才是准绳）。
  //
  // 对话 id 由调用方传入而不是取自 state：这个回调由流事件触发，而流回调
  // 闭包住的是"发送那一刻"的渲染——新对话的 id 要到 `session` 事件之后才存在，
  // 从 state 读会在新对话上永远拿到 null，从而查不到刚写好的断点。
  const refreshResumable = useCallback(async (target: string | null) => {
    if (!target) {
      setResumable(null);
      return;
    }
    try {
      const data = await loadConversationMessages(target);
      setResumable(data.resumable);
    } catch {
      // 取不到就按"没有可继续"处理；它只是一个便利入口，不该报错。
      setResumable(null);
    }
  }, []);

  // 恢复已存储对话的记录，让读者看到上次读到的地方，
  // 并延续同一记忆作用域。
  useEffect(() => {
    if (!conversationId) {
      setHistory([]);
      setHistoryConversationId(null);
      setCitations([]);
      setResumable(null);
      return;
    }
    // 如果用户已切换到另一个对话，则忽略迟到的响应：
    // 只有当前选中 id 对应的 effect 才能生效。
    let cancelled = false;
    void loadConversationMessages(conversationId)
      .then((data) => {
        if (cancelled) return;
        setHistory(data.messages);
        setResumable(data.resumable);
        setHistoryConversationId(conversationId);
      })
      .catch(() => {
        if (cancelled) return;
        setHistory([]);
        setResumable(null);
        setHistoryConversationId(conversationId);
      });
    // 数据来源也一并从存储中恢复，因此恢复（或重启）的
    // 对话仍会显示它所依据的数据。
    void fetchCitations(conversationId, null).then((data) => {
      if (!cancelled) setCitations(data);
    });
    return () => {
      cancelled = true;
    };
  }, [conversationId]);
  function rememberConversation(id: string | null) {
    setConversationId(id);
    // 已登录时才写入本地存储；键按用户区分。
    if (!user) return;
    const key = storageKeyFor(user.id);
    if (id) window.localStorage.setItem(key, id);
    else window.localStorage.removeItem(key);
  }

  // 在离开当前打开的对话之前，把面板当前展示的内容
  // 保存到该对话下。
  function cacheCurrentView() {
    const view = chatRef.current?.snapshot();
    if (conversationId && view && view.messages.length > 0) {
      viewCacheRef.current.set(conversationId, view);
    }
  }

  function startNewConversation() {
    // 只丢弃客户端的凭据：旧对话的历史仍保留在
    // 存储中，并可从列表中访问。
    cacheCurrentView();
    rememberConversation(null);
    setSessionId(null);
    setHistory([]);
    setCitations([]);
    setActivities([]);
    setResumable(null);
    setSessionVersion((version) => version + 1);
  }

  async function handleDeleteConversation(id: string) {
    try {
      await deleteConversation(id);
    } catch {
      // 下面的列表刷新才是准绳；这里失败只会
      // 让该行维持原样。
    }
    // 删除正在查看的对话会让 UI 回到全新状态，
    // 并丢弃本客户端为它缓存的一切。
    if (id === conversationId) startNewConversation();
    viewCacheRef.current.delete(id);
    await refreshConversations();
  }

  function selectConversation(id: string) {
    if (id === conversationId) return;
    cacheCurrentView();
    // 在历史加载期间立即恢复缓存的活动流。
    setActivities(activitiesFromView(viewCacheRef.current.get(id)));
    setFocusedTool(null);
    rememberConversation(id);
    setSessionId(null);
    setSessionVersion((version) => version + 1);
  }

  // 服务端会在新对话的第一轮分配 conversation id；采纳它，
  // 以便下一轮（以及列表）可以引用它。
  function handleSession(session: string, conversation: string | null) {
    setSessionId(session);
    if (conversation && conversation !== conversationId) {
      rememberConversation(conversation);
      void refreshConversations();
    }
  }

  // 稳定标识：每次渲染都新建数组会在每一轮重新触发子组件的
  // 历史同步 effect。只有在历史属于当前选中的对话时才暴露它，
  // 这样切换时绝不会填入旧记录。
  const restoredMessages = useMemo(
    () =>
      historyConversationId === conversationId
        ? history.map((item) => ({
            role: item.role,
            text: item.text,
            trace:
              item.role === "assistant" && item.turn
                ? traceFromStoredTurn(item.turn)
                : undefined,
          }))
        : [],
    [history, historyConversationId, conversationId],
  );

  const active = snapshot?.configs.find((config) => config.is_active) ?? null;

  // 认证门：未登录整页替换为登录/注册界面。
  if (!user) {
    return (
      <ConfigProvider locale={zhCN} theme={{ token: { motion: false, colorPrimary: "#176b63", colorInfo: "#176b63", borderRadius: 6, fontFamily: '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif', fontSize: 13 } }}>
        <LoginScreen
          onSuccess={(u) => {
            setUser(u);
            setSessionId(null);
            setHistory([]);
            setCitations([]);
            setActivities([]);
          }}
        />
      </ConfigProvider>
    );
  }

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
            <h1>FinHarness｜投研工作台</h1>
            <p className="app-positioning">覆盖个股、行业、宏观与量化因子研究；从问题到可回溯结论 <span>不构成投资建议</span></p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Button
              size="small"
              type={view === "chat" ? "primary" : "default"}
              onClick={() => setView("chat")}
            >
              工作台
            </Button>
            {/* 运行监控与管理同源门控（users.role='admin'）；普通用户只保留工作台。 */}
            {user.role === "admin" ? (
              <>
                <Button
                  size="small"
                  type={view === "monitor" ? "primary" : "default"}
                  onClick={() => setView("monitor")}
                >
                  运行监控
                </Button>
                <Button
                  size="small"
                  type={view === "admin" ? "primary" : "default"}
                  onClick={() => setView("admin")}
                >
                  管理
                </Button>
              </>
            ) : null}
            <button
              type="button"
              className="status-pill provider-chip"
              onClick={() => setSettingsOpen(true)}
              title="打开模型设置"
            >
              {active ? (
                <>
                  <Tag color="blue" style={{ marginInlineEnd: 6 }}>
                    {active.name}
                  </Tag>
                  {active.model}
                </>
              ) : (
                "完成模型设置"
              )}
            </button>
            <span className="status-pill" title={`当前用户：${user.username}`}>
              {user.username}
            </span>
            <Button size="small" onClick={() => void handleLogout()}>
              退出
            </Button>
          </div>
        </header>
        <section className="workspace">
          {view === "monitor" ? (
            <MonitorView />
          ) : view === "admin" ? (
            <AdminView />
          ) : (
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
                onFocusActivity={focusActivity}
                resumable={historyConversationId === conversationId ? resumable : null}
                onRefreshResumable={(target) => void refreshResumable(target)}
              />
            </div>
            <SourceSidebar activities={activities} citations={citations} focusedTool={focusedTool} />
            </div>
          )}
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

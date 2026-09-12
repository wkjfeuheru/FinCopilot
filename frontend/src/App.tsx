import { useCallback, useEffect, useState } from "react";
import { ConfigProvider, Tag } from "antd";
import zhCN from "antd/locale/zh_CN";
import { ChatPanel } from "./components/ChatPanel";
import { SessionBar } from "./components/SessionBar";
import { SettingsModal } from "./components/SettingsModal";
import { fetchConfig, type ConfigSnapshot } from "./api/config";

const EMPTY_CONFIG: ConfigSnapshot = { configured: false, active_id: null, configs: [] };

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionVersion, setSessionVersion] = useState(0);
  const [snapshot, setSnapshot] = useState<ConfigSnapshot | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

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

  useEffect(() => {
    void refreshConfig().then((data) => {
      if (!data.configured) setSettingsOpen(true);
    });
  }, [refreshConfig]);

  function startNewSession() {
    setSessionId(null);
    setSessionVersion((version) => version + 1);
  }

  // The modal reports whether it saved and activated a config; the parent owns
  // the resulting refresh/new-session, so the modal never mutates parent state
  // from inside its own async flow.
  function handleSettingsClose(savedAndActivated: boolean) {
    setSettingsOpen(false);
    void refreshConfig().then((data) => {
      if (savedAndActivated && data.active_id !== null) startNewSession();
    });
  }

  const active = snapshot?.configs.find((config) => config.is_active) ?? null;

  return (
    <ConfigProvider locale={zhCN} theme={{ token: { motion: false } }}>
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
          <SessionBar sessionId={sessionId} onNewSession={startNewSession} />
          <ChatPanel
            key={sessionVersion}
            sessionId={sessionId}
            onSession={setSessionId}
            configured={snapshot?.configured ?? true}
            onOpenSettings={() => setSettingsOpen(true)}
          />
        </section>
        <SettingsModal
          open={settingsOpen}
          onClose={handleSettingsClose}
          refreshConfig={refreshConfig}
        />
      </main>
    </ConfigProvider>
  );
}

export default App;

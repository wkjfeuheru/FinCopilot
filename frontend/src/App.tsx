import { useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { SessionBar } from "./components/SessionBar";

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionVersion, setSessionVersion] = useState(0);

  function startNewSession() {
    setSessionId(null);
    setSessionVersion((version) => version + 1);
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">FINANCIAL RESEARCH COPILOT</p>
          <h1>FinHarness</h1>
        </div>
        <span className="status-pill">研究工作区</span>
      </header>
      <section className="workspace"><SessionBar sessionId={sessionId} onNewSession={startNewSession} /><ChatPanel key={sessionVersion} sessionId={sessionId} onSession={setSessionId} /></section>
    </main>
  );
}

export default App;

import { FormEvent, useEffect, useRef, useState } from "react";
import { Button, Empty } from "antd";
import { artifactUrl, fetchCitations, respondChat, streamChat } from "../api/client";
import type { Citation } from "../api/client";
import { Interaction, InteractionPrompt } from "./InteractionPrompt";
import { Message, MessageList } from "./MessageList";
import { ToolStatus } from "./ToolStatus";
import type { Activity } from "./SourceSidebar";
import type { AgentStep, TurnTrace } from "./AgentTrace";

type Props = {
  sessionId: string | null;
  conversationId: string | null;
  /** History restored from the store when resuming a conversation. */
  initialMessages: Message[];
  onSession: (sessionId: string, conversationId: string | null) => void;
  configured: boolean;
  onOpenSettings: () => void;
  citations: Citation[];
  onCitations: (citations: Citation[]) => void;
  activities: Activity[];
  onActivities: (update: (current: Activity[]) => Activity[]) => void;
};

/** A file the engine produced (chart, report) that the user can download. */
type Artifact = { path: string; name: string };

function fileName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

/** Human-readable label for a tool/skill call in the activity feed. */
function activityLabel(name: string): string {
  if (name === "load_skill") return "加载技能";
  if (name === "list_skills") return "检索技能";
  if (name === "write_report") return "生成研报";
  if (name === "make_chart") return "绘制图表";
  if (name === "research_plan") return "制定研究计划";
  return `调用工具 ${name}`;
}

function traceKind(name: string): AgentStep["kind"] {
  if (name === "research_plan") return "plan";
  if (name === "load_skill" || name === "list_skills") return "skill";
  return "tool";
}

function traceLabel(name: string): string {
  if (name === "research_plan") return "制定研究计划";
  if (name === "load_skill") return "加载研究 Skill";
  if (name === "list_skills") return "检索可用 Skill";
  if (name === "write_report") return "生成研报并执行风险终审";
  if (name === "make_chart") return "绘制研究图表";
  return `调用 ${name}`;
}

function agentLabel(name: string): string {
  if (name === "risk") return "风险审阅子代理";
  return `${name} 子代理`;
}

export function ChatPanel({
  sessionId,
  conversationId,
  initialMessages,
  onSession,
  configured,
  onOpenSettings,
  citations,
  onCitations,
  activities,
  onActivities,
}: Props) {
  // Seeded from restored history so a resumed conversation shows where it left off.
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [interaction, setInteraction] = useState<Interaction | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const runStartedRef = useRef(0);
  const firstTokenRef = useRef<number | null>(null);
  // The current stream's handles, updated as the session event arrives. They let
  // each tool completion refresh the right conversation's citations.
  const streamIdsRef = useRef<{ session: string | null; conversation: string | null }>({
    session: sessionId,
    conversation: conversationId,
  });

  // Restored history arrives asynchronously, after this component has already
  // mounted with an empty list. Sync it in, but never clobber a conversation the
  // user has already started typing into in this session.
  useEffect(() => {
    setMessages((current) => (current.length === 0 ? initialMessages : current));
  }, [initialMessages]);

  function refreshCitations() {
    const { session, conversation } = streamIdsRef.current;
    void fetchCitations(conversation, session).then(onCitations);
  }

  function rememberArtifacts(raw: unknown) {
    const paths = (raw as string[] | undefined) ?? [];
    if (paths.length === 0) return;
    setArtifacts((current) => {
      const seen = new Set(current.map((item) => item.path));
      const added = paths
        .filter((path) => !seen.has(path))
        .map((path) => ({ path, name: fileName(path) }));
      return added.length ? [...current, ...added] : current;
    });
  }

  function updateActiveAssistant(update: (message: Message) => Message) {
    setMessages((current) => {
      const next = [...current];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === "assistant") {
          next[index] = update(next[index]);
          break;
        }
      }
      return next;
    });
  }

  function updateTrace(update: (trace: TurnTrace) => TurnTrace) {
    updateActiveAssistant((message) => message.trace ? { ...message, trace: update(message.trace) } : message);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const message = input.trim();
    if (!message || busy) return;
    setInput("");
    runStartedRef.current = performance.now();
    firstTokenRef.current = null;
    setMessages((current) => [
      ...current,
      { role: "user", text: message },
      {
        role: "assistant",
        text: "",
        trace: {
          planned: false,
          status: "running",
          steps: [{ key: "analysis", kind: "analysis", label: "理解问题并确定研究路径", status: "running" }],
        },
      },
    ]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    setNotice(null);
    try {
      await streamChat(message, conversationId, (event) => {
        if (event.event === "session") {
          const session = String(event.data.session_id);
          const conversation = event.data.conversation_id ? String(event.data.conversation_id) : null;
          streamIdsRef.current = { session, conversation };
          onSession(session, conversation);
        }
        if (event.event === "tool_status") {
          const name = String(event.data.name ?? "tool");
          const callId = String(event.data.call_id ?? name);
          const state = String(event.data.status ?? "");
          setStatus(`${name} ${state}`);
          if (state === "started") {
            updateTrace((trace) => ({
              ...trace,
              planned: trace.planned || name === "research_plan",
              steps: [
                ...trace.steps
                  .map((step) => step.key === "analysis" ? { ...step, status: "done" as const } : step)
                  .filter((step) => step.key !== callId),
                { key: callId, kind: traceKind(name), label: traceLabel(name), status: "running" },
              ],
            }));
            onActivities((current) => [
              ...current.filter((item) => item.key !== callId),
              { key: callId, label: activityLabel(name), status: "running" },
            ]);
          } else {
            rememberArtifacts(event.data.attachments);
            const attachments = (event.data.attachments as string[] | undefined) ?? [];
            const duration = Number(event.data.duration_ms ?? 0);
            const stepStatus = event.data.ok === false ? "error" as const : "done" as const;
            updateTrace((trace) => {
              const existing = trace.steps.some((step) => step.key === callId);
              const completed: AgentStep = {
                key: callId,
                kind: traceKind(name),
                label: traceLabel(name),
                status: stepStatus,
                durationMs: duration > 0 ? duration : undefined,
                detail: event.data.ok === false ? String(event.data.error ?? "执行失败") : undefined,
              };
              return {
                ...trace,
                planned: trace.planned || name === "research_plan",
                steps: existing
                  ? trace.steps.map((step) => step.key === callId ? { ...step, ...completed } : step)
                  : [...trace.steps, completed],
              };
            });
            onActivities((current) => [
              ...current.filter((item) => item.key !== callId),
              {
                key: callId,
                label: activityLabel(name),
                status: event.data.ok === false ? "error" : "done",
                detail:
                  event.data.ok === false
                    ? String(event.data.error ?? "执行失败")
                    : duration > 0
                      ? `耗时 ${duration} ms`
                      : undefined,
                attachments,
              },
            ]);
            if ((event.data.citations as string[] | undefined)?.length) refreshCitations();
          }
        }
        // Engine notices the UI should not swallow: window compaction and the
        // loop guard both change what the user is looking at.
        if (event.event === "context_compacted") {
          const before = Number(event.data.before_tokens ?? 0);
          const after = Number(event.data.after_tokens ?? 0);
          const degraded = Boolean(event.data.degraded);
          setNotice(
            `上下文已达上限，已压缩历史（${before} → ${after} tokens）${degraded ? "，摘要降级" : ""}`,
          );
          updateTrace((trace) => ({
            ...trace,
            steps: [...trace.steps, {
              key: `compact-${before}-${after}`,
              kind: "system",
              label: "压缩研究上下文",
              status: "done",
              detail: `${before.toLocaleString("zh-CN")} → ${after.toLocaleString("zh-CN")} Token${degraded ? " · 摘要降级" : ""}`,
            }],
          }));
          onActivities((current) => [
            ...current,
            {
              key: `compact-${current.length}`,
              label: "压缩上下文",
              status: "info",
              detail: `${before} → ${after} tokens${degraded ? "（摘要降级）" : ""}`,
            },
          ]);
        }
        if (event.event === "loop_guard") {
          const action = String(event.data.action ?? "refused");
          setNotice(
            action === "would_abort"
              ? `检测到重复调用，已终止本轮：${String(event.data.name ?? "")}`
              : `检测到重复调用，已跳过并提示模型改用已有结果：${String(event.data.name ?? "")}`,
          );
          updateTrace((trace) => ({
            ...trace,
            steps: [...trace.steps, {
              key: `guard-${String(event.data.call_id ?? trace.steps.length)}`,
              kind: "system",
              label: action === "would_abort" ? "终止重复调用" : "跳过重复调用",
              status: action === "would_abort" ? "error" : "info",
              detail: String(event.data.name ?? ""),
            }],
          }));
          onActivities((current) => [
            ...current,
            {
              key: `guard-${current.length}`,
              label: action === "would_abort" ? "重复调用，终止本轮" : "重复调用，已跳过",
              status: "info",
              detail: String(event.data.name ?? ""),
            },
          ]);
        }
        if (event.event === "interactive_request") {
          // The engine is paused awaiting an answer; surface the dialog.
          const prompt = String(event.data.prompt ?? "");
          setInteraction({
            requestId: String(event.data.request_id ?? ""),
            kind: String(event.data.kind ?? "question"),
            prompt,
            options: (event.data.options as string[] | undefined) ?? [],
          });
          updateTrace((trace) => ({
            ...trace,
            steps: [...trace.steps, {
              key: `ask-${String(event.data.request_id ?? trace.steps.length)}`,
              kind: "system",
              label: event.data.kind === "confirm" ? "等待操作确认" : "等待补充信息",
              status: "running",
              detail: prompt,
            }],
          }));
          onActivities((current) => [
            ...current,
            {
              key: `ask-${current.length}`,
              label: event.data.kind === "confirm" ? "等待写入确认" : "等待用户回答",
              status: "running",
              detail: prompt,
            },
          ]);
        }
        if (event.event === "text_reset") {
          // The round turned out to be a tool call; drop the draft that streamed
          // before it so only the final answer remains.
          updateActiveAssistant((assistant) => ({
            ...assistant,
            text: "",
            trace: assistant.trace ? {
              ...assistant.trace,
              steps: assistant.trace.steps.filter((step) => step.key !== "final"),
            } : undefined,
          }));
        }
        if (event.event === "delta") {
          if (firstTokenRef.current === null) firstTokenRef.current = performance.now();
          updateActiveAssistant((assistant) => {
            const trace = assistant.trace;
            const hasFinal = trace?.steps.some((step) => step.key === "final") ?? false;
            return {
              ...assistant,
              text: assistant.text + String(event.data.text ?? ""),
              trace: trace ? {
                ...trace,
                steps: hasFinal ? trace.steps : [
                  ...trace.steps.map((step) => step.key === "analysis" ? { ...step, status: "done" as const } : step),
                  { key: "final", kind: "final", label: "整理研究结论", status: "running" },
                ],
              } : undefined,
            };
          });
        }
        if (event.event === "error") {
          setInteraction(null);
          updateTrace((trace) => ({
            ...trace,
            status: "error",
            steps: trace.steps.map((step) => step.status === "running" ? { ...step, status: "error" as const } : step),
          }));
          setMessages((current) => [...current, { role: "error", text: String(event.data.message ?? "Unknown error") }]);
        }
        if (event.event === "done") {
          const usage = (event.data.usage as Record<string, unknown> | undefined) ?? {};
          const inputTokens = Number(usage.input_tokens ?? 0);
          const outputTokens = Number(usage.output_tokens ?? 0);
          const perAgent = (event.data.per_agent as Record<string, Record<string, unknown>> | undefined) ?? {};
          const agentSteps = Object.entries(perAgent).map(([name, value], index): AgentStep => ({
            key: `agent-${name}-${index}`,
            kind: "agent",
            label: agentLabel(name),
            status: "done",
            detail: `运行 ${Number(value.runs ?? 0)} 次`,
            tokens: Number(value.input_tokens ?? 0) + Number(value.output_tokens ?? 0),
          }));
          const agentRuns = Object.values(perAgent).reduce((sum, value) => sum + Number(value.runs ?? 0), 0);
          const succeeded = event.data.succeeded !== false;
          updateTrace((trace) => {
            const withoutAgents = trace.steps.filter((step) => step.kind !== "agent");
            const hasFinal = withoutAgents.some((step) => step.key === "final");
            const closedSteps = withoutAgents.map((step) => ({
              ...step,
              status: step.status === "running" ? (succeeded ? "done" as const : "error" as const) : step.status,
            }));
            return {
              ...trace,
              status: succeeded ? "done" : "error",
              steps: [
                ...closedSteps,
                ...agentSteps,
                ...(hasFinal ? [] : [{ key: "final", kind: "final" as const, label: "整理研究结论", status: succeeded ? "done" as const : "error" as const }]),
              ],
              metrics: {
                totalTokens: inputTokens + outputTokens,
                inputTokens,
                outputTokens,
                steps: Number(event.data.tool_calls ?? 0) + agentRuns + 1,
                firstTokenMs: firstTokenRef.current === null ? null : firstTokenRef.current - runStartedRef.current,
                totalDurationMs: performance.now() - runStartedRef.current,
                toolDurationMs: Number(event.data.tool_duration_ms ?? 0),
              },
            };
          });
          setStatus(null);
          setInteraction(null);
          refreshCitations();
        }
      }, controller.signal);
    } catch (error) {
      if ((error as Error).name !== "AbortError") setMessages((current) => [...current, { role: "error", text: (error as Error).message }]);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  async function handleRespond(requestId: string, response: string) {
    setInteraction(null);
    try {
      await respondChat(requestId, response);
    } catch (error) {
      setMessages((current) => [...current, { role: "error", text: `回答提交失败：${(error as Error).message}` }]);
    }
  }

  return (
    <section className="chat-panel">
      {!configured && messages.length === 0 && (
        <Empty
          className="config-empty-state"
          description="尚未配置模型供应商，配置后即可开始对话"
        >
          <Button type="primary" onClick={onOpenSettings}>
            去配置供应商
          </Button>
        </Empty>
      )}
      <MessageList messages={messages} citations={citations} />
      {notice && <div className="engine-notice">{notice}</div>}
      {artifacts.length > 0 && (
        <div className="artifact-list" aria-label="产出文件">
          <span className="artifact-title">产出文件</span>
          {artifacts.map((item) => (
            <a key={item.path} className="artifact-link" href={artifactUrl(item.path)} download>
              {item.name}
            </a>
          ))}
        </div>
      )}
      <ToolStatus status={status} />
      <form className="composer" onSubmit={submit}>
        <textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="输入研究问题" rows={3} disabled={busy} />
        <button type="submit" disabled={busy || !input.trim() || !configured}>{busy ? "处理中" : "发送"}</button>
      </form>
      <InteractionPrompt interaction={interaction} onRespond={handleRespond} busy={false} />
    </section>
  );
}

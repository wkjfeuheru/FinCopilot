import {
  FormEvent,
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
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
  /** 恢复对话时从存储中还原的历史消息。 */
  initialMessages: Message[];
  onSession: (sessionId: string, conversationId: string | null) => void;
  configured: boolean;
  onOpenSettings: () => void;
  citations: Citation[];
  onCitations: (citations: Citation[]) => void;
  activities: Activity[];
  onActivities: (update: (current: Activity[]) => Activity[]) => void;
  /**
   * 父组件为这个对话保存的快照。服务端只持久化
   * 可读轮次，因此执行 trace 由客户端内存快照保留；
   * 产出文件随之保存在各步的 attachments 中，刷新后
   * 仍能从持久化的轮次事件里还原。
   */
  cachedView?: ChatView;
};

/** 引擎产出的、用户可下载的文件（图表、研报）。 */
export type Artifact = { path: string; name: string };

/** 对话视图中仅存在于客户端的部分：消息（含 trace 与产出文件）。 */
export type ChatView = { messages: Message[] };

export type ChatPanelHandle = { snapshot: () => ChatView };

function fileName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

/** 消息 trace 中各步产出的文件，按出现顺序去重。
 *
 * 文件挂在步骤上（而非仅存在客户端状态里），因此实时一轮与刷新后
 * 从持久化轮次事件还原的历史，都会显示出同一份“产出文件”。
 */
function artifactsFromMessages(messages: Message[]): Artifact[] {
  const seen = new Set<string>();
  const items: Artifact[] = [];
  for (const message of messages) {
    for (const step of message.trace?.steps ?? []) {
      for (const path of step.attachments ?? []) {
        if (seen.has(path)) continue;
        seen.add(path);
        items.push({ path, name: fileName(path) });
      }
    }
  }
  return items;
}

/** 活动流中工具/Skill 调用的可读标签。 */
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

export const ChatPanel = forwardRef<ChatPanelHandle, Props>(function ChatPanel(
  {
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
    cachedView,
  },
  ref,
) {
  // 优先用父组件的快照初始化（这样切回时能恢复 trace 与
  // 产出文件），否则回退到还原的服务端历史。
  const [messages, setMessages] = useState<Message[]>(() =>
    cachedView && cachedView.messages.length > 0 ? cachedView.messages : initialMessages,
  );
  // 产出文件直接从消息 trace 派生：实时一轮与刷新后还原的历史
  // 走同一条路径，因此不再需要单独的客户端文件状态。
  const artifacts = useMemo(() => artifactsFromMessages(messages), [messages]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [interaction, setInteraction] = useState<Interaction | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const runStartedRef = useRef(0);
  const firstTokenRef = useRef<number | null>(null);
  // 当前流的句柄，随 session 事件到达而更新。它们让每次工具完成
  // 都能刷新正确对话的 citations。
  const streamIdsRef = useRef<{ session: string | null; conversation: string | null }>({
    session: sessionId,
    conversation: conversationId,
  });
  // 供父组件在切换时做快照的最新视图。保存在 ref 中，以便
  // 父组件能命令式读取，而不必在每个 token 时重新渲染。
  const latestViewRef = useRef<ChatView>({ messages });
  // 切换时父组件会重新挂载此面板，但它启动的流仍会继续
  // 运行；没有这道防护，其事件会泄漏到下一个对话。
  const mountedRef = useRef(true);

  useEffect(() => {
    latestViewRef.current = { messages };
  }, [messages]);

  useEffect(() => {
    // 挂载时也要重置：StrictMode 会先挂载、再清理、然后再挂载。
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useImperativeHandle(ref, () => ({ snapshot: () => latestViewRef.current }), []);

  function emitActivities(update: (current: Activity[]) => Activity[]) {
    if (mountedRef.current) onActivities(update);
  }

  // 还原的历史是异步到达的，此时本组件已经以空列表
  // 挂载完成。把它同步进来，但绝不要覆盖用户
  // 本次会话中已经开始输入的对话。
  useEffect(() => {
    setMessages((current) => (current.length === 0 ? initialMessages : current));
  }, [initialMessages]);

  function refreshCitations() {
    const { session, conversation } = streamIdsRef.current;
    void fetchCitations(conversation, session).then((data) => {
      if (mountedRef.current) onCitations(data);
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
            emitActivities((current) => [
              ...current.filter((item) => item.key !== callId),
              { key: callId, label: activityLabel(name), status: "running" },
            ]);
          } else {
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
                // 文件随步骤保存，使其与刷新后还原的历史走同一渲染路径。
                attachments: attachments.length ? attachments : undefined,
              };
              return {
                ...trace,
                planned: trace.planned || name === "research_plan",
                steps: existing
                  ? trace.steps.map((step) => step.key === callId ? { ...step, ...completed } : step)
                  : [...trace.steps, completed],
              };
            });
            emitActivities((current) => [
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
        // 引擎通知，UI 不应吞掉：窗口压缩与
        // 循环防护都会改变用户正在看的内容。
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
          emitActivities((current) => [
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
          emitActivities((current) => [
            ...current,
            {
              key: `guard-${current.length}`,
              label: action === "would_abort" ? "重复调用，终止本轮" : "重复调用，已跳过",
              status: "info",
              detail: String(event.data.name ?? ""),
            },
          ]);
        }
        if (event.event === "plan_progress") {
          const revision = Number(event.data.revision ?? 1);
          const done = Number(event.data.done ?? 0);
          const total = Number(event.data.total ?? 0);
          const drift = (event.data.drift as string[] | undefined) ?? [];
          const mismatch = (event.data.mismatch as string[] | undefined) ?? [];
          const stalled = Number(event.data.stalled_turns ?? 0);
          const detail = [
            `进度 ${done}/${total}`,
            revision > 1 ? `第 ${revision} 版` : "",
            drift.length ? `目标外：${drift.join("、")}` : "",
            mismatch.length ? `能力外：${mismatch.join("、")}` : "",
            stalled ? `停滞 ${stalled} 轮` : "",
          ].filter(Boolean).join(" · ");
          updateTrace((trace) => ({
            ...trace,
            planned: true,
            steps: trace.steps.some((step) => step.kind === "plan")
              ? trace.steps.map((step) => step.kind === "plan" ? { ...step, detail } : step)
              : [...trace.steps, {
                key: "plan-progress",
                kind: "plan" as const,
                label: "研究计划进度",
                status: "info" as const,
                detail,
              }],
          }));
          const offPlan = [...drift, ...mismatch];
          if (offPlan.length || stalled) {
            setNotice(offPlan.length
              ? `计划偏离：本轮触及了计划未涵盖的标的或能力（${offPlan.join("、")}）`
              : `计划已停滞 ${stalled} 轮，等待模型回写或修订`);
          }
        }
        if (event.event === "interactive_request") {
          // 引擎已暂停，等待回答；弹出该对话框。
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
          emitActivities((current) => [
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
          // 本轮结果是一次工具调用；丢弃此前流式输出的草稿，
          // 只保留最终答案。
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
          // 优先使用通俗易懂的提示，而非引擎原始消息；``reason``
          // 用于区分预算耗尽与供应商失败。
          const reasonText: Record<string, string> = {
            loop_detected: "检测到重复取数，已提前结束本轮",
            max_turns_exhausted: "达到轮次上限，已提前结束本轮",
            provider_error: "模型调用失败，本轮未能完成",
          };
          const reason = String(event.data.reason ?? "");
          const message = String(event.data.message ?? "Unknown error");
          setMessages((current) => [...current, { role: "error", text: reasonText[reason] ?? message }]);
        }
        if (event.event === "answer") {
          // 以单个事件送达的终止性答案（主要是失败运行的部分
          // 发现结果）。用赋值而非追加：成功时流式 delta 已经
          // 包含同样的文本，所以这里是空操作；
          // 失败时 delta 已被重置，所以这里用于还原发现结果。
          updateActiveAssistant((assistant) => ({
            ...assistant,
            text: String(event.data.text ?? assistant.text),
          }));
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
});

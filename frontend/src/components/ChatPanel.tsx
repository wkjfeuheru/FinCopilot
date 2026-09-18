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
import { artifactUrl, fetchCitations, respondChat, stopChat, streamChat } from "../api/client";
import type { Citation, ResumableTurn } from "../api/client";
import { Interaction, InteractionPrompt } from "./InteractionPrompt";
import { Message, MessageList } from "./MessageList";
import { ToolStatus } from "./ToolStatus";
import type { Activity } from "./SourceSidebar";
import type { AgentStep, TurnTrace } from "./AgentTrace";
import { planFromEvent, presentToolAction, presentToolProgress } from "../lib/researchPresentation";
import { ResearchWelcome } from "./ResearchWelcome";

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
  /**
   * 服务端为该对话保留的可继续断点（上一轮被停止）。刷新页面后它让
   * "继续研究"入口依然出现，而不是把那次中断当成一次普通失败。
   */
  resumable?: ResumableTurn | null;
  /** 续做一轮：把续做提示作为新的提问提交。 */
  onResume?: () => void;
  /**
   * 新一轮结束后，父组件据此刷新可继续状态。
   *
   * 显式传入对话 id：这是流事件回调，它闭包住的是**发送那一刻**的渲染，
   * 而新对话的 id 是流开始之后才由 `session` 事件采纳的。不传就会用 `null`
   * 去查询，从而永远查不到刚写好的断点。
   */
  onRefreshResumable?: (conversationId: string | null) => void;
};

/** 引擎产出的、用户可下载的文件（图表、研报）。 */
export type Artifact = { path: string; name: string };

/** 对话视图中仅存在于客户端的部分：消息（含 trace 与产出文件）。 */
export type ChatView = { messages: Message[] };

export type ChatPanelHandle = { snapshot: () => ChatView };

/** 空闲上限（毫秒）。后端心跳约 15s 一次，取一个宽裕的值：
 * 这么长时间连一个事件都没有，说明连接已经不可用。 */
const STREAM_IDLE_TIMEOUT_MS = 90_000;

/** 按下停止后，等待后端优雅收尾的宽限期（毫秒）。
 *
 * 后端是协作式停止：引擎在下一个等待点收尾并下发 done，通常很快。但若它
 * 正卡在一次很长的工具调用里，宽限期到了就直接断开连接兜底——服务端在断开
 * 路径上同样会落库并留下断点，因此兜底不会丢成果，只是少了那份"部分答案"
 * 事件的收尾叙事。 */
const STOP_GRACE_MS = 8_000;

/** 续做时提交的提示。它与断点一起构成"在同一计划上继续"：计划由服务端
 * 从断点恢复，这条提示只负责让模型把注意力放回未完成的步骤上。 */
const RESUME_PROMPT = "请接着上次未完成的部分继续研究，不要重复已经完成的步骤。";

/** 断点横幅的措辞，取决于上一轮是怎么停下的。
 *
 * `user_stopped` 是用户自己按的停止；其余（如硬取消的 `interrupted`、
 * 达到轮次上限）都是"被动中断"。两者都留有断点、都能续做，但说成同一句话
 * 会让用户误以为是自己按了停止。 */
function resumeBannerText(resumable: ResumableTurn): string {
  if (resumable.reason === "user_stopped") {
    return `上一轮已按你的要求停止（已完成 ${resumable.rounds} 轮），`;
  }
  return `上一轮未能完成（已完成 ${resumable.rounds} 轮），`;
}

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

/** 所有可见执行记录都使用同一份用户语言词典。 */
function activityLabel(name: string): string {
  return presentToolAction(name);
}

function traceKind(name: string): AgentStep["kind"] {
  if (name === "research_plan") return "plan";
  return "tool";
}

function traceLabel(name: string): string {
  return presentToolAction(name);
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
    resumable,
    onResume,
    onRefreshResumable,
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
  // 用户是否已请求停止本轮，以及为此启动的兜底计时器。后者在宽限期内
  // 收到 done 时取消，否则到点即硬断连接。
  const [stopping, setStopping] = useState(false);
  const stoppingRef = useRef(false);
  const stopTimerRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const runStartedRef = useRef(0);
  const firstTokenRef = useRef<number | null>(null);
  // 本轮是否已由 done/error 事件收敛到终态。传输层另行失败时（流被截断、
  // 连接被代理掐断），finally 需要据此补上终态，否则界面停在"运行中"。
  const settledRef = useRef(false);
  // 空闲看门狗：后端每 15s 心跳一次，长时间收不到任何事件即说明连接
  // 已死，不能无限期地等下去。
  const watchdogRef = useRef<number | null>(null);
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
      if (watchdogRef.current !== null) {
        window.clearTimeout(watchdogRef.current);
        watchdogRef.current = null;
      }
      if (stopTimerRef.current !== null) {
        window.clearTimeout(stopTimerRef.current);
        stopTimerRef.current = null;
      }
      // 切换/新建对话会以 sessionVersion 为 key 重新挂载本组件，但本组件
      // 启动的流仍在跑。先请服务端做协作式停止（本轮成果因此照常落库），
      // 再断开连接兜底；只断开而不请求停止会让这一轮的取数与结论白费。
      if (stoppingRef.current === false) {
        const { session, conversation } = streamIdsRef.current;
        void stopChat(conversation, session).catch(() => undefined);
        stoppingRef.current = true;
      }
      abortRef.current?.abort();
      abortRef.current = null;
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

  /** 每次收到事件都重置空闲计时；超时即中止，避免界面无限期"运行中"。 */
  function armWatchdog() {
    if (watchdogRef.current !== null) window.clearTimeout(watchdogRef.current);
    watchdogRef.current = window.setTimeout(() => {
      watchdogRef.current = null;
      abortRef.current?.abort();
      setMessages((current) => [
        ...current,
        { role: "error", text: "响应流长时间无数据，已中断本次请求，请重试。" },
      ]);
    }, STREAM_IDLE_TIMEOUT_MS);
  }

  function disarmWatchdog() {
    if (watchdogRef.current !== null) {
      window.clearTimeout(watchdogRef.current);
      watchdogRef.current = null;
    }
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
    void sendMessage(message);
  }

  /** 提交一条提问并消费它的流。与表单解耦，使"继续研究"走同一条路径。 */
  async function sendMessage(message: string) {
    runStartedRef.current = performance.now();
    firstTokenRef.current = null;
    stoppingRef.current = false;
    setStopping(false);
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
    settledRef.current = false;
    const controller = new AbortController();
    abortRef.current = controller;
    armWatchdog();
    setNotice(null);
    try {
      await streamChat(message, conversationId, (event) => {
        armWatchdog();
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
          setStatus(`${presentToolAction(name)}${state === "started" ? "中" : state === "completed" ? "已完成" : "未完成"}`);
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
        // 长时间工具的中间进展：更新正在运行那一步的说明，让"还在推进"可见。
        // 这也持续重置空闲看门狗——心跳注释帧做不到这件事，因此没有这类事件的
        // 长任务会被看门狗判为卡死而中止（见 handleToolProgress 与 armWatchdog）。
        if (event.event === "tool_progress") {
          const callId = String(event.data.call_id ?? "");
          const detail = presentToolProgress(event.data);
          if (callId && detail) {
            setStatus(detail);
            updateTrace((trace) => ({
              ...trace,
              steps: trace.steps.map((step) =>
                step.key === callId && step.status === "running" ? { ...step, detail } : step,
              ),
            }));
            emitActivities((current) =>
              current.map((item) =>
                item.key === callId && item.status === "running" ? { ...item, detail } : item,
              ),
            );
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
        // loop_guard 刻意不进 UI：同参数重复调用的去重是引擎自愈，用户侧
        // 没有可感知后果，展示出来只是把内部机制暴露给用户。升级为提前
        // 结束本轮时才需要说明，而那种情况已由 error 事件带通俗文案送达
        // （见下方 reasonText），因此这里无需再补一条提示。
        if (event.event === "plan_progress") {
          const plan = planFromEvent(event.data);
          const drift = (event.data.drift as string[] | undefined) ?? [];
          const mismatch = (event.data.mismatch as string[] | undefined) ?? [];
          const stalled = Number(event.data.stalled_turns ?? 0);
          updateTrace((trace) => ({
            ...trace,
            planned: true,
            plan: plan ?? trace.plan,
            steps: trace.steps.filter((step) => step.key !== "plan-progress"),
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
          settledRef.current = true;
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
          settledRef.current = true;
          // 用户主动停止的一轮带 reason="user_stopped" 收尾。它不是失败：
          // 不发 error 气泡、状态用 stopped 而非 error，并提示可以继续。
          const stopped = event.data.reason === "user_stopped";
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
            const resting = succeeded ? "done" as const : stopped ? "stopped" as const : "error" as const;
            const closedSteps = withoutAgents.map((step) => ({
              ...step,
              status: step.status === "running" ? resting : step.status,
            }));
            return {
              ...trace,
              status: succeeded ? "done" : stopped ? "stopped" : "error",
              plan: planFromEvent((event.data.plan as Record<string, unknown> | undefined) ?? {}) ?? trace.plan,
              steps: [
                ...closedSteps,
                ...agentSteps,
                ...(hasFinal ? [] : [{ key: "final", kind: "final" as const, label: "整理研究结论", status: resting }]),
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
          if (stopped) {
            setNotice("已停止本轮生成，已获取的数据与结论已保留，可继续研究。");
          }
          setStatus(null);
          setInteraction(null);
          refreshCitations();
          // 本轮结束（无论是停止还是正常完成），可继续状态由服务端说了算。
          // 用流里已知的对话 id，而不是发送时刻闭包里的那个（新对话时它是 null）。
          onRefreshResumable?.(streamIdsRef.current.conversation);
        }
      }, controller.signal);
    } catch (error) {
      // 用户停止会让我们主动断开连接，那是一个 AbortError；它不是错误，
      // 不该向用户报错。真正的传输失败仍照常提示。
      const aborted = (error as Error).name === "AbortError";
      if (!aborted && !stoppingRef.current) {
        setMessages((current) => [...current, { role: "error", text: (error as Error).message }]);
      }
    } finally {
      disarmWatchdog();
      if (stopTimerRef.current !== null) {
        window.clearTimeout(stopTimerRef.current);
        stopTimerRef.current = null;
      }
      // 兜底收敛：只要本轮没有走到 done/error，就不能让"运行中"留在屏幕上。
      // 传输失败、流被截断、空闲超时都会走到这里。
      if (mountedRef.current && !settledRef.current) {
        // 停止路径上断流是预期结果（宽限期到了的兜底），因此收敛为 stopped
        // 并给出可继续的提示，而不是渲染成一个错误。
        const stopped = stoppingRef.current;
        updateTrace((trace) => ({
          ...trace,
          status: stopped ? "stopped" : "error",
          steps: trace.steps.map((step) =>
            step.status === "running"
              ? { ...step, status: stopped ? ("stopped" as const) : ("error" as const) }
              : step,
          ),
        }));
        setStatus(null);
        setInteraction(null);
        if (stopped) {
          setNotice("已停止本轮生成，已获取的数据与结论已保留，可继续研究。");
        }
      }
      setBusy(false);
      setStopping(false);
      abortRef.current = null;
      // 停止靠断开兜底时，服务端的可继续断点要重新取一次才知道它写好了。
      if (stoppingRef.current) onRefreshResumable?.(streamIdsRef.current.conversation);
    }
  }

  /**
   * 请求停止本轮生成。
   *
   * 先请服务端做协作式停止：引擎在下一个等待点收尾，已有成果照常落库。
   * 若宽限期内没有等到终止帧（引擎卡在一次很长的工具调用里），再直接断开
   * 连接兜底——服务端在断开路径上同样会落库并写下断点。
   */
  function stop() {
    if (!busy || stoppingRef.current) return;
    stoppingRef.current = true;
    setStopping(true);
    // 宽限计时先起，避免停止请求本身卡住时界面无限等待。
    stopTimerRef.current = window.setTimeout(() => {
      stopTimerRef.current = null;
      abortRef.current?.abort();
    }, STOP_GRACE_MS);
    const { session, conversation } = streamIdsRef.current;
    void stopChat(conversation, session).catch(() => {
      // 请求失败也不阻塞：直接走断开兜底，停止的意图不该因此落空。
      if (stopTimerRef.current !== null) {
        window.clearTimeout(stopTimerRef.current);
        stopTimerRef.current = null;
      }
      abortRef.current?.abort();
    });
  }

  /** 在上一轮被停止后接着研究：计划由服务端从断点恢复。 */
  function resume() {
    if (busy) return;
    void sendMessage(RESUME_PROMPT);
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
      {messages.length === 0 ? (
        <>
          {!configured && (
            <Empty
              className="config-empty-state"
              description="尚未配置模型供应商，配置后即可开始研究"
            >
              <Button type="primary" onClick={onOpenSettings}>
                去完成模型设置
              </Button>
            </Empty>
          )}
          <ResearchWelcome onUsePrompt={setInput} />
        </>
      ) : <MessageList messages={messages} citations={citations} />}
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
      {/* 上一轮未完成但留有断点：给出续做入口，让用户不必重述问题。
          计划由服务端从断点恢复，因此续做不会重跑已完成的步骤。
          措辞随原因而变：用户主动停止与"被中断后恢复"是两回事，用同一句
          会让人以为是自己按了停止。 */}
      {!busy && resumable && (
        <div className="resume-banner">
          <span>
            {resumeBannerText(resumable)}已获取的数据与结论已保留。
          </span>
          <Button size="small" type="primary" onClick={resume}>
            继续研究
          </Button>
        </div>
      )}
      <form className="composer" onSubmit={submit}>
        <textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="输入研究问题" rows={3} disabled={busy} />
        {busy ? (
          <button type="button" className="stop-button" onClick={stop} disabled={stopping}>
            {stopping ? "正在停止…" : "停止"}
          </button>
        ) : (
          <button type="submit" disabled={!input.trim() || !configured}>发送</button>
        )}
      </form>
      <InteractionPrompt interaction={interaction} onRespond={handleRespond} busy={false} />
    </section>
  );
});

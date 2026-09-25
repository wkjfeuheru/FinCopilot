import {
  FormEvent,
  forwardRef,
  KeyboardEvent,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { Button, Empty } from "antd";
import { artifactUrl, fetchCitations, respondChat, stopChat, streamChat } from "../api/client";
import type { Citation, ResumableTurn } from "../api/client";
import {
  asPublicAgentState,
  presentAgentPhase,
  reduceAgentState,
  type PublicAgentState,
} from "../lib/agentState";
import { Interaction, InteractionPrompt } from "./InteractionPrompt";
import { enqueue, resolve } from "../lib/interactionQueue";
import { Message, MessageList } from "./MessageList";
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
  /** 续做一轮：显式 resume=true，不伪造用户提问。 */
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
  // 发送后把焦点还给输入框：连续追问是高频路径，省一次点击。
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  // 最近一次提交的提问，供错误气泡的重试按钮复用。
  const lastPromptRef = useRef<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  // 正在执行的工具/步骤。它进到正在生成的消息内部（而非只出现在
  // composer 上方）：视线停在消息区时也能看到"还在推进"。
  const [activeTool, setActiveTool] = useState<string | null>(null);
  // 待用户作答的提示（引擎暂停）。用队列而非单值：同一轮里并发的多个
  // 工具确认或提问会先后到达，单值覆盖会让先到的提示连同它的 request_id
  // 一起消失，那个请求便再也无法被应答，只能等满服务端 TTL 被判拒绝。
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  // 队列的同步镜像：事件在同一批里连续到达时，state 尚未提交，靠它给出
  // 即时的队列长度判断（决定看门狗是否该停摆）。
  const interactionsRef = useRef<Interaction[]>([]);
  // 本轮流是否仍在进行。收尾时置 false，使清空队列不再重新装上看门狗
  // （流已结束，装上只会在 90s 后报一次虚假的"无数据"中断）。
  const streamActiveRef = useRef(false);
  // 本轮最新的 public agent state（按 run_id + revision 收敛）。
  const agentStateRef = useRef<PublicAgentState | null>(null);
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

  // 空对话挂载时聚焦输入框，省一次点击；触屏设备跳过——自动聚焦会把
  // 软键盘拉出来顶走欢迎页。
  useEffect(() => {
    if (window.matchMedia("(hover: none)").matches) return;
    composerRef.current?.focus();
    // 只在挂载时执行一次；StrictMode 的双挂载聚焦两次无副作用。
  }, []);

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
        {
          role: "error",
          text: "响应流长时间无数据，已中断本次请求，请重试。",
          retryPrompt: lastPromptRef.current ?? undefined,
        },
      ]);
    }, STREAM_IDLE_TIMEOUT_MS);
  }

  function disarmWatchdog() {
    if (watchdogRef.current !== null) {
      window.clearTimeout(watchdogRef.current);
      watchdogRef.current = null;
    }
  }

  /** 队列里还有待作答的提示时让看门狗停摆：此时的静默是预期行为（等用户
   * 输入），而非连接已死；否则重新计时。等待的上界由服务端的 confirm TTL
   * 兜底——超时后服务端会下发 interaction_resolved，队列随之清空。 */
  function syncWatchdog() {
    if (interactionsRef.current.length > 0) disarmWatchdog();
    else if (streamActiveRef.current) armWatchdog();
  }

  /** 更新队列并同步看门狗；所有队列改动都经此，避免 state 与镜像脱节。 */
  function updateInteractions(next: Interaction[]) {
    interactionsRef.current = next;
    setInteractions(next);
    syncWatchdog();
  }

  function enqueueInteraction(item: Interaction) {
    updateInteractions(enqueue(interactionsRef.current, item));
  }

  function resolveInteraction(requestId: string | null) {
    updateInteractions(resolve(interactionsRef.current, requestId));
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

  /** 提交当前输入。表单提交与快捷键共用同一条入口；输入在生成期间
   * 保持可用（方便边等边准备下一个问题），这里再拦一次 busy。 */
  function submitMessage() {
    const message = input.trim();
    if (!message || busy) return;
    setInput("");
    composerRef.current?.focus();
    void sendMessage(message);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    submitMessage();
  }

  // Ctrl/Cmd+Enter 提交；Enter 保持换行（避免与中文输入法的候选确认
  // 打架），isComposing 排除输入法组合期间的事件。
  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submitMessage();
    }
  }

  /** 欢迎页场景卡：点击直接作为提问发出，不再只填充输入框。 */
  function sendPrompt(prompt: string) {
    if (busy) return;
    void sendMessage(prompt);
  }

  /** 从错误气泡一键重试：原样重发触发那轮失败的提问。 */
  function retryPrompt(prompt: string) {
    if (busy || !prompt) return;
    void sendMessage(prompt);
  }

  /** 提交一条提问并消费它的流。与表单解耦，使"继续研究"走同一条路径。 */
  async function sendMessage(message: string, options?: { resume?: boolean }) {
    const resume = options?.resume === true;
    runStartedRef.current = performance.now();
    firstTokenRef.current = null;
    stoppingRef.current = false;
    setStopping(false);
    agentStateRef.current = null;
    setMessages((current) => [
      ...current,
      // 显式 resume 不伪造用户气泡；普通发送才追加提问。
      ...(resume ? [] : [{ role: "user" as const, text: message }]),
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
    // 记住这轮的提问：错误气泡用它提供一键重试。resume 无用户提问可重试。
    lastPromptRef.current = resume ? null : message;
    const controller = new AbortController();
    abortRef.current = controller;
    streamActiveRef.current = true;
    armWatchdog();
    setNotice(null);
    try {
      await streamChat(message, conversationId, (event) => {
        // 每个事件都刷新空闲计时；但队列非空（正在等用户作答）时不计时，
        // 见 syncWatchdog。
        syncWatchdog();
        if (event.event === "session") {
          const session = String(event.data.session_id);
          const conversation = event.data.conversation_id ? String(event.data.conversation_id) : null;
          streamIdsRef.current = { session, conversation };
          onSession(session, conversation);
        }
        if (event.event === "state") {
          const incoming = asPublicAgentState(event.data);
          if (incoming) {
            const next = reduceAgentState(agentStateRef.current, incoming);
            if (next !== agentStateRef.current) {
              agentStateRef.current = next;
              const presented = presentAgentPhase(next);
              setStatus(presented.label);
              if (presented.status !== "running") {
                updateTrace((trace) => ({ ...trace, status: presented.status }));
              }
            }
          }
        }
        if (event.event === "tool_status") {
          const name = String(event.data.name ?? "tool");
          const callId = String(event.data.call_id ?? name);
          const state = String(event.data.status ?? "");
          setStatus(`${presentToolAction(name)}${state === "started" ? "中" : state === "completed" ? "已完成" : "未完成"}`);
          setActiveTool(state === "started" ? presentToolAction(name) : null);
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
            setActiveTool((current) => current ?? "正在推进");
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
          // 引擎已暂停，等待回答；入队该对话框（一次只呈现队首）。
          const prompt = String(event.data.prompt ?? "");
          enqueueInteraction({
            requestId: String(event.data.request_id ?? ""),
            kind: String(event.data.kind ?? "question"),
            prompt,
            options: (event.data.options as string[] | undefined) ?? [],
            multiSelect: event.data.multi_select === true,
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
        if (event.event === "interaction_resolved") {
          // 该提示已落定（批准/拒绝/超时）：清出队列并收尾对应步骤，
          // 否则"等待操作确认"会永远停在"进行中"。
          const requestId = String(event.data.request_id ?? "");
          const timedOut = event.data.timeout === true;
          const answer = event.data.answer === null || event.data.answer === undefined ? "" : String(event.data.answer);
          // 只有确认（confirm）才有"拒绝"语义；提问的自由文本回答里出现
          // "n" 只是一个普通答案，不能据此渲染成失败。
          const kind = interactionsRef.current.find((entry) => entry.requestId === requestId)?.kind;
          const denied = timedOut || (kind === "confirm" && (answer === "" || answer === "n"));
          resolveInteraction(requestId);
          updateTrace((trace) => ({
            ...trace,
            steps: trace.steps.map((step) => step.key === `ask-${requestId}` && step.status === "running"
              ? {
                  ...step,
                  status: denied ? ("error" as const) : ("done" as const),
                  detail: timedOut ? "超时未回答，已视为拒绝" : denied ? "已拒绝" : undefined,
                }
              : step),
          }));
          emitActivities((current) => current.map((item) => item.key === `ask-${requestId}`
            ? { ...item, status: denied ? ("error" as const) : ("done" as const) }
            : item));
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
          updateInteractions([]);
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
          setMessages((current) => [
            ...current,
            { role: "error", text: reasonText[reason] ?? message, retryPrompt: lastPromptRef.current ?? undefined },
          ]);
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
          setActiveTool(null);
          updateInteractions([]);
          refreshCitations();
          // 本轮结束（无论是停止还是正常完成），可继续状态由服务端说了算。
          // 用流里已知的对话 id，而不是发送时刻闭包里的那个（新对话时它是 null）。
          onRefreshResumable?.(streamIdsRef.current.conversation);
        }
      }, controller.signal, resume);
    } catch (error) {
      // 用户停止会让我们主动断开连接，那是一个 AbortError；它不是错误，
      // 不该向用户报错。真正的传输失败仍照常提示。
      const aborted = (error as Error).name === "AbortError";
      if (!aborted && !stoppingRef.current) {
        setMessages((current) => [
          ...current,
          { role: "error", text: (error as Error).message, retryPrompt: lastPromptRef.current ?? undefined },
        ]);
      }
    } finally {
      // 先声明本轮结束，再清空队列：否则清空触发的 syncWatchdog 会把
      // 看门狗重新装上，在流已经结束后留一个虚假的中断计时器。
      streamActiveRef.current = false;
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
        setActiveTool(null);
        updateInteractions([]);
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

  /** 在上一轮被停止后接着研究：显式 resume=true，不追加伪造用户提问。 */
  function resume() {
    if (busy) return;
    onResume?.();
    void sendMessage("", { resume: true });
  }

  async function handleRespond(requestId: string, response: string) {
    // 先出队，队首让位给下一个待处理提示；服务端随后下发的
    // interaction_resolved 按 id 幂等清尾，重复到达也不会出错。
    resolveInteraction(requestId);
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
          <ResearchWelcome onUsePrompt={sendPrompt} />
        </>
      ) : <MessageList messages={messages} citations={citations} onRetry={retryPrompt} />}
      {notice && <div className="engine-notice">{notice}</div>}
      {busy && activeTool && (
        <div className="inline-running-status">
          <span className="inline-running-dot" aria-hidden="true" />
          {activeTool === "正在推进" ? status : activeTool}
        </div>
      )}
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
      {/* ToolStatus 已由上方内联在消息区的运行状态行取代：视线停在消息
          流里就能看到"还在推进"，不必把目光挪回 composer 上方。 */}
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
        {/* 生成期间不禁用输入：一轮研究以分钟计，用户应能边等边准备
            下一个问题。真正防止误发的是 submitMessage 的 busy 守卫与
            发送/停止按钮的互斥切换。 */}
        <textarea
          ref={composerRef}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={onComposerKeyDown}
          placeholder="输入研究问题，Ctrl+Enter 发送"
          rows={3}
        />
        {busy ? (
          <button type="button" className="stop-button" onClick={stop} disabled={stopping}>
            {stopping ? "正在停止…" : "停止"}
          </button>
        ) : (
          <button type="submit" disabled={!input.trim() || !configured}>发送</button>
        )}
      </form>
      <InteractionPrompt
        key={interactions[0]?.requestId ?? "none"}
        interaction={interactions[0] ?? null}
        pendingCount={interactions.length}
        onRespond={handleRespond}
        busy={false}
      />
    </section>
  );
});

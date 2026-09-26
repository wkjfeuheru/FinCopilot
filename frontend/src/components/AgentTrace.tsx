import { useState } from "react";
import type { StoredTurn } from "../api/client";
import {
  asPublicAgentState,
  presentAgentPhase,
  reduceAgentState,
  type PublicAgentState,
} from "../lib/agentState";
import {
  asToolSummary,
  planFromEvent,
  presentLiveAction,
  presentSkillAction,
  presentToolAction,
  presentToolProgress,
  type ToolSummary,
} from "../lib/researchPresentation";
import type { ResearchPlan } from "../lib/researchPresentation";
import type { SpawnTaskProgress } from "../lib/liveStatus";
import { ResearchPlanLedger } from "./ResearchPlanLedger";

/** 一步/一轮的执行状态。``stopped`` 是用户主动停止，刻意区别于 ``error``：
 * 停止不是失败，界面不该用错误色与"出错"措辞来呈现它。 */
export type TraceStatus = "running" | "done" | "error" | "info" | "stopped";

export type AgentStep = {
  key: string;
  kind: "analysis" | "plan" | "skill" | "tool" | "agent" | "final" | "system";
  label: string;
  status: TraceStatus;
  detail?: string;
  durationMs?: number;
  tokens?: number;
  /** 该步产出的文件（图表、研报）。持久化在轮次事件中，使刷新后仍可还原。 */
  attachments?: string[];
  /** 工具的线名，覆盖条与侧栏按它分组。 */
  toolName?: string;
  summary?: ToolSummary;
  spawnTasks?: SpawnTaskProgress[];
};

export type TurnMetrics = {
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  steps: number;
  firstTokenMs: number | null;
  totalDurationMs: number;
  toolDurationMs: number;
};

export type TurnTrace = {
  planned: boolean;
  status: "running" | "done" | "error" | "stopped";
  steps: AgentStep[];
  plan?: ResearchPlan;
  metrics?: TurnMetrics;
};

const KIND_LABELS: Record<AgentStep["kind"], string> = {
  analysis: "分析",
  plan: "规划",
  skill: "研究方法",
  tool: "工具",
  agent: "子代理",
  final: "汇总",
  system: "系统",
};

function storedKind(name: string): AgentStep["kind"] {
  if (name === "research_plan") return "plan";
  return "tool";
}

function storedLabel(name: string, summary: ToolSummary | undefined, started: boolean): string {
  return presentLiveAction(name, summary, started ? "running" : "done");
}

function storedSpawnTasks(
  name: string,
  summary: ToolSummary | undefined,
  started: boolean,
  ok: boolean,
): SpawnTaskProgress[] | undefined {
  if (name !== "spawn_agent" || !summary?.tasks?.length) return undefined;
  const status: SpawnTaskProgress["status"] = started ? "running" : ok ? "done" : "error";
  return summary.tasks.map((task, index) => ({ index, task, status }));
}

function storedAgentLabel(name: string): string {
  return name === "risk" ? "风险审阅子代理" : "独立研究子代理";
}

/** 从随历史回答持久化的事件中重建可见的执行记录。 */
export function traceFromStoredTurn(turn: StoredTurn): TurnTrace {
  let planned = false;
  let status: TurnTrace["status"] = "done";
  let steps: AgentStep[] = [
    { key: "analysis", kind: "analysis", label: "理解问题并确定研究路径", status: "done" },
  ];
  let plan: ResearchPlan | undefined;
  let metrics: TurnMetrics | undefined;
  let publicState: PublicAgentState | null = null;

  const upsert = (next: AgentStep) => {
    steps = steps.some((step) => step.key === next.key)
      ? steps.map((step) => (step.key === next.key ? { ...step, ...next } : step))
      : [...steps, next];
  };

  for (const event of turn.events ?? []) {
    const data = event.data ?? {};
    if (event.event === "state") {
      const incoming = asPublicAgentState(data);
      if (incoming) publicState = reduceAgentState(publicState, incoming);
    }
    if (event.event === "tool_status") {
      const name = String(data.name ?? "tool");
      const callId = String(data.call_id ?? name);
      const started = data.status === "started";
      const summary = asToolSummary(data.summary);
      planned ||= name === "research_plan";
      upsert({
        key: callId,
        kind: storedKind(name),
        toolName: name,
        label: storedLabel(name, summary, started),
        status: started ? "running" : data.ok === false ? "error" : "done",
        durationMs: started ? undefined : Number(data.duration_ms ?? 0) || undefined,
        detail: data.ok === false ? String(data.error ?? "执行失败") : undefined,
        summary,
        spawnTasks: storedSpawnTasks(name, summary, started, data.ok !== false),
        // 工具产出的文件随事件持久化，重建 trace 时一并还原，
        // 否则刷新后产出文件一栏会消失。只在完成事件上写入，
        // 避免覆盖已记录的文件。
        ...(started
          ? {}
          : { attachments: (data.attachments as string[] | undefined) ?? [] }),
      });
    }
    if (event.event === "tool_progress") {
      // 进展事件只更新正在运行那一步的说明。历史回放与实时流走同一条渲染路径，
      // 否则刷新后长任务的进度说明会凭空消失。已存在的步骤保持其既有状态，
      // 避免进展事件晚于完成事件到达时把"已完成"回退成"运行中"。
      const callId = String(data.call_id ?? "");
      const detail = presentToolProgress(data);
      if (callId && detail) {
        if (steps.some((step) => step.key === callId)) {
          steps = steps.map((step) => (step.key === callId ? { ...step, detail } : step));
        } else {
          const name = String(data.name ?? "tool");
          upsert({
            key: callId,
            kind: storedKind(name),
            toolName: name,
            label: storedLabel(name, undefined, true),
            status: "running",
            detail,
          });
        }
      }
    }
    if (event.event === "context_compacted") {
      const before = Number(data.before_tokens ?? 0);
      const after = Number(data.after_tokens ?? 0);
      upsert({
        key: `compact-${before}-${after}`,
        kind: "system",
        label: "压缩研究上下文",
        status: "done",
        detail: `${before.toLocaleString("zh-CN")} → ${after.toLocaleString("zh-CN")} Token${data.degraded ? " · 摘要降级" : ""}`,
      });
    }
    // 路由注入与按需激活都是引擎的隐式动作，用户看不到工具调用，因此这里显式
    // 记一行——否则"模型为什么知道这个方法/这个工具"在界面上无从解释。
    if (event.event === "context_routed") {
      const skills = (data.skills as string[] | undefined) ?? [];
      if (skills.length) {
        upsert({
          key: `routed-${skills.join(",")}`,
          kind: "skill",
          toolName: "skill",
          label: presentSkillAction(skills, "done"),
          status: "done",
          detail: skills.join("、"),
        });
      }
    }
    if (event.event === "tool_activated") {
      upsert({
        key: `activated-${String(data.call_id ?? data.name)}`,
        kind: "system",
        label: "按需启用研究能力",
        status: "done",
        detail: presentToolAction(String(data.name ?? "")),
      });
    }
    // loop_guard 与实时流一致地不进 trace：去重属引擎自愈，不是用户可见的
    // 执行节点；升级为提前结束本轮时由 error 事件说明。
    if (event.event === "plan_progress") {
      planned = true;
      plan = planFromEvent(data) ?? plan;
      if (plan) {
        // 计划是独立台账，避免与普通执行动作混在同一条流水里。
        steps = steps.filter((step) => step.key !== "plan-progress");
      }
    }
    if (event.event === "interactive_request") {
      upsert({
        key: `ask-${String(data.request_id ?? steps.length)}`,
        kind: "system",
        label: data.kind === "confirm" ? "完成操作确认" : "完成补充信息",
        status: "done",
        detail: String(data.prompt ?? ""),
      });
    }
    if (event.event === "done") {
      const succeeded = data.succeeded !== false;
      // 用户主动停止的一轮不是失败：它带着已有成果正常收尾，因此还原为
      // stopped（黄/中性态 + "已停止"），而不是 error。刷新后重放的历史
      // 走的是这条路径，与实时流的处理必须一致。
      const stopped = data.reason === "user_stopped";
      const restingStatus = stopped ? ("stopped" as const) : ("error" as const);
      status = succeeded ? "done" : stopped ? "stopped" : "error";
      steps = steps.map((step) =>
        step.status === "running"
          ? { ...step, status: succeeded ? ("done" as const) : restingStatus }
          : step,
      );
      const perAgent =
        (data.per_agent as Record<string, Record<string, unknown>> | undefined) ?? {};
      let agentRuns = 0;
      Object.entries(perAgent).forEach(([name, value], index) => {
        const runs = Number(value.runs ?? 0);
        agentRuns += runs;
        upsert({
          key: `agent-${name}-${index}`,
          kind: "agent",
          label: storedAgentLabel(name),
          status: "done",
          detail: `运行 ${runs} 次`,
          tokens: Number(value.input_tokens ?? 0) + Number(value.output_tokens ?? 0),
        });
      });
      upsert({
        key: "final",
        kind: "final",
        label: "整理研究结论",
        status: succeeded ? "done" : restingStatus,
      });
      const usage = (data.usage as Record<string, unknown> | undefined) ?? {};
      const inputTokens = Number(usage.input_tokens ?? 0);
      const outputTokens = Number(usage.output_tokens ?? 0);
      metrics = {
        totalTokens: inputTokens + outputTokens,
        inputTokens,
        outputTokens,
        steps: Number(data.tool_calls ?? 0) + agentRuns + 1,
        firstTokenMs: turn.first_token_ms,
        totalDurationMs: Number(turn.total_duration_ms ?? 0),
        toolDurationMs: Number(data.tool_duration_ms ?? 0),
      };
      plan = planFromEvent((data.plan as Record<string, unknown> | undefined) ?? {}) ?? plan;
    }
  }

  // 高阶 phase 以最后一条有效 state 为准；done 仍负责指标与步骤细节。
  if (publicState !== null) {
    status = presentAgentPhase(publicState).status;
  }

  return { planned: planned || Boolean(plan), status, steps, plan, metrics };
}

function formatDuration(value: number | null): string {
  if (value === null) return "—";
  if (value < 1000) return `${Math.max(0, Math.round(value))} ms`;
  return `${(value / 1000).toFixed(value < 10000 ? 2 : 1)} s`;
}

function statusLabel(status: TraceStatus): string {
  if (status === "running") return "进行中";
  if (status === "done") return "已完成";
  if (status === "error") return "失败";
  if (status === "stopped") return "已停止";
  return "已记录";
}

function StepMark({ status }: { status: TraceStatus }) {
  return (
    <span className={`trace-step-mark trace-step-mark-${status}`} aria-hidden="true">
      {status === "done"
        ? "✓"
        : status === "error"
          ? "!"
          : status === "running"
            ? ""
            : status === "stopped"
              ? "■"
              : "·"}
    </span>
  );
}

export function TurnMetricsBar({ metrics }: { metrics: TurnMetrics }) {
  const items = [
    { label: "总 Token", value: metrics.totalTokens.toLocaleString("zh-CN"), title: `输入 ${metrics.inputTokens.toLocaleString("zh-CN")} · 输出 ${metrics.outputTokens.toLocaleString("zh-CN")}` },
    { label: "执行节点", value: String(metrics.steps) },
    { label: "首 Token 延迟", value: formatDuration(metrics.firstTokenMs) },
    { label: "总耗时", value: formatDuration(metrics.totalDurationMs), title: `工具累计耗时 ${formatDuration(metrics.toolDurationMs)}` },
  ];
  return (
    <div className="turn-metrics" aria-label="本轮运行指标">
      {items.map((item) => (
        <span className="turn-metric" key={item.label} title={item.title}>
          <strong>{item.value}</strong>
          <small>{item.label}</small>
        </span>
      ))}
    </div>
  );
}

export function AgentTrace({ trace }: { trace: TurnTrace }) {
  // 实时一轮默认展开，让用户看到 Agent 正在做什么；从历史还原的轮次
  // （状态已是 done/error）默认收起，切换对话时不会立刻铺满整页步骤。
  // 之后由用户自己的开合决定，不再随状态变化。
  const [expanded, setExpanded] = useState(() => trace.status === "running");
  const completed = trace.steps.filter((step) => step.status === "done").length;
  return (
    <details
      className={`turn-trace turn-trace-${trace.status}`}
      open={expanded}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>
        <span className="trace-summary-title">
          <span className="trace-pulse" aria-hidden="true" />
          分析与执行过程
        </span>
        <span className="trace-summary-meta">
          {trace.status === "running"
            ? "研究任务执行中"
            : trace.status === "stopped"
              ? `已停止 · ${completed} 个节点已完成`
              : `${completed} 个节点已完成`}
        </span>
      </summary>
      <div className="trace-content">
        {trace.plan && <ResearchPlanLedger plan={trace.plan} />}
        <ol className="trace-steps">
          {trace.steps.filter((step) => !(trace.plan && step.kind === "plan")).map((step) => (
            <li className={`trace-step trace-step-${step.status}`} key={step.key}>
              <StepMark status={step.status} />
              <span className="trace-step-kind">{KIND_LABELS[step.kind]}</span>
              <span className="trace-step-body">
                <strong>
                  {step.toolName
                    ? presentLiveAction(
                        step.toolName,
                        step.summary,
                        step.status === "running" ? "running" : "done",
                      )
                    : step.label}
                </strong>
                {step.detail && <small>{step.detail}</small>}
              </span>
              <span className="trace-step-tail">
                {step.tokens !== undefined && <span>{step.tokens.toLocaleString("zh-CN")} Token</span>}
                {step.durationMs !== undefined && <span>{formatDuration(step.durationMs)}</span>}
                <em>{statusLabel(step.status)}</em>
              </span>
            </li>
          ))}
        </ol>
      </div>
    </details>
  );
}

import { useState } from "react";
import type { StoredTurn } from "../api/client";

export type TraceStatus = "running" | "done" | "error" | "info";

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
  status: "running" | "done" | "error";
  steps: AgentStep[];
  metrics?: TurnMetrics;
};

const KIND_LABELS: Record<AgentStep["kind"], string> = {
  analysis: "分析",
  plan: "规划",
  skill: "Skill",
  tool: "工具",
  agent: "子代理",
  final: "汇总",
  system: "系统",
};

function storedKind(name: string): AgentStep["kind"] {
  if (name === "research_plan") return "plan";
  if (name === "load_skill" || name === "list_skills") return "skill";
  return "tool";
}

function storedLabel(name: string): string {
  if (name === "research_plan") return "制定研究计划";
  if (name === "load_skill") return "加载研究 Skill";
  if (name === "list_skills") return "检索可用 Skill";
  if (name === "write_report") return "生成研报并执行风险终审";
  if (name === "make_chart") return "绘制研究图表";
  return `调用 ${name}`;
}

function storedAgentLabel(name: string): string {
  return name === "risk" ? "风险审阅子代理" : `${name} 子代理`;
}

/** 从随历史回答持久化的事件中重建可见的执行记录。 */
export function traceFromStoredTurn(turn: StoredTurn): TurnTrace {
  let planned = false;
  let status: TurnTrace["status"] = "done";
  let steps: AgentStep[] = [
    { key: "analysis", kind: "analysis", label: "理解问题并确定研究路径", status: "done" },
  ];
  let metrics: TurnMetrics | undefined;

  const upsert = (next: AgentStep) => {
    steps = steps.some((step) => step.key === next.key)
      ? steps.map((step) => (step.key === next.key ? { ...step, ...next } : step))
      : [...steps, next];
  };

  for (const event of turn.events ?? []) {
    const data = event.data ?? {};
    if (event.event === "tool_status") {
      const name = String(data.name ?? "tool");
      const callId = String(data.call_id ?? name);
      const started = data.status === "started";
      planned ||= name === "research_plan";
      upsert({
        key: callId,
        kind: storedKind(name),
        label: storedLabel(name),
        status: started ? "running" : data.ok === false ? "error" : "done",
        durationMs: started ? undefined : Number(data.duration_ms ?? 0) || undefined,
        detail: data.ok === false ? String(data.error ?? "执行失败") : undefined,
        // 工具产出的文件随事件持久化，重建 trace 时一并还原，
        // 否则刷新后产出文件一栏会消失。只在完成事件上写入，
        // 避免覆盖已记录的文件。
        ...(started
          ? {}
          : { attachments: (data.attachments as string[] | undefined) ?? [] }),
      });
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
    if (event.event === "loop_guard") {
      const aborted = data.action === "would_abort";
      upsert({
        key: `guard-${String(data.call_id ?? steps.length)}`,
        kind: "system",
        label: aborted ? "终止重复调用" : "跳过重复调用",
        status: aborted ? "error" : "info",
        detail: String(data.name ?? ""),
      });
    }
    if (event.event === "plan_progress") {
      planned = true;
      const revision = Number(data.revision ?? 1);
      const done = Number(data.done ?? 0);
      const total = Number(data.total ?? 0);
      const drift = (data.drift as string[] | undefined) ?? [];
      const mismatch = (data.mismatch as string[] | undefined) ?? [];
      const stalled = Number(data.stalled_turns ?? 0);
      const detail = [
        `进度 ${done}/${total}`,
        revision > 1 ? `第 ${revision} 版` : "",
        drift.length ? `目标外：${drift.join("、")}` : "",
        mismatch.length ? `能力外：${mismatch.join("、")}` : "",
        stalled ? `停滞 ${stalled} 轮` : "",
      ].filter(Boolean).join(" · ");
      upsert({
        key: "plan-progress",
        kind: "plan",
        label: "研究计划进度",
        status: "info",
        detail,
      });
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
      status = succeeded ? "done" : "error";
      steps = steps.map((step) =>
        step.status === "running"
          ? { ...step, status: succeeded ? ("done" as const) : ("error" as const) }
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
        status: succeeded ? "done" : "error",
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
    }
  }

  return { planned, status, steps, metrics };
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
  return "已记录";
}

function StepMark({ status }: { status: TraceStatus }) {
  return (
    <span className={`trace-step-mark trace-step-mark-${status}`} aria-hidden="true">
      {status === "done" ? "✓" : status === "error" ? "!" : status === "running" ? "" : "·"}
    </span>
  );
}

function PlanProgress({ trace }: { trace: TurnTrace }) {
  const planningDone = trace.steps.some((step) => step.kind === "plan" && step.status === "done");
  const executionDone = trace.steps.some((step) => step.kind === "final");
  const finalDone = trace.steps.some((step) => step.kind === "final" && step.status === "done");
  const phaseDone = [planningDone, executionDone, finalDone].filter(Boolean).length;
  const progress = trace.status === "done" ? 100 : Math.min(92, Math.round((phaseDone / 3) * 100));

  return (
    <div className="plan-progress" aria-label={`复杂任务完成进度 ${progress}%`}>
      <div className="plan-progress-head">
        <span>复杂任务进度</span>
        <strong>{progress}%</strong>
      </div>
      <div className="plan-progress-track">
        <span style={{ width: `${progress}%` }} />
      </div>
      <div className="plan-phases">
        {[
          ["规划", planningDone],
          ["执行", executionDone],
          ["汇总", finalDone],
        ].map(([label, done], index) => (
          <span key={String(label)} className={done ? "done" : phaseDone === index ? "active" : ""}>
            <i />{label}
          </span>
        ))}
      </div>
    </div>
  );
}

export function TurnMetricsBar({ metrics }: { metrics: TurnMetrics }) {
  const items = [
    { label: "总 Token", value: metrics.totalTokens.toLocaleString("zh-CN"), title: `输入 ${metrics.inputTokens.toLocaleString("zh-CN")} · 输出 ${metrics.outputTokens.toLocaleString("zh-CN")}` },
    { label: "Steps", value: String(metrics.steps) },
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
          {trace.status === "running" ? "Agent 正在工作" : `${completed} 个节点已完成`}
        </span>
      </summary>
      <div className="trace-content">
        {trace.planned && <PlanProgress trace={trace} />}
        <ol className="trace-steps">
          {trace.steps.map((step) => (
            <li className={`trace-step trace-step-${step.status}`} key={step.key}>
              <StepMark status={step.status} />
              <span className="trace-step-kind">{KIND_LABELS[step.kind]}</span>
              <span className="trace-step-body">
                <strong>{step.label}</strong>
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

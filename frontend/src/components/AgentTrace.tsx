export type TraceStatus = "running" | "done" | "error" | "info";

export type AgentStep = {
  key: string;
  kind: "analysis" | "plan" | "skill" | "tool" | "agent" | "final" | "system";
  label: string;
  status: TraceStatus;
  detail?: string;
  durationMs?: number;
  tokens?: number;
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
  const [expanded, setExpanded] = useState(true);
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
import { useState } from "react";

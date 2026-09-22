/** 监控视图的纯格式化逻辑；独立成模块以便单测，不依赖 React。 */

/** 比率展示：0.1234 → "12.3%"；null/undefined → "—"。 */
export function formatRate(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

/** 计数展示：null → "—"。 */
export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return String(value);
}

/** 平均值展示：保留两位小数。 */
export function formatAverage(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(2);
}

/** 毫秒 → 人类可读时长。 */
export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}m ${rest}s`;
}

/** 运行状态 → 中文标签与语义色（沿用工作台既有语义）。 */
export function presentStatus(status: string): { label: string; tone: "ok" | "error" | "warn" | "info" } {
  switch (status) {
    case "done":
      return { label: "已完成", tone: "ok" };
    case "stopped":
      return { label: "用户停止", tone: "info" };
    case "error":
      return { label: "失败", tone: "error" };
    case "aborted":
      return { label: "连接中断", tone: "warn" };
    case "running":
      return { label: "运行中", tone: "info" };
    default:
      return { label: status, tone: "info" };
  }
}

const REASON_LABELS: Record<string, string> = {
  done: "正常完成",
  max_turns_exhausted: "轮次耗尽",
  loop_detected: "重复调用中止",
  provider_error: "模型服务错误",
  user_stop: "用户停止",
  disconnected: "连接断开",
  engine_error: "引擎异常",
  transport_failed: "传输失败",
  missing_terminal_event: "缺少终止事件",
};

/** 停止/交付原因 → 中文标签。 */
export function presentReason(reason: string | null | undefined): string {
  if (!reason) return "—";
  return REASON_LABELS[reason] ?? reason;
}

const EVENT_LABELS: Record<string, string> = {
  tool_status: "工具调用",
  tool_activated: "按需激活工具",
  context_compacted: "上下文压缩",
  context_routed: "方法论注入",
  loop_guard: "重复调用拦截",
  plan_progress: "计划进展",
  interactive_request: "等待交互确认",
  interaction_resolved: "交互已应答",
  answer: "交付答案",
  error: "错误",
  done: "运行结束",
};

export function presentEventKind(kind: string): string {
  return EVENT_LABELS[kind] ?? kind;
}

/** 一个事件是否标记"跑偏点"：计划偏航、能力偏离或重复调用拦截。 */
export function isDriftEvent(kind: string, payload: Record<string, unknown> | null): boolean {
  if (kind === "loop_guard") return true;
  if (kind !== "plan_progress") return false;
  const drift = payload?.drift;
  const mismatch = payload?.mismatch;
  const stalled = payload?.stalled_turns;
  return Boolean(
    (Array.isArray(drift) && drift.length > 0) ||
      (Array.isArray(mismatch) && mismatch.length > 0) ||
      (typeof stalled === "number" && stalled > 0),
  );
}

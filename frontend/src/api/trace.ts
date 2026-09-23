import { authedFetch } from "./http";

/** 一次 agent 运行的摘要行（监控 run 列表）。 */
export type TraceRun = {
  run_id: string;
  source: string;
  user_id: string;
  session_id: string | null;
  conversation_id: string | null;
  eval_case_id: string | null;
  input: string;
  answer: string;
  status: "running" | "done" | "stopped" | "error" | "aborted" | string;
  reason: string | null;
  succeeded: number | null;
  rounds: number | null;
  tool_calls: number | null;
  retry_count: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cache_hit_tokens: number | null;
  per_agent: Record<string, unknown> | null;
  citations: string[] | null;
  plan: Record<string, unknown> | null;
  started_at: string;
  finished_at: string | null;
  duration_ms: number | null;
};

/** 一个轮次：模型怎么想、调了什么工具、返回了什么。 */
export type TraceRound = {
  turn: number;
  thought: string;
  actions: { call_id: string; name: string; args: string }[];
  observations: {
    call_id: string;
    name: string;
    ok: boolean;
    error: string | null;
    preview: string;
    duration_ms: number;
  }[];
  input_tokens: number;
  output_tokens: number;
  llm_first_ms: number;
  llm_ms: number;
  answer: string;
};

/** 一个引擎事件（tool_status / plan_progress 跑偏 / loop_guard 重复调用……）。 */
export type TraceEvent = {
  seq: number;
  turn: number | null;
  kind: string;
  payload: Record<string, unknown> | null;
};

export type TraceRunDetail = TraceRun & {
  rounds_trace: TraceRound[];
  events: TraceEvent[];
};

/** 七项运行指标 + 拆解。 */
export type TraceMetrics = {
  generated_at: string;
  filters: Record<string, string | null>;
  total_runs: number;
  completion: {
    task_completion_rate: number | null;
    status_counts: Record<string, number>;
    reason_counts: Record<string, number>;
  };
  tool_calls_total: number;
  avg_rounds: number | null;
  tool_failure_rate: number | null;
  repeat_call_rate: number | null;
  loop_guard_events: number;
  safety_blocks: number;
  timeout_rate: number | null;
  per_tool: Record<
    string,
    { failed: number; blocked: number; total: number; failure_rate: number | null }
  >;
  drift_first_turn: Record<string, number>;
  error?: string;
};

export type TraceFilters = {
  source?: string;
  status?: string;
  user_id?: string;
  conversation_id?: string;
  since?: string;
  until?: string;
};

/**
 * 监控可用性。两种状态正交，前端据此给出不同的引导：
 * 未启用（提示开启方法）/ 无权限（提示找运维提权）/ 可用（渲染数据）。
 * 权限已统一为 users.role='admin'；admin_users 白名单已退役。
 */
export type TraceStatus = {
  enabled: boolean;
  is_admin: boolean;
};

export async function fetchTraceStatus(): Promise<TraceStatus> {
  const response = await authedFetch("/v1/trace/status");
  if (!response.ok) throw new Error(`加载监控状态失败：${response.status}`);
  return (await response.json()) as TraceStatus;
}

function query(filters: TraceFilters, extra?: Record<string, string>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries({ ...filters, ...extra })) {
    if (value) params.set(key, value);
  }
  const text = params.toString();
  return text ? `?${text}` : "";
}

/** 把失败响应转成可读错误：优先采用服务端的 detail（如"监控未启用"的开启方法）。 */
async function failure(response: Response, fallback: string): Promise<Error> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail) return new Error(body.detail);
  } catch {
    // 非 JSON 错误体：退回状态码。
  }
  return new Error(`${fallback}：${response.status}`);
}

export async function fetchTraceRuns(
  filters: TraceFilters = {},
  limit = 50,
  offset = 0,
): Promise<{ runs: TraceRun[]; total: number }> {
  const response = await authedFetch(
    `/v1/trace/runs${query(filters, { limit: String(limit), offset: String(offset) })}`,
  );
  if (!response.ok) throw await failure(response, "加载运行列表失败");
  return (await response.json()) as { runs: TraceRun[]; total: number };
}

export async function fetchTraceRun(runId: string): Promise<TraceRunDetail> {
  const response = await authedFetch(`/v1/trace/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) throw await failure(response, "加载运行详情失败");
  return (await response.json()) as TraceRunDetail;
}

export async function fetchTraceMetrics(filters: TraceFilters = {}): Promise<TraceMetrics> {
  const response = await authedFetch(`/v1/trace/metrics${query(filters)}`);
  if (!response.ok) throw await failure(response, "加载指标失败");
  return (await response.json()) as TraceMetrics;
}

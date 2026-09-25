/** Public agent-loop phases exposed on SSE `state` events / history. */
export type AgentPhase =
  | "hydrate"
  | "thinking"
  | "tooluse"
  | "awaitingconfirmation"
  | "compact"
  | "complete"
  | "error";

/** Typed subset of `public_state_view`; unknown extras are ignored by callers. */
export type PublicAgentState = {
  run_id: string;
  revision: number;
  phase: AgentPhase;
  turn: number;
  calls: Array<{ call_id: string; name: string; status: string }>;
  outcome?: { kind: string; reason: string | null; resumable: boolean } | null;
  error?: { kind: string; message: string } | null;
};

export type PresentedAgentPhase = {
  status: "running" | "stopped" | "done" | "error";
  label: string;
};

const PHASE_LABELS: Record<Exclude<AgentPhase, "complete" | "error">, string> = {
  hydrate: "恢复会话",
  thinking: "思考中",
  tooluse: "调用工具",
  awaitingconfirmation: "等待确认",
  compact: "压缩上下文",
};

/**
 * Keep the newest revision for the active run.
 *
 * - null current → accept incoming
 * - different run_id → keep current
 * - revision not strictly greater → keep current
 */
export function reduceAgentState(
  current: PublicAgentState | null,
  incoming: PublicAgentState,
): PublicAgentState {
  if (current === null) return incoming;
  if (incoming.run_id !== current.run_id) return current;
  if (incoming.revision <= current.revision) return current;
  return incoming;
}

/** Map a public state snapshot to UI status + Chinese label. */
export function presentAgentPhase(state: PublicAgentState): PresentedAgentPhase {
  if (state.phase === "complete" && state.outcome?.kind === "stopped") {
    return { status: "stopped", label: "已停止" };
  }
  if (state.phase === "error") {
    return {
      status: "error",
      label: state.error?.message?.trim() || "出错",
    };
  }
  if (state.phase === "complete") {
    return { status: "done", label: "已完成" };
  }
  return { status: "running", label: PHASE_LABELS[state.phase] };
}

/** Narrow an SSE/history payload to the typed public subset; return null if unusable. */
export function asPublicAgentState(
  data: Record<string, unknown>,
): PublicAgentState | null {
  const runId = data.run_id;
  const revision = data.revision;
  const phase = data.phase;
  const turn = data.turn;
  if (typeof runId !== "string" || !runId) return null;
  if (typeof revision !== "number" || !Number.isFinite(revision)) return null;
  if (typeof phase !== "string" || !isAgentPhase(phase)) return null;
  if (typeof turn !== "number" || !Number.isFinite(turn)) return null;

  const callsRaw = Array.isArray(data.calls) ? data.calls : [];
  const calls = callsRaw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
    .map((item) => ({
      call_id: String(item.call_id ?? ""),
      name: String(item.name ?? ""),
      status: String(item.status ?? ""),
    }));

  let outcome: PublicAgentState["outcome"];
  if (data.outcome === null) {
    outcome = null;
  } else if (data.outcome && typeof data.outcome === "object") {
    const raw = data.outcome as Record<string, unknown>;
    outcome = {
      kind: String(raw.kind ?? ""),
      reason: raw.reason === null || raw.reason === undefined ? null : String(raw.reason),
      resumable: raw.resumable === true,
    };
  }

  let error: PublicAgentState["error"];
  if (data.error === null) {
    error = null;
  } else if (data.error && typeof data.error === "object") {
    const raw = data.error as Record<string, unknown>;
    error = {
      kind: String(raw.kind ?? ""),
      message: String(raw.message ?? ""),
    };
  }

  return {
    run_id: runId,
    revision,
    phase,
    turn,
    calls,
    ...(outcome !== undefined ? { outcome } : {}),
    ...(error !== undefined ? { error } : {}),
  };
}

function isAgentPhase(value: string): value is AgentPhase {
  return (
    value === "hydrate" ||
    value === "thinking" ||
    value === "tooluse" ||
    value === "awaitingconfirmation" ||
    value === "compact" ||
    value === "complete" ||
    value === "error"
  );
}

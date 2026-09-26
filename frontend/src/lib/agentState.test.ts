import { describe, expect, it } from "vitest";
import {
  agentPhaseLabel,
  buildAgentStateTimeline,
  presentAgentPhase,
  reduceAgentState,
  resolveHighLevelTraceStatus,
  type AgentPhase,
  type AgentStateStep,
  type PublicAgentState,
} from "./agentState";

function state(
  overrides: Partial<PublicAgentState> & { phase?: AgentPhase },
): PublicAgentState {
  return {
    run_id: "r1",
    revision: 1,
    phase: "thinking",
    turn: 0,
    calls: [],
    ...overrides,
  };
}

describe("reduceAgentState", () => {
  it("keeps the newest state revision", () => {
    const older = state({ run_id: "r1", revision: 2, phase: "thinking" });
    const newer = state({ run_id: "r1", revision: 3, phase: "tooluse" });
    expect(reduceAgentState(older, newer)).toEqual(newer);
    expect(reduceAgentState(newer, older)).toEqual(newer);
  });

  it("accepts the first state when current is null", () => {
    const incoming = state({ revision: 1, phase: "hydrate" });
    expect(reduceAgentState(null, incoming)).toEqual(incoming);
  });

  it("ignores an event from a different run", () => {
    const current = state({ run_id: "r1", revision: 2, phase: "thinking" });
    const other = state({ run_id: "r2", revision: 9, phase: "tooluse" });
    expect(reduceAgentState(current, other)).toEqual(current);
  });
});

describe("presentAgentPhase", () => {
  it("maps stopped complete state without treating it as an error", () => {
    const value = state({
      phase: "complete",
      outcome: { kind: "stopped", reason: "user_stopped", resumable: true },
    });
    expect(presentAgentPhase(value)).toEqual({ status: "stopped", label: "已停止" });
  });

  it("gives distinct non-error labels for in-progress phases", () => {
    expect(presentAgentPhase(state({ phase: "hydrate" })).status).toBe("running");
    expect(presentAgentPhase(state({ phase: "thinking" })).status).toBe("running");
    expect(presentAgentPhase(state({ phase: "tooluse" })).status).toBe("running");
    expect(presentAgentPhase(state({ phase: "awaitingconfirmation" })).status).toBe(
      "running",
    );
    expect(presentAgentPhase(state({ phase: "compact" })).status).toBe("running");

    const labels = [
      presentAgentPhase(state({ phase: "hydrate" })).label,
      presentAgentPhase(state({ phase: "thinking" })).label,
      presentAgentPhase(state({ phase: "tooluse" })).label,
      presentAgentPhase(state({ phase: "awaitingconfirmation" })).label,
      presentAgentPhase(state({ phase: "compact" })).label,
    ];
    expect(new Set(labels).size).toBe(labels.length);
    expect(labels.every((label) => label !== "出错")).toBe(true);
  });

  it("keeps error phase as error", () => {
    expect(
      presentAgentPhase(
        state({
          phase: "error",
          error: { kind: "provider", message: "模型调用失败" },
        }),
      ),
    ).toEqual({ status: "error", label: "模型调用失败" });
  });
});

describe("resolveHighLevelTraceStatus", () => {
  it("prefers state-derived status when done disagrees (live path)", () => {
    // Mirrors stored-turn: done says user_stopped, last state says error → error.
    const agentState = state({
      revision: 2,
      phase: "error",
      error: { kind: "provider", message: "模型调用失败" },
    });
    expect(resolveHighLevelTraceStatus(agentState, "stopped")).toBe("error");
  });

  it("falls back to done status when no state events exist (legacy servers)", () => {
    expect(resolveHighLevelTraceStatus(null, "stopped")).toBe("stopped");
    expect(resolveHighLevelTraceStatus(null, "done")).toBe("done");
    expect(resolveHighLevelTraceStatus(null, "error")).toBe("error");
  });

  it("keeps state preference when a later state revises the phase after done", () => {
    const before = state({
      revision: 1,
      phase: "complete",
      outcome: { kind: "stopped", reason: "user_stopped", resumable: true },
    });
    expect(resolveHighLevelTraceStatus(before, "stopped")).toBe("stopped");

    const afterDone = reduceAgentState(
      before,
      state({
        revision: 2,
        phase: "error",
        error: { kind: "provider", message: "late failure" },
      }),
    );
    expect(resolveHighLevelTraceStatus(afterDone, "stopped")).toBe("error");
  });
});

describe("buildAgentStateTimeline", () => {
  const step = (
    revision: number,
    phase: string,
    turn: number | null,
    created_at: string,
  ): AgentStateStep => ({ revision, phase, turn, created_at });

  it("orders by revision and derives each step's duration from the next", () => {
    const timeline = buildAgentStateTimeline([
      step(3, "complete", 1, "2026-09-25T00:00:02.000Z"),
      step(1, "hydrate", 0, "2026-09-25T00:00:00.000Z"),
      step(2, "thinking", 0, "2026-09-25T00:00:00.500Z"),
    ]);
    expect(timeline.map((e) => e.phase)).toEqual(["hydrate", "thinking", "complete"]);
    expect(timeline.map((e) => e.duration_ms)).toEqual([500, 1500, null]);
    expect(timeline[1].turn).toBe(0);
  });

  it("labels phases in Chinese including terminal ones", () => {
    expect(agentPhaseLabel("thinking")).toBe("思考中");
    expect(agentPhaseLabel("complete")).toBe("已完成");
    expect(agentPhaseLabel("error")).toBe("出错");
    expect(agentPhaseLabel("custom")).toBe("custom");
  });

  it("yields null duration for an unparseable timestamp instead of NaN", () => {
    const timeline = buildAgentStateTimeline([
      step(1, "hydrate", 0, "not-a-date"),
      step(2, "complete", 0, "2026-09-25T00:00:01.000Z"),
    ]);
    expect(timeline[0].duration_ms).toBeNull();
  });

  it("returns an empty timeline for no states", () => {
    expect(buildAgentStateTimeline([])).toEqual([]);
  });
});

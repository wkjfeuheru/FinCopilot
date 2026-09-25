import { describe, expect, it } from "vitest";
import {
  presentAgentPhase,
  reduceAgentState,
  type AgentPhase,
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

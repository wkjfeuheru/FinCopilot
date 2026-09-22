import { describe, expect, it } from "vitest";
import { detailToTrace } from "./MonitorView";
import type { TraceRunDetail } from "../api/trace";

function baseDetail(overrides: Partial<TraceRunDetail> = {}): TraceRunDetail {
  return {
    run_id: "tr_1",
    source: "server",
    user_id: "u",
    session_id: null,
    conversation_id: null,
    eval_case_id: null,
    input: "茅台股价",
    answer: "1700 元",
    status: "done",
    reason: "done",
    succeeded: 1,
    rounds: 1,
    tool_calls: 1,
    retry_count: 0,
    input_tokens: 10,
    output_tokens: 5,
    cache_hit_tokens: 0,
    per_agent: null,
    citations: ["cid_1"],
    plan: null,
    started_at: "2026-09-19T00:00:00.000Z",
    finished_at: "2026-09-19T00:00:01.000Z",
    duration_ms: 1200,
    rounds_trace: [],
    events: [],
    ...overrides,
  };
}

describe("detailToTrace", () => {
  it("renders each round in order: thought, action, observation", () => {
    const trace = detailToTrace(
      baseDetail({
        rounds_trace: [
          {
            turn: 1,
            thought: "需要先取价",
            actions: [{ call_id: "c1", name: "get_quote", args: "symbol=600519" }],
            observations: [
              { call_id: "c1", name: "get_quote", ok: true, error: null, preview: "1700.5", duration_ms: 120 },
            ],
            input_tokens: 10,
            output_tokens: 5,
            llm_first_ms: 200,
            llm_ms: 900,
            answer: "",
          },
        ],
      }),
    );
    const labels = trace.steps.map((s) => s.label);
    expect(labels).toEqual(["第 1 轮 思考", "get_quote", "get_quote 返回"]);
    expect(trace.steps[0].detail).toBe("需要先取价");
    expect(trace.steps[1].detail).toBe("symbol=600519");
    expect(trace.steps[2].status).toBe("done");
  });

  it("marks failed observations as errors with the error text", () => {
    const trace = detailToTrace(
      baseDetail({
        rounds_trace: [
          {
            turn: 1,
            thought: "",
            actions: [{ call_id: "c1", name: "read_file", args: "path=x" }],
            observations: [
              { call_id: "c1", name: "read_file", ok: false, error: "文件不存在", preview: "", duration_ms: 5 },
            ],
            input_tokens: 1,
            output_tokens: 1,
            llm_first_ms: 0,
            llm_ms: 0,
            answer: "",
          },
        ],
      }),
    );
    const obs = trace.steps.find((s) => s.label === "read_file 返回");
    expect(obs?.status).toBe("error");
    expect(obs?.detail).toBe("文件不存在");
  });

  it("surfaces loop_guard and plan drift as highlighted drift points", () => {
    const trace = detailToTrace(
      baseDetail({
        events: [
          { seq: 1, turn: 3, kind: "tool_status", payload: { name: "get_quote", status: "started" } },
          { seq: 2, turn: 4, kind: "loop_guard", payload: { name: "get_quote", count: 3 } },
          { seq: 3, turn: 5, kind: "plan_progress", payload: { drift: ["000858"] } },
        ],
      }),
    );
    const driftSteps = trace.steps.filter((s) => s.label.includes("跑偏点"));
    expect(driftSteps).toHaveLength(2);
    expect(driftSteps.every((s) => s.status === "error")).toBe(true);
    expect(driftSteps[0].label).toContain("第 4 轮");
    // 普通 tool_status 事件不应成为跑偏点节点。
    expect(trace.steps.map((s) => s.label)).not.toContain("跑偏点：工具调用");
  });

  it("maps run status to trace status including stopped", () => {
    expect(detailToTrace(baseDetail({ status: "stopped" })).status).toBe("stopped");
    expect(detailToTrace(baseDetail({ status: "error" })).status).toBe("error");
    expect(detailToTrace(baseDetail({ status: "done" })).status).toBe("done");
  });
});

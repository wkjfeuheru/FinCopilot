import { describe, expect, it } from "vitest";
import { traceFromStoredTurn } from "./AgentTrace";
import type { StoredTurn } from "../api/client";

function turn(events: StoredTurn["events"]): StoredTurn {
  return { events, first_token_ms: null, total_duration_ms: 0 };
}

describe("traceFromStoredTurn", () => {
  it("keeps the engine's duplicate-call guard out of the visible trace", () => {
    // 去重是引擎自愈：用户侧没有可感知后果，把内部机制渲染成执行节点只会
    // 造成困惑。这条用例锁住它不被再次引入。
    const trace = traceFromStoredTurn(
      turn([
        {
          event: "loop_guard",
          data: { call_id: "call_1", name: "web_search", count: 3, action: "refused" },
        },
        {
          event: "loop_guard",
          data: { call_id: "call_2", name: "web_search", count: 4, action: "would_abort" },
        },
      ]),
    );

    const labels = trace.steps.map((step) => step.label);
    expect(labels).not.toContain("跳过重复调用");
    expect(labels).not.toContain("终止重复调用");
    expect(trace.steps.some((step) => step.key.startsWith("guard-"))).toBe(false);
  });

  it("still surfaces a duplicate-call abort as a failed turn", () => {
    // 中止本轮是用户可感知的：它必须仍能从终态 done 事件还原为失败状态，
    // 否则移除 loop_guard 步骤会让这类轮次看起来无声无息地结束了。
    // （历史只持久化 loop_guard 与 done，不含 error 事件。）
    const trace = traceFromStoredTurn(
      turn([
        { event: "loop_guard", data: { action: "would_abort", name: "web_search" } },
        { event: "done", data: { succeeded: false, reason: "loop_detected" } },
      ]),
    );

    expect(trace.status).toBe("error");
  });

  it("restores a user-stopped turn as stopped, not error", () => {
    // 用户主动停止不是失败：刷新后重放这条历史时，状态必须是 stopped，
    // 否则一次"我不想等了"会被渲染成"系统出错了"。
    const trace = traceFromStoredTurn(
      turn([
        {
          event: "tool_status",
          data: { call_id: "call_1", name: "get_quotes", status: "started" },
        },
        {
          event: "done",
          data: { succeeded: false, reason: "user_stopped", resumable: true },
        },
      ]),
    );

    expect(trace.status).toBe("stopped");
    // 仍在运行中的步骤（那个还没返回的工具）也要收敛为 stopped 而非 error。
    const pending = trace.steps.find((step) => step.key === "call_1");
    expect(pending?.status).toBe("stopped");
  });

  it("keeps a genuine failure as error", () => {
    // 对照：非停止原因的不成功仍应是 error，避免上面那条改写过度。
    const trace = traceFromStoredTurn(
      turn([
        { event: "done", data: { succeeded: false, reason: "provider_error" } },
      ]),
    );

    expect(trace.status).toBe("error");
  });

  it("derives high-level phase from the last state event while keeping done metrics", () => {
    // done 仍提供指标；高阶 phase 以最后一条 state 为准（此处覆盖 done 的 stopped）。
    const trace = traceFromStoredTurn(
      turn([
        {
          event: "tool_status",
          data: { call_id: "call_1", name: "get_quotes", status: "started" },
        },
        {
          event: "done",
          data: {
            succeeded: false,
            reason: "user_stopped",
            usage: { input_tokens: 10, output_tokens: 5 },
            tool_calls: 1,
            tool_duration_ms: 100,
          },
        },
        {
          event: "state",
          data: {
            run_id: "r1",
            revision: 2,
            phase: "error",
            turn: 1,
            calls: [],
            error: { kind: "provider", message: "模型调用失败" },
          },
        },
      ]),
    );

    expect(trace.status).toBe("error");
    expect(trace.metrics?.totalTokens).toBe(15);
    expect(trace.metrics?.toolDurationMs).toBe(100);
    const pending = trace.steps.find((step) => step.key === "call_1");
    expect(pending?.status).toBe("stopped");
  });
});


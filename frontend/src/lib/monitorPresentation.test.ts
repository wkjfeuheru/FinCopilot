import { describe, expect, it } from "vitest";
import {
  formatAverage,
  formatCount,
  formatDuration,
  formatRate,
  isDriftEvent,
  presentReason,
  presentStatus,
} from "./monitorPresentation";

describe("monitorPresentation", () => {
  it("formats rates with one decimal and renders missing values as a dash", () => {
    expect(formatRate(0.1234)).toBe("12.3%");
    expect(formatRate(1)).toBe("100.0%");
    expect(formatRate(null)).toBe("—");
    expect(formatRate(undefined)).toBe("—");
  });

  it("formats counts and averages defensively", () => {
    expect(formatCount(0)).toBe("0");
    expect(formatCount(null)).toBe("—");
    expect(formatAverage(3.456)).toBe("3.46");
    expect(formatAverage(null)).toBe("—");
  });

  it("formats durations across unit boundaries", () => {
    expect(formatDuration(500)).toBe("500 ms");
    expect(formatDuration(1500)).toBe("1.5 s");
    expect(formatDuration(65000)).toBe("1m 5s");
    expect(formatDuration(null)).toBe("—");
  });

  it("maps run status to readable labels with tones", () => {
    expect(presentStatus("done")).toEqual({ label: "已完成", tone: "ok" });
    expect(presentStatus("stopped")).toEqual({ label: "用户停止", tone: "info" });
    expect(presentStatus("error")).toEqual({ label: "失败", tone: "error" });
  });

  it("maps stop reasons to Chinese labels and passes unknowns through", () => {
    expect(presentReason("max_turns_exhausted")).toBe("轮次耗尽");
    expect(presentReason("loop_detected")).toBe("重复调用中止");
    expect(presentReason(null)).toBe("—");
    expect(presentReason("custom_reason")).toBe("custom_reason");
  });

  it("flags loop guard and plan drift as drift points", () => {
    expect(isDriftEvent("loop_guard", { name: "web_search" })).toBe(true);
    expect(isDriftEvent("plan_progress", { drift: ["600519"] })).toBe(true);
    expect(isDriftEvent("plan_progress", { mismatch: ["valuation"] })).toBe(true);
    expect(isDriftEvent("plan_progress", { stalled_turns: 3 })).toBe(true);
    // 正常计划进展不是跑偏点。
    expect(isDriftEvent("plan_progress", { drift: [], mismatch: [], stalled_turns: 0 })).toBe(false);
    expect(isDriftEvent("tool_status", { name: "get_quote" })).toBe(false);
  });
});

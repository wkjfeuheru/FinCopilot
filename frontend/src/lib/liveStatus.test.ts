import { describe, expect, it } from "vitest";
import { groupActivitiesByTool, reduceLiveRows } from "./liveStatus";
import type { LiveStep } from "./liveStatus";

function step(partial: Partial<LiveStep> & Pick<LiveStep, "key">): LiveStep {
  return {
    kind: "tool",
    label: "读取行业表现",
    status: "running",
    ...partial,
  };
}

describe("reduceLiveRows", () => {
  it("collapses fifteen parallel industry reads into one overlay row", () => {
    const steps: LiveStep[] = Array.from({ length: 15 }, (_, index) =>
      step({
        key: `call_${index}`,
        toolName: "get_industry_perf",
        summary: { industry: ["白酒", "电子", "煤炭", "银行", "钢铁"][index] ?? `行业${index}` },
      }),
    );
    const rows = reduceLiveRows(steps, "running");
    expect(rows).toHaveLength(1);
    expect(rows[0].label).toBe("正在读取白酒、电子、煤炭等 15 个行业表现");
    expect(rows[0].toolName).toBe("get_industry_perf");
  });

  it("shows at most three spawn tasks and notes the remainder", () => {
    const tasks = ["甲公司盈利", "乙公司估值", "丙公司现金流", "丁公司商誉", "戊公司分红"];
    const rows = reduceLiveRows(
      [
        step({
          key: "spawn_1",
          toolName: "spawn_agent",
          kind: "tool",
          summary: { tasks },
          spawnTasks: tasks.map((task, index) => ({ index, task, status: "running" as const })),
        }),
      ],
      "running",
    );
    expect(rows).toHaveLength(3);
    expect(rows[0].label).toBe("正在研究：甲公司盈利");
    expect(rows[2].detail).toBe("另有 2 个");
    expect(rows.every((row) => row.kind === "agent")).toBe(true);
  });

  it("summarizes a finished turn instead of listing every node", () => {
    const steps: LiveStep[] = [
      step({ key: "analysis", kind: "analysis", label: "理解问题并确定研究路径", status: "done" }),
      ...Array.from({ length: 12 }, (_, index) =>
        step({
          key: `call_${index}`,
          toolName: "get_industry_perf",
          status: "done",
          summary: { industry: "白酒" },
        }),
      ),
    ];
    const rows = reduceLiveRows(steps, "done");
    expect(rows).toHaveLength(1);
    expect(rows[0].label).toBe("研究完成 · 12 项调用");
  });
});

describe("groupActivitiesByTool", () => {
  it("folds same-tool calls into one expandable group", () => {
    const groups = groupActivitiesByTool([
      { key: "a", tool: "get_industry_perf", label: "正在读取白酒行业表现", status: "done" },
      { key: "b", tool: "get_industry_perf", label: "正在读取煤炭行业表现", status: "done" },
      { key: "c", tool: "get_quote", label: "正在读取 600519 最新行情", status: "running" },
    ]);
    expect(groups).toHaveLength(2);
    expect(groups[0].title).toBe("读取行业表现 · 2 次");
    expect(groups[0].count).toBe(2);
    expect(groups[0].children).toHaveLength(2);
    expect(groups[1].status).toBe("running");
  });
});

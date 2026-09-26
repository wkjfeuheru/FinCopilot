import { describe, expect, it } from "vitest";
import {
  planStatusMeta,
  presentLiveAction,
  presentSkillAction,
  presentToolAction,
  presentToolProgress,
} from "./researchPresentation";

describe("presentToolAction", () => {
  it("uses user-facing Chinese labels instead of internal tool names", () => {
    expect(presentToolAction("get_financials")).toBe("读取财务报表");
    expect(presentToolAction("write_report")).toBe("生成研究报告并复核");
  });

  it("names the report-reading tools so a failure points at a real step", () => {
    // 这两个工具此前缺失，界面回落到兜底的"执行研究步骤"，失败时看不出
    // 是哪一步出了问题。
    expect(presentToolAction("read_pdf")).toBe("精读研报正文");
    expect(presentToolAction("summarize_document")).toBe("生成文档摘要");
  });

  it("keeps unknown internal names out of the user interface", () => {
    expect(presentToolAction("experimental_private_tool")).toBe("执行研究步骤");
  });
});

describe("presentLiveAction", () => {
  it("names a running industry read with the sector", () => {
    expect(presentLiveAction("get_industry_perf", { industry: "白酒" }, "running")).toBe(
      "正在读取白酒行业表现",
    );
  });

  it("falls back to the generic phrase when history has no summary", () => {
    expect(presentLiveAction("get_industry_perf", undefined, "running")).toBe("正在读取行业表现");
    expect(presentLiveAction("get_industry_perf", undefined, "done")).toBe("已读取行业表现");
  });

  it("joins several industries and counts the rest", () => {
    expect(
      presentLiveAction(
        "get_industry_perf",
        [
          { industry: "白酒" },
          { industry: "电子" },
          { industry: "煤炭" },
          { industry: "银行" },
        ],
        "running",
      ),
    ).toBe("正在读取白酒、电子、煤炭等 4 个行业表现");
  });

  it("names a quote by symbol", () => {
    expect(presentLiveAction("get_quote", { symbol: "600519" }, "running")).toBe(
      "正在读取 600519 最新行情",
    );
  });
});

describe("presentSkillAction", () => {
  it("renders a Chinese scene name instead of the skill key", () => {
    expect(presentSkillAction(["industry-research"], "running")).toBe("正在加载行业研究技能");
    expect(presentSkillAction(["industry-research/prosperity"], "done")).toBe("已加载行业研究技能");
  });
});

describe("presentToolProgress", () => {
  it("renders spawn task progress as a research line", () => {
    expect(
      presentToolProgress({
        phase: "spawn",
        index: 0,
        total: 2,
        task: "分析 600519 盈利质量",
        status: "started",
      }),
    ).toBe("正在研究：分析 600519 盈利质量");
  });
});

describe("planStatusMeta", () => {
  it("maps every persisted plan state to a visible Chinese status", () => {
    expect(planStatusMeta("pending")).toMatchObject({ label: "待处理", tone: "pending" });
    expect(planStatusMeta("done")).toMatchObject({ label: "已完成", tone: "done" });
    expect(planStatusMeta("fail")).toMatchObject({ label: "未完成", tone: "fail" });
    expect(planStatusMeta("skipped")).toMatchObject({ label: "已跳过", tone: "skipped" });
  });
});

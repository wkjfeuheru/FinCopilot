import { describe, expect, it } from "vitest";
import { planStatusMeta, presentToolAction } from "./researchPresentation";

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

describe("planStatusMeta", () => {
  it("maps every persisted plan state to a visible Chinese status", () => {
    expect(planStatusMeta("pending")).toMatchObject({ label: "待处理", tone: "pending" });
    expect(planStatusMeta("done")).toMatchObject({ label: "已完成", tone: "done" });
    expect(planStatusMeta("fail")).toMatchObject({ label: "未完成", tone: "fail" });
    expect(planStatusMeta("skipped")).toMatchObject({ label: "已跳过", tone: "skipped" });
  });
});

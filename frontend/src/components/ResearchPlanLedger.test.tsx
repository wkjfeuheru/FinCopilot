import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ResearchPlanLedger } from "./ResearchPlanLedger";

describe("ResearchPlanLedger", () => {
  it("renders a full plan with explicit Chinese statuses and dependencies", () => {
    const html = renderToStaticMarkup(
      <ResearchPlanLedger
        plan={{
          plan_id: "plan_001",
          goal: "比较两家公司估值",
          revision: 2,
          done: 1,
          total: 3,
          steps: [
            { seq: 1, action: "读取估值数据", status: "done", dep: [] },
            { seq: 2, action: "完成同业比较", status: "pending", dep: [1] },
            { seq: 3, action: "形成条件化结论", status: "skipped", dep: [2] },
          ],
        }}
      />,
    );

    expect(html).toContain("比较两家公司估值");
    expect(html).toContain("第 2 版");
    expect(html).toContain("已完成");
    expect(html).toContain("待处理");
    expect(html).toContain("已跳过");
    expect(html).toContain("依赖步骤 1");
  });
});

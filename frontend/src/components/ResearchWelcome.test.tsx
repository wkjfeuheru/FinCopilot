import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ResearchWelcome } from "./ResearchWelcome";

describe("ResearchWelcome", () => {
  it("presents the four implemented research scenarios and the evidence workflow", () => {
    const html = renderToStaticMarkup(<ResearchWelcome onUsePrompt={() => undefined} />);

    expect(html).toContain("个股研究");
    expect(html).toContain("行业研究");
    expect(html).toContain("宏观研究");
    expect(html).toContain("量化因子");
    expect(html).toContain("问题");
    expect(html).toContain("复核");
    expect(html).toContain("不构成投资建议");
  });
});

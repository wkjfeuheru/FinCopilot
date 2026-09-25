import { describe, expect, it } from "vitest";

import { confirmLabel } from "./InteractionPrompt";

describe("confirmLabel", () => {
  it("labels the one-time allow and the deny options", () => {
    expect(confirmLabel("y")).toBe("允许");
    expect(confirmLabel("n")).toBe("拒绝");
  });

  it("labels the conversation-scoped egress remember option", () => {
    expect(confirmLabel("y_remember")).toBe("允许并本对话不再询问");
  });

  it("labels the session-scoped always-allow option with an explicit scope", () => {
    // 会话级授权的新会话即失效，文案必须点明"本会话内"，
    // 不能写成"不再询问"而让用户以为永久生效。
    expect(confirmLabel("y_session")).toBe("始终允许此类操作（本会话内）");
  });
});

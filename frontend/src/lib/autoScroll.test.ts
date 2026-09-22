import { describe, expect, it } from "vitest";
import { isNearBottom, SCROLL_FOLLOW_THRESHOLD_PX } from "./autoScroll";

describe("isNearBottom", () => {
  it("treats the exact bottom as following", () => {
    expect(isNearBottom(500, 300, 800)).toBe(true);
  });

  it("still follows within the threshold, but not one pixel beyond", () => {
    // scrollHeight - (scrollTop + clientHeight) === 80：恰好等于阈值仍跟随；
    // 81 就算"用户上翻离开"，必须停止跟随，否则用户往上读会被拉回底部。
    const atThreshold = 800 - 80;
    expect(isNearBottom(atThreshold, 300, 1100)).toBe(true);
    expect(isNearBottom(atThreshold - 1, 300, 1100)).toBe(false);
  });

  it("stops following when the user scrolled far up", () => {
    expect(isNearBottom(0, 300, 4000)).toBe(false);
  });

  it("uses the shared threshold constant so UI and tests stay in sync", () => {
    expect(SCROLL_FOLLOW_THRESHOLD_PX).toBe(80);
  });
});

import { describe, expect, it } from "vitest";
import { enqueue, resolve, type Interaction } from "./interactionQueue";

function item(requestId: string): Interaction {
  return {
    requestId,
    kind: "confirm",
    prompt: `run ${requestId}?`,
    options: ["y", "n"],
    multiSelect: false,
  };
}

describe("enqueue", () => {
  it("keeps every concurrent prompt instead of overwriting earlier ones", () => {
    // 单值覆盖正是那个 bug：三条并发提示只留最后一条，其余 request_id
    // 消失后再也无法被应答。
    let queue: Interaction[] = [];
    queue = enqueue(queue, item("req_a"));
    queue = enqueue(queue, item("req_b"));
    queue = enqueue(queue, item("req_c"));

    expect(queue.map((entry) => entry.requestId)).toEqual(["req_a", "req_b", "req_c"]);
  });

  it("ignores a duplicate id so a merged request is shown once", () => {
    let queue = enqueue([], item("req_a"));
    queue = enqueue(queue, item("req_a"));

    expect(queue).toHaveLength(1);
  });

  it("ignores an entry without a request id", () => {
    expect(enqueue([], item(""))).toEqual([]);
  });
});

describe("resolve", () => {
  it("removes only the answered prompt and leaves the rest queued", () => {
    const queue = [item("req_a"), item("req_b")];

    expect(resolve(queue, "req_a").map((entry) => entry.requestId)).toEqual(["req_b"]);
  });

  it("is a no-op for an id that is not queued", () => {
    const queue = [item("req_a")];

    expect(resolve(queue, "req_z")).toEqual(queue);
  });

  it("clears the whole queue when the turn ends without a specific id", () => {
    expect(resolve([item("req_a"), item("req_b")], null)).toEqual([]);
  });
});

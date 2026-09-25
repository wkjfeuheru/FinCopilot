import { afterEach, describe, expect, it, vi } from "vitest";
import { readEventStream, stopChat, streamChat } from "./client";
import type { ChatEvent } from "./client";

/** 把若干字节片段拼成一个可读的 SSE 响应体。 */
function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

function collect(): { events: ChatEvent[]; onEvent: (event: ChatEvent) => void } {
  const events: ChatEvent[] = [];
  return { events, onEvent: (event) => events.push(event) };
}

describe("readEventStream", () => {
  it("dispatches every frame and resolves once a terminal event arrives", async () => {
    const { events, onEvent } = collect();

    await readEventStream(
      streamOf(
        'event: session\ndata: {"session_id": "s_1"}\n\n',
        'event: delta\ndata: {"text": "hi"}\n\n',
        'event: done\ndata: {"succeeded": true}\n\n',
      ),
      onEvent,
    );

    expect(events.map((event) => event.event)).toEqual(["session", "delta", "done"]);
    expect(events[0].data.session_id).toBe("s_1");
  });

  it("does not lose a terminal frame that arrives without a trailing blank line", async () => {
    // 服务端发完终止帧即结束连接时，最后一帧可能没有结尾空行。旧实现在
    // 循环里把残留 buffer 丢弃，界面因此停在"运行中"。
    const { events, onEvent } = collect();

    await readEventStream(
      streamOf('event: delta\ndata: {"text": "hi"}\n\n', 'event: done\ndata: {"succeeded": true}'),
      onEvent,
    );

    expect(events.map((event) => event.event)).toEqual(["delta", "done"]);
  });

  it("rejects when the stream ends without any terminal event", async () => {
    // 连接被代理截断/静默关闭：必须抛错，让上层把界面收敛为失败，
    // 而不是永远显示"运行中"。
    const { onEvent } = collect();

    await expect(
      readEventStream(streamOf('event: delta\ndata: {"text": "partial"}\n\n'), onEvent),
    ).rejects.toThrow(/完成事件/);
  });

  it("ignores heartbeat comment frames", async () => {
    const { events, onEvent } = collect();

    await readEventStream(
      streamOf(
        ": keep-alive\n\n",
        'event: delta\ndata: {"text": "hi"}\n\n',
        ": keep-alive\n\n",
        'event: done\ndata: {"succeeded": true}\n\n',
      ),
      onEvent,
    );

    expect(events.map((event) => event.event)).toEqual(["delta", "done"]);
  });

  it("survives a frame split across two network chunks", async () => {
    const { events, onEvent } = collect();

    await readEventStream(
      streamOf(
        'event: del',
        'ta\ndata: {"text": "hi"}\n\nevent: done\ndata: {"succeeded": true}\n\n',
      ),
      onEvent,
    );

    expect(events).toEqual([
      { event: "delta", data: { text: "hi" } },
      { event: "done", data: { succeeded: true } },
    ]);
  });

  it("treats an error event as terminal", async () => {
    const { events, onEvent } = collect();

    await readEventStream(
      streamOf('event: error\ndata: {"reason": "provider_error"}\n\n'),
      onEvent,
    );

    expect(events.map((event) => event.event)).toEqual(["error"]);
  });

  it("does not surface a truncated trailing frame as a JSON error", async () => {
    // 半截帧（连接在帧中途断开）不应把 JSON 语法错误当成用户提示，
    // 而应由缺少终止事件这一判断统一处理。
    const { onEvent } = collect();

    await expect(
      readEventStream(
        streamOf('event: done\ndata: {"succeeded": tru'),
        onEvent,
      ),
    ).rejects.toThrow(/完成事件/);
  });
});

describe("stopChat", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stubFetch(response: Partial<Response> & { jsonText?: string }): ReturnType<typeof vi.fn> {
    const fetchMock = vi.fn(async () => ({
      ok: response.ok ?? true,
      status: response.status ?? 200,
      text: async () => response.jsonText ?? "",
      json: async () => JSON.parse(response.jsonText ?? "{}"),
    })) as unknown as ReturnType<typeof vi.fn>;
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  it("posts both ids so the server can address the in-flight session", async () => {
    const fetchMock = stubFetch({ jsonText: '{"stopping": true}' });

    const stopping = await stopChat("c_1", "s_1");

    expect(stopping).toBe(true);
    const [path, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe("/v1/chat/stop");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({
      conversation_id: "c_1",
      session_id: "s_1",
    });
  });

  it("reports false when the server has nothing in flight", async () => {
    stubFetch({ jsonText: '{"stopping": false}' });

    await expect(stopChat("c_1", null)).resolves.toBe(false);
  });

  it("treats 404 as already-stopped rather than an error", async () => {
    // 会话不可寻址（多半已自己结束）时照样算停止成功：对调用方而言，
    // "已经停了"与"本来就没事在跑"同义，不该弹错误提示。
    stubFetch({ ok: false, status: 404, jsonText: "会话或对话不存在" });

    await expect(stopChat("c_gone", null)).resolves.toBe(false);
  });

  it("throws on other failures so the caller can fall back to aborting", async () => {
    stubFetch({ ok: false, status: 500, jsonText: "boom" });

    await expect(stopChat("c_1", null)).rejects.toThrow("boom");
  });
});

describe("streamChat", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stubStreamFetch(): ReturnType<typeof vi.fn> {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('event: done\ndata: {"succeeded": true}\n\n'));
        controller.close();
      },
    });
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      body,
      text: async () => "",
    })) as unknown as ReturnType<typeof vi.fn>;
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  it("serializes resume=true in the request body", async () => {
    const fetchMock = stubStreamFetch();

    await streamChat("", "c_1", () => undefined, new AbortController().signal, true);

    const [path, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe("/v1/chat/stream");
    expect(JSON.parse(String(init.body))).toEqual({
      message: "",
      conversation_id: "c_1",
      resume: true,
    });
  });

  it("always includes resume=false for a normal send", async () => {
    const fetchMock = stubStreamFetch();

    await streamChat("hello", "c_1", () => undefined, new AbortController().signal);

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({
      message: "hello",
      conversation_id: "c_1",
      resume: false,
    });
  });
});


import { authedFetch } from "./http";

export type ChatEvent = { event: string; data: Record<string, unknown> };

export type ConversationSummary = {
  conversation_id: string;
  title: string | null;
  created_at: string;
  last_active_at: string;
};

export type StoredTurn = {
  events: ChatEvent[];
  first_token_ms: number | null;
  total_duration_ms: number;
};

/** 一条还原的可读轮次，及其可选的可观测执行记录。 */
export type HistoryMessage = {
  role: "user" | "assistant";
  text: string;
  turn?: StoredTurn;
};

/**
 * 上一轮被用户停止后留下的断点（服务端 `resumable`）。
 *
 * 它的存在意味着服务端仍持有可续做的研究计划与已落库的部分结论，因此
 * 刷新页面后界面依然要显示"继续研究"，而不是把那次中断当作一次普通失败。
 */
export type ResumableTurn = {
  reason: string;
  rounds: number;
  /** Legacy checkpoint plan; FSM history may omit this. */
  plan?: Record<string, unknown> | null;
  /** FSM public_state_view when a snapshot is resumable. */
  state?: Record<string, unknown> | null;
  updated_at: string;
};

export type ConversationHistory = {
  messages: HistoryMessage[];
  resumable: ResumableTurn | null;
};

/** 一条已获取数据的来源信息，展示在数据来源侧栏中。 */
export type Citation = {
  cid: string;
  tool: string;
  endpoint: string;
  symbol: string | null;
  params: Record<string, unknown>;
  rows: number;
  cols: number;
  from_cache: boolean;
  ts: string;
  fingerprint: string;
};

/** 终止性事件名：收到其中任意一个才说明本轮真的结束了。 */
const TERMINAL_EVENTS = new Set(["done", "error"]);

/**
 * 逐帧解析 SSE 字节流。
 *
 * 与直觉相反，连接正常关闭 **不等于** 本轮完成：服务端可能在中途失败、
 * 被代理截断，或者终止帧在传输中丢失。因此这里显式跟踪是否见过终止
 * 事件，未见即抛错，而不是让上层把"流断了"误当成"还在跑"——那会让界面
 * 永远停在运行中。
 *
 * 注释帧（以 ``:`` 开头的心跳）不含 ``event``/``data``，按规范忽略。
 */
export async function readEventStream(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawTerminal = false;

  const handleFrame = (frame: string) => {
    const name = frame.match(/^event: (.+)$/m)?.[1];
    const raw = frame.match(/^data: (.+)$/m)?.[1];
    if (!name || raw === undefined) return;
    let data: Record<string, unknown>;
    try {
      data = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      // 半截帧（连接在帧中途断开）：交给末尾的"缺少终止事件"判断处理，
      // 不要把 JSON 语法错误当作面向用户的提示抛出去。
      return;
    }
    if (TERMINAL_EVENTS.has(name)) sawTerminal = true;
    onEvent({ event: name, data });
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) handleFrame(frame);
    if (done) break;
  }
  // 关闭时 buffer 里可能还留着最后一帧：服务端发完终止帧即结束，若少了
  // 结尾空行，终止事件正好落在这里。不冲洗就会把它丢掉。
  if (buffer.trim()) handleFrame(buffer);

  if (!sawTerminal) {
    throw new Error("响应流在送达完成事件前结束，请重试");
  }
}

export async function streamChat(
  message: string,
  conversationId: string | null,
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
  resume = false,
): Promise<void> {
  const response = await authedFetch("/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // conversation_id 是持久化的凭据：发送它即可恢复一个
    // 执行会话已过期的对话。
    // resume=true 仅由"继续研究"发出：显式恢复可续做的 FSM/断点，
    // 普通新消息必须带 resume=false，避免隐式 recovery。
    body: JSON.stringify({ message, conversation_id: conversationId, resume }),
    signal,
  });
  if (!response.ok) {
    throw new Error((await response.text()) || `Request failed: ${response.status}`);
  }
  if (!response.body) throw new Error("Streaming response is unavailable");

  await readEventStream(response.body, onEvent);
}

/** 回答轮次中的请求（写入确认或模型提问）。 */
export async function respondChat(requestId: string, response: string): Promise<void> {
  const reply = await authedFetch("/v1/chat/respond", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: requestId, response }),
  });
  if (!reply.ok) {
    throw new Error((await reply.text()) || `Request failed: ${reply.status}`);
  }
}

/**
 * 请求服务端停止当前在飞的生成。
 *
 * 这是**协作式**停止：服务端置位中断信号，引擎在下一个等待点收尾，因此
 * 已有成果（本轮结论、数据）照常落库、断点记为可继续。它不依赖断开连接，
 * 所以停止不会让这一轮的工作白费。
 *
 * 返回 `true` 表示确实有一个在飞的生成被请求停止；`false` 表示它已经结束
 * 或本就没在运行——对调用方而言两者同义，无需区别对待。
 *
 * 这里是"软"停止。若引擎长时间不响应（例如卡在一次很长的工具调用里），
 * 调用方仍可再用 AbortController 断开连接兜底；服务端在断开路径上同样会
 * 落库并留下断点。
 */
export async function stopChat(
  conversationId: string | null,
  sessionId: string | null,
): Promise<boolean> {
  const response = await authedFetch("/v1/chat/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, session_id: sessionId }),
  });
  if (!response.ok) {
    // 404 只表示会话已不可寻址（多半是已经自己结束了）；对"停止"而言
    // 这不算失败，照常视为已停止。
    if (response.status === 404) return false;
    throw new Error((await response.text()) || `Request failed: ${response.status}`);
  }
  const body = (await response.json()) as { stopping?: boolean };
  return body.stopping === true;
}

/** 用于下载产出物（研报、图表）的 URL。 */
export function artifactUrl(path: string): string {
  return `/v1/artifacts?path=${encodeURIComponent(path)}`;
}

/** 为某个对话（优先）或当前会话获取的数据来源。 */
export async function fetchCitations(
  conversationId: string | null,
  sessionId: string | null,
): Promise<Citation[]> {
  const params = new URLSearchParams();
  if (conversationId) params.set("conversation_id", conversationId);
  else if (sessionId) params.set("session_id", sessionId);
  const query = params.toString();
  try {
    const response = await authedFetch(`/v1/citations${query ? `?${query}` : ""}`);
    // 404 只表示作用域尚不可知；此时侧栏为空是正确的。
    if (!response.ok) return [];
    const body = (await response.json()) as { citations: Citation[] };
    return body.citations;
  } catch {
    return [];
  }
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await authedFetch("/v1/conversations");
  if (!response.ok) throw new Error(`加载对话列表失败：${response.status}`);
  const body = (await response.json()) as { conversations: ConversationSummary[] };
  return body.conversations;
}

export async function loadConversationMessages(
  conversationId: string,
): Promise<ConversationHistory> {
  const response = await authedFetch(
    `/v1/conversations/${encodeURIComponent(conversationId)}/messages`,
  );
  if (response.status === 404) return { messages: [], resumable: null };
  if (!response.ok) throw new Error(`加载历史失败：${response.status}`);
  const body = (await response.json()) as {
    messages: HistoryMessage[];
    resumable?: ResumableTurn | null;
  };
  return { messages: body.messages, resumable: body.resumable ?? null };
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await authedFetch(`/v1/conversations/${encodeURIComponent(conversationId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    let detail = `删除失败：${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* 保留基于状态码的提示信息 */
    }
    throw new Error(detail);
  }
}

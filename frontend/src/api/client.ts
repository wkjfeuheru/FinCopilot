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

/** A restored readable turn plus its optional observable execution record. */
export type HistoryMessage = {
  role: "user" | "assistant";
  text: string;
  turn?: StoredTurn;
};

/** Provenance for one piece of fetched data, shown in the source sidebar. */
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

export async function streamChat(
  message: string,
  conversationId: string | null,
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // conversation_id is the durable handle: sending it resumes a conversation
    // whose execution session has already expired.
    body: JSON.stringify({ message, conversation_id: conversationId, mode: "default" }),
    signal,
  });
  if (!response.ok) {
    throw new Error((await response.text()) || `Request failed: ${response.status}`);
  }
  if (!response.body) throw new Error("Streaming response is unavailable");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const event = frame.match(/^event: (.+)$/m)?.[1];
      const data = frame.match(/^data: (.+)$/m)?.[1];
      if (event && data) onEvent({ event, data: JSON.parse(data) });
    }
    if (done) break;
  }
}

/** Answer a mid-turn request (write confirmation or a model question). */
export async function respondChat(requestId: string, response: string): Promise<void> {
  const reply = await fetch("/v1/chat/respond", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: requestId, response }),
  });
  if (!reply.ok) {
    throw new Error((await reply.text()) || `Request failed: ${reply.status}`);
  }
}

/** URL for downloading a produced artefact (report, chart). */
export function artifactUrl(path: string): string {
  return `/v1/artifacts?path=${encodeURIComponent(path)}`;
}

/** Sources fetched for a conversation (preferred) or the live session. */
export async function fetchCitations(
  conversationId: string | null,
  sessionId: string | null,
): Promise<Citation[]> {
  const params = new URLSearchParams();
  if (conversationId) params.set("conversation_id", conversationId);
  else if (sessionId) params.set("session_id", sessionId);
  const query = params.toString();
  try {
    const response = await fetch(`/v1/citations${query ? `?${query}` : ""}`);
    // A 404 just means the scope is not known yet; an empty sidebar is correct.
    if (!response.ok) return [];
    const body = (await response.json()) as { citations: Citation[] };
    return body.citations;
  } catch {
    return [];
  }
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await fetch("/v1/conversations");
  if (!response.ok) throw new Error(`加载对话列表失败：${response.status}`);
  const body = (await response.json()) as { conversations: ConversationSummary[] };
  return body.conversations;
}

export async function loadConversationMessages(
  conversationId: string,
): Promise<HistoryMessage[]> {
  const response = await fetch(
    `/v1/conversations/${encodeURIComponent(conversationId)}/messages`,
  );
  if (response.status === 404) return [];
  if (!response.ok) throw new Error(`加载历史失败：${response.status}`);
  const body = (await response.json()) as { messages: HistoryMessage[] };
  return body.messages;
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(`/v1/conversations/${encodeURIComponent(conversationId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    let detail = `删除失败：${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep the status-based message */
    }
    throw new Error(detail);
  }
}

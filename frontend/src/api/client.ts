export type ChatEvent = { event: string; data: Record<string, unknown> };

export type ConversationSummary = {
  conversation_id: string;
  title: string | null;
  created_at: string;
  last_active_at: string;
};

/** A restored message: user turns and final answers only (no tool frames). */
export type HistoryMessage = { role: "user" | "assistant"; text: string };

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

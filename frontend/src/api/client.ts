export type ChatEvent = { event: string; data: Record<string, unknown> };

export async function streamChat(
  message: string,
  sessionId: string | null,
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, mode: "default" }),
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

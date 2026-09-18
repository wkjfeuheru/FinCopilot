"""审计 hook：为每一次受治理的工具调用记录仅追加的 JSONL（docs 4.3）。

始终开启且不可移除——``settings.audit`` 选择的是路径，而不是开关。每一行都会
立即刷写，因此即使崩溃也会留下关于运行内容的完整记录。

脱敏与 ``trace_id`` 都走 ``observability`` 包：审计与结构化日志必须共享同一套
凭据遮蔽规则，也必须能被同一个 id 串联起来（docs 03.14.1）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from finharness.hooks.base import BaseHook
from finharness.observability.context import current_trace
from finharness.observability.redact import summarize_args
from finharness.types import ToolResult


class AuditLogWriter:
    """行缓冲的 JSONL 写入器；每个事件一行。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def write(self, record: dict) -> None:
        self._seq += 1
        payload = {"seq": self._seq, "ts": datetime.now().astimezone().isoformat(timespec="seconds")}
        payload.update(record)
        # 请求作用域内的审计行带上 trace_id，使其与同一次对话的日志可互相检索。
        trace = current_trace()
        if trace is not None and trace.trace_id and "trace_id" not in payload:
            payload["trace_id"] = trace.trace_id
        with open(self.path, "a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class AuditHook(BaseHook):
    """写入会话边界，以及每个工具结果一行。"""

    def __init__(self, writer: AuditLogWriter, *, session_id: str = "local", user_id: str = "") -> None:
        self.writer = writer
        self.session_id = session_id
        self.user_id = user_id

    def session_start(self, *, mode: str, provider: str, model: str) -> None:
        self.writer.write(
            {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "action": "session_start",
                "mode": mode,
                "provider": provider,
                "model": model,
            }
        )

    def session_end(self, *, total_tokens: int, tool_calls: int) -> None:
        self.writer.write(
            {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "action": "session_end",
                "total_tokens": total_tokens,
                "tool_calls": tool_calls,
            }
        )

    def generation_stopped(self, *, conversation_id: str, rounds: int = 0) -> None:
        """记录一次用户主动停止生成（docs 03.3）。

        与 ``session_end`` 分开：一次会话可以包含多轮，而停下的是其中某一轮。
        文档 §8 曾承诺断线时写入 ``aborted`` 审计行却从未实现；这条覆盖了用户
        主动停止这一真实路径（协作式停止经由轮次日志与断点留痕）。
        """
        self.writer.write(
            {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "action": "generation_stopped",
                "conversation_id": conversation_id,
                "rounds": rounds,
            }
        )

    async def post(
        self,
        tool,
        args: dict,
        result: ToolResult,
        *,
        action: str,
        verdict: str,
        duration_ms: float = 0.0,
        citations: list[str] | None = None,
        turn: int = 0,
        endpoint: str = "",
        rows: int = 0,
        cols: int = 0,
    ) -> None:
        self.writer.write(
            {
                "session_id": self.session_id,
                # 与 session 边界行对齐：逐调用行也要能独立回答"是谁"，
                # 而不必先 join 会话行才知道操作者。
                "user_id": self.user_id,
                "action": action,
                "turn": turn,
                "tool": getattr(tool, "name", "unknown"),
                "verdict": verdict,
                "args_summary": summarize_args(args),
                "duration_ms": duration_ms,
                "ok": bool(result.ok),
                "cids": list(citations or []),
                "rows": rows,
                "cols": cols,
                "endpoint": endpoint,
            }
        )
        # 研报的复核本身就是一次可审计事件：工具在 ``metadata`` 中声明结果
        # （写入/声明，绝不对模型可见），而一份已交付的研报是否真的经过了复核——
        # 还是降级为 "unreviewed"，亦或根本在没有复核者的情况下运行——必须仅凭
        # 审计日志就能回答。工具负责声明，hook 负责记录。
        review = getattr(result, "metadata", {}).get("review")
        if isinstance(review, dict):
            row = {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "action": "review",
                "turn": turn,
                "tool": getattr(tool, "name", "unknown"),
            }
            row.update({k: review[k] for k in ("status", "topic", "review_path", "error") if k in review})
            self.writer.write(row)

#!/usr/bin/env python
"""HTTP-driven end-to-end demo (M5).

Drives a real FinHarness process over its own HTTP/SSE API — the same surface a
browser uses — rather than calling the engine in-process. That choice is what
makes the demo honest: tool confirmation travels through the real
``interactive_request`` / ``POST /v1/chat/respond`` round trip, artefacts arrive
as ``tool_status`` attachments, and the closing metrics are read from the
server's own endpoints.

Two scenarios:

* **Demo A** — a comparative question: does the agent plan, fetch data and reach
  a grounded conclusion?
* **Demo B** — "turn that into a report": does it produce a charted docx with an
  appendix, and does the risk reviewer run?

Run against a fresh in-process server, or an already-running one:

    python scripts/demo.py                      # spawn a server, run both
    python scripts/demo.py --base-url http://127.0.0.1:8000
    python scripts/demo.py --demo b --json       # machine-readable metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

DEMO_A_PROMPT = (
    "请对比贵州茅台(600519)与五粮液(000858)当前的估值水平，说明哪只更贵，"
    "并给出你的判断依据。"
)
DEMO_B_PROMPT = (
    "请把本次研究整理成一份贵州茅台(600519)的估值研报，"
    "要有图表、数据来源附录和风险提示。"
)

# The reviewer runs a second model loop after the report is written, so Demo B
# needs a budget well beyond Demo A's.
DEMO_TIMEOUT_S = {"a": 300.0, "b": 900.0}


@dataclass
class TurnResult:
    """Everything observable about one HTTP-driven turn."""

    label: str
    wall_s: float
    ok: bool
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    per_agent: dict = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    tool_calls: int = 0
    tools: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    compactions: int = 0
    retry_count: int = 0
    reason: str | None = None
    confirms: int = 0
    error: str | None = None


def summarize(result: TurnResult) -> str:
    lines = [
        f"  状态：{'成功' if result.ok else '失败'}"
        + (f"（{result.reason}）" if result.reason else ""),
        f"  耗时：{result.wall_s:.1f}s",
        f"  token：输入 {result.input_tokens} / 输出 {result.output_tokens}"
        + (
            f"（前缀缓存命中 {result.cache_hit_tokens} / 未命中 {result.cache_miss_tokens}，"
            f"命中率 {result.cache_hit_tokens / (result.cache_hit_tokens + result.cache_miss_tokens):.0%}）"
            if result.cache_hit_tokens + result.cache_miss_tokens
            else ""
        ),
        f"  工具调用：{result.tool_calls} 次"
        + (f"（{'、'.join(result.tools)}）" if result.tools else ""),
        f"  引用：{len(result.citations)} 条",
        f"  确认往返：{result.confirms} 次",
    ]
    if result.compactions:
        lines.append(f"  上下文压缩：{result.compactions} 次")
    if result.retry_count:
        lines.append(f"  重试：{result.retry_count} 次")
    if result.per_agent:
        for name, entry in result.per_agent.items():
            lines.append(
                f"  子 Agent[{name}]：输入 {entry['input_tokens']} / "
                f"输出 {entry['output_tokens']}（{entry['runs']} 次）"
            )
    if result.attachments:
        lines.append("  产物：")
        lines.extend(f"    - {path}" for path in result.attachments)
    return "\n".join(lines)


class DemoClient:
    """Thin SSE client: submit a message, auto-approve writes, collect metrics."""

    def __init__(self, base_url: str, *, auto_approve: bool = True, verbose: bool = True):
        self.base_url = base_url.rstrip("/")
        self.auto_approve = auto_approve
        self.verbose = verbose
        # The streaming request holds the connection open while the server waits
        # for a confirmation, so the reply must travel on a second client.
        self._control = httpx.Client(timeout=30.0)
        # Pinned after the first turn so later turns resume the same conversation
        # rather than starting a fresh one — Demo B must build on Demo A.
        self.conversation_id: str | None = None

    def close(self) -> None:
        self._control.close()

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def run_turn(self, prompt: str, *, label: str, timeout: float) -> TurnResult:
        started = time.monotonic()
        result = TurnResult(label=label, wall_s=0.0, ok=False)
        tools_seen: list[str] = []
        attachments: list[str] = []

        try:
            with httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0)) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/stream",
                    json={
                        "message": prompt,
                        "mode": "default",
                        "conversation_id": self.conversation_id,
                    },
                ) as response:
                    response.raise_for_status()
                    event_name: str | None = None
                    for line in response.iter_lines():
                        if line.startswith("event:"):
                            event_name = line.split(":", 1)[1].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload = json.loads(line.split(":", 1)[1].strip())
                        if event_name == "session":
                            self.conversation_id = payload.get("conversation_id") or self.conversation_id
                        self._handle(event_name, payload, result, tools_seen, attachments)
        except httpx.TimeoutException:
            result.error = f"请求超时（>{timeout:.0f}s）"
        except httpx.HTTPStatusError as exc:
            result.error = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        except httpx.HTTPError as exc:
            result.error = f"连接失败：{exc}"

        result.wall_s = time.monotonic() - started
        result.tools = tools_seen
        result.attachments = attachments
        return result

    def _handle(
        self,
        event_name: str | None,
        payload: dict,
        result: TurnResult,
        tools_seen: list[str],
        attachments: list[str],
    ) -> None:
        if event_name == "tool_status":
            name = payload.get("name", "")
            if payload.get("status") == "started" and name not in tools_seen:
                tools_seen.append(name)
                self._say(f"  ▸ 调用 {name}")
            if payload.get("attachments"):
                attachments.extend(payload["attachments"])
        elif event_name == "interactive_request":
            self._approve(payload, result)
        elif event_name == "context_compacted":
            result.compactions += 1
            self._say(
                f"  ▸ 上下文压缩：{payload.get('before_tokens')} → "
                f"{payload.get('after_tokens')} token"
            )
        elif event_name == "answer":
            result.answer = payload.get("text", "")
        elif event_name == "error":
            result.error = payload.get("message")
            self._say(f"  ✗ {payload.get('message')}")
        elif event_name == "done":
            result.ok = bool(payload.get("succeeded"))
            result.reason = payload.get("reason")
            usage = payload.get("usage") or {}
            result.input_tokens = int(usage.get("input_tokens", 0))
            result.output_tokens = int(usage.get("output_tokens", 0))
            result.cache_hit_tokens = int(usage.get("cache_hit_tokens", 0))
            result.cache_miss_tokens = int(usage.get("cache_miss_tokens", 0))
            result.per_agent = dict(payload.get("per_agent") or {})
            result.citations = list(payload.get("citations") or [])
            result.tool_calls = int(payload.get("tool_calls", 0))
            result.compactions = int(payload.get("compactions", result.compactions))
            result.retry_count = int(payload.get("retry_count", 0))

    def _approve(self, payload: dict, result: TurnResult) -> None:
        request_id = payload.get("request_id", "")
        prompt = payload.get("prompt", "")
        self._say(f"  ▸ 服务端请求确认：{prompt}")
        if not self.auto_approve:
            return
        response = self._control.post(
            f"{self.base_url}/v1/chat/respond",
            json={"request_id": request_id, "response": "y"},
        )
        result.confirms += 1
        self._say(f"  ▸ 已批准（HTTP {response.status_code}）")


def wait_for_health(base_url: str, *, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/v1/health", timeout=2.0).status_code == 200:
                return True
        except httpx.HTTPError:
            time.sleep(0.3)
    return False


def start_server(settings_path: Path, host: str, port: int):
    """Start uvicorn in a background thread; returns (server, thread)."""
    import uvicorn

    from finharness.server.api import create_production_app

    app = create_production_app(str(settings_path))
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def provider_label(client: DemoClient) -> str:
    """Name the provider that actually served the run.

    Worth resolving explicitly: an activated database configuration takes
    precedence over the settings file's preset, so "which model answered" is
    not something the caller can infer from its own config.
    """
    try:
        payload = client._control.get(f"{client.base_url}/v1/config").json()
    except httpx.HTTPError:
        return "未知"
    active_id = payload.get("active_id")
    for record in payload.get("configs", []):
        if record.get("id") == active_id:
            return f"{record.get('name') or record.get('kind')} / {record.get('model')}"
    return "settings 预设（无激活的数据库配置）"


def print_closing_metrics(client: DemoClient, settings_path: Path) -> dict:
    """The demo's closing screen: what the run produced and what it cost."""
    stats = {}
    try:
        stats = client._control.get(f"{client.base_url}/v1/cache/stats").json()
    except httpx.HTTPError:
        pass

    audit_lines = 0
    audit_path = ROOT / "logs" / "audit.jsonl"
    if audit_path.is_file():
        audit_lines = sum(1 for _ in audit_path.open(encoding="utf-8"))

    provider = provider_label(client)
    lines = ["", "=" * 60, "收尾指标"]
    lines.append(f"  Provider：{provider}")
    lines.append(
        f"  缓存：{stats.get('entries', '?')} 条，命中 {stats.get('hits', '?')} / "
        f"未命中 {stats.get('misses', '?')}，命中率 {stats.get('hit_ratio', 0):.0%}"
    )
    lines.append(f"  审计条目：{audit_lines} 行（logs/audit.jsonl）")
    if client.conversation_id:
        lines.append(f"  对话 id：{client.conversation_id}")
    print("\n".join(lines), flush=True)
    return {
        "provider": provider,
        "cache": stats,
        "audit_lines": audit_lines,
        "conversation_id": client.conversation_id,
    }


def run_demos(client: DemoClient, which: str) -> list[TurnResult]:
    results: list[TurnResult] = []
    if which in ("a", "all"):
        print("\n【Demo A】多标的对比研究（应触发规划）", flush=True)
        print(f"  提问：{DEMO_A_PROMPT}", flush=True)
        result = client.run_turn(
            DEMO_A_PROMPT, label="Demo A", timeout=DEMO_TIMEOUT_S["a"]
        )
        print(summarize(result), flush=True)
        if result.answer:
            print(f"  结论摘要：{result.answer[:200].replace(chr(10), ' ')}", flush=True)
        results.append(result)

    if which in ("b", "all"):
        print("\n【Demo B】一句话出带图带附录 docx（并跑风险终审）", flush=True)
        print(f"  提问：{DEMO_B_PROMPT}", flush=True)
        result = client.run_turn(
            DEMO_B_PROMPT, label="Demo B", timeout=DEMO_TIMEOUT_S["b"]
        )
        print(summarize(result), flush=True)
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FinHarness HTTP-driven demo (M5)")
    parser.add_argument("--base-url", help="drive an already-running server instead of spawning one")
    parser.add_argument("--settings", default="settings.json", help="settings file for a spawned server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--demo", choices=["a", "b", "all"], default="all")
    parser.add_argument("--json", action="store_true", help="print a machine-readable metric summary")
    parser.add_argument("--quiet", action="store_true", help="suppress per-event progress")
    args = parser.parse_args(argv)

    server = None
    base_url = args.base_url
    if base_url is None:
        settings_path = Path(args.settings)
        if not settings_path.is_file():
            print(
                f"找不到配置文件 {settings_path}；请先复制 settings.example.json 为 "
                "settings.json 并配置 Provider 与密钥环境变量。",
                file=sys.stderr,
            )
            return 2
        base_url = f"http://{args.host}:{args.port}"
        print(f"启动服务：{base_url}（配置 {settings_path}）", flush=True)
        server, _thread = start_server(settings_path, args.host, args.port)
        if not wait_for_health(base_url):
            print("服务未在 30s 内就绪，退出。", file=sys.stderr)
            return 1
        print("服务就绪。", flush=True)

    client = DemoClient(base_url, verbose=not args.quiet)
    results: list[TurnResult] = []
    try:
        results = run_demos(client, args.demo)
        closing = print_closing_metrics(client, Path(args.settings))
    finally:
        client.close()
        if server is not None:
            server.should_exit = True

    if args.json:
        print(
            json.dumps(
                {
                    "base_url": base_url,
                    "results": [
                        {
                            "label": r.label,
                            "ok": r.ok,
                            "wall_s": round(r.wall_s, 1),
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                            "cache_hit_tokens": r.cache_hit_tokens,
                            "cache_miss_tokens": r.cache_miss_tokens,
                            "per_agent": {k: dict(v) for k, v in r.per_agent.items()},
                            "citations": len(r.citations),
                            "tool_calls": r.tool_calls,
                            "tools": r.tools,
                            "attachments": r.attachments,
                            "compactions": r.compactions,
                            "retry_count": r.retry_count,
                            "confirms": r.confirms,
                            "reason": r.reason,
                            "error": r.error,
                        }
                        for r in results
                    ],
                    "closing": closing,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0 if all(r.ok for r in results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())

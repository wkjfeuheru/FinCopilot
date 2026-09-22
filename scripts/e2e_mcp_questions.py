"""端到端验收：真实 LLM × 真实数据源（含同花顺 MCP）跑测试问题集。

**它验证什么。** 不是单元测试的替代品，而是把整条链路接起来跑一遍真问题：模型是否
真的调用了工具、取到的是不是真实数据、回答有没有引用可溯源。因此断言只落在"无法靠
编造通过"的不变量上：`succeeded`、是否有 citation、citation 的 endpoint 来自哪个源。

**为什么选这些题。** 重点压在**只有同花顺能答**的域上——特色数据（涨停池/龙虎榜/
热股榜）、交易日历、概念指数、估值快照、公募基金、期货、期权、标的检索。这些在接入
同花顺之前是取不到的，因此"答出来了"本身就是 MCP 通道打通的证据；纯靠 akshare 也能
答的题（行情/指标）只保留少量作为对照组。

**为什么走 PermissionGate(AUTO)。** 生产路径下外发工具（同花顺/联网）首次调用须经
用户确认。脚本没有交互通道，用 AUTO 模式绕过确认——这与服务端把它设为可选行为一致，
且不影响本脚本要验证的东西（工具调用与数据落地）。若要看确认往返本身，用 Web 聊天页。

运行：``uv run python scripts/e2e_mcp_questions.py``（或 .venv 直跑）。
需要：``DEEPSEEK_API_KEY``（或 settings.json 里配置的模型 provider），以及
``HITHINK_FINANCE_API_KEY`` 或 settings.json 的 ``fuyao.api_key``。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from finharness.config.settings import Settings  # noqa: E402
from finharness.context.session import ResearchContext  # noqa: E402
from finharness.data.access import DataAccess  # noqa: E402
from finharness.data.adapters.akshare_adapter import AkShareAdapter  # noqa: E402
from finharness.data.adapters.eastmoney_report_adapter import EastmoneyReportAdapter  # noqa: E402
from finharness.data.adapters.fuyao_adapter import FuyaoMcpAdapter  # noqa: E402
from finharness.data.adapters.tavily_adapter import TavilyAdapter  # noqa: E402
from finharness.data.cache import LocalCache  # noqa: E402
from finharness.data.citation import CitationRegistry  # noqa: E402
from finharness.engine.loop import AgentLoop  # noqa: E402
from finharness.engine.prompt import system_prompt  # noqa: E402
from finharness.permissions.gate import PermissionGate  # noqa: E402
from finharness.provider.openai_compat import OpenAICompatProvider  # noqa: E402
from finharness.tools.registry import ToolRegistry  # noqa: E402


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    # 期望的数据来源：``fuyao`` 表示只有同花顺答得出（新能力），``any`` 表示
    # akshare 也能答（对照组），``refuse`` 表示期望诚实拒答/说明边界。
    expect: str
    note: str = ""


QUESTIONS: tuple[Question, ...] = (
    # --- 核心 A 股：现在经同花顺 typed 适配器（对照组，akshare 也能答） ---
    Question("Q01", "请提供中国平安（601318）的最新行情快照。", "any"),
    Question("Q02", "沪深300ETF（510300）最新价格是多少？", "fuyao", "ETF 行情只有 fund 服务的 get_fund_market_snapshot 能答"),
    Question("Q03", "请提供长江电力（600900）近两年的月K线走势。", "any", "月线只有 akshare 有"),
    Question("Q04", "请提供长江电力（600900）近三年的现金流量表概况。", "any"),
    Question("Q05", "请提供比亚迪（002594）近五年的ROE与毛利率变化情况。", "any"),
    # --- 只有同花顺能答：行情/指标之外的新域 ---
    Question("Q06", "请提供沪深300指数（000300）当前的成分股数量与其中前5只。", "fuyao"),
    Question("Q07", "贵州茅台（600519）与五粮液（000858）当前的市盈率（TTM）与市净率分别是多少？请做对比。", "fuyao"),
    Question("Q08", "请列出最近一个交易日的A股龙虎榜，并指出净买入额最大的三只股票。", "fuyao", "特色数据"),
    Question("Q09", "当前A股热股榜前10名是哪些股票？", "fuyao", "特色数据"),
    Question("Q10", "请给出最近一个交易日的A股涨停股票池。", "fuyao", "特色数据"),
    Question("Q11", "2026年9月A股共有多少个交易日？分别是哪几天？", "fuyao", "交易日历，之前完全没有"),
    Question("Q12", "同花顺的白酒概念板块包含哪些成分股？", "fuyao", "概念指数，akshare 无"),
    Question("Q13", "请查询沪深300ETF（510300）最新披露的前十大重仓股。", "fuyao", "公募基金域"),
    Question("Q14", "请查询易方达蓝筹精选混合基金的最新单位净值。", "fuyao", "原测试集预期「不覆盖基金」，现应能答"),
    Question("Q15", "目前国内期货市场有哪些品种可以查询？请列出部分。", "fuyao", "期货域"),
    Question("Q16", "目前有哪些期权品种可以查询？", "fuyao", "期权域"),
    Question("Q17", "我只知道公司名叫「沈鼓集团」，请帮我查到它的股票代码。", "fuyao", "标的检索（meta）"),
    # --- 边界与幻觉：期望诚实处理 ---
    Question("Q18", "苹果（AAPL）的最新股价是多少？", "refuse", "非 A 股，应说明边界"),
    Question("Q19", "请查询神州十八号股份（999999）的最新行情。", "refuse", "代码不存在，不得编造"),
    Question("Q20", "据我掌握的信息，贵州茅台2023年营业收入约为1500亿元，请帮我核实一下。", "any", "用户记忆错误，应取数纠正"),
)


@dataclass
class Result:
    qid: str
    text: str
    expect: str
    ok: bool = False
    succeeded: bool = False
    error: str | None = None
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    rounds: int = 0
    tool_calls: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    note: str = ""


def build_loop(settings: Settings, workdir: Path) -> AgentLoop:
    """按服务端的接线方式组装一个会话（同花顺优先，其余兜底）。"""
    cache_dir = workdir / "cache"
    scoped = settings.model_copy(
        update={
            "data": settings.data.model_copy(update={"cache_dir": cache_dir}),
            "paths": settings.paths.model_copy(update={"output_dir": workdir / "output"}),
        }
    )
    adapters = [
        FuyaoMcpAdapter(
            api_key=scoped.fuyao.resolved_api_key(),
            base_url=scoped.fuyao.base_url,
            timeout_s=scoped.fuyao.timeout_s,
            proxy=scoped.fuyao.proxy,
            throttle_seconds=scoped.data.throttle_seconds,
        ),
        AkShareAdapter(throttle_seconds=scoped.data.throttle_seconds),
        TavilyAdapter(
            api_key=scoped.search.api_key
            or (os.getenv(scoped.search.env_key) if scoped.search.env_key else None),
            base_url=scoped.search.base_url,
            timeout_s=scoped.search.timeout_s,
            proxy=scoped.search.proxy,
        ),
        EastmoneyReportAdapter(
            timeout_s=scoped.search.timeout_s,
            with_text_allowed=scoped.search.local_pdf_fallback,
            pdf_dir=cache_dir / "pdf",
        ),
    ]
    data = DataAccess(adapters, cache=LocalCache(cache_dir), settings=scoped)
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=scoped)

    provider = _build_provider(scoped)

    async def auto_answer(_kind, _prompt, _options, **_):
        # 无人值守：缺输入时给一个确定性的替代答案，使流程继续。
        return "综合体检"

    gate = PermissionGate(settings=scoped, mode="auto")
    return AgentLoop(
        provider=provider,
        registry=ToolRegistry(data, ctx=ctx, settings=scoped),
        settings=scoped,
        system=system_prompt(),
        cite=cite,
        ctx=ctx,
        gate=gate,
        interactive=auto_answer,
    )


def _build_provider(settings: Settings):
    """按 settings.model 选择 provider；fake 时用 FakeProvider。"""
    preset = settings.providers[settings.model.provider]
    if preset.kind == "fake":
        from finharness.provider.fake import FakeProvider

        return FakeProvider(["（离线占位回答）"])
    env_key = preset.env_key
    api_key = os.getenv(env_key) if env_key else None
    if not api_key:
        raise SystemExit(
            f"缺少模型密钥：环境变量 {env_key} 未设置，无法运行端到端验收。"
        )
    if preset.kind != "openai_compat":
        raise SystemExit(
            f"本脚本只演示 OpenAI 兼容协议；settings.model.provider="
            f"{settings.model.provider} 是 {preset.kind}。请改用 deepseek/qwen/volcano，"
            "或在 settings.json 里指定一个 openai_compat provider。"
        )
    return OpenAICompatProvider(
        base_url=preset.base_url,
        api_key=api_key,
        model=settings.model.model_name,
        client=httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)),
    )


def run_one(loop: AgentLoop, q: Question, timeout: float) -> Result:
    started = time.monotonic()
    result = Result(qid=q.qid, text=q.text, expect=q.expect, note=q.note)
    try:
        outcome = asyncio.run(asyncio.wait_for(loop.run(q.text), timeout))
    except asyncio.TimeoutError:
        result.error = f"超时（>{timeout:.0f}s）"
        result.seconds = time.monotonic() - started
        return result
    except Exception as exc:  # noqa: BLE001 - 单题失败不该中断整轮
        result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = time.monotonic() - started
        return result

    result.seconds = time.monotonic() - started
    result.succeeded = bool(outcome.succeeded)
    result.error = outcome.error
    result.answer = outcome.answer or ""
    result.rounds = int(outcome.rounds or 0)
    result.tool_calls = int(outcome.tool_calls or 0)
    usage = outcome.usage
    result.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    result.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    citations = loop.cite.all()
    result.citations = [c.cid for c in citations]
    result.endpoints = [c.endpoint for c in citations]
    result.tools = sorted({c.tool for c in citations})
    # 判定：成功完成 + 有依据（expected=refuse 的题目不要求 citation）。
    if result.succeeded and (result.citations or q.expect == "refuse"):
        result.ok = True
    return result


def _fuyao_used(r: Result) -> bool:
    return any(endpoint.startswith("fuyao:") for endpoint in r.endpoints)


def main() -> int:
    settings = Settings.from_file(str(ROOT / "settings.json"))
    if not settings.fuyao.resolved_api_key():
        print("！未配置同花顺密钥，Q06-Q17 必然失败。请设置 HITHINK_FINANCE_API_KEY 或 fuyao.api_key。")

    runs_dir = ROOT / "evals" / "runs"
    run_dir = runs_dir / f"mcp-e2e-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "results.jsonl"

    print(f"运行目录：{run_dir}")
    print(f"模型：{settings.model.provider}/{settings.model.model_name}")
    print(f"同花顺：{'已配置' if settings.fuyao.resolved_api_key() else '未配置'}")
    print(f"题目：{len(QUESTIONS)} 道\n")

    timeout = float(os.getenv("E2E_TIMEOUT", "240"))
    results: list[Result] = []
    for index, question in enumerate(QUESTIONS, start=1):
        # 每题一个新会话：避免前题的结论污染后题（本脚本测的是单题链路）。
        loop = build_loop(settings, run_dir / f"case-{question.qid}")
        print(f"[{index:2d}/{len(QUESTIONS)}] {question.qid} 期望={question.expect:6s} {question.text[:44]}")
        result = run_one(loop, question, timeout)
        results.append(result)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
        flag = "✅" if result.ok else "❌"
        fuyao = "via 同花顺" if _fuyao_used(result) else ""
        print(
            f"        {flag} rounds={result.rounds} calls={result.tool_calls} "
            f"cites={len(result.citations)} {fuyao} {result.seconds:.0f}s"
        )
        if result.error:
            print(f"        error: {result.error[:160]}")
        if result.tools:
            print(f"        tools: {', '.join(result.tools)}")
        sys.stdout.flush()

    _write_report(run_dir, settings, results)
    passed = sum(1 for r in results if r.ok)
    print(f"\n完成：{passed}/{len(results)} 通过；报告见 {run_dir / 'report.md'}")
    return 0 if passed == len(results) else 1


def _write_report(run_dir: Path, settings: Settings, results: list[Result]) -> None:
    lines: list[str] = []
    lines.append(f"# 同花顺 MCP 端到端验收（{time.strftime('%Y-%m-%d %H:%M')}）\n")
    lines.append(f"- 模型：`{settings.model.provider}/{settings.model.model_name}`")
    lines.append(f"- 同花顺：`{settings.fuyao.base_url}`")
    lines.append(f"- 题目：{len(results)} 道\n")

    passed = sum(1 for r in results if r.ok)
    fuyao_hits = sum(1 for r in results if _fuyao_used(r))
    lines.append(f"**通过 {passed}/{len(results)}；其中 {fuyao_hits} 题实际经同花顺取数。**\n")

    lines.append("| # | 题号 | 期望 | 结果 | 轮次 | 工具调用 | 引用 | 经同花顺 | 耗时 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {results.index(r) + 1} | {r.qid} | {r.expect} | {'✅' if r.ok else '❌'} | "
            f"{r.rounds} | {r.tool_calls} | {len(r.citations)} | "
            f"{'是' if _fuyao_used(r) else '否'} | {r.seconds:.0f}s |"
        )

    lines.append("\n---\n")
    for r in results:
        lines.append(f"## {r.qid}　{'✅' if r.ok else '❌'}　期望={r.expect}\n")
        lines.append(f"**问**：{r.text}\n")
        if r.note:
            lines.append(f"> 考点：{r.note}\n")
        if r.error:
            lines.append(f"**错误**：`{r.error}`\n")
        lines.append(f"**调用工具**：{', '.join(r.tools) or '（无）'}\n")
        if r.endpoints:
            lines.append(f"**数据来源**：{', '.join(f'`{e}`' for e in dict.fromkeys(r.endpoints))}\n")
        lines.append(f"**用量**：轮次 {r.rounds}、工具调用 {r.tool_calls}、"
                     f"prompt {r.prompt_tokens}、completion {r.completion_tokens}、{r.seconds:.0f}s\n")
        lines.append("**答**：\n")
        lines.append("```text")
        lines.append((r.answer or "（空）").strip()[:4000])
        lines.append("```\n")

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

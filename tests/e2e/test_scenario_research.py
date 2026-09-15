"""在真实 provider 上的端到端验收：scenario 路由与新工具。

这些是 ``docs/测试问题集-功能与幻觉.md`` 中 G 系列验收问题转化为的可执行检查。
它们会访问真实的 LLM 和真实数据源，因此被标记为 ``smoke``，并且在缺少凭证和
网络时会被跳过。运行方式：``uv run pytest -m smoke``。

断言策略（与 README 中「模型输出存在波动」的说明以及现有 smoke 测试的风格一致）：
只断言在任何合理路径上都成立的不变量。

* 当**必须借助 tool 才能回答**时（一次 macro 读数、一次 backtest），会断言该 tool
  call——模型无法绕过它凭空编造。
* 当**skill 只是建议性**时（scenario 入口），只断言路由*没有出错*：一个问题
  可以在不加载 skill 的情况下被妥善回答，因此「加载了哪个 skill」属于清单中
  的人工判断，而非测试内容。

这些路由所依赖的确定性管道——skill 文件加载、backtest-then-chart、registry tiers——
在 ``tests/e2e/test_scenario_pipeline.py`` 中离线覆盖。
"""

from __future__ import annotations

import asyncio
import os

import pytest

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.tools.registry import ToolRegistry

pytestmark = pytest.mark.smoke

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
SYSTEM_PROMPT = system_prompt()

SCENARIOS = ("equity-research", "industry-research", "macro-research", "quant-factor")


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return key


def _build_loop(tmp_path, *, answer_ask: bool = False) -> AgentLoop:
    import httpx

    from finharness.context.session import ResearchContext

    provider = OpenAICompatProvider(
        base_url=DEEPSEEK_BASE_URL,
        api_key=_api_key(),
        model=DEEPSEEK_MODEL,
        client=httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)),
    )
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={"output_dir": tmp_path / "output"},
    )
    data = DataAccess(
        [AkShareAdapter(throttle_seconds=0.2)], cache=LocalCache(tmp_path / "cache"), settings=settings
    )
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)

    async def auto_answer(_kind, _prompt, _options):
        # 对 UI 会显示的确认对话框的确定性替身。
        return "综合体检"

    return AgentLoop(
        provider=provider,
        # 必须把 ctx 接入 registry——load_skill 正是在那里记录已加载的
        # scenario；生产服务器也这么做。
        registry=ToolRegistry(data, ctx=ctx, settings=settings),
        settings=settings,
        system=SYSTEM_PROMPT,
        cite=cite,
        ctx=ctx,
        interactive=auto_answer if answer_ask else None,
    )


def _run(loop: AgentLoop, question: str, timeout: int = 240):
    async def run():
        return await asyncio.wait_for(loop.run(question), timeout)

    return asyncio.run(run())


def _loaded_scenarios(loop: AgentLoop) -> set[str]:
    return {name.split("/")[0] for name in loop.ctx.loaded_skills}


def _tools_called(loop: AgentLoop) -> set[str]:
    """本会话中产生了 citation 的 tool 名称。"""
    return {citation.tool for citation in loop.cite.all()}


# --- scenario 路由 --------------------------------------------------------

def test_equity_research_question_is_grounded_in_fetched_data(tmp_path):
    """一个多维度的 equity 任务必须基于真实数据来回答。

    断言其有依据，且没有进入*错误的* scenario；equity skill 是否加载属于建议性
    内容，由人工判断（见模块 docstring）。
    """
    loop = _build_loop(tmp_path)

    outcome = _run(
        loop, "帮我分析贵州茅台(600519)近三年的盈利质量和当前估值水平，并说明主要财务风险。"
    )

    assert outcome.succeeded is True, outcome.error
    assert outcome.citations, "a data-backed answer must register citations"
    assert _loaded_scenarios(loop) <= {"equity-research"}, loop.ctx.loaded_skills


def test_industry_question_is_grounded_and_not_misrouted(tmp_path):
    loop = _build_loop(tmp_path)

    outcome = _run(loop, "分析一下白酒行业目前的竞争格局。")

    assert outcome.succeeded is True, outcome.error
    assert _loaded_scenarios(loop) <= {"industry-research"}, loop.ctx.loaded_skills


def test_simple_lookup_does_not_load_a_scenario(tmp_path):
    """单个数据点不是研究任务；不应加载任何 scenario。"""
    loop = _build_loop(tmp_path)

    outcome = _run(loop, "贵州茅台(600519)现在多少钱？只回答价格。")

    assert outcome.succeeded is True, outcome.error
    assert _loaded_scenarios(loop) == set(), loop.ctx.loaded_skills


# --- 新数据工具（必须借助 tool：稳健） ----------------------------------

def test_macro_question_fetches_real_indicators(tmp_path):
    """一次 macro 读数只能来自 get_macro_indicators。"""
    loop = _build_loop(tmp_path)

    outcome = _run(loop, "现在最新的制造业PMI、CPI和M2同比分别大概是多少？")

    assert outcome.succeeded is True, outcome.error
    assert "get_macro_indicators" in _tools_called(loop), _tools_called(loop)
    assert any(ch.isdigit() for ch in outcome.answer)


def test_industry_question_may_use_industry_data_tool(tmp_path):
    """行业问题有专门的数据路由；一旦使用，它就必须为回答提供依据。
    （并非每个行业问题都需要指数工具，因此这是一个弱检查：如果获取了行业数据，
    回答就带有 citation。）"""
    loop = _build_loop(tmp_path)

    outcome = _run(loop, "白酒行业指数近一年的表现怎么样？")

    assert outcome.succeeded is True, outcome.error
    assert outcome.citations


# --- backtest + chart（必须借助 tool：稳健） --------------------------------

def test_backtest_question_runs_the_tool_and_reports_discipline(tmp_path):
    """策略问题必须触达 run_backtest，而不是编造结果。"""
    loop = _build_loop(tmp_path)

    outcome = _run(
        loop,
        "帮我用20日均线上穿60日均线的策略回测一下贵州茅台(600519)，靠不靠谱？",
        timeout=300,
    )

    assert outcome.succeeded is True, outcome.error
    assert "run_backtest" in _tools_called(loop), _tools_called(loop)
    # 纪律性字段属于该 tool 自身输出的一部分。
    assert any(token in outcome.answer for token in ("样本外", "夏普", "回撤"))


def test_backtest_chart_request_produces_a_chart_file(tmp_path):
    loop = _build_loop(tmp_path)

    outcome = _run(
        loop,
        "回测贵州茅台(600519)的20/60日均线策略，并把策略和买入持有的净值曲线画在一张图里。",
        timeout=300,
    )

    assert outcome.succeeded is True, outcome.error
    charts = list((tmp_path / "output" / "charts").glob("*.png"))
    assert charts, "the chart request should have produced a chart file"

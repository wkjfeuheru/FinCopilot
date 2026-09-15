"""观测工具：运行真实问题并报告 ReAct 循环的可观测项。

不属于测试套件。针对每个问题打印：是否形成计划、plan_progress 事件
（偏离计划 / 停滞信号）、done payload 的 plan 字段、加载的场景技能、
调用的工具、引用，以及回合/token 计数。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

import httpx

from finharness.config.settings import Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.tools.registry import ToolRegistry
from finharness.types import EngineEvent

QUESTIONS = [
    ("F1-3 多步规划", "对比贵州茅台(600519)与五粮液(000858)近三年的盈利能力和估值水平，并归因 ROE 差异。"),
    ("G1-3/G3 行业", "分析一下白酒行业目前的竞争格局和景气度。"),
    ("G4/G5 量化回测", "用20日均线上穿60日均线策略回测贵州茅台(600519)，靠不靠谱？"),
]


class Sink:
    def __init__(self) -> None:
        self.events: list[EngineEvent] = []

    async def emit(self, event: EngineEvent) -> None:
        self.events.append(event)


def build(tmp: Path):
    settings = Settings(
        data={"cache_dir": tmp / "cache"}, paths={"output_dir": tmp / "out"}
    )
    data = DataAccess(
        [AkShareAdapter(throttle_seconds=0.3)], cache=LocalCache(tmp / "cache"), settings=settings
    )
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)
    sink = Sink()
    provider = OpenAICompatProvider(
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        model="deepseek-chat",
        client=httpx.AsyncClient(timeout=httpx.Timeout(150.0, connect=30.0)),
    )

    async def auto_answer(_kind, _prompt, _options):
        return "综合"

    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(data, ctx=ctx, settings=settings),
        settings=settings,
        system=system_prompt(),
        cite=cite,
        ctx=ctx,
        output=sink,
        interactive=auto_answer,
    )
    return loop, ctx, cite, sink


def report(label: str, question: str, outcome, loop, ctx, cite, sink) -> None:
    print(f"\n{'='*78}\n### {label}\nQ: {question}\n")
    usage = outcome.usage
    print(f"succeeded={outcome.succeeded} reason={outcome.reason} "
          f"turns={loop.turn} tool_calls={outcome.tool_calls}")
    print(f"skills loaded: {ctx.loaded_skills or '（未加载场景技能）'}")
    print(f"tools called: {sorted({c.tool for c in cite.all()}) or '（无取数）'}")

    progress = [e.data for e in sink.events if e.kind == "plan_progress"]
    if progress:
        print(f"plan_progress events ({len(progress)}):")
        for p in progress:
            flag = []
            if p.get("drift"):
                flag.append(f"drift={p['drift']}")
            if p.get("mismatch"):
                flag.append(f"mismatch={p['mismatch']}")
            if p.get("stalled_turns"):
                flag.append(f"stalled={p['stalled_turns']}")
            print(f"  - rev{p['revision']} {p['done']}/{p['total']} {' '.join(flag) or 'on-plan'}")
    else:
        print("plan_progress events: none")

    plan = ctx.plan
    print(f"plan: {('v%s %d steps' % (plan.revision, len(plan.steps))) if plan else '（未立计划）'}")
    done = [e for e in sink.events if e.kind == "done"]
    if done:
        print(f"done.plan payload: {done[-1].data.get('plan')}")
    total_mismatch = sum(len(p.get("mismatch") or []) for p in progress)
    total_drift = sum(len(p.get("drift") or []) for p in progress)
    print(f"signals: mismatch={total_mismatch} drift={total_drift}（应为 0 误报）")
    print(f"answer head: {(outcome.answer or '')[:160].replace(chr(10), ' ')}")
    print(f"token usage: in={usage.input_tokens} out={usage.output_tokens}")


def main() -> None:
    """运行全部观测问题，并打印每个问题的 ReAct 循环观测报告。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY not set"); return
    for label, question in QUESTIONS:
        tmp = Path(tempfile.mkdtemp())
        loop, ctx, cite, sink = build(tmp)
        try:
            outcome = asyncio.run(asyncio.wait_for(loop.run(question), 300))
        except Exception as exc:  # noqa: BLE001
            print(f"\n### {label}\nQ: {question}\n!! raised {type(exc).__name__}: {exc}")
            continue
        report(label, question, outcome, loop, ctx, cite, sink)


if __name__ == "__main__":
    main()

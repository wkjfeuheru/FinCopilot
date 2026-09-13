#!/usr/bin/env python
"""Measure prefix-cache effectiveness: state-in-system vs state-as-trailing-message.

Runs the same multi-step research question through two loop configurations and
reports the provider's own cache accounting for each. The "old" arm is produced
by monkeypatching the loop's request assembly back to the pre-optimization
layout (state concatenated onto the system prompt), so both arms run identical
code paths otherwise and the comparison is apples-to-apples.

    python scripts/measure_prefix_cache.py            # needs DEEPSEEK_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from finharness.config.settings import PermissionSettings, Settings  # noqa: E402
from finharness.context.session import ResearchContext  # noqa: E402
from finharness.data.access import DataAccess  # noqa: E402
from finharness.data.adapters.akshare_adapter import AkShareAdapter  # noqa: E402
from finharness.data.cache import LocalCache  # noqa: E402
from finharness.data.citation import CitationRegistry  # noqa: E402
from finharness.engine.loop import AgentLoop  # noqa: E402
from finharness.engine.prompt import system_prompt  # noqa: E402
from finharness.permissions.gate import PermissionGate  # noqa: E402
from finharness.provider.openai_compat import OpenAICompatProvider  # noqa: E402
from finharness.tools.registry import ToolRegistry  # noqa: E402

# One question that forces several sequential tool rounds — the shape where
# per-iteration prefix reuse matters most.
QUESTION = "请分步研究贵州茅台(600519)：先取行情与估值，再看财务指标，最后给一段综合结论。"


@dataclass
class Arm:
    label: str
    input_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    output_tokens: int
    wall_s: float
    tool_calls: int

    @property
    def hit_ratio(self) -> float:
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return (self.cache_hit_tokens / total) if total else 0.0


def _build_loop(settings, data, *, state_in_system: bool):
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)
    provider = OpenAICompatProvider(
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        model="deepseek-chat",
        client=httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=30.0)),
    )
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(data, ctx=ctx, settings=settings),
        settings=settings,
        system=system_prompt(),
        cite=cite,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
        session_id="prefix-measure",
    )
    if state_in_system:
        # Reproduce the pre-optimization layout without touching the source:
        # state glued to the system prompt, history unchanged.
        def old_system() -> str:
            return loop.system + loop.ctx.state_block()

        loop._system_prompt = old_system  # type: ignore[method-assign]
        loop._request_messages = lambda: loop.memory.snapshot()  # type: ignore[method-assign]
    return loop


async def _run_arm(settings, data, *, state_in_system: bool) -> Arm:
    loop = _build_loop(settings, data, state_in_system=state_in_system)
    started = time.monotonic()
    outcome = await asyncio.wait_for(loop.run(QUESTION), 600)
    wall = time.monotonic() - started
    snapshot = loop.stats.snapshot()
    return Arm(
        label="OLD (state in system)" if state_in_system else "NEW (state trailing)",
        input_tokens=snapshot.input_tokens,
        cache_hit_tokens=snapshot.cache_hit_tokens,
        cache_miss_tokens=snapshot.cache_miss_tokens,
        output_tokens=snapshot.output_tokens,
        wall_s=wall,
        tool_calls=outcome.tool_calls,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("需要设置 DEEPSEEK_API_KEY", file=sys.stderr)
        return 2

    from finharness.context.tokens import TokenCounter  # warm the vocabulary cache

    TokenCounter()

    settings = Settings(
        permission=PermissionSettings(default_mode="auto"),
        data={"cache_dir": ROOT / "data_cache"},
        paths={"output_dir": ROOT / "output"},
        audit={"log_path": ROOT / "logs" / "prefix-measure.jsonl"},
    )
    data = DataAccess(
        [AkShareAdapter(throttle_seconds=0.2)],
        cache=LocalCache(settings.data.cache_dir),
        settings=settings,
    )

    # Old first: its writes warm the market-data cache, so the new arm is not
    # penalised by cold fetches. Prefix caching is per-request-prefix and
    # unaffected by this ordering.
    arms = [asyncio.run(_run_arm(settings, data, state_in_system=True))]
    arms.append(asyncio.run(_run_arm(settings, data, state_in_system=False)))

    if args.json:
        import json

        print(json.dumps([arm.__dict__ | {"hit_ratio": arm.hit_ratio} for arm in arms],
                         ensure_ascii=False, indent=2))
        return 0

    print("\n前缀缓存测量（同一问题、同一数据源）")
    print("=" * 62)
    header = f"{'配置':<26}{'输入':>9}{'命中':>9}{'未命中':>9}{'命中率':>8}{'工具':>5}"
    print(header)
    for arm in arms:
        print(
            f"{arm.label:<26}{arm.input_tokens:>9}{arm.cache_hit_tokens:>9}"
            f"{arm.cache_miss_tokens:>9}{arm.hit_ratio:>7.0%}{arm.tool_calls:>5}"
        )
    old, new = arms
    print("-" * 62)
    print(f"命中率：{old.hit_ratio:.0%} → {new.hit_ratio:.0%}")
    print(f"未命中（全价）token：{old.cache_miss_tokens} → {new.cache_miss_tokens}")
    if old.input_tokens:
        # Lower is better; report the direction explicitly rather than a sign.
        delta = new.input_tokens - old.input_tokens
        word = "减少" if delta < 0 else "增加"
        print(f"输入 token 合计：{old.input_tokens} → {new.input_tokens}（{word} {abs(delta)}）")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

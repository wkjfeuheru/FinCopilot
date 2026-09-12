"""Risk-review sub-agent (docs 03.10).

The reviewer is the one place a second context is worth its cost. Compaction
already recovers window space, and parallel tool calls already cover speed, but
neither gives an *independent* read of a finished report: a reviewer that has not
written the report, and that can re-fetch the underlying data, catches things the
author cannot see in their own draft.

Isolation is the whole point, so it is enforced structurally:

* the sub-agent gets a fresh ``ResearchContext``, transcript and stats — it
  cannot see the main conversation's reasoning, and nothing it does leaks into
  the conversation's memory (``store=None``);
* its tool catalogue is narrowed by construction (``only=review_tool_names()``),
  so a name outside the read-only subset does not resolve at all;
* the citation registry *is* shared, deliberately: cids stay continuous, and a
  reviewer-verified figure can be cited in the report's next revision.

The reviewer never edits the report. It returns comments; the main agent decides
what to do with them (docs 03.10).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry

RISK_FOCUS = "risk"
RISK_SKILL = "risk-checklist"
# One round to read the report (and optionally fetch data), one to write the
# review. A reviewer that needs more than this is looping, not reviewing.
REVIEW_MAX_TURNS = 3


@dataclass(frozen=True, slots=True)
class SubAgentResult:
    """What a review returns: comments plus an honest account of its own cost."""

    focus: str
    summary: str
    citations: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    ok: bool = True
    error: str | None = None


class Coordinator:
    """Runs a focus-scoped review in an isolated context (docs 03.10)."""

    def __init__(
        self,
        *,
        provider: Any,
        data: DataAccess,
        settings: Settings,
        cite: CitationRegistry,
        counter: Any | None = None,
        on_usage: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.provider = provider
        self.data = data
        self.settings = settings
        self.cite = cite
        self.counter = counter
        self._on_usage = on_usage
        # Wired by the owning loop so sub-agent spend lands in the session's
        # totals. Absent wiring, the review still runs; it just is not billed.
        self._usage: Any | None = None
        self._stats: Any | None = None

    def bind_accounting(self, *, usage: Any, stats: Any) -> None:
        """Attach the main loop's counters so review cost is attributed there."""
        self._usage = usage
        self._stats = stats

    # -- risk review ----------------------------------------------------------
    async def review_risk(self, *, topic: str, markdown: str) -> SubAgentResult:
        """Review a rendered report; never raises, so the report always stands."""
        try:
            return await self._run_review(topic=topic, markdown=markdown)
        except Exception as exc:  # noqa: BLE001 - a failed review must not be fatal
            return SubAgentResult(
                focus=RISK_FOCUS,
                summary="",
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _run_review(self, *, topic: str, markdown: str) -> SubAgentResult:
        from finharness.context.session import ResearchContext
        from finharness.engine.loop import AgentLoop
        from finharness.permissions.gate import ReadOnlyGate
        from finharness.tools.registry import ToolRegistry, review_tool_names

        system = self._reviewer_system()
        # A derived settings object: tightening the reviewer's turn budget must
        # not touch the main loop's, and the reviewer shares the main cache dir.
        sub_settings = self.settings.model_copy(
            update={
                "context": self.settings.context.model_copy(
                    update={"max_turns": REVIEW_MAX_TURNS}
                )
            }
        )

        sub_cite = self.cite  # shared on purpose: cids stay continuous
        sub_ctx = ResearchContext(cite=sub_cite, settings=sub_settings)
        before = {item.cid for item in sub_cite.all()}

        sub_loop = AgentLoop(
            provider=self.provider,
            registry=ToolRegistry(
                self.data,
                ctx=sub_ctx,
                settings=sub_settings,
                only=set(review_tool_names()),
            ),
            settings=sub_settings,
            system=system,
            cite=sub_cite,
            ctx=sub_ctx,
            gate=ReadOnlyGate(),
            counter=self.counter,
            # No store: the review is an episode of its own, not conversation
            # memory. Its transcript dies with this call.
            store=None,
            session_id=f"{RISK_FOCUS}-review",
        )

        # Once the sub-loop exists, its recorded usage is the truth about what
        # this review cost — including a run that failed part-way after spending
        # tokens. So both the success and failure paths account from it, and only
        # a failure *before* the loop is built goes unbilled.
        try:
            outcome = await sub_loop.run(self._review_request(topic=topic, markdown=markdown))
        except Exception as exc:  # noqa: BLE001 - bill what was spent, then report
            result = self._result_from(sub_loop, before, ok=False, error=str(exc))
            self._account(result)
            return result

        result = self._result_from(
            sub_loop,
            before,
            ok=bool(outcome.succeeded),
            error=None if outcome.succeeded else (outcome.error or outcome.reason),
            summary=(outcome.answer or "").strip(),
        )
        self._account(result)
        return result

    def _result_from(
        self,
        sub_loop: Any,
        before: set[str],
        *,
        ok: bool,
        error: str | None,
        summary: str = "",
    ) -> SubAgentResult:
        """Build a result from the sub-loop's real counters, success or not."""
        return SubAgentResult(
            focus=RISK_FOCUS,
            summary=summary,
            citations=[
                item.cid for item in self.cite.all() if item.cid not in before
            ],
            input_tokens=int(getattr(sub_loop.usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(sub_loop.usage, "output_tokens", 0) or 0),
            turns=int(getattr(sub_loop, "turn", 0) or 0),
            ok=ok,
            error=error,
        )

    def _account(self, result: SubAgentResult) -> None:
        """Fold this run's tokens into the session totals and per-agent breakdown."""
        if self._usage is not None:
            self._usage.input_tokens += result.input_tokens
            self._usage.output_tokens += result.output_tokens
        if self._stats is not None:
            # add_usage moves the total; record_agent_usage only labels it.
            self._stats.add_usage(result.input_tokens, result.output_tokens)
            self._stats.record_agent_usage(
                result.focus, result.input_tokens, result.output_tokens
            )
        if self._on_usage is not None:
            self._on_usage(result.focus, result.input_tokens, result.output_tokens)

    # -- prompt assembly ------------------------------------------------------
    def _reviewer_system(self) -> str:
        """Role prompt plus the risk checklist, so the criteria are not restated.

        The checklist is the same asset the main agent uses to *write* the risk
        section; sharing it is what keeps the reviewer's criteria and the
        author's in step. A missing skill degrades to the role prompt alone
        rather than failing the review.
        """
        from finharness.engine.prompt import risk_review_prompt

        sections = [risk_review_prompt()]
        try:
            from finharness.tools.meta.skills import SkillRegistry

            registry = SkillRegistry(self.settings.paths.skills_dir)
            _meta, body, _record = registry.load(RISK_SKILL)
        except Exception:  # noqa: BLE001 - the review is still useful without it
            body = ""
        if body:
            sections.append("## 风险核查清单（判据来源）\n\n" + body)
        return "\n\n".join(sections)

    @staticmethod
    def _review_request(*, topic: str, markdown: str) -> str:
        return (
            f"请对以下研报做风险终审，主题：{topic}。\n"
            "报告正文如下（附录已略去）：\n\n"
            "<report>\n"
            f"{markdown}\n"
            "</report>\n\n"
            "按你的职责逐项核查并给出结论。"
        )

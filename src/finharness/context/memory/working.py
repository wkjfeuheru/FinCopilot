"""L1 working memory: the owner of the raw transcript and its token accounting.

Docs 03.6.4 (2): this is the only writer of the raw message list. ``AgentLoop``
keeps orchestration; the window maintenance (how full am I, must I compact,
replace a range with a digest) lives here so the loop does not grow a second
responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from finharness.config.settings import Settings
from finharness.context.tokens import TokenCounter
from finharness.types import Msg

# How many of the most recent rounds survive compaction verbatim.
KEEP_RECENT_ROUNDS = 2


@dataclass(frozen=True, slots=True)
class WindowUsage:
    """Two distinct numbers that must not be conflated.

    ``window_tokens`` is how full the next request would be — the figure
    compaction reduces. ``used_tokens`` is the session's cumulative spend, which
    compaction must NOT change, because it is the billing figure.
    """

    window_tokens: int
    used_tokens: int
    exact: bool


class WorkingMemory:
    """Raw transcript plus incremental token accounting."""

    def __init__(self, *, ctx=None, settings: Settings, counter: TokenCounter | None = None) -> None:
        self.settings = settings
        self.ctx = ctx
        self.counter = counter or TokenCounter(cache_dir=_token_cache_dir(settings))
        self.raw: list[Msg] = []
        self.used_tokens = 0
        self._exact = True

    # -- writes (the only mutation points) ------------------------------------
    def append(self, message: Msg) -> None:
        self.raw.append(message)
        self._account([message])

    def append_user(self, text: str) -> None:
        self.append(Msg.user(text))

    def append_assistant(self, message: Msg) -> None:
        self.append(message)

    def append_tool_result(self, call_id: str, content: str) -> None:
        self.append(Msg(role="tool_result", content=None, tool_results=[(call_id, content)]))

    # -- reads ----------------------------------------------------------------
    def snapshot(self) -> list[Msg]:
        """A read-only copy for the provider request."""
        return list(self.raw)

    def _account(self, messages: list[Msg]) -> None:
        for message in messages:
            for text in _message_texts(message):
                counted = self.counter.count(text)
                self.used_tokens += counted.tokens
                self._exact = self._exact and counted.exact

    def request_tokens(self, *, system: str, tools: list[dict]) -> int:
        """Estimated size of the next request (the number compaction targets).

        Measured rather than inferred: the provider ignores the ``usage`` object
        it is handed and reports its own totals afterwards, so the only way to
        know how full the window is *before* sending is to count here.
        """
        total = self.counter.count(system).tokens
        # _message_texts is a generator per message, so flatten before counting.
        for message in self.raw:
            total += self.counter.count_many(_message_texts(message))
        for schema in tools:
            function = schema.get("function", {})
            total += self.counter.count(str(function.get("description", ""))).tokens
            total += self.counter.count(str(function.get("parameters", ""))).tokens
        return total

    def usage(self, *, system: str, tools: list[dict]) -> WindowUsage:
        return WindowUsage(
            window_tokens=self.request_tokens(system=system, tools=tools),
            used_tokens=self.used_tokens,
            exact=self._exact,
        )

    def over_budget(self, *, system: str, tools: list[dict]) -> bool:
        window = self.settings.context.context_window_tokens
        if window <= 0:
            return False
        threshold = window * self.settings.context.compaction_ratio
        return self.request_tokens(system=system, tools=tools) >= threshold

    # -- window maintenance ---------------------------------------------------
    def squash(self, *, digest: Msg, keep_rounds: int = KEEP_RECENT_ROUNDS) -> int:
        """Replace everything but the last ``keep_rounds`` rounds with a digest.

        Returns how many messages were folded away. Cumulative ``used_tokens`` is
        deliberately left untouched: the tokens were still spent.
        """
        boundary = _recent_boundary(self.raw, keep_rounds)
        if boundary <= 0:
            return 0
        removed = boundary
        self.raw = [digest, *self.raw[boundary:]]
        return removed


def _recent_boundary(messages: list[Msg], keep_rounds: int) -> int:
    """Index where the last ``keep_rounds`` model exchanges begin.

    A "round" is one model round-trip: an assistant frame plus its tool results.
    Two consequences matter:

    * Counting *user* messages would find nothing foldable in a long research
      task (one question, many exchanges) — exactly the case compaction exists
      for.
    * The cut must land ON an assistant frame, never between an assistant frame
      and its tool results: a retained tool_result whose call_id was never
      announced by an assistant message is rejected by OpenAI-compatible APIs.
    """
    if keep_rounds <= 0:
        return len(messages)
    assistant_positions = [
        index for index, message in enumerate(messages) if message.role == "assistant"
    ]
    if len(assistant_positions) <= keep_rounds:
        return 0
    return assistant_positions[-keep_rounds]


def _message_texts(message: Msg):
    if message.content:
        yield message.content
    for tool_use in message.tool_uses:
        yield f"{tool_use.name}{tool_use.args}"
    for call_id, raw in message.tool_results:
        yield f"{call_id}{raw}"


def _token_cache_dir(settings: Settings) -> str:
    """Keep the tiktoken vocabulary beside the other caches."""
    return str(settings.data.cache_dir / "tiktoken")

import pytest

from finharness.engine.cost import SessionStats


class StepClock:
    """Deterministic clock advancing a fixed step per call."""

    def __init__(self, step_s: float = 0.25):
        self._value = 0.0
        self._step_s = step_s

    def __call__(self) -> float:
        value = self._value
        self._value += self._step_s
        return value


def test_session_stats_accumulates_usage_and_retries():
    stats = SessionStats()

    stats.add_usage(3, 5)
    stats.add_usage(4, 6)
    stats.add_retry()
    snapshot = stats.snapshot()

    assert snapshot.input_tokens == 7
    assert snapshot.output_tokens == 11
    assert snapshot.retry_count == 1


def test_session_stats_counts_every_request_but_times_only_real_runs():
    stats = SessionStats(clock=StepClock())

    stats.record_tool_request("missing_tool")
    stats.record_tool_request("get_quote")
    started_at = stats.now()
    duration_ms = stats.record_tool_duration("get_quote", started_at)

    snapshot = stats.snapshot()
    assert duration_ms == 250
    assert snapshot.tool_calls == 2
    assert snapshot.tool_duration_ms == 250
    assert dict(snapshot.per_tool) == {
        "missing_tool": {"count": 1, "duration_ms": 0},
        "get_quote": {"count": 1, "duration_ms": 250},
    }


def test_tool_durations_accumulate_per_tool_name():
    stats = SessionStats(clock=StepClock())

    first = stats.now()
    stats.record_tool_duration("get_quote", first)
    second = stats.now()
    stats.record_tool_duration("get_quote", second)

    snapshot = stats.snapshot()
    assert snapshot.per_tool["get_quote"] == {"count": 0, "duration_ms": 500}
    assert snapshot.tool_duration_ms == 500


def test_snapshot_is_detached_and_read_only():
    stats = SessionStats()
    stats.record_tool_request("get_quote")

    snapshot = stats.snapshot()
    stats.record_tool_request("get_quote")

    assert snapshot.tool_calls == 1
    with pytest.raises(TypeError):
        snapshot.per_tool["get_quote"]["count"] = 99

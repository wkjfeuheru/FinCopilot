"""Prometheus 指标：四类核心指标的标签与计数（docs 03.14.2）。"""

from finharness.observability.metrics import MetricsRecorder


def _sample(recorder: MetricsRecorder, prefix: str) -> dict[str, float]:
    """解析渲染结果，返回匹配前缀的样本（取最后一个样本名）。"""
    out: dict[str, float] = {}
    for line in recorder.render().decode("utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.rsplit(" ", 1)[0]
        value = float(line.rsplit(" ", 1)[1])
        if name.startswith(prefix):
            out[name] = value
    return out


def test_llm_tokens_are_counted_by_kind_and_call_type() -> None:
    recorder = MetricsRecorder()

    recorder.llm_finished(
        model="deepseek-chat", call_type="main", duration_s=1.0, first_token_s=0.2,
        input_tokens=100, output_tokens=20, cache_hit_tokens=80, cache_miss_tokens=20,
    )

    samples = _sample(recorder, "llm_tokens_total")
    assert samples['llm_tokens_total{call_type="main",kind="input",model="deepseek-chat"}'] == 100.0
    assert samples['llm_tokens_total{call_type="main",kind="output",model="deepseek-chat"}'] == 20.0
    assert samples['llm_tokens_total{call_type="main",kind="cache_hit",model="deepseek-chat"}'] == 80.0


def test_call_type_separates_compaction_and_subagent_cost() -> None:
    """压缩与子代理的开销不能被并进主循环，否则成本视图会失真。"""
    recorder = MetricsRecorder()

    recorder.llm_finished(model="m", call_type="compaction", duration_s=1.0, first_token_s=0.1, input_tokens=30)
    recorder.llm_finished(model="m", call_type="subagent", duration_s=1.0, first_token_s=0.1, input_tokens=50)

    samples = _sample(recorder, "llm_tokens_total")
    assert samples['llm_tokens_total{call_type="compaction",kind="input",model="m"}'] == 30.0
    assert samples['llm_tokens_total{call_type="subagent",kind="input",model="m"}'] == 50.0


def test_request_duration_histogram_records_status_label() -> None:
    recorder = MetricsRecorder()

    recorder.request_finished(status="ok", duration_s=1.5, model="m")
    recorder.request_finished(status="error", duration_s=0.5, model="m")

    samples = _sample(recorder, "agent_request_duration_seconds_count")
    assert samples['agent_request_duration_seconds_count{model="m",status="ok"}'] == 1.0
    assert samples['agent_request_duration_seconds_count{model="m",status="error"}'] == 1.0


def test_tool_calls_and_errors_are_labeled() -> None:
    recorder = MetricsRecorder()

    recorder.tool_finished(tool="get_quote", status="ok", duration_s=0.3)
    recorder.tool_finished(tool="get_quote", status="timeout", duration_s=30.0)
    recorder.error(error_type="tool_failure")

    calls = _sample(recorder, "agent_tool_calls_total")
    assert calls['agent_tool_calls_total{status="ok",tool="get_quote"}'] == 1.0
    assert calls['agent_tool_calls_total{status="timeout",tool="get_quote"}'] == 1.0
    assert _sample(recorder, "agent_request_errors_total")['agent_request_errors_total{error_type="tool_failure"}'] == 1.0


def test_each_recorder_has_its_own_registry() -> None:
    """自建 registry 使两次应用构建不会撞上重复注册。"""
    first = MetricsRecorder()
    second = MetricsRecorder()

    first.error(error_type="auth_error")

    assert first.registry is not second.registry
    assert _sample(second, "agent_request_errors_total") == {}

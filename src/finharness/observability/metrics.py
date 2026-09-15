"""Prometheus 指标（docs 03.14.2）。

四类核心指标对应图里的设计：

* ``agent_request_duration_seconds``——Histogram，看 P50/P95/P99 而非均值。
* ``agent_request_errors_total``——按 ``error_type`` 分类计数。
* ``llm_tokens_total``——按 ``model``/``kind``/``call_type`` 统计，成本直接来源。
* ``agent_tool_calls_total``——按 ``tool``/``status`` 统计调用量与成功率。

刻意使用**自建 ``CollectorRegistry``** 而非 prometheus_client 的全局默认
registry：测试会多次 ``create_app``，复用全局 registry 会在第二次注册时抛
``Duplicated timeseries``。自建 registry 让每个应用实例各有一份指标。
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

__all__ = ["MetricsRecorder"]

# 请求可能是分钟级（含多轮工具调用），因此桶要覆盖到 5 分钟。
_REQUEST_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0)
# LLM 单次调用通常几秒到几十秒。
_LLM_BUCKETS = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 120.0)
# 首 token 延迟决定体感，关注低端分布。
_FIRST_TOKEN_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0)
# 工具大多在亚秒到几十秒；数据抓取偶有分钟级。
_TOOL_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


class MetricsRecorder:
    """持有本应用实例的指标族，并暴露 ``Observer`` 需要的写入方法。"""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.request_duration = Histogram(
            "agent_request_duration_seconds",
            "一次用户请求（含全部模型轮次与工具调用）的耗时",
            labelnames=("model", "status"),
            buckets=_REQUEST_BUCKETS,
            registry=self.registry,
        )
        self.request_errors = Counter(
            "agent_request_errors_total",
            "按错误类型统计的请求失败数",
            labelnames=("error_type",),
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "llm_tokens_total",
            "按模型、token 种类与调用类型统计的 token 数",
            labelnames=("model", "kind", "call_type"),
            registry=self.registry,
        )
        self.tool_calls = Counter(
            "agent_tool_calls_total",
            "按工具名与结果状态统计的调用量",
            labelnames=("tool", "status"),
            registry=self.registry,
        )
        self.tool_duration = Histogram(
            "agent_tool_duration_seconds",
            "工具执行耗时",
            labelnames=("tool",),
            buckets=_TOOL_BUCKETS,
            registry=self.registry,
        )
        self.llm_duration = Histogram(
            "agent_llm_duration_seconds",
            "单次模型调用耗时",
            labelnames=("model", "call_type"),
            buckets=_LLM_BUCKETS,
            registry=self.registry,
        )
        self.llm_first_token = Histogram(
            "agent_llm_first_token_seconds",
            "首个 token 到达的延迟",
            labelnames=("model", "call_type"),
            buckets=_FIRST_TOKEN_BUCKETS,
            registry=self.registry,
        )

    # -- Observer 接口 ----------------------------------------------------------

    def request_finished(
        self, *, status: str, duration_s: float, model: str = ""
    ) -> None:
        self.request_duration.labels(model=model or "unknown", status=status).observe(duration_s)

    def llm_finished(
        self,
        *,
        model: str,
        call_type: str,
        duration_s: float,
        first_token_s: float | None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
    ) -> None:
        label = model or "unknown"
        self.llm_duration.labels(model=label, call_type=call_type).observe(duration_s)
        if first_token_s is not None:
            self.llm_first_token.labels(model=label, call_type=call_type).observe(first_token_s)
        if input_tokens:
            self.llm_tokens.labels(model=label, kind="input", call_type=call_type).inc(input_tokens)
        if output_tokens:
            self.llm_tokens.labels(model=label, kind="output", call_type=call_type).inc(output_tokens)
        if cache_hit_tokens:
            self.llm_tokens.labels(model=label, kind="cache_hit", call_type=call_type).inc(cache_hit_tokens)
        if cache_miss_tokens:
            self.llm_tokens.labels(model=label, kind="cache_miss", call_type=call_type).inc(cache_miss_tokens)

    def tool_finished(self, *, tool: str, status: str, duration_s: float) -> None:
        label = tool or "unknown"
        self.tool_calls.labels(tool=label, status=status).inc()
        self.tool_duration.labels(tool=label).observe(duration_s)

    def error(self, *, error_type: str) -> None:
        self.request_errors.labels(error_type=error_type or "unknown").inc()

    # -- 渲染 -------------------------------------------------------------------

    def render(self) -> bytes:
        """生成 Prometheus 文本格式的抓取内容。"""
        return generate_latest(self.registry)

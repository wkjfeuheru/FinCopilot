# Engine M1：重试与会话统计实施计划

## Summary

实现 Provider 流式重试和会话累计统计，同时保持现有 Agent Loop、SSE 与 Provider 调用接口兼容。

本阶段统计输入/输出 token、重试次数、工具请求次数和分工具耗时；不计算金额，`cost_cny` 保持 `0.0`。

## Global Constraints

- 不访问真实 Provider API；所有测试使用本地 Fake/脚本化 Provider 和工具替身。
- 首次请求之外最多重试 4 次，总请求上限 5 次。
- 只有当前尝试尚未产生任何归一化 chunk，且异常为可重试 ProviderError 时才重试。
- 已产生任意 chunk 后发生异常立即上抛；CancelledError、非 Provider 异常和不可重试错误立即传播。
- 仅成功收到 MESSAGE_END 后登记本次 usage；失败尝试不得虚构 token。
- 所有工具请求计入 tool_calls，只有真正进入 tool.run() 的调用计入耗时。
- cost_cny 固定 0.0；不实现真实成本、重试事件、stream.py 加固、Context、Gate、hooks、audit、citations。

## Task 1: Provider 错误契约与流重试器

扩展 `ProviderError(message, *, retryable=None, retry_after_s=None)`；RateLimitError/ServerError 默认可重试，AuthError 不可重试，NetworkError 由抛出位置指定。连接失败、连接重置和首字节超时可重试；idle timeout、非法 JSON、损坏 SSE、非法响应结构不可重试。429 解析 Retry-After，支持非负秒数和 HTTP 日期，过去日期归零，非法值忽略。OpenAI/Anthropic 保持异常类型并补元数据。

在 `engine/retry.py` 增加：

```python
@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 4
    base_delay_s: float = 1.0
    cap_delay_s: float = 30.0

async def stream_with_retry(stream_factory, *, policy, sleep=asyncio.sleep,
                           jitter=random.uniform, on_retry=None): ...
```

退避为 `min(cap, base * 2**retry_index) + jitter(0, base) + retry_after_s`；成功重试不发 error，on_retry 仅登记统计。

## Task 2: 会话统计与 Loop 集成

在 `engine/cost.py` 增加不可变 `SessionStatsSnapshot(input_tokens, output_tokens, retry_count, tool_calls, tool_duration_ms)` 与 `SessionStats`，提供 usage/retry/tool request 计数和实际工具执行的单调时钟计时；快照不可反向修改内部映射。

`AgentLoop` 增加可选 `retry_policy`、`stats`，通过 `stream_with_retry()` 调用 Provider；保留 `usage` 兼容入口。累计 retry、工具请求和分工具耗时；工具状态带 duration_ms，ToolResult(ok=False) 最终状态为 failed；未知/拒绝无耗时。`AgentTurnOutcome` 增加 `retry_count`、`tool_duration_ms`；done 增加同字段并保持 cost_cny=0.0。

## Task 3: 测试与兼容回归

新增 Provider 错误元数据、Retry-After、重试器、统计器和 Loop 集成测试；覆盖重试恢复/耗尽、零 chunk 边界、取消、累计统计、ok=False 工具失败和 SSE done 字段。运行 `uv run pytest tests/provider tests/engine tests/server -q`、`uv run pytest -q`、`git diff --check`。

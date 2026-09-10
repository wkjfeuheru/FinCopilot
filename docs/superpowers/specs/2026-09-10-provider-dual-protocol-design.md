# Provider 双协议流式适配设计

## 目标与范围

本阶段完成 Provider 层的离线可验证闭环：

- 保留并完善 DeepSeek 的 OpenAI 兼容流式调用；
- 将 Kimi 配置为 OpenAI 兼容端点（`https://api.moonshot.cn/v1`）；
- 将 GLM 配置为 OpenAI 兼容端点（`https://open.bigmodel.cn/api/paas/v4`）；
- 新增通用 `AnthropicCompatProvider`，用于任何支持 `/v1/messages` 的端点；
- 统一 SSE、文本增量、工具调用增量、结束事件、超时与错误语义；
- 补齐本地 SSE 回放测试。

本阶段不实现重试策略、非流式摘要接口、价格计算、真实厂商冒烟测试，也不改动 `AgentLoop` 的业务循环。

## 已裁定的设计决策

1. Provider 配置同时预置 `deepseek` 与 `kimi`，默认仍为 `deepseek`。
2. Kimi 使用当前官方公开的 OpenAI 兼容接口；`AnthropicCompatProvider` 不绑定未验证的 Kimi URL。
3. 每一次成功调用必须且只能产出一个 `MESSAGE_END`。缺失 usage 时使用零值 usage，仍完成工具调用归并。
4. 首个 SSE 字节超时默认 30 秒，后续流空闲超时默认 60 秒，均可配置。
5. 工具调用增量使用厂商无关的 `ToolUseDelta`，不暴露 OpenAI 或 Anthropic 原始事件结构。
6. SSE 或 HTTP 中可识别的错误直接抛出 `ProviderError` 子类；暂不输出当前引擎不会处理的 `StreamEvent.ERROR`。
7. `anthropic-version` 为配置项，默认 `2023-06-01`。
8. `temperature`、`max_tokens` 由 `ModelSettings` 单一管理，两个协议均透传。
9. 单测全部使用本地 `httpx.MockTransport` 回放；真实厂商调用留到 E2E/演示验收。
10. GLM 复用 `OpenAICompatProvider`，使用 `ZHIPU_API_KEY`；模型名由全局 `model.model_name` 选择，不改变默认模型。

## 架构

采用“共享流式底座、协议适配留在具体 Provider”的结构：

```
Settings + build_provider
        |
        +-- OpenAICompatProvider ---- 请求/响应格式转换 ----+
        |                                                  |
        +-- AnthropicCompatProvider - 请求/响应格式转换 ----+--> event_stream.py
                                                               SSE 分帧、超时、工具调用累积
                                                               |
                                                               +--> StreamChunk
```

不采用每个 Provider 独立解析 SSE 的方案，以避免超时、结束和工具归并行为分叉；也不引入声明式厂商 DSL，避免当前两协议场景的过度抽象。

## 数据契约

在 `types.py` 增加 `ToolUseDelta`：

```python
@dataclass(slots=True)
class ToolUseDelta:
    index: int
    call_id: str | None = None
    name_delta: str = ""
    arguments_delta: str = ""
```

Provider 成功流的事件顺序为零或多个 `TEXT_DELTA` / `TOOL_USE_DELTA`，随后唯一一个 `MESSAGE_END`。结束事件携带完整的 `ModelUsage`，其中 `tool_uses` 是已解析完成的 `ToolUse` 列表。

截断 SSE、非法 JSON、未闭合或无法解析的工具参数均是协议错误，抛出带有协议上下文的 `NetworkError`，不生成不完整的工具调用。

## 配置与 Provider 构建

`ProviderSettings` 增加以下字段：

- `api_version: str | None`；仅 Anthropic 兼容端使用，默认 `2023-06-01`；
- `first_byte_timeout_s: float = 30`；
- `idle_timeout_s: float = 60`。

`ModelSettings` 的 `model_name`、`temperature` 与 `max_tokens` 会传给构造器。`build_provider()` 根据 `kind` 创建 `OpenAICompatProvider` 或 `AnthropicCompatProvider`；未知类型、缺失 Provider、非法 URL 或缺失密钥都在启动前报出 `SettingsError`。

`settings.example.json` 展示 `deepseek`、`kimi` 和 `glm` 三项。Kimi 的 `kind` 为 `openai_compat`、环境变量为 `MOONSHOT_API_KEY`，而非 Anthropic 兼容类型；GLM 同样为 `openai_compat`，使用 `https://open.bigmodel.cn/api/paas/v4` 和 `ZHIPU_API_KEY`。选择 GLM 时，用户将 `model.provider` 改为 `glm`，并按需将 `model.model_name` 设为例如 `glm-5.3-flash`。

## 协议转换

### OpenAI 兼容端

- `system` 作为首条 `messages` 消息；
- `tool_result` 转为 `role: tool` 和 `tool_call_id`；
- assistant 工具调用转为 `tool_calls`；
- 工具 schema 直接使用当前 OpenAI function 格式；
- 请求携带 `model`、`temperature`、`max_tokens`、`stream: true`，并请求 usage；
- 解析 `delta.content`、`delta.tool_calls`、usage 和 `[DONE]`。

### Anthropic 兼容端

- 请求目标为 `{base_url}/messages`；
- 头部为 `x-api-key` 与 `anthropic-version`；
- `system` 放在请求顶层；
- assistant 的文本和 `tool_use` 还原为 content blocks；
- `tool_result` 合并为 user 的 `tool_result` content blocks；
- OpenAI function schema 转为 Anthropic 的 `name`、`description`、`input_schema`；
- 解析 `content_block_start`、`content_block_delta` 与 `message_delta`。

## 流式底座与错误

`provider/event_stream.py` 负责解析标准 SSE 字段（含多行 `data:`），从原始字节流施加首字节与空闲超时，并维护每一个工具调用的 ID、名称和 JSON 参数缓冲区。

HTTP 401/403 映射为 `AuthError`，429 映射为 `RateLimitError`，5xx 映射为 `ServerError`；连接失败、超时、断连与协议损坏映射为 `NetworkError`。Provider 不负责重试，由后续 engine 重试层消费这些异常类型。

## 测试与验收

新增或扩展 `tests/provider/`，全部使用 `httpx.MockTransport`：

- 两个协议的请求结构、system 消息、工具结果、工具 schema、生成参数；
- 文本 SSE、分片工具参数、标准化 `ToolUseDelta` 和完整 `ToolUse`；
- OpenAI `[DONE]` 未带 usage 时的唯一 `MESSAGE_END`；
- Anthropic 文本与工具调用事件映射；
- 401、403、429、5xx、SSE error、非法 JSON、截断流、首字节和空闲超时；
- Provider 注册、DeepSeek/Kimi/GLM 选择和配置/密钥校验。

验收命令：

```powershell
uv run pytest tests/provider
uv run pytest
```

两条命令均须通过，且不要求任何真实厂商 API Key。

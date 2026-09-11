# Provider 双协议流式适配 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 DeepSeek、Kimi、GLM 提供统一 OpenAI 兼容流式调用，并新增通用 Anthropic `/v1/messages` 适配器，使上层只消费稳定的 `StreamChunk` 和 `ToolUse`。

**Architecture:** `provider/event_stream.py` 负责 SSE 分帧、超时和分片工具调用 JSON 累积；具体 Provider 只转换请求与厂商事件。`Settings` 管理端点和超时，`ModelSettings` 是模型参数唯一来源，`build_provider()` 按 `kind` 注入实现。

**Tech Stack:** Python 3.13、httpx、asyncio、dataclass、pytest、httpx.MockTransport。

## Global Constraints

- 保持 `Provider.stream(system, messages, tools, usage)` 与 `AgentLoop` 不变。
- 每个成功流必须且只能输出一个 `StreamEvent.MESSAGE_END`；缺失 usage 时 token 为零。
- Provider 仅抛出 `AuthError`、`RateLimitError`、`ServerError`、`NetworkError`；不输出 `StreamEvent.ERROR`。
- 首帧超时默认 30 秒，后续空闲超时默认 60 秒，均由 `ProviderSettings` 配置。
- 测试使用 `httpx.MockTransport` 和本地异步迭代器，不访问真实厂商 API。
- Kimi 和 GLM 均使用 `openai_compat`；GLM 使用 `https://open.bigmodel.cn/api/paas/v4` 与 `ZHIPU_API_KEY`。
- `anthropic-version` 由 `ProviderSettings.api_version` 提供，默认 `2023-06-01`。
- 除代码标识符和专业词汇外，用户可见文本、注释、文档使用中文。

---

## 文件职责图

| 文件 | 变更 | 职责 |
|---|---|---|
| `src/finharness/types.py` | 修改 | 厂商无关的 `ToolUseDelta`。 |
| `src/finharness/provider/event_stream.py` | 新建 | SSE 分帧、读超时、工具调用累积和 JSON 验证。 |
| `src/finharness/config/settings.py` | 修改 | 端点、密钥、协议版本、超时的加载和校验。 |
| `src/finharness/provider/registry.py` | 修改 | 依 `kind` 组装 Provider 并注入模型参数。 |
| `src/finharness/provider/openai_compat.py` | 修改 | OpenAI 请求和事件标准化。 |
| `src/finharness/provider/anthropic_compat.py` | 新建 | Anthropic Messages 请求和事件标准化。 |
| `settings.example.json` | 修改 | DeepSeek、Kimi、GLM 的安全样例。 |
| `tests/provider/test_event_stream.py` | 新建 | 共享 SSE 和工具累积器。 |
| `tests/provider/test_openai_compat.py` | 修改 | OpenAI 离线 SSE 回放。 |
| `tests/provider/test_anthropic_compat.py` | 新建 | Anthropic 离线 SSE 回放。 |
| `tests/provider/test_registry.py` | 修改 | 三厂商选择与注册错误。 |
| `tests/config/test_settings.py` | 修改 | 扩展设置加载和校验。 |

## Task 1: 共享事件与工具调用累积器

**Files:**

- Modify: `src/finharness/types.py`
- Create: `src/finharness/provider/event_stream.py`
- Create: `tests/provider/test_event_stream.py`

**Interfaces:**

- Produces: `ToolUseDelta(index: int, call_id: str | None = None, name_delta: str = "", arguments_delta: str = "")`。
- Produces: `async def iter_sse_data(lines: AsyncIterator[str], *, first_byte_timeout_s: float, idle_timeout_s: float) -> AsyncIterator[str]`。
- Produces: `ToolUseAccumulator.add(delta: ToolUseDelta) -> None`、`ToolUseAccumulator.replace_arguments(index: int, arguments: str = "") -> None` 与 `ToolUseAccumulator.build() -> list[ToolUse]`。

- [ ] **Step 1: 写失败测试**

```python
def test_tool_use_accumulator_merges_fragmented_arguments():
    accumulator = ToolUseAccumulator()
    accumulator.add(ToolUseDelta(index=0, call_id="call_1", name_delta="get_"))
    accumulator.add(ToolUseDelta(index=0, name_delta="quote", arguments_delta='{"symbol":"'))
    accumulator.add(ToolUseDelta(index=0, arguments_delta='600519"}'))

    assert accumulator.build() == [ToolUse("call_1", "get_quote", {"symbol": "600519"})]

def test_tool_use_accumulator_rejects_invalid_json():
    accumulator = ToolUseAccumulator()
    accumulator.add(ToolUseDelta(index=0, call_id="call_1", name_delta="get_quote", arguments_delta="{"))
    with pytest.raises(NetworkError, match="tool arguments"):
        accumulator.build()

def test_tool_use_accumulator_can_replace_initial_arguments():
    accumulator = ToolUseAccumulator()
    accumulator.add(ToolUseDelta(index=0, call_id="call_1", name_delta="get_quote", arguments_delta="{}"))
    accumulator.replace_arguments(0)
    accumulator.add(ToolUseDelta(index=0, arguments_delta='{"symbol":"600519"}'))
    assert accumulator.build()[0].args == {"symbol": "600519"}
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest tests/provider/test_event_stream.py -v`

Expected: FAIL，提示共享模块或类型尚不存在。

- [ ] **Step 3: 实现共享类型和累积器**

在 `types.py` 的 `ToolUse` 后加入：

```python
@dataclass(slots=True)
class ToolUseDelta:
    index: int
    call_id: str | None = None
    name_delta: str = ""
    arguments_delta: str = ""
```

`ToolUseAccumulator` 按 `index` 保存 ID、名称和参数缓冲。`add()` 仅拼接非空值；`replace_arguments()` 清空或设置指定 index 的参数缓冲，用于 Anthropic 的初始 `input` 与后续 `input_json_delta` 不能拼接的情形；`build()` 按 index 排序，要求 ID/名称存在，以 `json.loads(arguments or "{}")` 解析且要求结果为 `dict`，否则抛出 `NetworkError("Provider returned invalid tool arguments")`。

- [ ] **Step 4: 写 SSE 分帧与超时测试**

```python
async def test_iter_sse_data_merges_multiline_data_and_ignores_comments():
    async def lines():
        for line in (": keepalive", "data: {\"a\":", "data: 1}", ""):
            yield line
    assert [value async for value in iter_sse_data(lines(), first_byte_timeout_s=1, idle_timeout_s=1)] == ['{"a":\n1}']

async def test_iter_sse_data_applies_first_byte_timeout():
    async def lines():
        await asyncio.sleep(0.02)
        yield "data: late"
    with pytest.raises(NetworkError, match="first response byte"):
        async for _ in iter_sse_data(lines(), first_byte_timeout_s=0.001, idle_timeout_s=1):
            pass
```

- [ ] **Step 5: 实现分帧与超时**

每次用 `await anext(lines)` 读一行；第一次包在 `asyncio.timeout(first_byte_timeout_s)`，后续包在 `asyncio.timeout(idle_timeout_s)`。超时转换为分别含 `first response byte` 或 `stream idle` 的 `NetworkError`。忽略 `:` 注释和未知字段；连续 `data:` 行以 `"\n".join()` 合并，空行产生一个事件，EOF 时提交未完成帧。

- [ ] **Step 6: 验证并提交**

Run: `uv run pytest tests/provider/test_event_stream.py -v`

Expected: PASS。

```powershell
git add src/finharness/types.py src/finharness/provider/event_stream.py tests/provider/test_event_stream.py
git commit -m "feat: add normalized provider stream helpers"
```

## Task 2: Provider 配置与注册

**Files:**

- Modify: `src/finharness/config/settings.py`
- Modify: `src/finharness/provider/registry.py`
- Modify: `settings.example.json`
- Modify: `tests/config/test_settings.py`
- Modify: `tests/provider/test_registry.py`

**Interfaces:**

- Produces: `ProviderSettings(kind, base_url, env_key, api_version, first_byte_timeout_s, idle_timeout_s)`。
- Produces: `build_provider(path, *, client=None) -> Provider`，向具体 Provider 注入模型名、温度、最大 token 和超时。

- [ ] **Step 1: 写 GLM 和设置校验的失败测试**

```python
def test_settings_loads_glm_provider(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"model": {"provider": "glm", "model_name": "glm-5.3-flash"}, "providers": {
        "glm": {"kind": "openai_compat", "base_url": "https://open.bigmodel.cn/api/paas/v4", "env_key": "ZHIPU_API_KEY"}
    }}), encoding="utf-8")
    settings = Settings.from_file(path)
    assert settings.model.provider == "glm"
    assert settings.providers["glm"].first_byte_timeout_s == 30.0

def test_build_provider_constructs_glm(monkeypatch, tmp_path):
    monkeypatch.setenv("ZHIPU_API_KEY", "secret")
    provider = build_provider(write_settings(tmp_path, provider="glm"), client=httpx.AsyncClient())
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.base_url == "https://open.bigmodel.cn/api/paas/v4"
```

- [ ] **Step 2: 运行并确认失败**

Run: `uv run pytest tests/config/test_settings.py tests/provider/test_registry.py -v`

Expected: FAIL，当前加载器只对 DeepSeek 预置和解析。

- [ ] **Step 3: 实现通用设置加载和校验**

扩展为：

```python
@dataclass(slots=True)
class ProviderSettings:
    kind: str = "openai_compat"
    base_url: str = ""
    env_key: str = ""
    api_version: str | None = None
    first_byte_timeout_s: float = 30.0
    idle_timeout_s: float = 60.0
```

默认 providers 包含 DeepSeek、Kimi、GLM；文件中的每个 provider 与默认值合并。`validate()` 校验所选 provider 存在、kind 属于 `openai_compat` 或 `anthropic_compat`、URL 是绝对 HTTP(S) URL、两个超时大于零；仅在 `require_api_key=True` 时校验选中 Provider 的环境变量。

- [ ] **Step 4: 实现注册器和样例配置**

`build_provider()` 按 `kind` 创建 OpenAI 或 Anthropic Provider，均传入：

```python
model=settings.model.model_name,
temperature=settings.model.temperature,
max_tokens=settings.model.max_tokens,
first_byte_timeout_s=config.first_byte_timeout_s,
idle_timeout_s=config.idle_timeout_s,
```

Anthropic 额外使用 `api_version=config.api_version or "2023-06-01"`。在样例中加入 Kimi（`MOONSHOT_API_KEY`）和 GLM（`ZHIPU_API_KEY`）条目及各自端点。

- [ ] **Step 5: 扩展负例并验证**

增加未知 provider/kind、非法 URL、非正超时、缺失 `ZHIPU_API_KEY` 和 `anthropic_compat` 组装测试。

Run: `uv run pytest tests/config/test_settings.py tests/provider/test_registry.py -v`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/finharness/config/settings.py src/finharness/provider/registry.py settings.example.json tests/config/test_settings.py tests/provider/test_registry.py
git commit -m "feat: configure selectable model providers"
```

## Task 3: 完善 OpenAI 兼容流式 Provider

**Files:**

- Modify: `src/finharness/provider/openai_compat.py`
- Modify: `tests/provider/test_openai_compat.py`

**Interfaces:**

- Consumes: `ToolUseDelta`、`ToolUseAccumulator`、`iter_sse_data` 和 Task 2 注入的模型/超时参数。
- Produces: `OpenAICompatProvider(..., model: str, temperature: float, max_tokens: int, first_byte_timeout_s: float = 30, idle_timeout_s: float = 60)`。
- Produces: 文本产生 `StreamChunk(TEXT_DELTA, str)`，工具分片产生 `StreamChunk(TOOL_USE_DELTA, ToolUseDelta)`，以唯一 `StreamChunk(MESSAGE_END, ModelUsage)` 结束。

- [ ] **Step 1: 写请求体和分片工具调用的失败测试**

```python
def test_openai_provider_passes_model_parameters_and_normalizes_tool_delta():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["temperature"] == 0.25
        assert payload["max_tokens"] == 512
        return sse_response([
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_", "arguments": "{\"symbol\":\""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "quote", "arguments": "600519\"}"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ])

    chunks = asyncio.run(collect_from(handler))
    assert chunks[0].data == ToolUseDelta(index=0, call_id="call_1", name_delta="get_", arguments_delta='{"symbol":"')
    assert chunks[-1].data.tool_uses == [ToolUse("call_1", "get_quote", {"symbol": "600519"})]
```

- [ ] **Step 2: 写无 usage 结束和 HTTP 错误测试**

```python
def test_openai_done_without_usage_emits_one_message_end():
    chunks = asyncio.run(collect_sse(["data: [DONE]", ""]))
    assert [chunk.event for chunk in chunks] == [StreamEvent.MESSAGE_END]
    assert chunks[0].data.input_tokens == 0

@pytest.mark.parametrize(("status", "error_type"), [(401, AuthError), (403, AuthError), (429, RateLimitError), (500, ServerError)])
def test_openai_http_status_is_classified(status, error_type):
    with pytest.raises(error_type):
        asyncio.run(collect_status(status))
```

- [ ] **Step 3: 运行并确认失败**

Run: `uv run pytest tests/provider/test_openai_compat.py -v`

Expected: FAIL，构造函数、标准化增量和 `[DONE]` 契约尚未满足。

- [ ] **Step 4: 重构请求转换与流事件循环**

保留首条 `system`、assistant `tool_calls` 和 tool result 的 `role: tool`。请求体必须含 `model`、`temperature`、`max_tokens`、`stream: true`、`stream_options: {"include_usage": true}`。逐个消费 `iter_sse_data(response.aiter_lines(), ...)`：解析 JSON，文本立即 yield；每个工具调用转为 `ToolUseDelta`、加入累积器、立即 yield；保存 usage。收到 `[DONE]` 或正常 EOF 后只 yield 一次 `MESSAGE_END(ModelUsage(..., tool_uses=accumulator.build()))`。

`error` 字段、无 choices 的错误 payload、非法 JSON 均为 `NetworkError`，但 error code/type 含 `rate_limit` 时映射 `RateLimitError`，含 `authentication`、`permission`、`api_key` 时映射 `AuthError`，含 `server` 时映射 `ServerError`；HTTP 401/403、429、5xx 同样分别映射现有三类异常；现有 `httpx.HTTPError` 包装为 `NetworkError`。

- [ ] **Step 5: 验证并提交**

Run: `uv run pytest tests/provider/test_openai_compat.py -v`

Expected: PASS，包含旧文本/usage 用例和新增工具、无 usage、HTTP 错误用例。

```powershell
git add src/finharness/provider/openai_compat.py tests/provider/test_openai_compat.py
git commit -m "feat: normalize OpenAI compatible streams"
```

## Task 4: 新增 Anthropic Messages 兼容 Provider

**Files:**

- Create: `src/finharness/provider/anthropic_compat.py`
- Create: `tests/provider/test_anthropic_compat.py`
- Modify: `src/finharness/provider/__init__.py`

**Interfaces:**

- Consumes: Task 1 流底座和 Task 2 注册器注入的配置。
- Produces: `AnthropicCompatProvider`，遵循与 `OpenAICompatProvider` 相同的 `Provider.stream()` 输出契约。

- [ ] **Step 1: 写请求转换失败测试**

```python
def test_anthropic_provider_converts_system_tools_and_tool_results():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "secret"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body["system"] == "系统提示"
        assert body["tools"] == [{"name": "get_quote", "description": "报价", "input_schema": {"type": "object"}}]
        assert body["messages"][-1]["content"] == [{"type": "tool_result", "tool_use_id": "call_1", "content": "600519: 100"}]
        return sse_response([{"type": "message_delta", "usage": {"input_tokens": 3, "output_tokens": 2}}])
```

- [ ] **Step 2: 写文本、工具及错误事件失败测试**

```python
def test_anthropic_provider_normalizes_text_and_tool_use():
    chunks = asyncio.run(collect_sse_events([
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_quote", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"symbol":"600519"}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"input_tokens": 3, "output_tokens": 2}},
    ]))
    assert chunks[0].data == ToolUseDelta(index=0, call_id="toolu_1", name_delta="get_quote")
    assert chunks[-1].data.tool_uses == [ToolUse("toolu_1", "get_quote", {"symbol": "600519"})]

def test_anthropic_error_event_raises_classified_error():
    with pytest.raises(RateLimitError):
        asyncio.run(collect_sse_events([{"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}]))
```

- [ ] **Step 3: 运行并确认失败**

Run: `uv run pytest tests/provider/test_anthropic_compat.py -v`

Expected: FAIL，目标模块不存在。

- [ ] **Step 4: 实现消息和工具 schema 转换**

请求为 `POST {base_url.rstrip('/')}/messages`，headers 使用 `x-api-key`、`anthropic-version`、`content-type: application/json`。顶层设置 `model`、`system`、`temperature`、`max_tokens`、`stream: true`、`messages`、`tools`。

每个 OpenAI function schema 转为：

```python
{
    "name": tool["function"]["name"],
    "description": tool["function"].get("description", ""),
    "input_schema": tool["function"]["parameters"],
}
```

普通 user/assistant 文本为 `{role, content}`。含 `tool_uses` 的 assistant 用文本 block（内容非空时）加 `{type: "tool_use", id, name, input}` block；`tool_result` 合并为一个 user message，其 content 是 `{type: "tool_result", tool_use_id, content}` block 列表。

- [ ] **Step 5: 实现 SSE 映射与错误分类**

用 `iter_sse_data` 读取 JSON。`content_block_start` 的 tool_use 向累积器加入完整 ID、名称与初始 input，并 yield `ToolUseDelta`；若该调用首次收到 `input_json_delta`，先调用 `replace_arguments(index)` 清除初始 input，再追加 `partial_json` 并 yield 仅含参数片段的增量，避免初始 `{}` 与 JSON 片段拼成无效 JSON。`text_delta` yield 文本；在 `message_start` 或 `message_delta` 出现 usage 时更新已知字段。`error.type` 含 `rate_limit` 时抛 `RateLimitError`，含 `authentication` 或 `permission` 时抛 `AuthError`，含 `server` 时抛 `ServerError`，其他错误抛 `NetworkError`。正常 `message_stop` 或 EOF 后唯一输出 `MESSAGE_END`。

- [ ] **Step 6: 验证并提交**

Run: `uv run pytest tests/provider/test_anthropic_compat.py -v`

Expected: PASS。

```powershell
git add src/finharness/provider/anthropic_compat.py src/finharness/provider/__init__.py tests/provider/test_anthropic_compat.py
git commit -m "feat: add Anthropic compatible provider"
```

## Task 5: 集成回归与文档验收

**Files:**

- Modify: `README.md`
- Modify: `docs/modules/03.2-provider.md`
- Modify: `tests/provider/test_registry.py`
- Modify: `tests/provider/test_fake.py`（仅当 Task 1 的类型变更影响现有测试时）

**Interfaces:**

- Consumes: Tasks 1–4 的稳定接口。
- Produces: 对三厂商、两协议、离线测试边界的一致说明。

- [ ] **Step 1: 写三厂商预置回归测试**

```python
@pytest.mark.parametrize(("provider_name", "env_key", "expected_url"), [
    ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    ("kimi", "MOONSHOT_API_KEY", "https://api.moonshot.cn/v1"),
    ("glm", "ZHIPU_API_KEY", "https://open.bigmodel.cn/api/paas/v4"),
])
def test_openai_compatible_provider_presets(monkeypatch, tmp_path, provider_name, env_key, expected_url):
    monkeypatch.setenv(env_key, "secret")
    provider = build_provider(write_preset_settings(tmp_path, provider_name), client=httpx.AsyncClient())
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.base_url == expected_url
```

- [ ] **Step 2: 运行 Provider 测试集**

Run: `uv run pytest tests/provider -v`

Expected: PASS；不访问网络、不要求真实密钥、没有未处理协程警告。

- [ ] **Step 3: 更新用户与模块文档**

`README.md` 明确：复制 `settings.example.json` 后通过 `model.provider` 选择 `deepseek`、`kimi`、`glm`，通过 `model.model_name` 选择模型，密钥仅由环境变量提供。说明 Anthropic 兼容端是用户可配置的通用端点，日常测试不调用真实 API。

`docs/modules/03.2-provider.md` 记录实际实现状态：OpenAI 兼容端覆盖 DeepSeek/Kimi/GLM；加入 AnthropicCompatProvider、ToolUseDelta、唯一 MESSAGE_END、30/60 秒超时、错误映射。不要宣称重试、非流式 API 或真实厂商测试已实现。

- [ ] **Step 4: 运行全量验证**

Run: `uv run pytest`

Expected: PASS。若失败，修复受 Provider 变更影响的兼容性后重新运行；不跳过失败用例。

- [ ] **Step 5: 检查差异并提交**

Run: `git diff --check`

Expected: 无输出、退出码 0。

```powershell
git add README.md docs/modules/03.2-provider.md tests/provider/test_registry.py tests/provider/test_fake.py
git commit -m "docs: document supported model providers"
```

## 计划自检

| 设计要求 | 对应任务 |
|---|---|
| `ToolUseDelta` 与完整 `ToolUse` | Task 1 |
| SSE 分帧、首帧/空闲超时 | Task 1 |
| DeepSeek/Kimi/GLM 配置、选择与校验 | Task 2、Task 5 |
| 模型参数单一来源 | Task 2、Task 3、Task 4 |
| OpenAI 完成语义与异常分类 | Task 3 |
| Anthropic 请求/响应与版本头 | Task 4 |
| 离线回放和全量回归 | Tasks 3–5 |
| 明确排除重试、非流式与真实 API | Global Constraints、Task 5 |

占位符扫描：无 `TODO`、`TBD` 或未定义的后续工作；每个任务均包含具体接口、测试、命令和提交范围。

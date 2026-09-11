# Task 1 报告：共享事件与工具调用累积器

## 变更文件

- `src/finharness/types.py`：新增 `ToolUseDelta` 数据契约。
- `src/finharness/provider/event_stream.py`：新增 `ToolUseAccumulator` 与 `iter_sse_data`。
- `tests/provider/test_event_stream.py`：新增离线行为测试，覆盖参数片段累积、参数替换、无效 JSON、SSE 多行聚合、注释忽略和首字节超时。

## RED 证据

执行：

```text
uv --cache-dir .superpowers/uv-cache run --offline pytest tests/provider/test_event_stream.py -v
```

结果：测试收集失败，`ModuleNotFoundError: No module named 'finharness.provider.event_stream'`。该失败发生在生产模块创建前，符合预期。

## GREEN 证据

再次执行同一命令：

```text
5 passed in 0.03s
```

全量离线回归：

```text
uv --cache-dir .superpowers/uv-cache run --offline pytest -q
38 passed, 2 warnings in 0.97s
```

警告为现有 FastAPI/Starlette 与 httpx 兼容性弃用警告，与本任务无关。

## 自检

- 累积器按 `index` 保存缓冲区，非空 `call_id`、名称和参数片段才追加。
- `replace_arguments` 会替换指定索引的参数缓冲区。
- `build` 按索引排序，要求调用 ID 与名称，解析 `arguments or "{}"`，并拒绝无效 JSON 或非对象参数，统一抛出 `NetworkError("Provider returned invalid tool arguments")`。
- SSE 逐行 `anext`，首读和后续读分别使用首字节/空闲超时；注释与未知字段忽略；空行提交事件；EOF 提交未完成帧。
- 未修改 `Provider.stream` 或 `AgentLoop`。

## Commit 状态

已尝试限定范围的提交：

```text
git add src/finharness/types.py src/finharness/provider/event_stream.py tests/provider/test_event_stream.py
git commit -m "feat: add provider event stream primitives"
```

失败：`fatal: Unable to create 'F:/python project/FinCopilot/.git/index.lock': Permission denied`。按要求未采取绕过措施。

## 审查补充（2026-09-11）

新增离线测试覆盖以下边界：

- 连接在首行后空闲时抛出包含 `stream idle` 的 `NetworkError`。
- EOF 时提交未以空行结束的 `data` 帧。
- `id`、`retry` 等未知 SSE 字段不影响 `data` 结果。
- 空参数解析为 `{}`、工具调用按 `index` 排序、缺少调用 ID 或名称时拒绝、非对象 JSON 参数时拒绝。

新增测试先在当前实现上运行。它们均通过，表明这些已实现的行为此前仅缺测试覆盖，生产代码无需修改。`iter_sse_data` 的超时边界已复查：第一次且仅第一次 `anext(lines)` 使用 `first_byte_timeout_s`，之后的每次读取使用 `idle_timeout_s`；注释或未知字段的首行不会改变该语义。

执行：

```text
uv --cache-dir .superpowers/uv-cache run --offline pytest tests/provider/test_event_stream.py -v
```

结果：`12 passed in 0.06s`。

随后再次进行限定范围提交，成功创建：

```text
6fb869e test: cover event stream edge cases
```

本报告目录受项目 `.gitignore` 忽略，因此将以强制添加报告文件的限定提交保存最终证据。

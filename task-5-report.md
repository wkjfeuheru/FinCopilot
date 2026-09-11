# Task 5 集成回归与文档验收报告

## 范围

仅修改 README、provider 模块文档、provider registry 回归测试及本报告；未调用真实厂商 API。

## 回归测试

- 新增 `test_build_provider_assembles_openai_compatible_presets` 参数化用例，覆盖：
  - deepseek → `https://api.deepseek.com/v1` / `DEEPSEEK_API_KEY`
  - kimi → `https://api.moonshot.cn/v1` / `MOONSHOT_API_KEY`
  - glm → `https://open.bigmodel.cn/api/paas/v4` / `ZHIPU_API_KEY`
- 测试使用 `httpx.MockTransport`，显式关闭 `AsyncClient`，无协程告警。
- TDD RED 证据：临时将 kimi 期望 URL 改为 `/v1.invalid` 后运行该测试，结果 `1 failed, 2 passed`；随后恢复正确期望并通过。

## 验收命令

```text
uv run pytest tests/provider --basetemp .tmp/base -q
84 passed in 0.47s

uv run pytest --basetemp .tmp/base -q
118 passed, 2 warnings in 1.86s

git diff --check
exit 0
```

全量测试的 2 条 warning 来自依赖自身的 Starlette/httpx 与 anyio 弃用提示，不是 coroutine warning。

## 文档验收

README 已说明复制 `settings.example.json`、选择 `model.provider`/`model.model_name`、仅通过环境变量提供密钥、可配置 Anthropic 兼容端点及离线测试约束。provider 文档已描述实际的 DeepSeek/Kimi/GLM OpenAI-compatible、AnthropicCompatProvider、ToolUseDelta、恰好一次成功 `MESSAGE_END`、30/60 秒超时和错误映射；未宣称 retries、非流式 API 或真实厂商测试。

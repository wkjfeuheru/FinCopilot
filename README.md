# FinHarness

FinHarness 是基于 FastAPI 后端和 React 前端的金融研究 copilot。

## 配置与开发

先复制 `settings.example.json` 为 `settings.json`，然后在 `model.provider` 中选择 `deepseek`、`kimi` 或 `glm`，并在 `model.model_name` 中填写目标模型名。密钥只通过环境变量提供：

- DeepSeek：`DEEPSEEK_API_KEY`
- Kimi：`MOONSHOT_API_KEY`
- GLM：`ZHIPU_API_KEY`

Anthropic 兼容端点可在 `providers` 中配置，`kind` 设置为 `anthropic_compat` 并指定 `base_url`、`env_key` 等字段。缺少密钥不会自动回退到其他厂商；离线测试显式注入 `FakeProvider` 或使用 HTTP mock。

```powershell
uv sync --extra dev
uv run uvicorn finharness.server.api:create_production_app --factory --reload --port 8000
cd frontend
npm install
npm run dev
```

常规 provider 测试不调用真实厂商 API。

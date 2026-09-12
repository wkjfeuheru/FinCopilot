# FinHarness

FinHarness 是基于 FastAPI 后端和 React 前端的金融研究 copilot。

## 配置与开发

先复制 `settings.example.json` 为仓库根目录的 `settings.json`，然后在 `model.provider` 中选择 `deepseek`、`kimi`、`glm`、`volcano` 或 `qwen`，并在 `model.model_name` 中填写目标模型名。密钥只通过环境变量提供：

- DeepSeek：`DEEPSEEK_API_KEY`
- Kimi：`MOONSHOT_API_KEY`
- GLM：`ZHIPU_API_KEY`
- 火山方舟（Volcano Ark）：`ARK_API_KEY`
- 阿里云百炼（DashScope 兼容模式）：`DASHSCOPE_API_KEY`

默认情况下，DeepSeek、火山方舟与阿里云百炼使用 OpenAI 兼容协议，Kimi 与 GLM 使用 Anthropic 兼容协议。也可以在 `providers` 中配置自定义端点。缺少密钥不会自动回退到其他厂商；离线测试须显式选择 `fake`、注入 `FakeProvider` 或使用 HTTP mock。

配置覆盖只接受白名单中的单下划线环境变量，例如 `FINH_MODEL_PROVIDER=glm`、`FINH_SERVER_PORT=8123`。列表和对象使用 JSON，例如 `$env:FINH_DATA_ADAPTER_ORDER='["akshare"]'`。未知的 `FINH_` 变量和双下划线写法会直接报错。详细契约见 `docs/modules/03.1-config.md`。

```powershell
uv sync --extra dev
uv run uvicorn finharness.server.api:create_production_app --factory --reload --port 8000
cd frontend
npm install
npm run dev
```

常规 provider 测试不调用真实厂商 API。

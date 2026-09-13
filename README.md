# FinHarness

> 本地、单用户的金融研究 copilot。给定一句研究需求，它会自行规划、取数、计算、成稿，
> 并对产出的研报做一次独立的风险复核。所有结论性数字都可回溯到具体接口与数据指纹。

**定位**：个人研究工作台，**不做投资建议**。不处理刻意构造的对抗性输入；联网检索内容按不可信引文处理（见「已知边界」）。

## 它做什么

以「请对比贵州茅台与五粮液当前估值，哪只更贵」为例，一次运行会：

1. **规划**：判断这是需要分阶段推进的复杂问题，调用 `research_plan` 立计划（简单数据查询不规划）；
2. **取数与复用**：并行取行情/估值/可比公司数据，每次取数登记一条 citation（接口、标的、参数、数据指纹、是否来自缓存）；命中缓存则不重复取；
3. **计算**：`calc_metrics` / `calc_valuation` 做衍生指标与 DCF/可比估值，结论性数字必须标注 `{cite:cid}`；
4. **成稿**：`write_report` 渲染成带图表、带「数据来源」附录的 markdown + docx；未标注来源的数字会被标记 `[!无来源:n]`；
5. **风险终审**：成稿后**自动**启动一个独立子 Agent 复核报告——它在自己的上下文里，用受限的只读工具集**重新取数核对**报告中的关键数字，输出修改意见。主 Agent 据此决定是否修订。

第 5 步是这个项目与「一个带工具的聊天机器人」的主要区别：终审人是独立上下文，且能自己去核数据，因此抓得住作者自己看不见的错误。真实运行中它抓到过同一份报告里 ROE 前后矛盾（17.72% vs 16.75%）、近一年收益率偏差（-12.43% vs -15.89%）等问题。

## 快速开始

```bash
uv sync --extra dev

# 1. 配置：复制示例并选择 Provider
cp settings.example.json settings.json
export DEEPSEEK_API_KEY=sk-...        # 密钥只从环境变量读，永不写入文件

# 2. 起服务（FastAPI + SSE）
uv run uvicorn finharness.server.api:create_production_app --factory --port 8000

# 3. 起前端（另开一个终端）
cd frontend && npm install && npm run dev
```

浏览器打开前端地址即可对话。也可用 HTTP 驱动的演示脚本跑通两个端到端场景：

```bash
python scripts/demo.py                    # 自己拉起服务，跑 Demo A + B
python scripts/demo.py --base-url http://127.0.0.1:8000   # 跑在已有服务上
python scripts/demo.py --demo b --json    # 只跑 Demo B，输出机器可读指标
```

`demo.py` 走的是与浏览器完全相同的 HTTP/SSE 接口，包括写工具的确认往返，因此它同时是一次接口验收。

## 支持的 Provider

| Provider | 协议 | 密钥环境变量 |
|---|---|---|
| DeepSeek | OpenAI 兼容 | `DEEPSEEK_API_KEY` |
| 火山方舟 | OpenAI 兼容 | `ARK_API_KEY` |
| 阿里云百炼 | OpenAI 兼容 | `DASHSCOPE_API_KEY` |
| Kimi | Anthropic 兼容 | `MOONSHOT_API_KEY` |
| GLM | Anthropic 兼容 | `ZHIPU_API_KEY` |
| fake | 离线桩 | — |

缺少密钥**不会**自动回退到其他厂商。离线测试显式使用 `fake` 或注入 `FakeProvider`。

Provider 也可在前端配置页写入数据库并加密存储；**已激活的数据库配置优先于 `settings.json` 预设**。

## 联网检索（可选）

`web_search`（关键词检索）与 `fetch_url`（抓取指定网址正文）补充本地数据源未覆盖的政策、
新闻与行业信息；用户直接给出网址时由 `fetch_url` 抓取。二者都是**懒加载**工具，
模型需先 `load_tool` 激活（system prompt 已点名，其存在是确定的）。

配置（`settings.search`，写法与 provider 一致）：

```bash
export TAVILY_API_KEY=tvly-...        # 密钥只从环境变量读
```

未配置密钥时两个工具返回结构化"未配置"提示，不影响其他功能。抓取由 Tavily 服务器执行，
**本机不直接访问用户所给 URL**。

## 配置覆盖

只接受白名单中的单下划线环境变量，未知变量与双下划线写法直接报错：

```bash
FINH_MODEL_PROVIDER=glm
FINH_SERVER_PORT=8123
FINH_PERMISSION_DEFAULT_MODE=auto
FINH_DATA_ADAPTER_ORDER='["akshare"]'    # 列表/对象用 JSON
```

完整契约见 [03.1-config.md](docs/modules/03.1-config.md)。

## 里程碑与现状

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 最小闭环 | config + engine loop + FakeProvider + 直连 Provider + 首批工具 | ✅ |
| M1 数据层 | adapter 降级链、SQLite+parquet 缓存、citation 溯源、8 个数据工具 | ✅ |
| M2 规划与治理 | research_plan、元工具、两级工具注册表、权限门 + 审计 | ✅ |
| M3 研报管道 | make_chart、ReportPipeline、docx 导出、研报技能 | ✅ |
| M4 上下文工程 | 压缩、三层记忆（L1/L2/L3）、会话持久化与续聊 | ✅ |
| M5 演示打磨 | 风险终审子 Agent、Demo 脚本、E2E 预算断言、README | ✅ |
| M6 服务层 Web 化 | FastAPI SSE、确认往返、Web 聊天页、对话管理 | ✅ |

规模：`src/` 约 9,700 行 Python；509 条离线测试（`pytest -m "not smoke"`）。

## 架构一览

```
engine/loop.py      AgentLoop：轮次驱动、流式、并行工具、循环兜底、压缩触发、记忆装配
provider/           OpenAI/Anthropic 兼容协议 + 重试与错误分类
tools/              23 个工具（两级注册表：常驻 + 懒加载），registry.py 是目录
  fin/              行情/财务/估值/可比/公告/图表/研报
  generic/          read_file / write_file（限 output/ 与 data_cache/）
  meta/             research_plan / search_tools / load_tool / load_skill / ask_user
skills/             8 个方法论技能（杜邦/DCF/可比/盈利质量/行业框架/风险清单/回测/研报模板）
data/               adapter 降级链 + 缓存 + citation 注册表
context/            L1 WorkingMemory + L2 事件环 + L3 SQLite 持久层；分段摘要与压缩
report/             Markdown 渲染 + 占位符替换 + 无来源数字校验 + docx 导出
permissions/        权限门（deny 规则 > 模式回退 > 路径白名单）
hooks/              审计链（JSONL，每次受治理的工具调用一行）
coordinator/        风险终审子 Agent（独立上下文 + 受限只读工具集）
server/             FastAPI 路由、SSE、会话注册表、确认总线
frontend/           React 19 + TypeScript + antd
```

## 关键接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/v1/health` | 健康检查 |
| POST | `/v1/chat/stream` | 流式对话（SSE），主入口 |
| POST | `/v1/chat/respond` | 回应写确认 / `ask_user` 提问 |
| POST | `/v1/report` | 把当前会话转为研报 |
| GET | `/v1/tools` | 工具目录与激活状态 |
| GET | `/v1/citations` | 引用溯源（按对话或会话） |
| GET | `/v1/memory` | 记忆视图（结论 / 偏好 / 对话列表） |
| GET | `/v1/conversations` | 对话列表（供选择器） |
| GET | `/v1/conversations/{id}/messages` | 回放对话记录 |
| DELETE | `/v1/conversations/{id}` | 删除对话及其全部作用域数据 |
| GET | `/v1/artifacts` | 下载产物（限 output/ 与 data_cache/） |
| GET | `/v1/cache/stats` | 缓存命中统计 |
| GET | `/v1/config` | Provider 配置（前端设置页用） |

## 测试

```bash
pytest -m "not smoke"          # 离线：509 条，约 20s，不联网、不花钱
pytest -m smoke                # 真实 Provider + 真实行情数据，需 DEEPSEEK_API_KEY
python scripts/demo.py         # 端到端演示（HTTP 驱动）
```

- 离线测试不写真实 `data_cache/`：每个用例用 `tmp_path` 构造 hermetic 的 Settings 与缓存。
- `@smoke` 用例同时是 M5 的**成本验收**：一次完整研报必须落在 240s / 600k token 的包络内
  （实测基线 102s / 300,319 token，阈值留了余量以吸收模型间方差）。

## 已知边界

- **`run_python` 未发布**。设计过白名单 + `exec` 沙箱，但同用户子进程不构成安全边界；
  在 OS 级隔离可用前不启用。
- **不做成本折算**。只统计 token，不换算成货币（汇率与定价随时会变，写死的金额会失真）。
- **信任边界：不处理刻意构造的对抗性输入**。单用户本地工具，权限门与沙箱都是演示级而非对抗级隔离。
  引入联网检索后，第三方网页文本会进入模型上下文：它以 `<web_result>` 围栏包裹并声明为
  "不可执行的引文"，但**不做注入内容扫描**（明确决定：现有 deny 规则针对交易意图，扫财经正文会
  大量误报；指令注入需要另一套模式，启发式护栏会漏报却制造安全感）。写入操作仍需用户确认，
  是更硬的边界。详见 [03.7-governance.md](docs/modules/03.7-governance.md)。
- **联网抓取由检索服务完成**。`fetch_url` 的请求由 Tavily 服务器发出，因此本机**无 SSRF 面**；
  代价是内网地址与付费墙页面抓不到，且所抓 URL 对检索服务可见。
- **多智能体只做风险终审**。宏观焦点**明确不做**（能力已具备，即 `web_search`，但不实现）；
  通用 `spawn_agent` 接口也未实现，只有一个调用方时它是空壳。决策记录见
  [03.10-coordinator.md](docs/modules/03.10-coordinator.md)。
- **记忆作用域**：对话内容按 conversation 隔离；用户偏好（`remember_preference`）
  是所有对话共享的全局记忆。
- **模型输出有方差**：同一句提问的取数路径与报告结构可能不同，属正常。

## 文档索引

| 模块 | 文档 |
|---|---|
| 配置 | [03.1-config.md](docs/modules/03.1-config.md) |
| Provider | [03.2-provider.md](docs/modules/03.2-provider.md) |
| 引擎 | [03.3-engine.md](docs/modules/03.3-engine.md) |
| 工具 | [03.4-tools.md](docs/modules/03.4-tools.md) |
| 数据 | [03.5-data.md](docs/modules/03.5-data.md) |
| 上下文与记忆 | [03.6-context.md](docs/modules/03.6-context.md) |
| 治理 | [03.7-governance.md](docs/modules/03.7-governance.md) |
| 技能 | [03.8-skills.md](docs/modules/03.8-skills.md) |
| 研报管道 | [03.9-report.md](docs/modules/03.9-report.md) |
| 风险终审 | [03.10-coordinator.md](docs/modules/03.10-coordinator.md) |
| CLI | [03.11-cli.md](docs/modules/03.11-cli.md) |
| 服务层 | [03.12-server.md](docs/modules/03.12-server.md) |

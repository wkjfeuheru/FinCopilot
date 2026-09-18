# FinHarness

> 基于Agent Loop的金融研究 copilot。给定一句研究需求，它会自行规划、取数、计算、成稿，
> 并对产出的研报做一次独立的风险复核。所有结论性数字都可回溯到具体接口与数据指纹。

**定位**：个人研究工作台，**不做投资建议**。不处理刻意构造的对抗性输入；联网检索内容按不可信引文处理（见「已知边界」）。

## 它做什么

以「请对比贵州茅台与五粮液当前估值，哪只更贵」为例，一次运行会：

1. **规划**：判断这是需要分阶段推进的复杂问题，调用 `research_plan` 立计划（简单数据查询不规划）；
2. **取数与复用**：并行取行情/估值/可比公司数据，每次取数登记一条 citation（接口、标的、参数、数据指纹、是否来自缓存）；命中缓存则不重复取；
3. **计算**：`calc_metrics` / `calc_valuation` 做衍生指标与 DCF/可比估值，结论性数字必须标注 `{cite:cid}`；
4. **成稿**：`write_report` 渲染成带图表、带「数据来源」附录的 markdown + docx；未标注来源的数字会被标记 `[!无来源:n]`；
5. **风险终审**：成稿后**自动**启动一个独立子 Agent 复核报告——它在自己的上下文里，用受限的只读工具集**重新取数核对**报告中的关键数字，输出修改意见。主 Agent 据此决定是否修订。
6. **并行子代理**：`spawn_agent` 把若干彼此独立的并行任务分派给子代理，每个子代理在**自己的上下文**里完成、只回结论，因此大量中间材料（多份长文档、多条独立分析线）不占主上下文。子代理只读、**不取数**：材料由主 Agent 取好后写在任务里或以本地路径给出。

第 5 步是这个项目与「一个带工具的聊天机器人」的主要区别：终审人是独立上下文，且能自己去核数据，因此抓得住作者自己看不见的错误。真实运行中它抓到过同一份报告里 ROE 前后矛盾（17.72% vs 16.75%）、近一年收益率偏差（-12.43% vs -15.89%）等问题。第 6 步把同一个机制推广到任意可隔离任务——**理由始终是隔离，不是并发**：主 Agent 本来就能在一轮内并行调多个工具。

## 快速开始

```bash
uv sync --extra dev

# 可选：可观测性后端（Prometheus 指标 / LangSmith 追踪）
uv sync --extra dev --extra observability

# 1. 配置：复制示例并选择 Provider
cp settings.example.json settings.json
export DEEPSEEK_API_KEY=sk-...        # 密钥只从环境变量读，永不写入文件

# 2. 起服务（FastAPI + SSE）
uv run uvicorn finharness.server.api:create_production_app --factory --port 8000

# 3. 起前端（另开一个终端）
cd frontend && npm install && npm run dev
```

浏览器打开前端地址即可对话。开发服务器把 `/v1` 请求代理到后端，目标地址由
环境变量 `FINHARNESS_API_ORIGIN` 决定（默认 `http://127.0.0.1:8001`）；后端起在
别的端口时改这个变量即可，例如 `FINHARNESS_API_ORIGIN=http://127.0.0.1:8000 npm run dev`。

首次使用需要注册一个账号：登录页可切换
「注册新账号」，用户名 2–32 个字符，密码至少 8 位。所有对话、偏好与
模型供应商配置都按账号隔离（见 [docs/modules/03.13-auth.md](docs/modules/03.13-auth.md)）。
若这是从单用户版本升级而来，第一个注册的账号会自动继承原有的对话与配置。

### 生产部署（单端口）

不依赖 Node 开发服务器，前端构建产物由 FastAPI 直接托管：

```bash
cd frontend && npm install && npm run build   # 产出 frontend/dist（已按 vendor 分包）
uv run uvicorn finharness.server.api:create_production_app --factory --port 8001
```

服务检测到 `frontend/dist` 后，`/` 返回页面、`/assets/*` 返回静态资源；页面与
`/v1` 接口同源，浏览器直接携带会话 Cookie，无需 CORS 与代理。若未执行前端构建，
`/` 会返回占位页，其余接口不受影响。

也可用 HTTP 驱动的演示脚本跑通两个端到端场景（脚本会先注册/登录一个
`demo` 账号，因为所有 `/v1` 接口都要求认证）：

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

## 联网检索与研报（可选）

`web_search`（关键词检索）补充本地数据源未覆盖的政策、新闻与行业信息；
`get_research_reports` 抓取**东方财富研报**——按类型（行业/个股）、行业、机构、时间段与标题
关键词筛选，返回标题/机构/评级/日期/链接，并可选用 `with_text` 抓取 PDF 全文。
二者都是**按需注入**的工具：直接调用即可，系统会在调用时让它可用，无需预热轮
（用 `search_tools` 检索也能一并激活，并得到参数清单）。

`web_search` 配置（`settings.search`，写法与 provider 一致）：

```bash
export TAVILY_API_KEY=tvly-...        # 推荐：密钥只从环境变量读
```

也可在 `settings.json`（已被 git 忽略）里直接写 `search.api_key`，适合不方便设环境变量的本地环境，
此时密钥以明文落盘。`search.proxy` 可显式指定检索请求走的代理；**不填则自动使用系统代理**——
`httpx` 只读环境变量、不读 Windows 注册表，而 A 股数据源用的 `requests` 会读系统代理，
不自动对齐会导致"国内源通、检索源不通"。

未配置密钥时 `web_search` 返回结构化"未配置"提示，不影响其他功能（`get_research_reports`
**不需要密钥**）。`web_search` 的检索请求由 Tavily 服务器发出，本机不直接访问目标地址。
例外是研报**全文**：`with_text=true` 时由本机抓取 PDF 并用 `pypdf` 抽取正文，该能力默认开启
（`search.local_pdf_fallback`）且带私网阻断（逐跳校验重定向，拦 loopback／私网／云元数据地址），
关闭后 `with_text=true` 会明确报错；仅取元数据不受影响。

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

规模：`src/` 约 1.9 万行 Python（`wc -l`）；895 条离线测试（`pytest -m "not smoke"`）。

## 架构一览

```
engine/loop.py      AgentLoop：轮次驱动、流式、并行工具、循环兜底、压缩触发、记忆装配
provider/           OpenAI/Anthropic 兼容协议 + 重试与错误分类
tools/              32 个工具（两级注册表：21 常驻 + 11 按需），declare.py 是声明层
  declare.py        @tool / @param：元数据与参数的唯一声明处
  fin/              行情/财务/估值/可比/公告/图表/研报（含 report_pipeline.py 渲染 + docx 导出）
  generic/          read_file（限 output/ 与本人 data_cache/） / write_file（仅 output/）
  meta/             research_plan / search_tools / ask_user / spawn_agent / …
skills/             4 个投研场景（个股/行业/宏观/量化）：各含 SKILL.md + references 方法论 + assets 报告模板
                    由路由层按问题意图自动注入（不提供加载工具）
data/               adapter 降级链 + 缓存 + citation 注册表
context/            L1 WorkingMemory + L2 事件环 + SQLite 持久层；分段摘要与压缩；LTM 跨对话记忆（情节+语义，蒸馏/注入/向量召回）
utils/              跨层通用件：pandas/akshare 运行时垫片、Markdown 图片链接转义
permissions/        权限门（deny 规则 > 缓存拒写 > 读/写分流 > 模式回退 > output 白名单）
                    网络外发（联网检索、研报全文）首次确认，可在对话内免问
hooks/              审计链（JSONL，每次受治理的工具调用一行，含子代理）
coordinator/        风险终审子 Agent（独立上下文 + 受限只读工具集；review.py 终审编排）
observability/      三层观测：结构化 JSON 日志（trace_id 贯穿）+ Prometheus 指标 + LangSmith 追踪
server/             FastAPI 路由、SSE、会话注册表、确认总线
frontend/           React 19 + TypeScript + antd
```

## 关键接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/v1/health` | 健康检查 |
| POST | `/v1/chat/stream` | 流式对话（SSE），主入口 |
| POST | `/v1/chat/respond` | 回应写确认 / `ask_user` 提问 |
| POST | `/v1/chat/stop` | 停止当前生成（协作式；保留已取得的数据与结论，可继续） |
| POST | `/v1/report` | 把当前会话转为研报 |
| GET | `/v1/tools` | 工具目录与激活状态 |
| GET | `/v1/citations` | 引用溯源（按对话或会话） |
| GET | `/v1/memory` | 记忆视图（结论 / 偏好 / 对话列表） |
| GET | `/v1/conversations` | 对话列表（供选择器） |
| GET | `/v1/conversations/{id}/messages` | 回放对话记录；另含 `resumable`（上一轮被停止时） |
| DELETE | `/v1/conversations/{id}` | 删除对话及其全部作用域数据 |
| GET | `/v1/artifacts` | 下载产物（限 output/ 与 data_cache/） |
| GET | `/v1/cache/stats` | 缓存命中统计 |
| GET | `/v1/config` | Provider 配置（前端设置页用） |
| GET | `/metrics` | Prometheus 指标（仅在 `observability.metrics.enabled=true` 时注册） |

## 测试

```bash
pytest -m "not smoke"          # 离线：895 条，约 30s，不联网、不花钱
pytest -m smoke                # 真实 Provider + 真实行情数据，需 DEEPSEEK_API_KEY
python scripts/demo.py         # 端到端演示（HTTP 驱动）
```

- 离线测试不写真实 `data_cache/`：每个用例用 `tmp_path` 构造 hermetic 的 Settings 与缓存。
- `@smoke` 用例同时是 M5 的**成本验收**：一次完整研报必须落在 240s / 600k token 的包络内
  （实测基线 102s / 300,319 token，阈值留了余量以吸收模型间方差）。

### Agent 评估体系

四维度评估（任务完成率 / 推理路径正确性 / 效率 / 安全性），详见 [03.13-eval.md](docs/modules/03.13-eval.md)：

```bash
python -m finharness.eval list  --set smoke|core|full   # 列出题集
python -m finharness.eval check                          # 仅校验用例 YAML
python -m finharness.eval run --set selfcheck --offline  # 零成本管线自检
python -m finharness.eval run --set smoke                # 真实 Provider（需密钥）
```

- 用例在 `evals/cases/*.yaml`，判定规则从 `docs/测试问题集-功能与幻觉.md` 的
  「通过标准/典型失败信号」翻译而来；判定看**行为模式**而非字面文本。
- 产出 `evals/runs/<ts>_<set>/report.md`：四维度表、加权综合分、红线门禁结论、失败用例轨迹。
- 引擎为轨迹评估记录每步 `Thought/Action/Observation`（含被拒绝的调用），见 `AgentTurnOutcome.trace`。
- **首次真实运行前需校准 `evals/config.yaml` 的效率预算**（当前为估计基线）。

## 已知边界

- **`run_python` 未发布**。设计过白名单 + `exec` 沙箱，但同用户子进程不构成安全边界；
  在 OS 级隔离可用前不启用。
- **不做成本折算**。只统计 token，不换算成货币（汇率与定价随时会变，写死的金额会失真）。
- **信任边界：不处理刻意构造的对抗性输入**。单用户本地工具，权限门与沙箱都是演示级而非对抗级隔离。
  引入联网检索后，第三方网页文本会进入模型上下文：它以 `<web_result>` 围栏包裹并声明为
  "不可执行的引文"，但**不做注入内容扫描**（明确决定：现有 deny 规则针对交易意图，扫财经正文会
  大量误报；指令注入需要另一套模式，启发式护栏会漏报却制造安全感）。写入操作仍需用户确认，
  是更硬的边界；网络外发（联网检索、研报全文下载）首次调用也需确认，用户可授权"本对话内
  不再询问"。详见 [03.7-governance.md](docs/modules/03.7-governance.md)。
- **联网检索由检索服务完成**。`web_search` 的请求由 Tavily 服务器发出，该路径本机
  **无 SSRF 面**；代价是内网地址与付费墙页面抓不到，且查询对检索服务可见。
  例外是研报**全文**（`get_research_reports` 的 `with_text=true`）：它让本机访问文档 CDN，
  故以"私网阻断 + 长度/时长上限 + 有界重试"收窄受影响面，并可用
  `search.local_pdf_fallback=false` 完全关闭。已知局限：地址校验为请求前解析而非固定对端，
  非 DNS-rebinding 免疫。
- **多智能体做两件事：风险终审 + 通用任务扇出**，共同理由是**上下文隔离**（不是并发）。
  宏观焦点**明确不做**（能力已具备，即 `web_search`，但不实现）。通用 `spawn_agent`
  已在第二个焦点出现后按原定条件落地；子代理一律只读、**不含取数层**（风险终审是唯一例外，
  它需要独立核数）。决策记录见 [03.10-coordinator.md](docs/modules/03.10-coordinator.md)。
- **记忆作用域**：对话内容按 conversation 隔离；**跨对话长期记忆（LTM）** 与用户偏好是该用户
  所有对话共享的，但绝不跨用户。LTM 分两层：
  - **情节记忆**（`ltm_episodes`：做过什么）——任务结果每轮由结论自动写入，关键决策与对话片段
    在对话闲置后由 LLM 懒蒸馏产出；
  - **语义记忆**（`ltm_facts`：知道什么）——事实、概念与偏好，与情节**共用同一次**蒸馏调用，
    `(user, key)` 覆盖式更新，因此用户后来的口径会取代旧口径。配了 embedding 端点与 Qdrant 时
    按语义相似度召回（也可只用本地向量，或退化为键匹配——见 03.6 §3.6.4「4.2」）。
  用户可经 `/v1/memory` 查看/编辑/删除任一记忆条目，或让 agent 用 `search_memory` /
  `update_memory` / `forget_memory` 操作。详见
  [03.6-context.md](docs/modules/03.6-context.md) §3.6.4「4.1」「4.2」。
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
| 评估体系 | [03.13-eval.md](docs/modules/03.13-eval.md) |
| 可观测性 | [03.14-observability.md](docs/modules/03.14-observability.md) |

---
name: equity-research
description: 个股深度研究场景：财报解读（盈利/成长/健康/效率）、盈利质量核查、估值（DCF/可比）与多标的对比聚合。复杂个股研究任务进入本场景，按流程加载方法论文件。触发词：个股、财报、盈利、ROE、杜邦、盈利质量、估值、贵不贵、内在价值、深度研究、对比分析。
inputs:
  - symbol（6位A股代码；多标的对比时逐个执行后聚合）
  - 研究关注点（盈利质量 / 成长性 / 财务健康 / 估值 / 综合体检）
outputs:
  - 对话内 markdown 结论（表格+要点，每个结论性数字带 {cite:cid}）
  - 用户明确要求报告时：读 assets/report-template.md 后经 write_report 成稿
use_cases:
  - 解读个股财报：盈利归因、质量核查、成长性判断
  - 估值判断：DCF 内在价值或同业相对估值
  - 多只个股横向对比（逐标的分析后聚合）
examples:
  - 帮我深度分析一下贵州茅台
  - 600519 这份财报质量怎么样？ROE 为什么下滑？
  - 茅台和五粮液对比一下财务和估值
allowed_tools: [get_financials, get_indicators, get_valuation, get_peers, get_announcements, get_market_news, calc_metrics, calc_valuation, make_chart, write_report, web_search, read_file]
content_estimate: 1400
version: 1
---

# 个股深度研究（场景入口）

> 本文件只负责**流程与边界**。分析方法在 `references/` 下按需加载，报告模板在 `assets/`。

## 场景界定

**进入本场景**：研究对象是一只（或几只）A 股个股，且任务超出单点数据查询——需要财报解读、
盈利质量判断、估值分析、或多标的对比聚合。

**不进入**：
- 单点事实查询（"600519 现在多少钱""最近一年跌了多少"）→ 直接取数作答，不加载本场景。
- 研究主体是行业本身 → `industry-research`；宏观环境/政策 → `macro-research`。
- 单条公告/事件的快问快答（"这份公告什么意思"）→ 直接 `get_announcements` / `web_search`
  查证作答，无需方法论文件；只有当事件需要放进财务框架评估影响时才回到本场景。

**边界处理**：个股研究中需要行业背景（如判断毛利率是否异常需要同业对照）→ 用 `get_peers`
取同业数据对照即可，**不切换场景**；行业层面的系统性结论才去 `industry-research`。

## 分析流程

**先确认关注点**：用户未指明研究维度（如只说"分析一下 600519"）时，先调用 `ask_user`
弹出交互窗口让用户选择关注点——盈利质量 / 成长性 / 财务健康 / 运营效率 / 估值 / 综合体检
（可多选），再据此选步骤。用户已明确关注点（如"ROE 为什么下滑""贵不贵"）时**直接进入**，
不再询问。多标的对比时同样先确认对比维度。

多维度个股研究满足规划条件，先 `research_plan`（版本随研究推进修订）。然后**按关注点选步骤，
不是每单都走全流程**：

| 步骤 | 做什么 | 加载什么 / 调用什么 |
|---|---|---|
| 0 取数 | 财务三表、指标、估值、同业 | `get_financials` / `get_indicators` / `get_valuation` / `get_peers` |
| 1 盈利 | ROE 归因（杜邦）+ 盈利质量与造假核查 | references/profitability.md；`calc_metrics(method="dupont")` |
| 2 成长 | 增速拆解与可持续性判断 | references/growth.md |
| 3 健康 | 偿债能力与现金流 | references/health.md |
| 4 效率 | 三大周转与营运 | references/efficiency.md |
| 5 估值 | DCF 绝对估值 / 可比相对估值 | references/valuation.md；`calc_valuation` |
| 6 行业对照 | 按行业类型挑关注指标 | references/industry-focus.md |

**多标的对比模式**：对每只标的**分别执行**上述步骤（可比组取数可复用），最后由你聚合对比——
对比结论要落在"差异是什么、由什么驱动"，不是两张表并排了事。工具调用逐只执行，
不得因某个对比表未覆盖某标的就略过它、或把它写成"取不到"。

## 参考文件索引

| 文件 | 方法论 | 何时加载 |
|---|---|---|
| references/profitability.md | 盈利能力：指标体系 + 杜邦归因 + 盈利质量与造假核查 | 分析盈利、ROE 归因、利润质量 |
| references/growth.md | 成长性：增速拆解与可持续性 | 判断增长的真实性与可持续性 |
| references/health.md | 财务健康度：偿债与现金流 | 评估财务风险 |
| references/efficiency.md | 运营效率：三大周转 | 周转异常，或零售/制造类标的 |
| references/valuation.md | 估值：DCF + 可比 + 陷阱 | 一切估值问题 |
| references/industry-focus.md | 行业差异化关注点速查 | 定位行业类型后选择关注指标 |
| assets/report-template.md | 个股研报模板 | 用户要求成稿时 |

**加载纪律**：`load_skill("equity-research", file="references/<文件名>")` 按需逐份加载，
一般一单 2–3 份即可；同一文件只加载一次（重复调用返回复用提示）。不要为"可能有用"预加载。

## 输出契约

- **默认对话内作答**：markdown 表格 + 要点，结论性数字带 `{cite:cid}`；问"对比一下/解释差异"
  不等于要报告，**不得自行升级成稿**。
- **配图按需增强解释**（不必等用户点名）：估值历史分位/PE 走势 → 折线图；同业横向对比
  （PE/PB/ROE）→ 分组柱；多期盈利或杜邦因子趋势 → 折线。单点数字或已说清的表格不配图。
  图先 `make_chart` 生成再引用路径；单位与口径随数字标注。
- **用户明确说"报告/研报/成稿/导出"** → 读 `assets/report-template.md` → 按其章节盘点补齐素材 →
  按上述判据配图 → `write_report` 成稿（写入前经用户确认；成稿后系统自动运行风控终审）。
- 结论只做条件性判断，不下买卖建议；无来源的数字不得出现。

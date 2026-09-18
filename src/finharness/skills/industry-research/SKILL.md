---
name: industry-research
description: 行业研究场景：产业链分析、竞争格局、行业空间测算与景气度跟踪。以行业本身为研究对象时进入。触发词：行业、产业链、赛道、竞争格局、景气度、行业空间、集中度、上中下游。
inputs:
  - industry（行业名）或 symbol（由个股定位所属行业）
  - 研究关注点（产业链 / 竞争格局 / 景气度 / 行业空间 / 综合）
outputs:
  - 对话内 markdown 结论（三层框架互相印证，结论带 {cite:cid}）
  - 用户要求报告时：读 assets/report-template.md 后经 write_report 成稿
use_cases:
  - 分析一个行业的结构与竞争格局
  - 判断行业当前景气位置与跟踪指标
  - 产业链拆解与价值分布
examples:
  - 分析一下新能源汽车产业链
  - 白酒行业现在的竞争格局怎么样？
  - 光伏行业的景气度如何？
allowed_tools: [get_industry_perf, get_industry_constituents, get_peers, get_valuation, get_indicators, get_financials, get_market_news, web_search, make_chart, write_report, read_file]
content_estimate: 1300
version: 1
---

# 行业研究（场景入口）

> 本文件只负责**流程与边界**。分析方法在 `references/` 下按需加载，报告模板在 `assets/`。

## 场景界定

**进入本场景**：研究对象是**行业本身**——产业结构、竞争格局、产业链、景气度、行业空间。

**不进入**：
- 研究对象是具体个股（行业只是背景）→ `equity-research`（用 `get_peers` 做同业对照即可）。
- 宏观环境、跨行业的政策与外部冲击 → `macro-research`；但某政策对**特定行业**的影响分析属于本场景。
- "XX 行业有哪些股票/龙头是谁"这类列表查询 → 直接取数作答。

## 分析流程

**先确认关注点**：用户未指明研究维度时，先调用 `ask_user` 让用户选择——产业链 / 竞争格局 /
景气度 / 行业空间 / 综合（可多选），再据此选步骤；用户已明确（如"光伏行业景气度如何"）
时**直接进入**，不再询问。

满足规划条件（多维度/分阶段）时先 `research_plan`。**按关注点选步骤**：

| 步骤 | 做什么 | 加载什么 / 调用什么 |
|---|---|---|
| 1 定位 | 确定行业边界与主要参与者 | `get_industry_perf`（行业概览/行情）／`get_peers`（由个股定位行业） |
| 2 结构 | 生命周期阶段 + 竞争格局与集中度 | references/lifecycle.md；competition.md |
| 3 产业链 | 上中下游拆解、价值分布、话语权 | references/value-chain.md |
| 4 景气 | 景气跟踪指标体系与当前判断 | references/prosperity.md |
| 5 空间 | 行业空间测算（TAM/渗透率/量价） | references/market-space.md |
| 6 综合 | 关键成功因素 → 投资逻辑归纳 | references/csf.md |

**数据基础**：行业指数行情用 `get_industry_perf`（申万行业，支持一级与二级如"白酒"；
省略行业名返回申万一级行业总览与估值横向对比），成分股用 `get_industry_constituents`
（也可作横截面股票池）；行业特有风险写报告时从 references/risk-identification.md 取候选。

## 参考文件索引

| 文件 | 方法论 | 何时加载 |
|---|---|---|
| references/lifecycle.md | 行业生命周期：阶段判断与各阶段投资逻辑 | 判断行业所处阶段 |
| references/value-chain.md | 产业链：拆解、价值分布、话语权 | 产业链类问题 |
| references/competition.md | 竞争格局：市场结构、集中度、五力 | 格局与竞争类问题 |
| references/prosperity.md | 景气度：跟踪指标体系与周期定位 | 景气度类问题 |
| references/market-space.md | 行业空间测算：TAM-SAM-SOM/渗透率/量价 | 空间与增长测算 |
| references/csf.md | 关键成功因素：不同行业的核心 KPI | 从结构到投资逻辑的收口 |
| references/risk-identification.md | 行业风险识别（候选清单） | 成稿的风险章节 |
| assets/report-template.md | 行业研报模板 | 用户要求成稿时 |

**加载纪律**：这些文档由系统按问题所涉能力自动注入，无需请求加载——
一般一单 2–3 份；同一文件只加载一次。

## 输出契约

- **默认对话内作答**：结论须互相印证（如"五力弱"与"格局分散且份额下滑"并存时要指出矛盾），
  数字带 `{cite:cid}`；不得把分析自行升级成报告。
- **配图按需增强解释**（不必等用户点名）：行业指数走势/景气指标序列 → 折线图
  （`get_industry_perf` 数据经 `make_chart` 绘制）；集中度 CR5 或份额横向对比 → 分组柱。
  单点数字或已说清的表格不配图；图先 `make_chart` 生成再引用路径。
- **用户明确要求报告** → 读 `assets/report-template.md` → 盘点补齐素材 → 按上述判据配图 →
  `write_report` 成稿（写入前确认，成稿后自动风控终审）。
- 行业判断落到数据（集中度、ROE 中位数、价格趋势），不停留在定性描述；不下买卖建议。

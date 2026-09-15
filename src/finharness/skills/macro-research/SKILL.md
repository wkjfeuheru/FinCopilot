---
name: macro-research
description: 宏观研究场景：经济周期定位、货币/财政/监管政策解读、外部环境影响、宏观指标对资产的影响逻辑。以宏观环境或政策为研究对象时进入。触发词：宏观、经济周期、通胀、降息、加息、货币政策、社融、M2、财政、行业监管、美联储、汇率。
inputs:
  - 宏观主题（事件 / 指标 / 政策 / 综合研判）
  - 关注的资产或行业（用于传导分析）
outputs:
  - 对话内 markdown 结论（周期定位→政策评估→资产影响链，带 {cite:cid}）
  - 用户要求报告时：读 assets/report-template.md 后经 write_report 成稿
use_cases:
  - 解读宏观事件（降息、通胀数据发布、政治局会议）
  - 评估货币/财政政策立场与传导
  - 分析宏观指标对某资产/行业的影响逻辑
examples:
  - 这次降准对市场意味着什么？
  - 当前经济处于周期什么位置？
  - 美联储加息对 A 股有什么影响？
allowed_tools: [get_macro_indicators, get_quote, get_kline, get_peers, get_market_news, web_search, make_chart, write_report, read_file]
content_estimate: 1300
version: 1
---

# 宏观研究（场景入口）

> 本文件只负责**流程与边界**。分析方法在 `references/` 下按需加载，报告模板在 `assets/`。

## 场景界定

**进入本场景**：研究对象是**宏观环境本身**——经济周期位置、货币/财政/监管政策、
外部环境（美联储/汇率/地缘/大宗），以及它们向资产或行业的传导。

**不进入**：
- 只取一个宏观数字（"今天 SHIBOR 多少"）→ 直接取数作答。
- 个股/行业为研究对象 → `equity-research` / `industry-research`；但"宏观变量 X 对
  该行业/标的的影响"这类传导分析属本场景。
- 宏观指标的**定义/科普** → 直接对话回答，无需方法论。

## 分析流程

**先确认关注点**：用户未指明研究对象时，先调用 `ask_user` 让用户选择——周期定位 / 货币政策 /
财政政策 / 监管环境 / 外部环境 / 综合研判（可多选），以及需要分析的资产或行业，再据此选步骤；
用户已明确（如"这次降准意味着什么"）时**直接进入**，不再询问。

满足规划条件时先 `research_plan`。标准链路：**定位周期 → 评估政策 → 识别关键变量 →
推导资产影响**（源自底本四步应用框架）。按需选步骤：

| 步骤 | 做什么 | 加载什么 / 调用什么 |
|---|---|---|
| 1 周期定位 | 增长/通胀/景气的当前位置 | references/cycle.md；`get_macro_indicators` |
| 2 政策评估 | 货币/财政立场与传导 | references/monetary.md / fiscal.md |
| 3 监管环境 | 行业监管与外部冲击 | references/regulation.md / external.md |
| 4 指标查数 | 指标口径、频率、取数 | references/indicators.md |
| 5 资产影响 | 传导链推导（周期/政策→资产/行业） | 各方法论文件内的"对投资的影响"节 + cycle.md |

**数据基础**：宏观指标用 `get_macro_indicators`（PMI/CPI/PPI/M2/社融/LPR/SHIBOR/国债收益率/汇率/GDP）。
指标为月度/季度发布，**引用时必须标注发布机构与所属期间**，并注意发布滞后
（如 GDP 按季、PMI 月末发布）；数字只能来自本次取数，不得用记忆数字顶替。

## 参考文件索引

| 文件 | 方法论 | 何时加载 |
|---|---|---|
| references/cycle.md | 经济周期判断：四阶段识别与资产映射 | 周期定位类问题 |
| references/monetary.md | 货币政策：工具箱、立场判断、传导 | 利率/流动性/降准降息类 |
| references/fiscal.md | 财政政策：工具箱与行业指向 | 财政/赤字/专项债类 |
| references/regulation.md | 监管环境：领域、方法、案例 | 行业政策影响类 |
| references/external.md | 外部环境：美联储/汇率/地缘/大宗 | 外部冲击类 |
| references/indicators.md | 核心经济指标速查（口径/频率/取数） | 任何取宏观数、对口径有疑问时 |
| assets/report-template.md | 宏观研报模板 | 用户要求成稿时 |

**加载纪律**：`load_skill("macro-research", file="references/<文件名>")` 按需逐份加载，
一般一单 2–3 份；同一文件只加载一次。

## 输出契约

- **默认对话内作答**：按"定位→政策→变量→影响"链组织结论，数字带 `{cite:cid}`；
  不得把分析自行升级成报告。
- **配图按需增强解释**（不必等用户点名）：**单个**宏观指标的历史趋势 → 折线图
  （`get_macro_indicators` 返回长表，画图时一次只取一个指标、或用 `series` 指定一列）。
  **不同口径／不同频率的指标不在同一张图叠加**（PMI 指数、CPI 百分比、社融亿元量纲不同），
  需对比时改用表格或分图；图注须写明指标名与所属期间。单点数字不配图。
- **用户明确要求报告** → 读 `assets/report-template.md` → 盘点补齐素材 → 按上述判据配图 →
  `write_report` 成稿（写入前确认，成稿后自动风控终审）。
- 资产影响写成**条件判断与传导链**（"宽松利好成长风格"是传导结论而非买卖建议）；
  不下买卖建议、不给目标价。

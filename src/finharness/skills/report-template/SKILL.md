---
name: report-template
description: 研报结构编排：把各投研分析技能的产出组织成带图表与引用附录的成稿。触发词：研报、报告、成稿、写一份报告、整理成报告。
inputs:
  - 本次会话已完成的各分析技能产出（结论 + cids + 表格 + 图表路径）
  - 未完成时：先按 related_skills 补齐所需分析
outputs:
  - output/<topic>_<YYYYMMDD>.md（canonical，可读可 diff）
  - output/<topic>_<YYYYMMDD>.docx（交付物）
use_cases:
  - 用户明确要求把研究整理成报告
  - 问答已积累足够结论，需要成稿交付
examples:
  - 把刚才的分析整理成一份研报
  - 给我出一份贵州茅台的估值分析报告
# 本技能是聚合型：它复用下列分析技能的产出，按需加载
related_skills:
  - dupont-analysis      # 财务摘要/盈利归因
  - dcf-valuation        # 估值章节（绝对估值）
  - valuation-comps      # 估值章节（相对估值）
  - earnings-quality     # 财务摘要的利润质量校验
  - industry-framework   # 公司/行业摘要
  - risk-checklist       # 风险提示章节
allowed_tools: [write_report, make_chart, load_skill, get_indicators, get_financials]
content_estimate: 1200
version: 2
---

# 研报结构编排

> **本技能不产生分析结论，只负责编排。** 每章内容都应由对应的分析技能产出；
> 若某章尚无素材，**先 `load_skill` 加载对应技能完成该分析**，再回来成稿。
> 不得为了填满章节而编造内容——宁可缩短报告并说明缺什么。

## 章节 → 技能 → 占位符 对照

| 章节 | 依赖技能 | 产出占位符 | 缺失时怎么办 |
|---|---|---|---|
| 1 核心观点 | （综合各章） | — | 每条观点须引用本会话已有的 cids |
| 2 公司/行业摘要 | `industry-framework` | `{list:industry_map}` | 缺行业判断 → 加载该技能 |
| 3 财务摘要 | `dupont-analysis` + `earnings-quality` | `{table:dupont}` | 缺分解或质量校验 → 加载对应技能 |
| 4 估值分析 | `dcf-valuation` **或** `valuation-comps` | `{table:dcf_sens}` / `{table:comps}` | 缺估值 → 先取财务数据再加载估值技能 |
| 5 风险提示 | `risk-checklist` | `{list:risks}` | 必做；缺 → 加载该技能 |
| 6 附录 | `cite.to_appendix_md()`（由管道自动生成） | — | 无需手工填写 |

**取舍原则**：估值章节按标的特性二选一——
现金流稳定 → `dcf-valuation`（绝对估值）；周期或轻资产 → `valuation-comps`（相对估值）。
两者都做可作为交叉验证，但不是必需。

## 标准章节（按序）

1. **核心观点**（3–5 条）：每条一句话结论 + 关键数字，**每条必须带 citation**。
2. **公司/行业摘要**：主营业务、行业地位、竞争格局要点。
3. **财务摘要**：关键指标表（营收、净利润、ROE、毛利率、负债率），标注报告期；附盈利质量结论。
4. **估值分析**：方法说明 + 估值表 + 图表；三情景，不替读者下结论。
5. **风险提示**：≥5 条，每条含**触发条件**。
6. **附录**：数据来源（citation 全量）+ 免责声明。

## 占位符约定

| 占位符 | 含义 | 由谁提供 |
|---|---|---|
| `{cite:cit_000001}` | 引用 citation，渲染为 `[n]` 并入附录 | 正文直接写 cid |
| `{chart:output/charts/x.png}` | 嵌入图表 | `make_chart` 返回的路径 |
| `{table:dupont}` | 杜邦分解表 | `dupont-analysis` |
| `{table:dcf_sens}` | DCF 敏感性矩阵 | `dcf-valuation` |
| `{table:comps}` | 可比公司表 | `valuation-comps` |
| `{list:industry_map}` | 行业三层框架 | `industry-framework` |
| `{list:quality_checks}` | 盈利质量清单 | `earnings-quality` |
| `{list:risks}` | 风险清单 | `risk-checklist` |
| `{table:perf}` | 回测绩效表 | `backtest-protocol` |

## 编排工作流

```
① 盘点：本次会话已有哪些结论/cids/表格？对照上表找缺口
② 补齐：对每个缺口 load_skill(<对应技能>)，按该技能指引取数分析
   —— 复用已有 cids，避免重复取数（citation 里已有数据的直接引用）
③ 配图：需要图表的章节先调 make_chart 生成，拿到路径
④ 成稿：调 write_report(topic, core_view[], sections[], risks[])
   —— 入参为结构化内容，管道负责引用编号、附录、docx 导出
⑤ 确认：写入前经用户确认（write_report 无 path 参数，必走确认）
```

**注意顺序**：先补齐分析 → 再配图 → 最后成稿。不要在分析缺口未补时先调 `write_report`。

## 写作纪律

- **每个结论性数字都必须能回溯到 citation**；无来源数字会被管道标注 `[!无来源:n]`。
- 图表必须先由 `make_chart` 生成，再在正文引用其路径。
- 正文用 markdown；标题层级最多到 `####`（docx 只映射到 Heading4）。
- **不下投资建议，不做买卖推荐**。
- 素材不足时缩短报告并写明缺什么，不编造、不凑字数。

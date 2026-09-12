---
name: report-template
description: 研报结构模板：核心观点→摘要→分章分析→估值→风险提示，含图表与引用占位符约定。
allowed_tools: [write_report, make_chart, get_indicators]
content_estimate: 800
version: 1
---

# 研报结构模板

## 标准章节

1. **核心观点**（3-5 句）：一句话结论 + 关键依据数字，必须带 citation。
2. **公司/行业摘要**：主营业务、行业地位、最新财务概览。
3. **分章分析**：按分析框架展开（盈利质量、成长性、竞争力）。
4. **估值**：给出方法与三情景，不替读者下结论。
5. **风险提示**：≥5 条，每条含触发条件。

## 占位符约定

| 占位符 | 含义 |
|---|---|
| `{cite:x}` | 引用 citation id，渲染为附录编号 |
| `{table:dupont}` | 插入杜邦分解表 |
| `{table:dcf_sens}` | 插入 DCF 敏感性矩阵 |
| `{chart:path}` | 插入图表，path 为 `output/charts/` 下路径 |
| `{list:risks}` | 插入风险清单 |

## 无引用校验

任何结论性数字都必须能回溯到 citation；
缺少 `{cite:...}` 的结论段会在导出时被标记。

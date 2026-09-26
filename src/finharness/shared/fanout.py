"""多实体分析意图的确定性识别（docs 03.10 / 03.3）。

为什么需要它：`spawn_agent` 的触发只写在提示词里时，实测**不可靠**——真实用户不会
说"请用子代理"，而"分别评估 A、B、C 三家"这类自然问法下，模型倾向于自己串行取数作答
（实测隐式触发率 0/3，加硬规则也只 1/3）。既然"是否该扇出"能从文本确定性判定，就把它
交给引擎，而不是指望模型的自觉。

判据（窄、可解释）：**出现 ≥2 个不同实体，且带有"逐实体分析"类动词**。

* 实体 = 6 位股票代码 ∪ 常见行业词。
* 动词 = 分析/评估/对比/梳理/研究… 等。
* 只要几个数字/字段（"三家股价"）不触发——那属于同轮并行取数；单一实体不触发。

刻意保持保守：宁可漏判（退回自由裁量的串行作答），也不误判（把"查三家股价"逼成扇出）。
"""

from __future__ import annotations

import re

# 6 位 A 股代码（与 plan_progress 同款口径）。
_SYMBOL_RE = re.compile(r"\b\d{6}\b")

# 常见行业/板块词。**刻意只收"不会嵌进公司名"的辨识度高的词**：像"银行""证券"
# "医药""汽车"这类会出现在公司名里（招商银行、兴业证券、复星医药…），一旦收录，
# "分析招商银行(600036)"会被误判成两个实体。这类歧义词的请求仍有 6 位代码兜底
# （工行/招行等都带代码），因此剔除它们几乎不损失召回，却消除误判。
_INDUSTRY_LEXICON = frozenset(
    {
        "白酒", "食品饮料", "医疗器械", "生物医药", "消费",
        "半导体", "光伏", "风电", "储能", "电池", "电力设备",
        "新能源", "国防", "军工", "新能源汽车",
    }
)

# "逐实体分析"类动词/名词。只要几个数字不算。
_ANALYSIS_TERMS = (
    "分析", "评估", "对比", "比较", "梳理", "研究", "归因", "判断",
    "格局", "前景", "质量", "竞争力", "估值", "盈利", "趋势", "表现",
    "驱动", "怎么看", "如何看", "深度",
)


def entities_in_text(text: str) -> set[str]:
    """文本中出现的实体标识：6 位代码 + 已知行业词。"""
    lowered = str(text or "")
    found: set[str] = set(_SYMBOL_RE.findall(lowered))
    found.update(word for word in _INDUSTRY_LEXICON if word in lowered)
    return found


def fanout_intent(text: str) -> bool:
    """该文本是否"应当扇出"——即逐实体深分析。

    仅当**同时**满足：实体数 ≥2，且出现逐实体分析类动词。任一不满足都返回 False
    （退回自由裁量：模型可自行选择串行或扇出）。
    """
    message = str(text or "")
    if len(entities_in_text(message)) < 2:
        return False
    return any(term in message for term in _ANALYSIS_TERMS)

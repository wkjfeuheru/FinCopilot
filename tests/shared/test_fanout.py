"""多实体分析意图识别（docs 03.10）。

它是引擎侧**确定性扇出**的判据：命中即预激活 spawn_agent 并下发强制编排指令。
判据刻意保守——宁可漏判（退回自由裁量的串行作答），也不误判（把"查三家股价"
逼成扇出）。
"""

from __future__ import annotations

from finharness.shared.fanout import entities_in_text, fanout_intent


def test_multi_entity_analysis_triggers():
    assert fanout_intent("分别评估招商银行(600036)、兴业银行(601166)、平安银行(000001)的资产质量")
    assert fanout_intent("对比茅台(600519)与五粮液(000858)近三年盈利能力变化，并归因")
    assert fanout_intent("把这份组合里的行业逐个梳理：白酒、光伏、银行各自的竞争格局")
    assert fanout_intent("对比四个行业板块今年以来的表现并说明各自驱动因素：半导体、新能源、医药、消费")


def test_shallow_multi_entity_fetch_does_not_trigger():
    """只要几个数字/字段——同轮并行取数即可，不该扇出。"""
    assert not fanout_intent("贵州茅台(600519)和五粮液(000858)现在分别多少钱？")
    assert not fanout_intent("比亚迪(002594)最新一期财报的营收、净利润和经营现金流分别是多少？")


def test_single_entity_does_not_trigger():
    assert not fanout_intent("分析贵州茅台(600519)近三年 ROE 的变化趋势")
    assert not fanout_intent("全面分析招商银行(600036)最新财务与估值")


def test_company_names_embedding_industry_words_do_not_false_positive():
    """"招商银行"含"银行"，但那是一个实体、不是两个——词表刻意剔除了这类歧义词。"""
    ents = entities_in_text("全面分析招商银行(600036)最新财务与估值")
    assert ents == {"600036"}


def test_no_entities_or_no_analysis_term_does_not_trigger():
    assert not fanout_intent("帮我看看这家公司怎么样")
    # 有多个实体但只是取数（无分析动词）。
    assert not fanout_intent("600519、000858、000568 的收盘价")


def test_lithium_cross_check_query_is_not_multi_entity_fanout():
    """单行业交叉印证不得走多实体 fanout_intent；发现 spawn 靠描述而非强制扇出。"""
    assert not fanout_intent("搜索一些关于锂电池产能过剩的讨论交叉印证，给出一些你的观点。")

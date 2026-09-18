"""数据时效元数据：把「这是截至哪一期的数据」变成可证明的事实。

背景：用户问「当前最新的 PMI/CPI/M2」，回答给出上一月的值并被读为陈旧。问题不在取数
（月频指标按月发布，当月值本就未发布），而在于回答无法自证时效——数字旁边只有一个裸期间。
这里钉住的是：每条序列各自报自己的最新期（不被日频序列掩盖）、期数写法按频率区分、
发布机构与节奏随行给出。
"""

import pandas as pd

from finharness.data.freshness import (
    Freshness,
    SeriesFreshness,
    SeriesSpec,
    freshness_from_grouped_frame,
    frequency_label,
)


def _long_frame() -> pd.DataFrame:
    rows = []
    for months_back in range(3):
        moment = pd.Timestamp("2026-08-01") - pd.DateOffset(months=months_back)
        rows.append(
            {"date": moment, "indicator": "pmi_manufacturing", "label": "制造业PMI",
             "value": 49.8 - months_back, "unit": "指数"}
        )
        rows.append(
            {"date": moment, "indicator": "m2_yoy", "label": "M2同比",
             "value": 7.5 - months_back, "unit": "%"}
        )
    for moment in pd.date_range(end="2026-09-17", periods=5, freq="D")[::-1]:
        rows.append(
            {"date": moment, "indicator": "bond_10y", "label": "10年期国债收益率",
             "value": 2.1, "unit": "%"}
        )
    return pd.DataFrame(rows)


def test_each_series_reports_its_own_latest_period():
    """日频序列的最近交易日不得把月频序列的数据期顶高。

    若只看整表最大日期，国债收益率的 9-17 会掩盖 PMI 的 8 月，读者便无从判断
    月频指标是否落在正确的期数上。
    """
    frame = _long_frame()

    freshness = freshness_from_grouped_frame(
        frame, group_col="indicator", period_col="date", label_col="label", value_col="value"
    )

    periods = {item.label: item.period_iso for item in freshness.series}
    assert periods["制造业PMI"] == "2026-08-01"
    assert periods["M2同比"] == "2026-08-01"
    assert periods["10年期国债收益率"] == "2026-09-17"


def test_period_is_rendered_in_the_unit_the_reader_expects():
    """月频写「2026年8月」而非 2026-08-01；日频保留到日。"""
    monthly = SeriesFreshness(label="制造业PMI", period_iso="2026-08-01", frequency="monthly")
    quarterly = SeriesFreshness(label="GDP同比", period_iso="2026-07-01", frequency="quarterly")
    daily = SeriesFreshness(label="10年期国债收益率", period_iso="2026-09-16", frequency="daily")
    yearly = SeriesFreshness(label="年度值", period_iso="2026-03-01", frequency="yearly")

    assert monthly.period_display == "2026年8月"
    assert quarterly.period_display == "2026年3季度"
    assert daily.period_display == "2026-09-16"
    assert yearly.period_display == "2026年"


def test_describe_carries_publisher_and_cadence():
    item = SeriesFreshness(
        label="制造业PMI",
        period_iso="2026-08-01",
        frequency="monthly",
        publisher="中国物流与采购联合会",
        cadence="当月最后一日发布",
    )

    text = item.describe()

    assert "2026年8月" in text
    assert "中国物流与采购联合会" in text
    assert "当月最后一日发布" in text


def test_note_discloses_fetch_time_and_cache_state():
    """抓取时刻与缓存命中必须可见，否则「这份数据在本地停了多久」无从判断。"""
    freshness = Freshness(
        series=(SeriesFreshness(label="M2同比", period_iso="2026-08-01", frequency="monthly"),),
        fetched_at="2026-09-17T10:00:00+08:00",
        from_cache=True,
    )

    note = freshness.note()

    assert "2026-09-17T10:00:00+08:00" in note
    assert "缓存命中" in note


def test_a_live_fetch_is_distinguishable_from_a_cache_hit():
    live = Freshness(
        series=(SeriesFreshness(label="M2同比", period_iso="2026-08-01"),),
        fetched_at="2026-09-17T10:00:00+08:00",
        from_cache=False,
    )

    assert "本次实时取数" in live.note()


def test_note_is_empty_without_series():
    """无序列时不得产出一个空洞的时效段落——状态块里没有比有一句废话更好。"""
    assert Freshness().note() == ""
    assert not Freshness()


def test_metadata_is_attached_when_the_group_is_known():
    freshness = freshness_from_grouped_frame(
        _long_frame(),
        group_col="indicator",
        period_col="date",
        label_col="label",
        meta={"m2_yoy": SeriesSpec("monthly", "中国人民银行", "次月中旬发布（约10-15日）")},
    )

    m2 = next(item for item in freshness.series if item.label == "M2同比")
    assert m2.publisher == "中国人民银行"
    assert m2.cadence == "次月中旬发布（约10-15日）"
    # 未提供元数据的序列仍报告期数，只是不附机构。
    bond = next(item for item in freshness.series if item.label == "10年期国债收益率")
    assert bond.period_iso == "2026-09-17"
    assert bond.publisher == ""


def test_null_values_do_not_pin_a_stale_period():
    """最新一行若数值为空，应回退到最近一个有值的期——否则会把未填充的期当成数据期。"""
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-09-01"), "indicator": "cpi_yoy", "label": "CPI同比", "value": None},
            {"date": pd.Timestamp("2026-08-01"), "indicator": "cpi_yoy", "label": "CPI同比", "value": 0.8},
        ]
    )

    freshness = freshness_from_grouped_frame(
        frame, group_col="indicator", period_col="date", label_col="label", value_col="value"
    )

    assert freshness.series[0].period_iso == "2026-08-01"


def test_missing_columns_and_empty_frames_degrade_to_no_metadata():
    """缺少必需列或空表时返回空时效，而不是抛异常打断取数。"""
    assert freshness_from_grouped_frame(pd.DataFrame(), group_col="indicator", period_col="date").series == ()
    assert freshness_from_grouped_frame(
        pd.DataFrame([{"a": 1}]), group_col="indicator", period_col="date"
    ).series == ()


def test_frequency_label_falls_back_to_the_raw_slug():
    assert frequency_label("monthly") == "月度"
    assert frequency_label("daily") == "日度"
    assert frequency_label("fortnightly") == "fortnightly"

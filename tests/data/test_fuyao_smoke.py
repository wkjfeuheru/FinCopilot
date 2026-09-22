"""同花顺 MCP 的真实端点冒烟测试。

**这不是集成测试套件的替代品。** 适配器与客户端的全部解析逻辑都在
``tests/data/test_mcp_client.py`` / ``tests/data/test_fuyao_adapter.py`` 里用
``httpx.MockTransport`` 覆盖了，那些测试常在且不需要网络。

本文件要验证的是一件 mock 无法验证的事：**线上服务端到底怎么说话**。同花顺的公开
文档只写了客户端配置为 ``"type": "http"`` 加 ``X-api-key`` 请求头，没有公布握手
细节。对着真实端点跑一次，就能把"我们对协议的理解"与"服务端的实际行为"对上一次。

**已核实的线上行为**（这些断言即是对它的记录，也是回归防线）：

* 传输是标准 **Streamable HTTP**：``initialize`` → 响应头带 ``Mcp-Session-Id`` →
  ``notifications/initialized`` → 后续请求带回该会话 id。
* ``protocolVersion`` 回显 ``2024-11-05``；响应是 **JSON**（``application/json``），
  不是 SSE（客户端两条路径都实现了，因此这点变化不会打破它）。
* 未认证时 ``initialize`` **仍然成功**（HTTP 200 + 会话 id），鉴权推迟到
  ``tools/call``——失败以信封里的 ``code=2003`` 表达（HTTP 仍是 200），因此
  ``FuyaoMcpAdapter._unwrap`` 的 code 分类是这条路径上唯一的安全网。
* 每个数据集都带 ``inputSchema``，且若干部署了**实质性默认值**（如快照的
  ``thscodes`` 默认为两只具体个股）——这就是派发器必须披露继承默认值的原因。

缺少 ``HITHINK_FINANCE_API_KEY`` 时全部跳过——没有密钥的人不该因为一个用不到的集成
而看到红灯。运行方式：``uv run pytest -m smoke tests/data/test_fuyao_smoke.py``。

**注意**：除最后两条外都需要真实密钥，因此默认（无 key）运行时会跳过；最后两条
只验证"错误凭据被正确分类"与"错误端点快速失败"，不需要 key，因此在 CI 里也会执行。
"""

from __future__ import annotations

import os

import pytest

from finharness.config.settings import Settings
from finharness.data.adapters.mcp_client import McpHttpClient
from finharness.data.mapping import FUYAO_SERVICE_PATHS

pytestmark = pytest.mark.smoke

# 冒烟测试只用最小、有界的请求：文档明确要求认证检查不要用全市场快照这类重查询。
PROBE_SYMBOL = "600519"


@pytest.fixture(scope="module")
def adapter():
    """真实适配器；无密钥或未启用时跳过。

    密钥的两个来源都要试：环境变量（推荐路径）与本仓库 ``settings.json`` 里的内联
    ``fuyao.api_key``（git 已忽略，见配置契约 docs 03.1）。后者必须经
    ``Settings.from_file`` 加载——裸构造 ``Settings()`` 只读代码默认值，不会碰任何
    配置文件，用它判定"是否配置了密钥"会把本地配置误判为缺失（本文件曾因此整组误跳过）。
    """
    settings = Settings.from_file()
    api_key = settings.fuyao.resolved_api_key()
    if not api_key:
        pytest.skip(
            f"未配置同花顺密钥（{settings.fuyao.env_key} 或 settings.json 的 fuyao.api_key），"
            "跳过真实端点冒烟测试"
        )
    if not settings.fuyao.enabled:
        pytest.skip("settings.fuyao.enabled=false，跳过真实端点冒烟测试")

    from finharness.data.adapters.fuyao_adapter import FuyaoMcpAdapter

    return FuyaoMcpAdapter(
        api_key=api_key,
        base_url=settings.fuyao.base_url,
        timeout_s=settings.fuyao.timeout_s,
        proxy=settings.fuyao.proxy,
        throttle_seconds=0.5,
    )


def test_the_handshake_works_against_the_real_gateway(adapter):
    """握手成功即证明客户端对 Streamable HTTP 的理解与服务端一致。

    这是本文件最重要的断言：其余测试依赖它，而它验证的正是公开文档没写的那部分。
    """
    tools = adapter.list_datasets("a-share")

    assert tools, "同花顺 a-share 服务应至少返回一个数据集"
    names = {tool.get("name") for tool in tools}
    # 文档列出的端点必须真的在目录里：如果目录为空或形状不符，说明握手或解析出了问题。
    assert "get_a_share_prices_snapshot" in names


def test_every_documented_service_answers_a_tools_list(adapter):
    """六个服务都该能列出工具；这同时覆盖各自端点路径的正确性。"""
    for service in FUYAO_SERVICE_PATHS:
        tools = adapter.list_datasets(service)
        assert tools, f"{service} 服务未返回任何数据集"


def test_the_catalog_exposes_input_schemas_so_dispatch_can_validate(adapter):
    """派发器靠 ``inputSchema`` 校验参数，因此目录必须带上它。"""
    tools = adapter.list_datasets("a-share")
    snapshot = next(
        tool for tool in tools if tool.get("name") == "get_a_share_prices_snapshot"
    )

    schema = snapshot.get("inputSchema") or snapshot.get("input_schema")
    assert isinstance(schema, dict), "数据集条目应带 inputSchema"
    assert "thscodes" in schema.get("properties", {})


def test_a_real_quote_lands_on_the_internal_contract(adapter):
    """真实响应是否真的落到 date/open/high/low/close/volume/amount 上。

    mock 只能验证"按我们理解的形状解析正确"，这一条验证"服务端确实按那个形状回答"。
    """
    result = adapter.fetch_quote(PROBE_SYMBOL)

    assert len(result.df) == 1
    row = result.df.iloc[0]
    assert str(row["symbol"]) == PROBE_SYMBOL
    for column in ("date", "open", "high", "low", "close", "volume"):
        assert column in result.df.columns, column
    assert row["close"] == row["close"]  # 非 NaN
    assert row["close"] > 0


def test_a_real_kline_series_is_ordered_and_bounded(adapter):
    result = adapter.fetch_kline(PROBE_SYMBOL, "day", "qfq", 1)

    assert len(result.df) > 20
    dates = list(result.df["date"])
    assert dates == sorted(dates, reverse=True), "内部契约要求最新在前"


def test_real_financials_fold_into_the_wide_statement_shape(adapter):
    """三表字段是否真的折成"指标行 × 报告期列"，且字段标签已翻译。"""
    result = adapter.fetch_financials(PROBE_SYMBOL, "利润", 3)

    frame = result.df
    assert frame.columns[0] == "指标"
    labels = set(frame["指标"])
    assert "营业收入" in labels, f"未看到已翻译的字段标签：{sorted(labels)[:10]}"
    # 报告期列是 YYYYMMDD，且最新在前。
    periods = [c for c in frame.columns if str(c).isdigit() and len(str(c)) == 8]
    assert periods, "应至少有一个 YYYYMMDD 报告期列"
    assert periods == sorted(periods, reverse=True)


def test_real_indicators_are_folded_and_labelled(adapter):
    """指标端点是一期一请求，折叠后应有中文指标名列与 date 列。"""
    result = adapter.fetch_indicators(PROBE_SYMBOL, 1, None)

    frame = result.df
    assert "date" in frame.columns
    assert len(frame) > 0
    assert any("净资产收益率" in str(c) for c in frame.columns), list(frame.columns)


def test_a_real_dataset_call_through_the_generic_channel(adapter):
    """通用通道对着一个轻量端点跑通：证明长尾派发这条路径真的可用。

    交易日历不接收参数、结果规模可控，因此适合做认证探测（文档建议避免重查询）。
    """
    result = adapter.fetch_dataset(
        "a-share", "get_a_share_calendar_trading_days", {}
    )

    assert len(result.df) > 0


def test_a_typo_in_a_dataset_name_is_reported_not_silently_empty(adapter):
    """错的端点名必须报错。

    若无此保证，"上游改了名字"会表现为一份空数据，而那看起来像"今天没有涨停股"。
    """
    from finharness.data.adapters.base import AdapterError

    with pytest.raises(AdapterError):
        adapter.fetch_dataset("a-share", "get_a_share_not_a_real_dataset", {})


def test_an_invalid_key_is_reported_as_a_non_retryable_failure():
    """凭据错误的分类必须正确：重试不会让它变好，而回退链要继续往下走。"""
    from finharness.data.adapters.base import AdapterError
    from finharness.data.adapters.fuyao_adapter import FuyaoMcpAdapter

    bogus = FuyaoMcpAdapter(
        api_key="definitely-not-a-valid-key",
        base_url=Settings().fuyao.base_url,
        timeout_s=15.0,
        throttle_seconds=0.0,
    )

    with pytest.raises(AdapterError) as exc:
        bogus.fetch_quote(PROBE_SYMBOL)

    assert exc.value.retryable is False, "凭据错误不该触发重试"


def test_the_client_reports_a_bad_path_as_unavailable_rather_than_hanging():
    """端点路径错误应快速失败并让回退链继续，而不是耗尽调用方预算。"""
    from finharness.data.adapters.base import AdapterError

    client = McpHttpClient(
        url=Settings().fuyao.base_url + "/mcp/definitely-not-a-service",
        api_key=os.getenv("HITHINK_FINANCE_API_KEY") or "x",
        timeout_s=15.0,
    )

    with pytest.raises(AdapterError):
        client.list_tools()

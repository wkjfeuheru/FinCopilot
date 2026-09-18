"""system prompt 是一项产品资产；守护它必须包含的内容。

它规定了 agent 何时制定 plan、如何引用，以及何时停止。如果未来的
修改删掉了其中某个部分，agent 会无声地失去该行为，因此这里对必需的
指令进行了断言。
"""

import pytest

from finharness.engine.prompt import PROMPT_PATH, PromptNotFoundError, system_prompt


@pytest.fixture(autouse=True)
def _clear_cache():
    system_prompt.cache_clear()
    yield
    system_prompt.cache_clear()


def test_prompt_asset_exists_and_is_substantial():
    text = system_prompt()

    assert len(text) > 300
    assert PROMPT_PATH.is_file()


def test_prompt_defines_when_to_plan():
    text = system_prompt()

    assert "research_plan" in text
    # 它必须说明何时"不要"制定 plan，否则琐碎问题也要为 plan turn 付出代价。
    assert "不要" in text or "无需" in text
    assert "简单事实" in text


def test_prompt_requires_plan_write_back():
    text = system_prompt()

    # 从未更新的 plan 会永远显示为 pending；prompt 必须说明
    # 要将状态与结论写回（docs 03.3.9）。
    assert "update_plan_step" in text
    assert "record_conclusion" in text


def test_prompt_carries_the_citation_and_traceability_rules():
    text = system_prompt()

    assert "{cite:" in text          # 占位符语法
    assert "citation" in text.lower()
    assert "溯源" in text or "回溯" in text
    assert "无来源" in text           # 无来源数字的标记


def test_prompt_requires_convergence():
    """收敛是应对会话超长运行的直接解药。"""
    text = system_prompt()

    assert "收敛" in text
    assert "重复" in text


def test_prompt_routes_skills_and_reports():
    text = system_prompt()

    for scenario in (
        "equity-research",
        "industry-research",
        "macro-research",
        "quant-factor",
    ):
        assert scenario in text
    assert "write_report" in text


def test_prompt_states_when_not_to_produce_deliverables():
    """缺少这一边界，agent 会自行把一次分析升级为一份 report。"""
    text = system_prompt()

    assert "调用边界" in text
    assert "make_chart" in text
    # report 是可选项，而不是分析的默认产出。
    assert "明确说出" in text
    assert "不得把分析自行升级成成稿" in text
    assert "不要" in text


def test_prompt_makes_charts_scenario_driven_not_strictly_opt_in():
    """图表用于增强解释；当结论本身就是一个趋势或一次对比时，模型可以主动
    绘图——report 仍然是可选项。"""
    text = system_prompt()
    section = text.split("## 工具与技能的调用边界", 1)[1].split("\n## ", 1)[0]

    assert "配图按场景判断" in text
    # 各 skill 所依赖的具体触发条件。
    assert "净值曲线" in section
    assert "series" in section


def test_prompt_flags_an_unreviewed_report():
    text = system_prompt()

    assert "未经独立复核" in text


def test_prompt_states_output_discipline():
    text = system_prompt()

    assert "买卖建议" in text


def test_prompt_names_the_research_report_tool():
    """report 工具是懒加载的，因此 prompt 是模型发现它的途径。"""
    text = system_prompt()
    section = text.split("是按需注入的", 1)[1].split("\n## ", 1)[0]

    assert "get_research_reports" in section
    # 它必须说明这些参数存在，否则模型不会进行筛选。
    assert "行业" in section and "机构" in section
    # 并说明正文是第三方文本。
    assert "不得臆测" in section or "仅作事实参考" in section


def test_prompt_explains_the_handle_then_summarize_workflow():
    """研报正文不再内联：prompt 必须交代拿到句柄后的下一步。

    否则模型只会看到"路径 + 首页预览"，却不知道自己手上还有一个可摘要、
    可精读的正文——这会退化成"只有首页摘要"的老问题。
    """
    text = system_prompt()
    section = text.split("是按需注入的", 1)[1].split("\n## ", 1)[0]

    assert "summarize_document" in section
    assert "read_pdf" in section
    # 说明了正文为何不内联，以及截断时如何取回。
    assert "截断" in section and "truncated" in section


def test_prompt_explains_when_to_spawn_and_what_it_costs():
    """spawn_agent 是懒加载且代价高昂的；prompt 是模型唯一能同时了解到
    其触发条件（是隔离，而非提速）与边界（子代理不取数）的地方。"""
    text = system_prompt()
    section = text.split("## 子代理与上下文隔离", 1)[1].split("\n## ", 1)[0]

    assert "spawn_agent" in section
    # 理由必须是隔离，并且要排除单纯的并行抓取。
    assert "隔离" in section
    assert "不要" in section and "并行" in section
    # 子代理不取数：素材是交给它们的。
    assert "不取数" in section


def test_prompt_names_the_backtest_tool_and_its_discipline():
    """run_backtest 同样是懒加载的；prompt 必须指向它，但不能承诺
    可以执行任意代码。"""
    text = system_prompt()
    section = text.split("是按需注入的", 1)[1].split("\n## ", 1)[0]

    assert "run_backtest" in section
    assert "不执行任意代码" in section


# --- 角色、能力边界与拒绝策略（prompt 设计阶段） --------

def test_prompt_defines_a_specific_role_and_capability_boundary():
    """含糊的"你是一个助手"会让模型随意发挥。"""
    text = system_prompt()

    assert "角色与能力边界" in text
    assert "A 股" in text
    # 必须把两面都写明，否则模型无法判断该拒绝什么。
    assert "你能做" in text or "能做" in text
    assert "不能做" in text or "不能" in text
    # 点名这些能力缺口，才让边界真正可操作。
    assert "港股" in text or "美股" in text
    assert "预测" in text


def test_prompt_states_the_knowledge_boundary():
    """参数中携带数据；模型自身的记忆绝不能作为数字来源。"""
    text = system_prompt()

    assert "知识边界" in text
    assert "本次取数" in text


def test_prompt_shows_the_citation_format_by_example():
    """仅有描述，比描述加一个具体示例要弱。"""
    text = system_prompt()

    assert "正确示例" in text
    assert "错误示例" in text
    # 这些示例必须真正使用它们所教的占位符语法。
    assert text.count("{cite:") >= 3


def test_prompt_report_example_matches_the_real_schema():
    """JSON 示例必须始终能通过 write_report 参数模型的校验。

    偏离 schema 的示例会让模型学到一种会被工具拒绝的数据形状，
    这比完全没有示例更糟。
    """
    import json
    import re

    from finharness.tools.fin.writer import WriteReportTool

    blocks = re.findall(r"```json\n(.*?)```", system_prompt(), re.DOTALL)
    assert blocks, "the prompt should carry a JSON example for write_report"

    for block in blocks:
        WriteReportTool.input_model.model_validate(json.loads(block))


def test_prompt_forbids_calling_unfetched_data_unavailable():
    """当模型只是没有取数时却说"数据不可用"，会产生误导。

    读者无法区分"数据源没有这个数字"与"agent 跳过了这次调用"；后者读起来
    像是一种根本不属实的数据限制。
    """
    text = system_prompt()
    role_section = text.split("## 角色与能力边界", 1)[1].split("\n## ", 1)[0]

    assert "没取" in role_section
    assert "未取得" in role_section


def test_prompt_has_a_refusal_policy():
    """缺少这一点，模型会越界作答，而不是拒绝。

    拒绝是能力边界的后果，因此该策略位于能力边界这一节之内，而不是作为
    独立主题；本测试断言的是这一位置，而不仅仅是措辞出现在了文件某处。
    """
    text = system_prompt()
    role_section = text.split("## 角色与能力边界", 1)[1].split("\n## ", 1)[0]

    assert "做不到" in role_section
    assert "不要勉强" in role_section or "不硬撑" in role_section
    assert "ask_user" in role_section          # 缺少输入 -> 澄清
    # 反幻觉规则，作为底线明确写出。
    assert "不得猜测" in role_section or "不得编造" in role_section
    assert "取不到" in role_section


def test_prompt_has_no_standalone_refusal_section():
    """守护这一合并：该策略不得自行游离出去。"""
    text = system_prompt()

    assert "## 超出能力范围时" not in text


def test_missing_asset_raises(tmp_path):
    with pytest.raises(PromptNotFoundError):
        system_prompt(tmp_path / "nope.md")


def test_empty_asset_raises(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")

    with pytest.raises(PromptNotFoundError):
        system_prompt(empty)


def test_server_uses_the_same_prompt_as_the_loader():
    """单一来源：随包发布的 prompt 与 server 常量必须一致。"""
    from finharness.server.api import DEFAULT_SYSTEM_PROMPT

    assert DEFAULT_SYSTEM_PROMPT == system_prompt()

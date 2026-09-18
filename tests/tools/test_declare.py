"""声明层：``@tool`` / ``@param`` 是元数据与参数的唯一声明处。

这些测试守的是**漂移**——旧写法里参数的名称、类型与默认值要在输入模型与
``_dispatch`` 签名里各写一遍，两处不一致时既不报错也不报警，只是在模型眼里多出
一个不存在的参数。声明层把签名变成唯一真相，这里逐项验证它确实做到了。
"""

import pytest

from finharness.tools.base import BaseTool
from finharness.tools.declare import (
    DECLARED_TOOLS,
    Capability,
    DeclarationError,
    Tier,
    ToolGroup,
    param,
    tool,
)


@tool(
    name="declared_probe",
    description="探针工具",
    capability=Capability.FILE,
    group=ToolGroup.GENERIC,
)
class _ProbeTool(BaseTool):
    @param("symbol", desc="股票代码")
    @param("years", desc="回溯年数")
    @param("label", desc="标签", default="默认")
    async def _dispatch(
        self, *, symbol: str, years: int = 3, label: str = "默认"
    ) -> None:  # pragma: no cover - 只验证声明
        return None


@tool(
    name="declared_data_probe",
    description="数据探针",
    capability=Capability.MARKET,
    tier=Tier.LAZY,
    data_tool=True,
)
class _DataProbeTool(BaseTool):
    @param("symbol", desc="股票代码")
    async def _dispatch(self, *, symbol: str) -> None:  # pragma: no cover - 只验证声明
        return None


def test_the_decorator_projects_metadata_onto_the_class():
    assert _ProbeTool.name == "declared_probe"
    assert _ProbeTool.description == "探针工具"
    assert _ProbeTool.capability is Capability.FILE
    assert _ProbeTool.group is ToolGroup.GENERIC
    # 权限未声明时取默认值，而不是留空。
    assert _ProbeTool.permission.value == "read"


def test_types_and_defaults_come_from_the_signature():
    """参数的类型与默认值以签名为准，因此模型看到的就是实现真正接受的。"""
    fields = _ProbeTool.input_model.model_fields

    assert fields["symbol"].is_required()
    assert fields["years"].default == 3
    assert fields["label"].default == "默认"


def test_descriptions_come_from_the_param_declaration():
    fields = _ProbeTool.input_model.model_fields

    assert fields["symbol"].description == "股票代码"
    assert fields["label"].description == "标签"


def test_declared_names_exclude_self_and_match_the_signature_order():
    """声明顺序即签名顺序，使生成的 schema 属性序稳定（模型侧的前缀缓存依赖这一点）。"""
    names = list(_ProbeTool.input_model.model_fields)

    assert names == ["symbol", "years", "label"]


def test_optional_parameters_use_a_factory_without_becoming_required():
    """``default_factory`` 是签名写不出的默认值，它不得把参数变成必填。"""
    import datetime

    def _factory() -> str:
        return datetime.date.today().isoformat()

    @tool(name="factory_probe", description="d", capability=Capability.FILE)
    class _FactoryTool(BaseTool):
        @param("since", desc="起始日期", default_factory=_factory)
        async def _dispatch(self, *, since: str) -> None:  # pragma: no cover
            return None

    field = _FactoryTool.input_model.model_fields["since"]
    assert field.is_required() is False
    assert field.get_default(call_default_factory=True) == _factory()


def test_a_data_tool_gets_the_detail_strategy_parameter_for_free():
    names = list(_DataProbeTool.input_model.model_fields)

    assert "detail" in names
    # 它是策略参数（由渲染侧消费），因此工具主体不接收它。
    assert "detail" in _DataProbeTool.__tool_spec__.strategy_names
    assert _DataProbeTool.data_tool is True


def test_detail_is_absent_from_a_non_data_tool():
    assert "detail" not in _ProbeTool.input_model.model_fields
    assert _ProbeTool.__tool_spec__.strategy_names == ()


def test_a_tier_is_carried_by_the_declaration():
    assert _DataProbeTool.tier is Tier.LAZY
    assert _ProbeTool.tier is Tier.RESIDENT


def test_a_typo_in_a_param_name_fails_at_import_time():
    """参数名写错必须立刻炸掉。

    静默忽略的后果最难查：声明看起来生效了，模型却始终拿不到那项约束。
    """

    with pytest.raises(DeclarationError):

        @tool(name="typo_probe", description="d", capability=Capability.FILE)
        class _Typo(BaseTool):
            @param("symbolz", desc="笔误")  # 签名里是 symbol
            async def _dispatch(self, *, symbol: str) -> None:  # pragma: no cover
                return None


def test_a_custom_validator_flows_into_validation():
    """自定义校验与 Pydantic 自身的校验走同一条失败路径。"""

    def _positive(value: int) -> int:
        if value <= 0:
            raise ValueError("必须为正")
        return value

    @tool(name="validator_probe", description="d", capability=Capability.FILE)
    class _Validated(BaseTool):
        @param("count", desc="数量", validate=_positive)
        async def _dispatch(self, *, count: int) -> None:  # pragma: no cover
            return None

    assert _Validated.input_model.model_validate({"count": 3}).count == 3
    with pytest.raises(Exception):
        _Validated.input_model.model_validate({"count": 0})


def test_a_cross_field_rule_runs_after_field_validation():
    """跨字段规则由 ``params_model`` 承载（``summarize_document`` 即此模式）。

    ``@param`` 声明的是单字段约束；"两个来源必须给一个"这类规则约束的是字段的联合
    取值，因此它仍属于一个手写模型，只是模型现在从整体上充当工具的输入契约。
    """
    from pydantic import BaseModel, model_validator

    class _Pair(BaseModel):
        left: str | None = None
        right: str | None = None

        @model_validator(mode="after")
        def rule(self) -> "_Pair":
            if not (self.left or self.right):
                raise ValueError("至少给一个")
            return self

    @tool(
        name="cross_probe",
        description="d",
        capability=Capability.FILE,
        params_model=_Pair,
    )
    class _Cross(BaseTool):
        async def _dispatch(self, *, left=None, right=None) -> None:  # pragma: no cover
            return None

    assert _Cross.input_model.model_validate({"left": "x"}).left == "x"
    with pytest.raises(Exception):
        _Cross.input_model.model_validate({})


def test_model_validator_argument_wires_a_cross_field_rule():
    """``model_validator=`` 让生成出来的模型带上跨字段规则。"""
    from pydantic import BaseModel

    def _require_one(value: BaseModel) -> BaseModel:
        if not (getattr(value, "left", None) or getattr(value, "right", None)):
            raise ValueError("至少给一个")
        return value

    @tool(
        name="wired_probe",
        description="d",
        capability=Capability.FILE,
        model_validator=_require_one,
    )
    class _Wired(BaseTool):
        @param("left", desc="左")
        @param("right", desc="右")
        async def _dispatch(self, *, left: str | None = None, right: str | None = None) -> None:  # pragma: no cover
            return None

    assert _Wired.input_model.model_validate({"right": "y"}).right == "y"
    with pytest.raises(Exception):
        _Wired.input_model.model_validate({})


def test_the_signature_is_the_only_source_of_parameter_types():
    """签名改成什么，模型就看到什么——没有第二处可漂移。

    这是本项目里被替换掉的那种真实缺陷：手写输入模型写着 ``int``，而实现收 ``str``，
    两者各自正确、合起来是错的。
    """
    assert _ProbeTool.input_model.model_fields["years"].annotation is int
    assert _ProbeTool.input_model.model_fields["symbol"].annotation is str


def test_declarations_are_recorded_globally_for_lookup():
    """注册层与能力层按名称反查声明，无需导入具体工具模块。"""
    assert DECLARED_TOOLS["declared_probe"] is _ProbeTool.__tool_spec__

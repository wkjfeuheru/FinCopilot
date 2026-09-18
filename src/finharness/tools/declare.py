"""工具声明层：``@tool`` / ``@param`` 是工具元数据与参数的**唯一**声明处。

这是三层结构的最底层，也是「调用层发指令 → 注册层找工具」得以成立的前提：注册层
（``registry``）、发现层（``search_tools``）与路由层读的是同一份声明。在此之前，同一
件事在三处各写一遍——``capabilities.TOOL_CAPABILITY`` 记录能力、``registry.DEFAULT_LAZY_TOOLS``
记录层级、``settings.tools.resident|lazy`` 记录运维覆盖——新增一个工具要同步改三处，
而漏改任何一处都只在运行期以"用错工具"或"工具列得出来却调不到"的形式暴露。

参数也只声明一次：**名称、类型与默认值取自 ``_dispatch`` 签名**，``@param`` 只补
Pydantic 无法表达的描述与约束（``desc`` / ``min_length`` / 自定义校验）。旧写法里
``AnnouncementsInput.symbol/since/top_n`` 必须与 ``_dispatch(*, symbol, since, top_n=20)``
手工对齐，两处漂移时既不报错也不报警，只是在模型眼里多出一个不存在的参数。

装饰器的产物是一个 ``ToolSpec``：它随类一起被声明，并被登记进 ``DECLARED_TOOLS``，
供注册层与能力层按名称反查，而无需导入具体工具模块。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Callable, Literal, get_type_hints

from pydantic import BaseModel, Field, create_model
from pydantic import model_validator as _model_validator
from pydantic.functional_validators import AfterValidator

# 装饰器把参数声明挂到 ``_dispatch`` 的这个属性上；名称带模块前缀以免与业务属性撞车。
_PARAMS_ATTR = "__finharness_params__"
# 区分"未声明默认值"与"默认值就是 None"。
_UNSET: Any = object()


class PermissionLevel(str, Enum):
    READ = "read"
    WRITE = "write"


class ToolGroup(str, Enum):
    FIN_DATA = "金融-数据"
    FIN_CALC = "金融-计算"
    FIN_OUTPUT = "金融-输出"
    GENERIC = "通用"
    META = "元"


class Capability(str, Enum):
    """工具用途。当两个工具可互相替代时，它们共享同一能力；能力不同意味着工作本身
    确实不同。

    它定义在声明模块，是因为"这个工具是干什么的"本身就是一项声明；``tools/capabilities.py``
    在此之上提供*按能力使用它*的推理（哪些能力代表真正的研究工作、如何从文本推断意图）。
    """

    MARKET = "行情"
    FINANCIAL = "财务"
    VALUATION = "估值"
    PEER = "同业"
    NEWS = "新闻"
    ANNOUNCEMENT = "公告"
    RESEARCH_REPORT = "研报"
    MACRO = "宏观"
    INDUSTRY = "行业"
    COMPUTE = "计算"
    OUTPUT = "输出"
    FILE = "文件"
    WEB = "联网"
    META = "元"


class Tier(str, Enum):
    """注册层级，决定 schema 何时进入请求。

    ``RESIDENT`` 每轮随请求携带；``LAZY`` 只在被发现层检索命中或被路由层推断需要之后
    才注入（docs 03.4.3）。区分它们的唯一理由是 token：常驻 schema 每轮都要重发。
    """

    RESIDENT = "resident"
    LAZY = "lazy"


class DeclarationError(RuntimeError):
    """工具声明本身有缺陷（而非某次调用出错），因此在导入期就会暴露。"""


# 数据工具共享的展示策略参数。``detail`` 是同一次取数的宽度开关，而不是另一种能力，
# 因此它属于**策略**而非分发参数：工具主体不接收它，由 ``BaseTool.run`` 抽出交给渲染器。
# 集中声明在这里，使 12 个数据工具不必各写一遍（也不必各写错一遍）。
_DETAIL_PARAM = "detail"
_DETAIL_DESCRIPTION = (
    "返回详细程度：summary（默认，摘要+近期明细）或 "
    'full（更多行列，数据已缓存不重复取数）'
)


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """一个参数的声明。类型与默认值默认取签名，此处只存**补充**信息。"""

    name: str
    description: str = ""
    # 显式覆盖签名注解。签名有时比真实契约更窄——如 ``sections: list[dict]`` 的真实元素
    # 是一个结构化模型。写更精确的注解，模型才拿得到嵌套字段的说明。
    annotation: Any = None
    # 显式覆盖签名默认值。函数签名里写不出 ``default_factory``，因此需要
    # ``@param(..., default_factory=...)``（如"默认近半年"这类随当前日期变化的默认值）。
    default: Any = _UNSET
    default_factory: Callable[[], Any] | None = None
    # 字段级自定义校验：接受值、返回（可能规范化后的）值，非法时抛 ValueError。
    validate: Callable[[Any], Any] | None = None
    # 策略参数不在 ``_dispatch`` 签名上——它由渲染侧消费。标注出来，``run`` 才知道在
    # 调用工具主体前先把它摘掉。
    strategy: bool = False
    field_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """一个工具的完整声明，由 ``@tool`` 生成。"""

    name: str
    description: str
    capability: Capability
    tier: Tier
    group: ToolGroup
    permission: PermissionLevel
    timeout: int | None = None
    result_tokens: int | None = None
    output_schema_note: str = ""
    review_eligible: bool = True
    needs_interactive: bool = False
    needs_coordinator: bool = False
    data_tool: bool = False
    # 由签名与 ``@param`` 合成的 Pydantic 模型。它是**生成物**，不是手写的输入类：
    # 存在的意义是让校验与 JSON Schema 继续走 Pydantic 那条已被验证过的路径。
    input_model: type[BaseModel] = BaseModel
    params: tuple[ParamSpec, ...] = ()

    @property
    def strategy_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.params if spec.strategy)

    @property
    def required_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, info in self.input_model.model_fields.items() if info.is_required()
        )

    @property
    def optional_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, info in self.input_model.model_fields.items() if not info.is_required()
        )


# 声明登记表：``@tool`` 在类定义时写入，供注册层与能力层按名称反查，避免为了让
# capabilities 认识一个新工具而反向导入整棵工具模块树（那会引入循环导入）。
DECLARED_TOOLS: dict[str, ToolSpec] = {}


def declared(name: str) -> ToolSpec:
    """按名称取回声明；未声明即抛错。

    抛错而非返回 ``None``：查不到声明说明该工具绕过了 ``@tool``，那是声明缺陷。
    静默的默认值会让它带着一个空能力继续"看起来正常"地跑下去。
    """
    try:
        return DECLARED_TOOLS[name]
    except KeyError as exc:
        raise DeclarationError(f"工具 {name!r} 未通过 @tool 声明") from exc


def param(
    name: str,
    *,
    annotation: Any = None,
    desc: str = "",
    default: Any = _UNSET,
    default_factory: Callable[[], Any] | None = None,
    validate: Callable[[Any], Any] | None = None,
    strategy: bool = False,
    **field_kwargs: Any,
) -> Callable[[Callable], Callable]:
    """为一个 ``_dispatch`` 参数补充描述与约束。

    只补**签名说不出来的**东西：类型与默认值以签名为准，因此二者永不漂移。``annotation``
    用于签名比真实契约更窄的情形（嵌套结构），``default`` 与 ``default_factory`` 用于签名
    无法表达的情形。``field_kwargs`` 原样转交 ``pydantic.Field``（如 ``min_length``）。
    """

    def decorate(func: Callable) -> Callable:
        existing: dict[str, ParamSpec] = dict(getattr(func, _PARAMS_ATTR, {}))
        existing[name] = ParamSpec(
            name=name,
            description=desc,
            annotation=annotation,
            default=default,
            default_factory=default_factory,
            validate=validate,
            strategy=strategy,
            field_kwargs=dict(field_kwargs),
        )
        setattr(func, _PARAMS_ATTR, existing)
        return func

    return decorate


def tool(
    *,
    name: str,
    description: str,
    capability: Capability,
    tier: Tier | str = Tier.RESIDENT,
    group: ToolGroup | str = ToolGroup.GENERIC,
    permission: PermissionLevel | str = PermissionLevel.READ,
    timeout: int | None = None,
    result_tokens: int | None = None,
    output_schema_note: str = "",
    review_eligible: bool = True,
    needs_interactive: bool = False,
    needs_coordinator: bool = False,
    data_tool: bool = False,
    model_validator: Callable[[Any], Any] | None = None,
    params_model: type[BaseModel] | None = None,
) -> Callable[[type], type]:
    """声明一个工具：元数据、层级、能力与参数，一次说清。

    参数 schema 由 ``_dispatch`` 签名合成（``@param`` 补充描述），因此签名是参数的唯一
    真相。``params_model`` 是逃生口：跨字段的**结构**（如报告章节列表）仍可交给一个
    手写的 Pydantic 模型作为嵌套类型，但工具自身的输入结构不应再手写——那正是本装饰器
    要消灭的重复。

    ``model_validator`` 用于跨字段校验（如"两个来源必须给一个"），在字段级校验之后运行，
    语义与手写模型上的 ``@model_validator(mode="after")`` 一致。
    """

    def decorate(cls: type) -> type:
        dispatch = cls.__dict__.get("_dispatch") or getattr(cls, "_dispatch", None)
        declared_params: dict[str, ParamSpec] = dict(getattr(dispatch, _PARAMS_ATTR, {}))

        if params_model is not None:
            params = tuple(declared_params.values())
            model = params_model
        else:
            params, model = _build_params(
                name,
                dispatch,
                declared_params,
                data_tool=data_tool,
                model_validator=model_validator,
            )

        spec = ToolSpec(
            name=name,
            description=description,
            capability=Capability(capability),
            tier=Tier(tier),
            group=ToolGroup(group),
            permission=PermissionLevel(permission),
            timeout=timeout,
            result_tokens=result_tokens,
            output_schema_note=output_schema_note,
            review_eligible=review_eligible,
            needs_interactive=needs_interactive,
            needs_coordinator=needs_coordinator,
            data_tool=data_tool,
            input_model=model,
            params=params,
        )

        # 同时把声明投影回类属性：权限链与子代理子集在**类**（而非实例）上读
        # ``name``/``group``/``permission``/``review_eligible``，类属性是让那些读取
        # 不加改动就继续成立的最省事做法（``property`` 在类上访问只会返回描述符本身）。
        cls.__tool_spec__ = spec
        cls.name = spec.name
        cls.description = spec.description
        cls.capability = spec.capability
        cls.tier = spec.tier
        cls.group = spec.group
        cls.permission = spec.permission
        cls.timeout = spec.timeout
        cls.result_tokens = spec.result_tokens
        cls.output_schema_note = spec.output_schema_note
        cls.review_eligible = spec.review_eligible
        cls.needs_interactive = spec.needs_interactive
        cls.needs_coordinator = spec.needs_coordinator
        cls.data_tool = spec.data_tool
        cls.input_model = spec.input_model
        DECLARED_TOOLS[name] = spec
        return cls

    return decorate


@dataclass(slots=True)
class _Plan:
    """一个待生成的字段：注解 + 默认值 + 描述/约束。"""

    name: str
    annotation: Any
    spec: ParamSpec
    default: Any = _UNSET
    default_factory: Callable[[], Any] | None = None


def _build_params(
    tool_name: str,
    dispatch: Callable | None,
    declared_params: dict[str, ParamSpec],
    *,
    data_tool: bool,
    model_validator: Callable[[Any], Any] | None,
) -> tuple[tuple[ParamSpec, ...], type[BaseModel]]:
    """把签名与 ``@param`` 合成一个 Pydantic 模型。"""
    plans: list[_Plan] = []
    seen: set[str] = set()

    # 数据工具的 ``detail`` 排在最前，与它还是 ``DataInput`` 基类字段时的位置一致，
    # 因此生成的 schema 属性序不变。
    if data_tool:
        detail = ParamSpec(name=_DETAIL_PARAM, description=_DETAIL_DESCRIPTION, strategy=True)
        plans.append(_Plan(_DETAIL_PARAM, Literal["summary", "full"], detail, default="summary"))
        seen.add(_DETAIL_PARAM)

    if dispatch is not None:
        hints = _type_hints(dispatch)
        for pname, parameter in inspect.signature(dispatch).parameters.items():
            if pname == "self" or parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            spec = declared_params.get(pname) or ParamSpec(name=pname)
            annotation = spec.annotation
            if annotation is None:
                annotation = hints.get(pname, parameter.annotation)
                if annotation is inspect.Parameter.empty:
                    annotation = Any
            plan = _Plan(pname, annotation, spec)
            if parameter.default is not inspect.Parameter.empty:
                plan.default = parameter.default
            plans.append(plan)
            seen.add(pname)

    # 声明了却不在签名上的参数，只可能是策略参数（渲染侧消费）。其余情况是笔误，
    # 必须在导入期就炸掉，而不是留到模型调用时才发现该参数从未生效。
    for pname, spec in declared_params.items():
        if pname in seen:
            continue
        if not spec.strategy:
            raise DeclarationError(
                f"工具 {tool_name}：参数 {pname!r} 既不在 _dispatch 签名上，也未标为策略参数"
            )
        plans.append(_Plan(pname, spec.annotation if spec.annotation is not None else Any, spec))

    fields: dict[str, Any] = {}
    for plan in plans:
        fields[plan.name] = (_annotated(plan), _field(plan))

    validators: dict[str, Any] = {}
    if model_validator is not None:
        validators["_cross_field"] = _model_validator(mode="after")(model_validator)

    model = create_model(f"{tool_name}_params", __validators__=validators, **fields)
    return tuple(plan.spec for plan in plans), model


def _field(plan: _Plan) -> Any:
    """构造一个字段的 ``FieldInfo``（注解由调用方单独给出）。

    显式声明优先于签名默认值：签名写不出 ``default_factory``，而"默认近半年"这类默认值
    必须随调用日期变化，若退回签名就会把该参数变成必填——那是一个静默的行为回退。
    """
    options: dict[str, Any] = dict(plan.spec.field_kwargs)
    if plan.spec.description and "description" not in options:
        options["description"] = plan.spec.description

    if plan.spec.default is not _UNSET:
        options["default"] = plan.spec.default
    elif plan.spec.default_factory is not None:
        options["default_factory"] = plan.spec.default_factory
    elif plan.default is not _UNSET:
        options["default"] = plan.default

    if plan.spec.validate is not None:
        return Field(**options), AfterValidator(plan.spec.validate)
    return Field(**options)


def _annotated(plan: _Plan) -> Any:
    """字段注解；带自定义校验时把校验器挂到注解上。

    用 ``AfterValidator`` 而不是在 ``Field`` 上做后置处理，是为了让校验失败与 Pydantic
    自身的失败走同一条错误路径（同一个 ``ValidationError``、同一种 loc 形态），
    ``_validation_message`` 因此无需为两者分别拼装文案。
    """
    if plan.spec.validate is None:
        return plan.annotation
    return Annotated[plan.annotation, AfterValidator(plan.spec.validate)]



def _type_hints(dispatch: Callable) -> dict[str, Any]:
    """解析签名注解。

    工具模块普遍有 ``from __future__ import annotations``，于是注解在运行期是字符串；
    不解析就会把一个字面量 ``'str'`` 当作类型交给 Pydantic。
    """
    try:
        return get_type_hints(dispatch)
    except Exception:  # noqa: BLE001 - 注解不可解析时退回原始签名，交给 Pydantic 报错
        return {}

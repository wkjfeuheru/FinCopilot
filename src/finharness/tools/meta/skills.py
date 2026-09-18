"""技能目录与场景路由（docs 03.8）。

*技能* 是一个场景包：一份 ``SKILL.md`` 掌管流程（步骤、场景边界、何时查阅哪个文件），
方法论位于同级的 ``references/*.md`` 中，报告模板位于 ``assets/*.md``。

加载不再是一个模型可调用的工具，而是引擎的**路由**动作：引擎从用户消息与计划意图推断
需要哪些能力，据此注入对应的场景流程与方法论。理由有两条。其一，模型不该管理自己的
提示词里放什么——那是引擎的职责。其二，"该不该加载"取决于意图，而意图在模型发出工具
调用之前就已经可从文本判定；把它做成工具只会让每份方法论都多付一次往返。

注入是幂等的，且按目标可观测：同一目标每个会话只注入一次，每次都留档。

Frontmatter schema：

    name / description            —— 这是哪个场景、何时进入
    inputs / outputs              —— 语义契约
    use_cases / examples          —— 供检索层选择场景的辅助信息
    related_skills                —— 组合技能；仅作为提示呈现，不自动加载
    allowed_tools                 —— 正文预期使用的建议工具
    content_estimate / version    —— 预算提示与修订版本

可加载文件列表*不*在 frontmatter 中声明：目录本身就是唯一权威来源，因此文件既不
可能没有文档记录地存在，也不可能被声明为缺失。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from finharness.tools.declare import Capability

FILE_DIRS = ("references", "assets")


class SkillError(RuntimeError):
    """当技能文件缺失或格式错误时抛出。"""


@dataclass(frozen=True, slots=True)
class SkillMeta:
    name: str
    description: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    use_cases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    related_skills: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    content_estimate: int = 0
    version: int = 1
    path: str = ""
    files: tuple[str, ...] = ()

    def selection_hint(self) -> str:
        """供列表/检索输出使用的紧凑多行描述。"""
        lines = [self.description]
        if self.use_cases:
            lines.append("  适用：" + "；".join(self.use_cases))
        return "\n".join(lines)


@dataclass(slots=True)
class SkillLoadRecord:
    """一次被观测到的加载；可观测性保障的依据。"""

    name: str
    ts: str
    duration_ms: float
    reused: bool


def _split_frontmatter(text: str, *, source: Path) -> tuple[dict, str]:
    """拆分 SKILL.md 的 YAML frontmatter 与正文；格式不合法时抛 SkillError。"""
    if not text.startswith("---"):
        raise SkillError(f"SKILL.md 缺少 frontmatter：{source}")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillError(f"SKILL.md frontmatter 未闭合：{source}")
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise SkillError(f"SKILL.md frontmatter 不是合法 YAML：{source}：{exc}") from exc
    if not isinstance(meta, dict):
        raise SkillError(f"SKILL.md frontmatter 必须是映射：{source}")
    return meta, parts[2].strip()


def _as_tuple(value: object) -> tuple[str, ...]:
    """把 frontmatter 值统一规整为字符串元组。"""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)  # type: ignore[union-attr]


def _skill_files(source: Path) -> tuple[str, ...]:
    """枚举某个场景的可加载文件（references/*.md、assets/*.md）。"""
    found: list[str] = []
    for sub in FILE_DIRS:
        directory = source.parent / sub
        if directory.is_dir():
            found.extend(f"{sub}/{path.name}" for path in sorted(directory.glob("*.md")))
    return tuple(found)


def _to_meta(meta: dict, *, source: Path) -> SkillMeta:
    """由 frontmatter 映射构造 SkillMeta，并补全其文件清单。"""
    return SkillMeta(
        name=str(meta.get("name") or source.parent.name),
        description=str(meta.get("description") or ""),
        inputs=_as_tuple(meta.get("inputs")),
        outputs=_as_tuple(meta.get("outputs")),
        use_cases=_as_tuple(meta.get("use_cases")),
        examples=_as_tuple(meta.get("examples")),
        related_skills=_as_tuple(meta.get("related_skills")),
        allowed_tools=_as_tuple(meta.get("allowed_tools")),
        content_estimate=int(meta.get("content_estimate") or 0),
        version=int(meta.get("version") or 1),
        path=str(source),
        files=_skill_files(source),
    )


class SkillRegistry:
    """技能目录，按目标实现可观测、幂等的加载。"""

    def __init__(self, skills_dir: str | Path) -> None:
        self.root = Path(skills_dir)
        self._metas: dict[str, SkillMeta] = {}
        self._bodies: dict[str, str] = {}
        self._loaded: dict[str, SkillLoadRecord] = {}
        self._history: list[SkillLoadRecord] = []
        self._scan()

    def _scan(self) -> None:
        """扫描技能根目录下所有 SKILL.md，解析并缓存其元数据与正文。"""
        if not self.root.is_dir():
            return
        for path in sorted(self.root.glob("*/SKILL.md")):
            try:
                meta, body = _split_frontmatter(path.read_text(encoding="utf-8"), source=path)
            except (SkillError, OSError):
                # 格式错误的技能不能拖垮整个目录。
                continue
            parsed = _to_meta(meta, source=path)
            self._metas[parsed.name] = parsed
            self._bodies[parsed.name] = body

    # -- 目录 ------------------------------------------------------------
    def names(self) -> list[str]:
        return list(self._metas)

    def get(self, name: str) -> SkillMeta | None:
        return self._metas.get(name)

    def list_skills(self) -> list[SkillMeta]:
        return list(self._metas.values())

    def describe(self) -> str:
        """目录的可读描述，供 CLI `/skills` 与排障使用。

        它不再是一个模型可调用的工具（模型不经工具调用去了解自己有哪些方法），但
        "这个包里有什么"仍需要一个可打印的视图。
        """
        if not self._metas:
            return "（技能库为空）"
        blocks: list[str] = []
        for meta in self._metas.values():
            lines = [f"- {meta.name}：{meta.selection_hint()}"]
            if meta.files:
                lines.append("  可加载文件：" + "、".join(meta.files))
            blocks.append("\n".join(lines))
        return "\n".join(blocks)


    def search(self, query: str, *, limit: int = 5) -> list[tuple[SkillMeta, int]]:
        """关键词检索，返回 (技能, 得分) 对。

        ``use_cases`` 与 ``examples`` 比单纯的描述承载更多选择信号，因此命中它们的
        权重高于仅命中描述。
        """
        terms = [term for term in query.lower().replace("，", " ").split() if term]
        scored: list[tuple[SkillMeta, int]] = []
        for meta in self._metas.values():
            name_l = meta.name.lower()
            desc_l = meta.description.lower()
            case_l = " ".join(meta.use_cases).lower()
            ex_l = " ".join(meta.examples).lower()
            score = 0
            for term in terms:
                if term in name_l:
                    score += 4
                if term in case_l or term in ex_l:
                    score += 3
                if term in desc_l:
                    score += 2
            if score:
                scored.append((meta, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].name))
        return scored[:limit]

    # -- 加载（按目标：幂等 + 可观测） --------------------------
    @staticmethod
    def _load_key(name: str, file: str | None) -> str:
        """构造按目标区分的加载键（技能名，或 ``技能名/文件``）。"""
        return name if not file else f"{name}/{file}"

    def is_loaded(self, key: str) -> bool:
        return key in self._loaded

    def resolve_file(self, name: str, file: str) -> str:
        """校验所请求的文件确属该场景，并返回其相对路径。"""
        rel = file.replace("\\", "/").strip("/")
        meta = self._metas.get(name)
        if meta is None:
            raise SkillError(f"未找到技能：{name}")
        if rel not in meta.files:
            allowed = "、".join(meta.files) or "（无可加载文件）"
            raise SkillError(f"技能 {name} 下没有文件：{file}；可加载：{allowed}")
        return rel

    def load(
        self, name: str, *, file: str | None = None
    ) -> tuple[SkillMeta, str, SkillLoadRecord]:
        """返回某目标的正文及一条加载记录。重复加载是廉价的空操作。

        不带 ``file`` 时加载场景流程（SKILL.md）；带 ``file`` 时加载该场景随包文件之一
        （``references/...``／``assets/...``）。
        """
        started = time.monotonic()
        meta = self._metas.get(name)
        if meta is None:
            raise SkillError(f"未找到技能：{name}")

        if file:
            rel = self.resolve_file(name, file)
            target = Path(meta.path).parent / rel
            try:
                text = target.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise SkillError(f"读取技能文件失败：{target}") from exc
            key = self._load_key(name, rel)
        else:
            key = name
            text = self._bodies.get(name, "")

        reused = self.is_loaded(key)
        if not reused:
            # 幂等性：正文只读取一次并缓存，因此重试调用不可能产生不同结果。
            self._bodies.setdefault(key, text)
        record = SkillLoadRecord(
            name=key,
            ts=datetime.now().astimezone().isoformat(timespec="seconds"),
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            reused=reused,
        )
        self._loaded[key] = record
        self._history.append(record)
        return meta, text, record

    def load_history(self) -> list[SkillLoadRecord]:
        return list(self._history)

    def dependency_hint(self, meta: SkillMeta) -> str:
        """点名组合技能的提示；它们不会被自动加载。"""
        if not meta.related_skills:
            return ""
        return "本技能可复用：" + "、".join(meta.related_skills)


# --- 路由（引擎侧） ----------------------------------------------------------

# 能力 -> 该方法论所在文件。这是"注册-发现-路由"里**路由**那一层的映射表：意图（能力）
# 到载体（场景与方法论文档）的对应关系。
#
# 关键词表不在这里重写：它由 ``capabilities`` 持有，因为"这句话在说什么"与"这次用错没
# 用错工具"必须是同一个判断。这里只回答"既然如此，该读哪一份方法论"。
_METHODOLOGY_BY_CAPABILITY: dict[Capability, tuple[str, str]] = {
    Capability.FINANCIAL: ("equity-research", "references/profitability.md"),
    Capability.VALUATION: ("equity-research", "references/valuation.md"),
    Capability.PEER: ("equity-research", "references/industry-focus.md"),
    Capability.INDUSTRY: ("industry-research", "references/competition.md"),
    Capability.MACRO: ("macro-research", "references/cycle.md"),
    Capability.COMPUTE: ("quant-factor", "references/single-series.md"),
}

# 需要多少个研究能力同时命中，才值得连场景流程（SKILL.md）一起注入。
#
# 场景流程描述的是"一次完整研究的步骤编排"，对单点提问是无用的开销；方法论则回答"这个
# 指标该怎么读"，单点提问正需要它。两道门槛因此不同，这也是简单提问（"茅台 ROE 为什么
# 掉这么多"）能拿到方法论、却不会被塞进一份权益研究流程的原因。
_SKILL_FLOW_THRESHOLD = 2

# 成稿请求的关键词。报告模板只在真的要出成稿时才注入：它是排版契约，对"解释一下差异"
# 这类对话内作答是纯开销，而 prompt 明确写了只有用户点名要报告才走成稿流程。
_REPORT_KEYWORDS = ("研报", "报告", "成稿", "导出", "写一份", "出一份")
# 成稿请求命中时使用的默认场景（用户要报告但没指明研究维度）。
_DEFAULT_REPORT_SKILL = "equity-research"
_REPORT_TEMPLATE = "assets/report-template.md"


def report_requested(text: str) -> bool:
    """该文本是否明确要求出成稿。"""
    lowered = str(text or "").lower()
    return any(word in lowered for word in _REPORT_KEYWORDS)


@dataclass(frozen=True, slots=True)
class RoutedSkill:
    """路由结果的一项：要注入哪个技能的哪份正文。"""

    skill: str
    file: str | None
    key: str


def route(
    *,
    capabilities: set[Capability],
    hinted: tuple[str, ...] = (),
    report: bool = False,
) -> list[RoutedSkill]:
    """把"这次需要什么能力"解析为要注入的方法论。

    ``capabilities`` 是引擎从用户消息与计划意图推断出的能力集合；``hinted`` 是计划步骤
    显式写下的技能名（``skill_hint``），它优先于推断——模型点名了就照给；``report`` 表示
    用户明确要成稿，此时额外注入报告模板。

    返回按注入顺序排列的目标列表：先场景流程（若跨过门槛），后方法论，最后模板。
    """
    targets: list[RoutedSkill] = []
    seen: set[str] = set()

    def add(skill: str, file: str | None) -> None:
        key = skill if file is None else f"{skill}/{file}"
        if key in seen:
            return
        seen.add(key)
        targets.append(RoutedSkill(skill=skill, file=file, key=key))

    for name in hinted:
        cleaned = str(name or "").strip()
        if cleaned:
            add(cleaned, None)

    for capability in sorted(capabilities, key=lambda item: item.value):
        mapping = _METHODOLOGY_BY_CAPABILITY.get(capability)
        if mapping is None:
            continue
        skill, file = mapping
        if len(capabilities & _researched_capabilities()) >= _SKILL_FLOW_THRESHOLD:
            add(skill, None)
        add(skill, file)

    if report:
        # 成稿时模板是必备契约：字号、章节与引用附录的形状都由它决定。用户要了报告却
        # 没能指明研究维度时，用默认场景的模板而不是不给模板。
        skill = _report_skill(capabilities, hinted)
        add(skill, _REPORT_TEMPLATE)

    return targets


def _report_skill(capabilities: set[Capability], hinted: tuple[str, ...]) -> str:
    """成稿应使用哪个场景的模板。

    优先计划点名过的场景，其次按能力推断（行业问题用行业模板），最后退回默认。
    取第一个已识别的场景即可：模板之间的差异小于"有没有模板"的差异。
    """
    for name in hinted:
        cleaned = str(name or "").strip()
        if cleaned:
            return cleaned
    for capability in sorted(capabilities, key=lambda item: item.value):
        mapping = _METHODOLOGY_BY_CAPABILITY.get(capability)
        if mapping is not None:
            return mapping[0]
    return _DEFAULT_REPORT_SKILL


def _researched_capabilities() -> set[Capability]:
    """研究类能力集合。

    以函数而非模块常量取回，避免 ``capabilities`` 与本模块在导入期相互引用。
    """
    from finharness.tools.capabilities import RESEARCH_CAPABILITIES

    return RESEARCH_CAPABILITIES


def skill_files(skills_dir: str | Path) -> dict[str, tuple[str, ...]]:
    """技能名 -> 可加载文件清单，供路由层与目录展示共用。"""
    registry = SkillRegistry(skills_dir)
    return {meta.name: meta.files for meta in registry.list_skills()}

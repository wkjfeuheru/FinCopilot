"""技能目录及其元工具（docs 03.8）。

*技能* 是一个场景包：一份 ``SKILL.md`` 掌管流程（步骤、场景边界、何时查阅哪个文件），
方法论位于同级的 ``references/*.md`` 中，报告模板位于 ``assets/*.md``。加载按需且分
两阶段：``list_skills`` 只读 frontmatter（并枚举随包文件）而不触碰正文；``load_skill``
注入一份正文——场景流程，或单个参考文件。

加载是幂等的，且按目标可观测：重复加载零成本，每次加载都可追溯。

Frontmatter schema：

    name / description            —— 这是哪个场景、何时进入
    inputs / outputs              —— 语义契约
    use_cases / examples          —— 供模型选择的辅助信息
    related_skills                —— 组合技能；仅作为提示呈现，不自动加载
    allowed_tools                 —— 正文预期使用的建议工具
    content_estimate / version    —— 预算提示与修订版本

可加载文件列表*不*在 frontmatter 中声明：目录本身就是唯一权威来源，因此文件既不
可能没有文档记录地存在，也不可能被声明为缺失。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup

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
        if not self._metas:
            return "（技能库为空）"
        blocks: list[str] = []
        for meta in self._metas.values():
            lines = [f"- {meta.name}：{meta.selection_hint()}"]
            if meta.files:
                lines.append(
                    "  可加载文件：" + "、".join(meta.files) + "（load_skill 的 file 参数）"
                )
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
        return "本技能可复用：" + "、".join(meta.related_skills) + "（按需自行 load_skill）"


# --- 工具 ------------------------------------------------------------------

class ListSkillsInput(BaseModel):
    """无参数：列出目录的代价始终很低。"""


class ListSkillsTool(BaseTool):
    name = "list_skills"
    description = "列出可用的投研场景技能（名称+用途+可加载的参考/模板文件），不加载正文。"
    input_model = ListSkillsInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self) -> RawData:
        """返回技能目录的可读描述（名称、用途与可加载文件）。"""
        registry = SkillRegistry(self.data.settings.paths.skills_dir)
        return RawData(kind="text", text=registry.describe(), endpoint="skills:list")


class LoadSkillInput(BaseModel):
    name: str = Field(description="场景技能名，如 equity-research")
    file: str | None = Field(
        default=None,
        description=(
            "要加载的场景文件路径，如 references/valuation.md 或 assets/report-template.md；"
            "省略则加载场景流程（SKILL.md）。可用文件见 list_skills。"
        ),
    )


class LoadSkillTool(BaseTool):
    name = "load_skill"
    description = (
        "加载场景技能的流程（SKILL.md）或其参考/模板文件（references/、assets/ 下的 md）。"
        "重复加载同一目标幂等，零成本。"
    )
    input_model = LoadSkillInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self, *, name: str, file: str | None = None) -> RawData:
        """加载技能流程或其中某个参考/模板文件，组装带复用提示与正文的文本载荷。"""
        registry = SkillRegistry(self.data.settings.paths.skills_dir)
        meta, body, record = registry.load(name, file=file)

        # 会话上下文记录加载过的目标以供系统提示状态块使用，并告知本次是否为重复加载。
        key = record.name
        already_in_session = self.ctx is not None and key in self.ctx.loaded_skills
        if self.ctx is not None:
            self.ctx.add_skill(key)

        reused = record.reused or already_in_session
        sections: list[str] = []
        if file:
            sections.append(
                f"已加载（复用，未重复注入）：{key}" if reused else f"参考文件：{key}"
            )
        else:
            sections.append("已加载（复用，未重复注入）：" + name if reused else "技能：" + name)
            if meta.inputs:
                sections.append("输入：" + "；".join(meta.inputs))
            if meta.outputs:
                sections.append("产出：" + "；".join(meta.outputs))
            hint = registry.dependency_hint(meta)
            if hint:
                sections.append(hint)
        sections.append(body)
        return RawData(
            kind="text",
            text="\n\n".join(sections),
            endpoint="skills:load",
            params={"name": meta.name, "file": file or "", "reused": record.reused},
        )

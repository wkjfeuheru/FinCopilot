"""Skill catalogue and its meta tools (docs 03.8).

Skills are methodology content addressed like tools: discoverable by search,
documented with an explicit semantic contract, and loaded on demand. Loading is
idempotent and observed, so a retried load costs nothing and every load is
traceable.

Frontmatter schema:

    name / description            — what it is and when to use it (drives search)
    inputs / outputs              — the semantic contract
    use_cases / examples          — selection aids for the model
    related_skills                — composed skills; surfaced as a hint, not auto-loaded
    allowed_tools                 — suggested tools the body expects
    content_estimate / version    — budget hint and revision
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


class SkillError(RuntimeError):
    """Raised when a skill file is missing or malformed."""


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

    def selection_hint(self) -> str:
        """Compact multi-line description used by list/search output."""
        lines = [self.description]
        if self.use_cases:
            lines.append("  适用：" + "；".join(self.use_cases))
        return "\n".join(lines)


@dataclass(slots=True)
class SkillLoadRecord:
    """One observed load; the basis for the observability guarantee."""

    name: str
    ts: str
    duration_ms: float
    reused: bool


def _split_frontmatter(text: str, *, source: Path) -> tuple[dict, str]:
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
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)  # type: ignore[union-attr]


def _to_meta(meta: dict, *, source: Path) -> SkillMeta:
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
    )


class SkillRegistry:
    """Skill catalogue with observed, idempotent loads."""

    def __init__(self, skills_dir: str | Path) -> None:
        self.root = Path(skills_dir)
        self._metas: dict[str, SkillMeta] = {}
        self._bodies: dict[str, str] = {}
        self._loaded: dict[str, SkillLoadRecord] = {}
        self._history: list[SkillLoadRecord] = []
        self._scan()

    def _scan(self) -> None:
        if not self.root.is_dir():
            return
        for path in sorted(self.root.glob("*/SKILL.md")):
            try:
                meta, body = _split_frontmatter(path.read_text(encoding="utf-8"), source=path)
            except (SkillError, OSError):
                # A malformed skill must not take down the catalogue.
                continue
            parsed = _to_meta(meta, source=path)
            self._metas[parsed.name] = parsed
            self._bodies[parsed.name] = body

    # -- catalogue ------------------------------------------------------------
    def names(self) -> list[str]:
        return list(self._metas)

    def get(self, name: str) -> SkillMeta | None:
        return self._metas.get(name)

    def list_skills(self) -> list[SkillMeta]:
        return list(self._metas.values())

    def describe(self) -> str:
        if not self._metas:
            return "（技能库为空）"
        return "\n".join(f"- {meta.name}：{meta.selection_hint()}" for meta in self._metas.values())

    def search(self, query: str, *, limit: int = 5) -> list[tuple[SkillMeta, int]]:
        """Keyword search returning (skill, score) pairs.

        ``use_cases`` and ``examples`` carry more selection signal than the bare
        description, so a hit there is weighted above a plain description hit.
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

    # -- loading (idempotent + observed) --------------------------------------
    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def load(self, name: str) -> tuple[SkillMeta, str, SkillLoadRecord]:
        """Return the body plus a load record. Repeat loads are cheap no-ops."""
        started = time.monotonic()
        meta = self._metas.get(name)
        if meta is None:
            raise SkillError(f"未找到技能：{name}")

        reused = self.is_loaded(name)
        if not reused:
            # Idempotence: the body is read once and cached, so a retried call
            # cannot produce a different result.
            self._bodies.setdefault(name, self._bodies.get(name, ""))
        record = SkillLoadRecord(
            name=name,
            ts=datetime.now().astimezone().isoformat(timespec="seconds"),
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            reused=reused,
        )
        self._loaded[name] = record
        self._history.append(record)
        return meta, self._bodies.get(name, ""), record

    def load_history(self) -> list[SkillLoadRecord]:
        return list(self._history)

    def dependency_hint(self, meta: SkillMeta) -> str:
        """A hint naming composed skills; they are not loaded automatically."""
        if not meta.related_skills:
            return ""
        return "本技能可复用：" + "、".join(meta.related_skills) + "（按需自行 load_skill）"


# --- tools ------------------------------------------------------------------

class ListSkillsInput(BaseModel):
    """No parameters: listing the catalogue is always cheap."""


class ListSkillsTool(BaseTool):
    name = "list_skills"
    description = "列出可用的投研方法论技能（名称+用途+适用场景），不加载正文。"
    input_model = ListSkillsInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self) -> RawData:
        registry = SkillRegistry(self.data.settings.paths.skills_dir)
        return RawData(kind="text", text=registry.describe(), endpoint="skills:list")


class LoadSkillInput(BaseModel):
    name: str = Field(description="技能名，如 dupont-analysis")


class LoadSkillTool(BaseTool):
    name = "load_skill"
    description = "加载指定方法论技能的完整正文（重复加载幂等，零成本）。"
    input_model = LoadSkillInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self, *, name: str) -> RawData:
        registry = SkillRegistry(self.data.settings.paths.skills_dir)
        meta, body, record = registry.load(name)

        # The session context records skills for the system-prompt state block,
        # and tells us whether this was a repeat load.
        already_in_session = self.ctx is not None and name in self.ctx.loaded_skills
        if self.ctx is not None:
            self.ctx.add_skill(name)

        header = (
            "已加载（复用，未重复注入）：" + name
            if record.reused or already_in_session
            else "技能：" + name
        )
        sections = [header]
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
            params={"name": meta.name, "reused": record.reused},
        )

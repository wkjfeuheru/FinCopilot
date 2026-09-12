"""Skill catalogue and its meta tools (docs 03.8).

``list_skills`` reads only frontmatter so the catalogue stays cheap; ``load_skill``
injects the body once and reports reuse afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    allowed_tools: tuple[str, ...] = ()
    content_estimate: int = 0
    version: int = 1
    path: str = ""


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


def _to_meta(meta: dict, *, source: Path) -> SkillMeta:
    return SkillMeta(
        name=str(meta.get("name") or source.parent.name),
        description=str(meta.get("description") or ""),
        allowed_tools=tuple(meta.get("allowed_tools") or ()),
        content_estimate=int(meta.get("content_estimate") or 0),
        version=int(meta.get("version") or 1),
        path=str(source),
    )


class SkillLibrary:
    """Reads skills from a directory; tolerant of a missing or empty catalogue."""

    def __init__(self, skills_dir: str | Path) -> None:
        self.root = Path(skills_dir)

    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(self.root.glob("*/SKILL.md"))

    def list_skills(self) -> list[SkillMeta]:
        metas: list[SkillMeta] = []
        for path in self._files():
            try:
                meta, _ = _split_frontmatter(path.read_text(encoding="utf-8"), source=path)
            except (SkillError, OSError):
                continue
            metas.append(_to_meta(meta, source=path))
        return metas

    def load(self, name: str) -> tuple[SkillMeta, str]:
        """Return frontmatter and body for one skill."""
        for path in self._files():
            if path.parent.name != name:
                continue
            meta, body = _split_frontmatter(path.read_text(encoding="utf-8"), source=path)
            return _to_meta(meta, source=path), body
        raise SkillError(f"未找到技能：{name}")

    def describe(self) -> str:
        metas = self.list_skills()
        if not metas:
            return "（技能库为空）"
        return "\n".join(f"- {m.name}：{m.description}" for m in metas)


# --- tools ------------------------------------------------------------------

class ListSkillsInput(BaseModel):
    """No parameters: listing the catalogue is always cheap."""


class ListSkillsTool(BaseTool):
    name = "list_skills"
    description = "列出可用的投研方法论技能清单（名称+用途），不加载正文。"
    input_model = ListSkillsInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self) -> RawData:
        library = SkillLibrary(self.data.settings.paths.skills_dir)
        return RawData(kind="text", text=library.describe(), endpoint="skills:list")


class LoadSkillInput(BaseModel):
    name: str = Field(description="技能名，如 dupont-analysis")


class LoadSkillTool(BaseTool):
    name = "load_skill"
    description = "加载指定方法论技能的完整正文（重复加载会提示已加载，零成本）。"
    input_model = LoadSkillInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self, *, name: str) -> RawData:
        library = SkillLibrary(self.data.settings.paths.skills_dir)
        meta, body = library.load(name)
        # Reuse is free and visible: the context records the skill once.
        already = self.ctx is not None and name in self.ctx.loaded_skills
        if self.ctx is not None:
            self.ctx.add_skill(name)
        header = f"已加载（复用）：{name}\n\n" if already else f"技能：{name}\n\n"
        return RawData(kind="text", text=header + body, endpoint="skills:load", params={"name": meta.name})

"""系统提示词：单一定义，从文件加载（见 prompts/system.md）。

提示词是一项产品资产——它决定 agent 何时制定计划、如何引用数据、何时停止——
因此它存放在可评审的 markdown 文件中，而非字符串字面量中，并且每个入口
（服务端、测试）都加载这同一份文本。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system.md"
RISK_REVIEW_PATH = Path(__file__).resolve().parent.parent / "prompts" / "risk_review.md"
# 该检查清单是评审者的评判标准。它曾经是一个 skill；现在它是一项
# prompt 资产，因为其唯一读取者是评审 sub-agent，而非场景研究流程。
RISK_CHECKLIST_PATH = Path(__file__).resolve().parent.parent / "prompts" / "risk-checklist.md"
# 通用 sub-agent 角色，用于纯粹为隔离上下文而分派任务时（docs 03.10）。
WORKER_PATH = Path(__file__).resolve().parent.parent / "prompts" / "worker.md"


class PromptNotFoundError(RuntimeError):
    """当系统提示词资产缺失或为空时抛出。"""


def _read_prompt(target: Path, label: str) -> str:
    """读取一个 prompt 资产，失败则显式报错；空 prompt 是缺陷，而非一种状态。"""
    try:
        text = target.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PromptNotFoundError(f"无法读取{label}文件：{target}") from exc
    if not text:
        raise PromptNotFoundError(f"{label}文件为空：{target}")
    return text


@lru_cache(maxsize=1)
def system_prompt(path: str | Path | None = None) -> str:
    """返回系统提示词文本；若资产不可用则抛出异常。"""
    return _read_prompt(Path(path) if path is not None else PROMPT_PATH, "系统提示词")


@lru_cache(maxsize=1)
def risk_review_prompt(path: str | Path | None = None) -> str:
    """风险终审 sub-agent 的系统提示词（一个独立角色，而非一种模式）。

    缓存原因与 ``system_prompt`` 相同：每次评审都会读取它，但它在一次运行
    期间从不改变。
    """
    return _read_prompt(Path(path) if path is not None else RISK_REVIEW_PATH, "风险终审提示词")


@lru_cache(maxsize=1)
def risk_checklist_prompt(path: str | Path | None = None) -> str:
    """风险清单：评审 sub-agent 的评判标准。

    其唯一使用方是评审者，评审者会将其注入自己的系统提示词，因此它随
    prompt 一起存放，而非放在场景目录中。
    """
    return _read_prompt(Path(path) if path is not None else RISK_CHECKLIST_PATH, "风险清单")


@lru_cache(maxsize=1)
def worker_prompt(path: str | Path | None = None) -> str:
    """通用 sub-agent 角色提示词（docs 03.10）。

    用于为隔离上下文而分派任务时，而非针对像风险评审这样的具名聚焦点。
    """
    return _read_prompt(Path(path) if path is not None else WORKER_PATH, "子代理提示词")

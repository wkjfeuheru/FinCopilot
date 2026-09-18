"""拒绝模式与沙箱扫描（docs 03.7.2）。

两类规则并行运行：

1. ``DEFAULT_DENY_PATTERNS`` —— 针对交易动作的意图级防御。它们扫描
   ``run_python`` 代码以及每个工具的参数快照，因此即便框架没有注册任何交易
   工具，把交易指令偷偷塞进数据参数的 prompt 注入也会被捕获。
2. 沙箱导入白名单，仅适用于 ``run_python``。

命中时会连同匹配的规则一并报告，使拒绝保持可解释。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 交易意图防御。应用于 run_python 代码与工具参数值。
DEFAULT_DENY_PATTERNS: tuple[str, ...] = (
    r"easytrader",
    r"vnpy",
    r"券商\s*(API|接口|api)",
    r"实盘",
    r"(自动|程序化)?下单",
    r"(买入|卖出|委托|撤单)\s*[\(（]",
    r"submit\w*order",
    r"place\w*order",
    r"trade\w*api",
)

# run_python 的导入白名单（标准库 + 分析栈）。
SANDBOX_IMPORT_ALLOW: frozenset[str] = frozenset(
    {
        "pandas", "numpy", "matplotlib", "statistics", "math",
        "json", "re", "datetime", "collections", "itertools", "functools",
    }
)

# 标识符级拒绝名单：进程、网络、文件系统与动态执行的逃生通道。按单词边界
# 匹配，因此 "cost" 不会被误认成 "os"。
SANDBOX_FORBIDDEN_IDENTIFIERS: tuple[str, ...] = (
    "os", "sys", "subprocess", "socket", "requests", "urllib", "httpx",
    "open", "eval", "exec", "compile", "pickle", "shutil", "pathlib",
    "importlib", "__import__", "globals", "locals", "setattr", "delattr",
)

_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE)

# 递归扫描的深度上限。参数快照来自模型，可能深且宽；限制深度使扫描成本
# 有界，同时覆盖现实中的嵌套参数结构（如报告 sections）。
_MAX_SCAN_DEPTH = 8


@dataclass(frozen=True, slots=True)
class RuleHit:
    """单条规则违规，附带足以解释该拒绝的细节。"""

    rule_index: int
    pattern: str
    where: str  # "code" | "args" | "import" | "identifier"

    def reason(self) -> str:
        return f"命中了第 {self.rule_index} 条规则：{self.pattern}（位置：{self.where}）"


def scan_text(
    text: str, *, where: str = "code", patterns: tuple[str, ...] | None = None
) -> RuleHit | None:
    """把拒绝模式应用于一段文本。

    ``patterns`` 为 None 时使用内置默认集；运维可传入自定义集覆盖它
    （见 ``PermissionGate`` 的 ``deny_patterns``）。
    """
    for index, pattern in enumerate(patterns or DEFAULT_DENY_PATTERNS, start=1):
        if re.search(pattern, text, re.IGNORECASE):
            return RuleHit(rule_index=index, pattern=pattern, where=where)
    return None


def scan_value(
    value, *, where: str = "args", patterns: tuple[str, ...] | None = None, depth: int = 0
) -> RuleHit | None:
    """递归扫描任意参数值。

    嵌套必须被遍历：``write_report`` 的 ``sections`` 是 list[dict]，
    只扫顶层字符串会让注入把交易意图藏在内层对象里逃过检查。
    深度设上限，避免被深层/自引用结构拖住。
    """
    if depth > _MAX_SCAN_DEPTH:
        return None
    if isinstance(value, str):
        return scan_text(value, where=where, patterns=patterns)
    if isinstance(value, dict):
        for key, item in value.items():
            hit = scan_text(str(key), where=where, patterns=patterns)
            if hit is not None:
                return hit
            hit = scan_value(item, where=where, patterns=patterns, depth=depth + 1)
            if hit is not None:
                return hit
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            hit = scan_value(item, where=where, patterns=patterns, depth=depth + 1)
            if hit is not None:
                return hit
    return None


def scan_args(args: dict, *, patterns: tuple[str, ...] | None = None) -> RuleHit | None:
    """把拒绝模式应用于参数快照中的每一个值（含嵌套结构）。"""
    return scan_value(args, where="args", patterns=patterns)


def scan_sandbox(code: str) -> RuleHit | None:
    """依据导入白名单与标识符拒绝名单检查 run_python 代码。"""
    for match in _IMPORT_RE.finditer(code):
        module = (match.group(1) or match.group(2) or "").split(".")[0]
        if module and module not in SANDBOX_IMPORT_ALLOW:
            return RuleHit(rule_index=0, pattern=f"import {module}", where="import")

    for index, identifier in enumerate(SANDBOX_FORBIDDEN_IDENTIFIERS, start=1):
        if re.search(rf"\b{re.escape(identifier)}\b", code):
            return RuleHit(rule_index=index, pattern=identifier, where="identifier")
    return None

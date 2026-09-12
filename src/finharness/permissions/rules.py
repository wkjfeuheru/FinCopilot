"""Deny patterns and sandbox scanning (docs 03.7.2).

Two rule families run in parallel:

1. ``DEFAULT_DENY_PATTERNS`` — intent-level defences against trading actions.
   They scan ``run_python`` code and every tool's argument snapshot, so a prompt
   injection that smuggles a trade instruction into a data parameter is caught
   even though the framework registers no trading tools.
2. The sandbox import allowlist, which applies to ``run_python`` only.

Hits are reported with the matched rule so refusals stay explainable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Trading-intent defences. Applied to run_python code and tool argument values.
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

# Import allowlist for run_python (stdlib + analysis stack).
SANDBOX_IMPORT_ALLOW: frozenset[str] = frozenset(
    {
        "pandas", "numpy", "matplotlib", "statistics", "math",
        "json", "re", "datetime", "collections", "itertools", "functools",
    }
)

# Identifier-level denylist: process, network, filesystem and dynamic-execution
# escape hatches. Matched on word boundaries so "cost" is not mistaken for "os".
SANDBOX_FORBIDDEN_IDENTIFIERS: tuple[str, ...] = (
    "os", "sys", "subprocess", "socket", "requests", "urllib", "httpx",
    "open", "eval", "exec", "compile", "pickle", "shutil", "pathlib",
    "importlib", "__import__", "globals", "locals", "setattr", "delattr",
)

_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class RuleHit:
    """A single rule violation, with enough detail to explain the refusal."""

    rule_index: int
    pattern: str
    where: str  # "code" | "args" | "import" | "identifier"

    def reason(self) -> str:
        return f"命中了第 {self.rule_index} 条规则：{self.pattern}（位置：{self.where}）"


def scan_text(text: str, *, where: str = "code") -> RuleHit | None:
    """Apply the deny patterns to one text blob."""
    for index, pattern in enumerate(DEFAULT_DENY_PATTERNS, start=1):
        if re.search(pattern, text, re.IGNORECASE):
            return RuleHit(rule_index=index, pattern=pattern, where=where)
    return None


def scan_args(args: dict) -> RuleHit | None:
    """Apply the deny patterns to every value in an argument snapshot."""
    for value in args.values():
        if isinstance(value, str):
            hit = scan_text(value, where="args")
            if hit is not None:
                return hit
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    hit = scan_text(item, where="args")
                    if hit is not None:
                        return hit
    return None


def scan_sandbox(code: str) -> RuleHit | None:
    """Check run_python code against the import allowlist and identifier denylist."""
    for match in _IMPORT_RE.finditer(code):
        module = (match.group(1) or match.group(2) or "").split(".")[0]
        if module and module not in SANDBOX_IMPORT_ALLOW:
            return RuleHit(rule_index=0, pattern=f"import {module}", where="import")

    for index, identifier in enumerate(SANDBOX_FORBIDDEN_IDENTIFIERS, start=1):
        if re.search(rf"\b{re.escape(identifier)}\b", code):
            return RuleHit(rule_index=index, pattern=identifier, where="identifier")
    return None

"""权限模式与裁决（docs 03.7.1）。"""

from __future__ import annotations

from enum import Enum


class PermissionMode(str, Enum):
    DEFAULT = "default"
    PLAN = "plan"
    AUTO = "auto"


class Verdict(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"

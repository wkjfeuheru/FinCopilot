"""因子表达式引擎（声明式，不执行代码）。"""

from finharness.factor.engine import (
    FUNCTION_NAMES,
    MAX_DEPTH,
    MAX_WINDOW,
    MARKET_VARIABLES,
    FactorEngine,
    FactorError,
    FactorInfo,
    is_cross_sectional,
)

__all__ = [
    "FactorEngine",
    "FactorError",
    "FactorInfo",
    "FUNCTION_NAMES",
    "MARKET_VARIABLES",
    "MAX_WINDOW",
    "MAX_DEPTH",
    "is_cross_sectional",
]

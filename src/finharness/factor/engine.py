"""声明式因子表达式引擎（文档 03.4，量化因子场景）。

因子被写为针对行情/财务变量的一个小型表达式，例如::

    ts_mean(close, 20) / ts_mean(close, 60) - 1
    rank(ts_std(close / ts_delay(close, 1) - 1, 20))

表达式使用 :mod:`ast` 解析，并针对一份白名单运算符进行求值——绝不使用
``eval``/``exec``，且任何属性访问、下标或对未列出名称的调用都无法通过解析。
求值器操作一个按日期索引的 ``DataFrame``，每个标的一列，因此同一表达式
既能服务于单序列（单列）研究，也能服务于横截面（多列）研究：

* **时间序列** 运算符沿每一列独立滚动；
* **横截面** 运算符在每一行内跨列作用；
* 算术与逐元素运算符均为逐元素计算。
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MAX_WINDOW = 250
MAX_DEPTH = 12

# 表达式可引用的变量。行情变量为小写；财务字段为指标/财务工具所用的 snake_case 名称。
MARKET_VARIABLES = ("open", "high", "low", "close", "volume", "amount", "turnover")

# 时间序列运算符：(series, window[, series]) -> series
_TS_OPS: dict[str, tuple[int, Callable[..., pd.DataFrame]]] = {}
# 横截面运算符：(frame, ...) -> frame
_CS_OPS: dict[str, tuple[int, Callable[..., pd.DataFrame]]] = {}
# 逐元素运算符：(*frames) -> frame
_EW_OPS: dict[str, tuple[int, Callable[..., pd.DataFrame]]] = {}


def _ts(name: str, arity: int):
    def register(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        _TS_OPS[name] = (arity, func)
        return func

    return register


def _cs(name: str, arity: int):
    def register(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        _CS_OPS[name] = (arity, func)
        return func

    return register


def _ew(name: str, arity: int):
    def register(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        _EW_OPS[name] = (arity, func)
        return func

    return register


@_ts("ts_mean", 2)
def _ts_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window).mean()


@_ts("ts_std", 2)
def _ts_std(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window).std()


@_ts("ts_sum", 2)
def _ts_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window).sum()


@_ts("ts_min", 2)
def _ts_min(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window).min()


@_ts("ts_max", 2)
def _ts_max(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window).max()


@_ts("ts_delay", 2)
def _ts_delay(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.shift(window)


@_ts("ts_delta", 2)
def _ts_delta(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame - frame.shift(window)


@_ts("ts_pct_change", 2)
def _ts_pct_change(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    prior = frame.shift(window)
    return frame / prior.replace(0, np.nan) - 1


@_ts("ts_rank", 2)
def _ts_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    def last_rank(values: np.ndarray) -> float:
        if not np.isfinite(values[-1]):
            return np.nan
        return float((values <= values[-1]).sum()) / len(values)

    return frame.rolling(window).apply(last_rank, raw=True)


@_ts("ts_corr", 3)
def _ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window).corr(y)


@_ts("ts_cov", 3)
def _ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window).cov(y)


@_ts("ts_skew", 2)
def _ts_skew(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window).skew()


@_cs("rank", 1)
def _cs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """在每个行（日期）内计算横截面百分位排名。"""
    return frame.rank(axis=1, pct=True)


@_cs("zscore", 1)
def _cs_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """在每个行内进行横截面标准化。"""
    mean = frame.mean(axis=1)
    std = frame.std(axis=1)
    return frame.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


@_cs("quantile", 2)
def _cs_quantile(frame: pd.DataFrame, q: float) -> pd.DataFrame:
    """横截面 q 分位值，并在该行内广播。"""
    row_q = frame.quantile(q, axis=1)
    return pd.DataFrame(
        np.repeat(row_q.to_numpy()[:, None], frame.shape[1], axis=1),
        index=frame.index,
        columns=frame.columns,
    )


@_ew("abs", 1)
def _ew_abs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.abs()


@_ew("log", 1)
def _ew_log(frame: pd.DataFrame) -> pd.DataFrame:
    return np.log(frame.where(frame > 0))


@_ew("sign", 1)
def _ew_sign(frame: pd.DataFrame) -> pd.DataFrame:
    return np.sign(frame)


@_ew("sqrt", 1)
def _ew_sqrt(frame: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(frame.where(frame >= 0))


@_ew("exp", 1)
def _ew_exp(frame: pd.DataFrame) -> pd.DataFrame:
    return np.exp(frame)


@_ew("clip", 3)
def _ew_clip(frame: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    return frame.clip(lower=low, upper=high)


@_ew("where", 3)
def _ew_where(condition: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a.where(condition.astype(bool), b)


_TS_NAMES = frozenset(_TS_OPS)
_CS_NAMES = frozenset(_CS_OPS)
_EW_NAMES = frozenset(_EW_OPS)
FUNCTION_NAMES = _TS_NAMES | _CS_NAMES | _EW_NAMES

_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b.replace(0, np.nan) if isinstance(b, pd.DataFrame) else a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
}
_CMPOPS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


class FactorError(ValueError):
    """表达式无法解析或求值时抛出。"""


@dataclass(frozen=True, slots=True)
class FactorInfo:
    expression: str
    variables: tuple[str, ...]
    functions: tuple[str, ...]
    cross_sectional: bool


def _validate(node: ast.AST, depth: int = 0) -> None:
    """在求值前拒绝任何白名单之外的语法。"""
    if depth > MAX_DEPTH:
        raise FactorError(f"表达式嵌套过深（上限 {MAX_DEPTH} 层）")
    if isinstance(node, ast.Expression):
        _validate(node.body, depth)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool)):
            raise FactorError("表达式只允许数字与布尔常量")
    elif isinstance(node, ast.Name):
            return  # 在求值时针对变量 frame 解析
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise FactorError(f"不支持的运算符：{type(node.op).__name__}")
        _validate(node.left, depth + 1)
        _validate(node.right, depth + 1)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            raise FactorError(f"不支持的一元运算符：{type(node.op).__name__}")
        _validate(node.operand, depth + 1)
    elif isinstance(node, ast.Compare):
        if len(node.ops) != 1 or type(node.ops[0]) not in _CMPOPS:
            raise FactorError("只支持单次比较（> < >= <= == !=）")
        _validate(node.left, depth + 1)
        _validate(node.comparators[0], depth + 1)
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            _validate(value, depth + 1)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise FactorError("只允许调用白名单函数，不允许属性访问")
        if node.func.id not in FUNCTION_NAMES:
            raise FactorError(f"不支持的函数：{node.func.id}")
        if node.keywords:
            raise FactorError("函数调用不支持关键字参数")
        for arg in node.args:
            _validate(arg, depth + 1)
    else:
        raise FactorError(f"不支持的语法：{type(node).__name__}")


class FactorEngine:
    """在变量 frame 上解析并求值因子表达式。"""

    def parse(self, expression: str) -> ast.Expression:
        """解析表达式并做白名单校验，返回 AST。"""
        text = (expression or "").strip()
        if not text:
            raise FactorError("因子表达式不能为空")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise FactorError(f"因子表达式语法错误：{exc.msg}") from exc
        _validate(tree)
        return tree

    def describe(self, expression: str) -> FactorInfo:
        """解析表达式并汇总其引用的变量、函数及是否为横截面因子。"""
        tree = self.parse(expression)
        # 函数名在语法上也是 Name 节点；先收集调用目标，
        # 以便将它们从变量集合中排除。
        call_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        variables: set[str] = set()
        functions: list[str] = []
        cross = [False]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                functions.append(node.func.id)
                if node.func.id in _CS_NAMES:
                    cross[0] = True
            elif isinstance(node, ast.Name) and node.id not in call_names:
                variables.add(node.id)
        return FactorInfo(
            expression=expression,
            variables=tuple(sorted(variables)),
            functions=tuple(functions),
            cross_sectional=cross[0],
        )

    def evaluate(self, expression: str, variables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """针对 ``variables``（日期 x 标的的 frame）求值 ``expression``。"""
        tree = self.parse(expression)
        result = self._eval(tree.body, variables)
        if isinstance(result, pd.DataFrame):
            return result
        # 常量表达式广播为变量的形状。
        if variables:
            sample = next(iter(variables.values()))
            return pd.DataFrame(
                float(result), index=sample.index, columns=sample.columns
            )
        raise FactorError("表达式没有引用任何变量")

    # -- 求值 -----------------------------------------------------------------
    def _eval(self, node: ast.AST, variables: dict[str, pd.DataFrame]) -> Any:
        """递归求值已通过白名单校验的 AST 节点。"""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            if node.id in MARKET_VARIABLES:
                raise FactorError(f"数据中缺少变量：{node.id}")
            raise FactorError(f"未知变量：{node.id}（可用：{'/'.join(MARKET_VARIABLES)} 及财务字段）")
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, variables)
            right = self._eval(node.right, variables)
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand, variables)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            return ~operand.astype(bool)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, variables)
            right = self._eval(node.comparators[0], variables)
            return _CMPOPS[type(node.ops[0])](left, right)
        if isinstance(node, ast.BoolOp):
            values = [self._eval(value, variables) for value in node.values]
            result = values[0].astype(bool)
            for value in values[1:]:
                if isinstance(node.op, ast.And):
                    result = result & value.astype(bool)
                else:
                    result = result | value.astype(bool)
            return result
        if isinstance(node, ast.Call):
            name = node.func.id
            args = [self._eval(arg, variables) for arg in node.args]
            return self._call(name, args)
        raise FactorError(f"不支持的语法：{type(node).__name__}")

    @staticmethod
    def _as_frame(value: Any, reference: pd.DataFrame | None = None) -> pd.DataFrame:
        """将标量/Series 统一为 DataFrame；标量需提供 reference 以广播形状。"""
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, pd.Series):
            return value.to_frame()
        if reference is not None:
            return pd.DataFrame(
                float(value), index=reference.index, columns=reference.columns
            )
        raise FactorError("标量无法单独参与运算")

    def _call(self, name: str, args: list[Any]) -> pd.DataFrame:
        """按名称在时间序列/横截面/逐元素注册表中查找并调用运算符。"""
        if name in _TS_OPS:
            arity, func = _TS_OPS[name]
            if len(args) != arity:
                raise FactorError(f"{name} 需要 {arity} 个参数")
            frames = [self._as_frame(arg) for arg in args[:-1]]
            window = self._window(args[-1])
            if len(frames) == 1:
                return func(frames[0], window)
            return func(frames[0], frames[1], window)
        if name in _CS_OPS:
            arity, func = _CS_OPS[name]
            if len(args) != arity:
                raise FactorError(f"{name} 需要 {arity} 个参数")
            frame = self._as_frame(args[0])
            if arity == 1:
                return func(frame)
            return func(frame, float(args[1]))
        if name in _EW_OPS:
            arity, func = _EW_OPS[name]
            if len(args) != arity:
                raise FactorError(f"{name} 需要 {arity} 个参数")
            # 标量（例如 ``where`` 的各个分支）会广播为第一个 frame 参数的形状。
            reference = next((arg for arg in args if isinstance(arg, pd.DataFrame)), None)
            return func(*[self._as_frame(arg, reference) for arg in args])
        raise FactorError(f"不支持的函数：{name}")

    @staticmethod
    def _window(value: Any) -> int:
        """校验并返回窗口参数，取值需在 1~MAX_WINDOW 之间。"""
        if isinstance(value, (pd.DataFrame, pd.Series)):
            raise FactorError("窗口参数必须是整数常量")
        try:
            window = int(value)
        except (TypeError, ValueError) as exc:
            raise FactorError("窗口参数必须是整数") from exc
        if window < 1 or window > MAX_WINDOW:
            raise FactorError(f"窗口参数须在 1~{MAX_WINDOW} 之间，得到 {window}")
        return window


def is_cross_sectional(expression: str) -> bool:
    return FactorEngine().describe(expression).cross_sectional

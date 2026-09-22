"""边界守卫：锁住"agent 可达面里没有任意代码执行"这一性质（隔离方案 P1-2）。

**为什么这组测试就是 P1-2 的交付物。** 原路线图写的 P1-2 是"执行沙箱（容器/
受限账户 + 密钥目录不可达）"。实施前先核实它到底要围住什么，结论是**没有对象**：
本仓库刻意不注册任何执行类工具（``run_python`` 不发布），工具层也不含
``subprocess``/``os.system``/裸 ``eval``。给"不存在的代码执行"建一个容器，得到的
是一个空容器，而报告里会多出一条"沙箱已完成"——那是用配置冒充边界。

真正值得守的是**让 P1-2 不必要的那条性质本身**，因为它很容易被一次"顺手加个
工具"破坏，而破坏之后没有任何测试会响。因此这里把它写成可执行的守卫：

* 工具层（agent 唯一能触达的代码）不得出现执行原语；
* 唯一接受模型表达式的地方（``factor_expr``）必须是声明式白名单求值，
  而不是 ``eval``——并显式断言它挡住了属性访问、下标与未列名函数。

P1-2 剩下的部分（把**整个应用**放进受限容器、按 ``Workspace.mount_plan()`` 挂载）
属于部署形态，记在 Phase 2；它由 ``tests/workspace/test_contract.py`` 的挂载清单
测试守着输入契约，不在本文件重复。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "finharness" / "tools"

# 真正危险的**裸内置**调用（名字解析为内置，不是某个对象的属性）。
# 刻意只列裸调用：``re.compile`` 是属性调用，与内置 ``compile`` 完全不同，
# 按名字一刀切会把正则编译全部误报。
_FORBIDDEN_BUILTINS = {"eval", "exec", "compile", "__import__", "breakpoint"}

# 危险的**属性**调用：按 ``对象.方法`` 的形式匹配，避免把 ``coordinator.spawn``
# （子代理扇出）误当成进程派生。
_FORBIDDEN_ATTR_CALLS = {
    "os.system",
    "os.popen",
    "os.execv",
    "os.spawnv",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "subprocess.check_output",
}

_FORBIDDEN_IMPORTS = {"subprocess", "pty", "multiprocessing", "ctypes"}


def _attribute_path(node: ast.AST) -> str | None:
    """把 ``a.b.c`` 形式的属性访问还原成点号路径；其它形式返回 None。"""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _iter_tool_sources():
    for path in sorted(TOOLS_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_the_tool_layer_contains_no_execution_primitives():
    """工具层一旦出现 subprocess/裸 eval/进程派生，P1-2 的"无需沙箱"前提即失效。

    判定刻意区分"裸内置调用"与"属性调用"：``re.compile`` 与内置 ``compile``
    是两回事，``coordinator.spawn``（子代理扇出）与进程派生也是两回事。
    一刀切按名字匹配会把这两类都误报，从而让守卫噪音化、最终被绕过。
    """
    offenders: list[str] = []

    for path, tree in _iter_tool_sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in _FORBIDDEN_IMPORTS:
                    offenders.append(
                        f"{path.name}:{node.lineno} from {node.module} import ..."
                    )
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _FORBIDDEN_BUILTINS:
                    offenders.append(f"{path.name}:{node.lineno} 裸 {func.id}()")
                else:
                    dotted = _attribute_path(func)
                    if dotted in _FORBIDDEN_ATTR_CALLS:
                        offenders.append(f"{path.name}:{node.lineno} {dotted}()")

    assert not offenders, (
        "工具层出现了执行原语，agent 可达面不再“无代码执行”：\n  "
        + "\n  ".join(offenders)
    )


def test_no_execution_tool_is_registered():
    """注册表里不得有执行类工具；``run_python`` 保持不发布。"""
    from finharness.tools.registry import ALL_TOOL_CLASSES

    names = {tool_cls.name for tool_cls in ALL_TOOL_CLASSES}

    for suspicious in ("run_python", "run_shell", "bash", "exec", "python_repl"):
        assert suspicious not in names, f"{suspicious} 被注册了：任意执行面重新出现"


def test_the_factor_expression_engine_is_declarative_not_eval():
    """唯一接受模型表达式的入口必须是 AST 白名单，而不是 eval。

    这是 P1-2"没有沙箱对象"的实际依据：``run_backtest`` 的 ``factor_expr`` 来自
    模型，但它只能构成一个受限表达式语言，不能构成任意代码。
    """
    import finharness.factor.engine as engine

    source = pathlib.Path(engine.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # 不允许出现以 eval/exec 命名的**调用**（ast.parse(mode="eval") 是解析模式，
    # 不是内置 eval，因此按调用名判定）。
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec"}, (
                f"因子引擎在 {node.lineno} 行调用了 {node.func.id}()："
                "它必须是 AST 白名单求值"
            )


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd').read()",
        "close.__class__",
        "close[0]",
        "lambda: 1",
        "print('x')",
    ],
)
def test_the_factor_engine_rejects_escape_attempts(expression):
    """逃逸尝试必须在解析期被拒，且不能进入求值。"""
    from finharness.factor.engine import FactorEngine, FactorError

    with pytest.raises(FactorError):
        FactorEngine().parse(expression)


def test_the_factor_engine_accepts_a_legitimate_expression():
    """守卫不能把正常表达式一起挡掉（否则上面那组测试可被"全拒"讨好）。"""
    from finharness.factor.engine import FactorEngine

    engine = FactorEngine()
    info = engine.describe("ts_mean(close, 20) / ts_mean(close, 60) - 1")

    assert "close" in info.variables
    assert "ts_mean" in info.functions

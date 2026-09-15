"""追踪上下文：trace_id 生成、绑定与跨 asyncio 任务传播（docs 03.14.1）。"""

import asyncio

from finharness.observability.context import (
    bind_trace,
    current_trace,
    new_trace_id,
    update_turn,
)


def test_new_trace_id_has_stable_prefix() -> None:
    first, second = new_trace_id(), new_trace_id()
    assert first.startswith("tr_") and second.startswith("tr_")
    assert first != second


def test_bind_trace_generates_and_preserves_id() -> None:
    context = bind_trace(session_id="s_1", conversation_id="c_1")
    assert context.trace_id.startswith("tr_")
    assert context.session_id == "s_1"

    # 二次绑定复用同一个 id：一次运行内多次绑定共享身份。
    again = bind_trace(session_id="s_1")
    assert again.trace_id == context.trace_id


def test_update_turn_keeps_identity_and_sets_turn() -> None:
    bind_trace(session_id="s_1")
    update_turn(3)
    trace = current_trace()
    assert trace is not None and trace.turn == 3


def test_trace_id_propagates_into_child_task() -> None:
    """``create_task`` 复制 context，因此引擎任务天然带上同一个 trace_id。"""

    async def main() -> tuple[str, str]:
        bind_trace(session_id="s_1", conversation_id="c_1")
        parent = current_trace()
        assert parent is not None

        async def child() -> str:
            trace = current_trace()
            assert trace is not None
            return trace.trace_id

        return parent.trace_id, await asyncio.create_task(child())

    parent_id, child_id = asyncio.run(main())
    assert parent_id == child_id

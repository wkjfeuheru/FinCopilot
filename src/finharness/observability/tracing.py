"""LangSmith 追踪后端（docs 03.14.3）。

用 LangSmith 的**低层 ``RunTree`` API**，不引入 LangChain：本项目没有使用
LangChain，而 RunTree 足以表达"请求 → LLM / 工具 / 子代理"这棵 Span 树。

两条不变量：

* **绝不阻塞事件循环**。``post()`` 是网络 I/O，因此一律丢给后台线程，
  流式循环里只做内存中的 ``end()``。
* **绝不因追踪失败而中断一轮**。所有异常都被吞掉并降级为一条 debug 日志。
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

__all__ = ["LangSmithTracer"]

_log = logging.getLogger("finharness.observability.tracing")

# 单例线程池：追踪上报是尽力而为的低优先级 I/O，串行执行即可，且不至于
# 每次请求都新建线程。
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ls-trace")

def _try_import_run_tree() -> Any | None:
    """langsmith 是否已安装；未安装返回 None。"""
    try:
        from langsmith.run_trees import RunTree

        return RunTree
    except Exception:  # noqa: BLE001 - 未安装 langsmith 即不可用
        return None


class LangSmithTracer:
    """把 ``Observer`` 的 Span 生命周期映射到 LangSmith RunTree。"""

    def __init__(self, *, project: str = "finharness", env_key: str = "LANGSMITH_API_KEY") -> None:
        self.project = project
        self.env_key = env_key
        self._run_tree = _try_import_run_tree()
        self.api_key = os.getenv(env_key) if env_key else None

    @property
    def available(self) -> bool:
        """langsmith 已安装且凭据存在时才可用。"""
        return self._run_tree is not None and bool(self.api_key)

    # -- Observer 接口 ----------------------------------------------------------

    def start_run(
        self, *, name: str, run_type: str, inputs: dict, parent: Any | None
    ) -> Any | None:
        """创建一个 RunTree；``parent`` 为空即为根 run。"""
        if not self.available:
            return None
        run_tree = self._run_tree
        try:
            if parent is not None:
                return parent.create_child(name=name, run_type=run_type, inputs=inputs)
            return run_tree(
                name=name,
                run_type=run_type,
                inputs=inputs,
                session_name=self.project,
            )
        except Exception:  # noqa: BLE001 - 追踪尽力而为
            _log.debug("langsmith start_run failed", exc_info=True)
            return None

    def finish_run(
        self,
        handle: Any,
        *,
        attributes: dict,
        outputs: dict | None = None,
        error: str | None = None,
    ) -> None:
        """结束 run 并异步入队上报；本方法绝不阻塞调用方。"""
        try:
            payload = {k: v for k, v in attributes.items() if k != "outputs"}
            if outputs is None:
                outputs = attributes.get("outputs")
            handle.end(
                outputs=outputs or {"status": attributes.get("status", "ok")},
                error=error,
                metadata=payload or None,
            )
        except Exception:  # noqa: BLE001 - 追踪尽力而为
            _log.debug("langsmith end failed", exc_info=True)
            return
        # 网络上报放到后台线程：流式循环只等待内存里的 end()。
        try:
            _EXECUTOR.submit(self._post, handle)
        except Exception:  # noqa: BLE001 - 线程池不可用时放弃上报
            _log.debug("langsmith submit failed", exc_info=True)

    @staticmethod
    def _post(handle: Any) -> None:
        try:
            handle.post()
        except Exception:  # noqa: BLE001 - 上报失败不影响任何业务
            _log.debug("langsmith post failed", exc_info=True)

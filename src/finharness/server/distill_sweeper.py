"""LTM 懒蒸馏的后台扫描器（docs 03.6.4）。

设计曾承诺的"每 60s 会话 TTL 回收"并不存在——``SessionRegistry`` 是惰性
淘汰，且没有可挂钩的销毁回调。本扫描器不做会话回收，只做一件事：周期性
找出闲置超过 ``ltm.distill_idle_s`` 且尚未成功蒸馏的对话，交给
``EpisodeDistiller`` 补课。它是双保险的前一翼（后一翼：用户开新对话时
在 loop_factory 里兜底）。

对话本身绝不因蒸馏被阻塞或修改；蒸馏失败只累加台账计数。
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

from finharness.config.settings import Settings
from finharness.context.memory.distill import EpisodeDistiller
from finharness.context.memory.store import MemoryStore
from finharness.observability import get_logger

# 扫描周期。独立于 distill_idle_s：调小前者可以更快发现闲置对话，
# 调大后者避免把仍在进行的对话过早蒸馏。
SCAN_INTERVAL_S = 60.0


def _idle_before(settings: Settings, *, now: datetime | None = None) -> str:
    """闲置判定的时间下界（ISO 字符串，与库中的时间戳同格式）。"""
    moment = now or datetime.now(UTC)
    return (moment - timedelta(seconds=int(settings.ltm.distill_idle_s))).isoformat(
        timespec="seconds"
    )


async def distill_user_backlog(
    *,
    provider: Any,
    store: MemoryStore,
    settings: Settings,
    user_id: str,
    exclude_conversation: str | None = None,
    observer: Any | None = None,
    on_usage: Any | None = None,
    index: Any | None = None,
    now: datetime | None = None,
) -> int:
    """为一位用户补蒸馏全部待处理对话；返回处理成功的对话数。

    供两处调用：后台扫描器（每用户一批）与 loop_factory 兜底（开新对话
    时一次性补齐）。异常逐对话吞掉——一个对话的蒸馏失败不应终止
    同批其余对话。
    """
    distiller = EpisodeDistiller(
        provider=provider, store=store, settings=settings,
        observer=observer, on_usage=on_usage, index=index,
    )
    candidates = store.ltm_distill_candidates(
        user_id=user_id,
        exclude_conversation=exclude_conversation,
        max_attempts=int(settings.ltm.distill_max_attempts),
        idle_before=_idle_before(settings, now=now),
        limit=int(settings.ltm.distill_batch) if exclude_conversation is None else 50,
    )
    done = 0
    for conversation_id in candidates:
        outcome = await distiller.distill_conversation(conversation_id, user_id=user_id)
        if outcome.error is None and not outcome.skipped:
            done += 1
    return done


def start_distill_sweeper(
    *, app_state: Any, provider_resolver: Any, settings: Settings, observer: Any | None = None
) -> asyncio.Task | None:
    """启动周期扫描任务；返回 Task 以便测试取消。

    provider 按用户解析（数据库配置优先）：蒸馏与对话使用同一模型配置，
    因此走 ProviderResolver 而非进程级单例。解析失败（未配置）时本轮跳过，
    绝不让扫描器把服务端拖垮。
    """
    store: MemoryStore = app_state.memory_store
    index = getattr(app_state, "semantic_index", None)
    log = get_logger("finharness.server.distill")

    async def sweep() -> None:
        while True:
            await asyncio.sleep(SCAN_INTERVAL_S)
            try:
                await sweep_once(
                    store=store,
                    provider_resolver=provider_resolver,
                    settings=settings,
                    observer=observer,
                    index=index,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 扫描器绝不致命
                log.exception("ltm_distill_sweep_failed")

    return asyncio.create_task(sweep(), name="ltm-distill-sweeper")


async def sweep_once(
    *,
    store: MemoryStore,
    provider_resolver: Any,
    settings: Settings,
    observer: Any | None = None,
    index: Any | None = None,
    now: datetime | None = None,
) -> int:
    """一轮扫描：找出有待蒸馏对话的用户，逐用户补一批。返回蒸馏成功的对话数。"""
    log = get_logger("finharness.server.distill")
    users = store.ltm_users_with_candidates(
        max_attempts=int(settings.ltm.distill_max_attempts),
        idle_before=_idle_before(settings, now=now),
        limit=int(settings.ltm.distill_batch),
    )
    done = 0
    for user_id in users:
        try:
            provider = provider_resolver.current(user_id)
        except Exception:  # noqa: BLE001 - 未配置 provider 的用户跳过
            continue
        done += await distill_user_backlog(
            provider=provider,
            store=store,
            settings=settings,
            user_id=user_id,
            observer=observer,
            index=index,
        )
    if done:
        log.info("ltm_distill_swept", extra={"conversations": done})
    return done


async def stop_distill_sweeper(task: asyncio.Task | None) -> None:
    """取消扫描任务并等待其退出（服务端关闭/测试清理用）。"""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

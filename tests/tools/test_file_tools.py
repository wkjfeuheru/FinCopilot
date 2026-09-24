"""read_file / write_file：白名单、体积上限与不阻塞事件循环（docs 03.4）。

体积上限与线程化是两个独立的性质：上限决定"多大算过界"，线程化决定"阻塞期间
是否占住事件循环"。两者都必须可观测，否则一条挂住的调用仍会拖慢整个进程。
"""

from __future__ import annotations

import asyncio
import threading

from finharness.config.settings import ContextSettings, Settings
from finharness.data.access import DataAccess
from finharness.tools.generic.files import (
    MAX_READ_BYTES,
    MAX_WRITE_BYTES,
    ReadFileTool,
    WriteFileTool,
)
from tests.conftest import settings_with_cache


def make_settings(tmp_path) -> Settings:
    return settings_with_cache(
        tmp_path,
        context=ContextSettings(),
        paths={"output_dir": tmp_path / "output"},
    )


def make_tools(tmp_path):
    settings = make_settings(tmp_path)
    data = DataAccess([], settings=settings)
    return ReadFileTool(data), WriteFileTool(data), settings


def run(coro):
    return asyncio.run(coro)


# -- 读取 ---------------------------------------------------------------------

def test_it_reads_a_file_inside_the_output_dir(tmp_path):
    reader, _, settings = make_tools(tmp_path)
    target = settings.paths.output_dir / "note.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello", encoding="utf-8")

    result = run(reader.run(path=str(target)))

    assert result.ok is True, result.error
    assert "hello" in result.content


def test_it_truncates_long_text_to_the_read_cap(tmp_path):
    reader, _, settings = make_tools(tmp_path)
    target = settings.paths.output_dir / "big.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x" * (MAX_READ_BYTES * 5), encoding="utf-8")

    result = run(reader.run(path=str(target)))

    assert result.ok is True, result.error
    assert len(result.content) < MAX_READ_BYTES * 2


def test_it_refuses_a_path_outside_the_allowed_roots(tmp_path):
    reader, _, _ = make_tools(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")

    result = run(reader.run(path=str(outside)))

    assert result.ok is False
    assert "nope" not in (result.content or "")


# -- 写入 ---------------------------------------------------------------------

def test_write_refuses_content_over_the_cap(tmp_path):
    _, writer, settings = make_tools(tmp_path)
    target = settings.paths.output_dir / "too-big.txt"

    result = run(writer.run(path=str(target), content="y" * (MAX_WRITE_BYTES + 1)))

    assert result.ok is False
    assert not target.exists(), "超限内容不得落盘"


def test_write_at_the_cap_succeeds(tmp_path):
    _, writer, settings = make_tools(tmp_path)
    target = settings.paths.output_dir / "just-fits.txt"

    result = run(writer.run(path=str(target), content="y" * MAX_WRITE_BYTES))

    assert result.ok is True, result.error
    assert target.stat().st_size == MAX_WRITE_BYTES


# -- 不阻塞事件循环（多租户下的跨租户 DoS 防线）--------------------------------

def test_read_runs_off_the_event_loop_thread(tmp_path):
    """读取必须在线程中完成，使超时可取消、阻塞期间不占住事件循环。"""
    reader, _, settings = make_tools(tmp_path)
    target = settings.paths.output_dir / "whoami.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("thread-id", encoding="utf-8")

    seen: dict[str, int] = {}
    original = ReadFileTool.__dict__["_read_text"].__func__

    def spy(path):
        seen["worker"] = threading.get_ident()
        return original(path)

    ReadFileTool._read_text = staticmethod(spy)
    try:
        result = run(reader.run(path=str(target)))
    finally:
        ReadFileTool._read_text = staticmethod(original)

    assert result.ok is True, result.error
    assert seen["worker"] != threading.main_thread().ident, "读取必须发生在工作线程"


def test_write_runs_off_the_event_loop_thread(tmp_path):
    _, writer, settings = make_tools(tmp_path)
    target = settings.paths.output_dir / "whoami.txt"

    seen: dict[str, int] = {}
    original = WriteFileTool.__dict__["_write"].__func__

    def spy(path, content):
        seen["worker"] = threading.get_ident()
        return original(path, content)

    WriteFileTool._write = staticmethod(spy)
    try:
        result = run(writer.run(path=str(target), content="ok"))
    finally:
        WriteFileTool._write = staticmethod(original)

    assert result.ok is True, result.error
    assert seen["worker"] != threading.main_thread().ident, "写入必须发生在工作线程"


def test_a_slow_read_does_not_block_other_coroutines(tmp_path):
    """慢读取进行时另一个协程仍能被调度——"超时形同虚设"的回归测试。

    必须在**读取期间**计数：累计 tick 数会把读取前后本就存在的 await 也算进去，
    于是阻塞版本也能通过。这里只比较调用内部那一段的推进量。
    """
    reader, _, settings = make_tools(tmp_path)
    target = settings.paths.output_dir / "slow.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("data", encoding="utf-8")

    import time

    original = ReadFileTool.__dict__["_read_text"].__func__
    ticks = 0
    progress_during_read: dict[str, int] = {}

    def slow(path):
        start = ticks
        time.sleep(0.4)
        progress_during_read["during"] = ticks - start
        return original(path)

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    async def scenario():
        ticker_task = asyncio.create_task(ticker())
        try:
            await asyncio.wait_for(reader.run(path=str(target)), timeout=5)
        finally:
            ticker_task.cancel()
        return progress_during_read.get("during", 0)

    ReadFileTool._read_text = staticmethod(slow)
    try:
        during = run(scenario())
    finally:
        ReadFileTool._read_text = staticmethod(original)

    assert during >= 3, (
        f"读取期间事件循环被占住，其他协程无法推进（仅 {during} 次 tick）"
    )

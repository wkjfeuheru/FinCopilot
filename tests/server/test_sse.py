import asyncio
import json

from finharness.server.api import QueueSink, _event_name
from finharness.server.sse import HEARTBEAT_S, encode_comment, encode_event
from finharness.types import EngineEvent


def test_encode_event_uses_sse_frame_format():
    assert encode_event("delta", {"text": "line\nnext"}) == (
        "event: delta\ndata: "
        + json.dumps({"text": "line\nnext"}, ensure_ascii=False)
        + "\n\n"
    )


def test_encode_comment_is_an_sse_comment_frame():
    """心跳必须是注释帧：不含 event/data，客户端按规范忽略它。"""
    frame = encode_comment()

    assert frame.startswith(":")
    assert frame.endswith("\n\n")
    assert "event:" not in frame
    assert "data:" not in frame


def test_heartbeat_interval_is_positive():
    assert HEARTBEAT_S > 0


def test_event_name_state_is_passthrough():
    assert _event_name("state") == "state"


def test_queue_sink_replays_state_events():
    sink = QueueSink()
    payload = {"run_id": "r1", "revision": 3, "phase": "thinking"}

    async def emit():
        await sink.emit(EngineEvent("state", payload))
        await sink.emit(EngineEvent("done", {"reason": "done"}))
        # Drain so emit() completes; QueueSink always puts on the queue.
        await sink.queue.get()
        await sink.queue.get()

    asyncio.run(emit())

    assert {"event": "state", "data": payload} in sink.replay_events
    assert {"event": "done", "data": {"reason": "done"}} in sink.replay_events

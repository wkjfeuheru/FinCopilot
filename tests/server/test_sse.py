import json

from finharness.server.sse import HEARTBEAT_S, encode_comment, encode_event


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

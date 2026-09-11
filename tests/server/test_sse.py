import json

from finharness.server.sse import encode_event


def test_encode_event_uses_sse_frame_format():
    assert encode_event("delta", {"text": "line\nnext"}) == (
        "event: delta\ndata: "
        + json.dumps({"text": "line\nnext"}, ensure_ascii=False)
        + "\n\n"
    )

from fastapi.testclient import TestClient

from finharness.provider.fake import FakeProvider
from finharness.server.api import app, create_app


def test_health_endpoint_returns_service_status() -> None:
    response = TestClient(app).get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_stream_returns_session_and_answer_events() -> None:
    client = TestClient(create_app(FakeProvider(["hello", " world"])))

    response = client.post("/v1/chat/stream", json={"message": "question"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: session" in response.text
    assert 'event: delta\ndata: {"text": "hello"}' in response.text
    assert 'event: answer\ndata: {"text": "hello world"}' in response.text
    assert "event: done" in response.text


def test_chat_stream_reports_provider_error_as_sse_event() -> None:
    client = TestClient(create_app(FakeProvider([], error=RuntimeError("provider down"))))

    response = client.post("/v1/chat/stream", json={"message": "question"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "provider down" in response.text


def test_tools_endpoint_lists_m0_financial_tools() -> None:
    client = TestClient(create_app(FakeProvider(["ok"])))

    response = client.get("/v1/tools")

    assert response.json() == {"tools": ["get_quote", "get_kline", "get_indicators"]}


def test_second_turn_reuses_session_history() -> None:
    provider = FakeProvider(["answer"])
    client = TestClient(create_app(provider))
    first = client.post("/v1/chat/stream", json={"message": "first"})
    session_id = first.text.split('"session_id": "', 1)[1].split('"', 1)[0]

    second = client.post(
        "/v1/chat/stream",
        json={"session_id": session_id, "message": "second"},
    )

    assert second.status_code == 200
    assert [message.content for message in provider.requests[-1]] == ["first", "answer", "second"]

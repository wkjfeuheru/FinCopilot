"""日志脱敏：键名、值模式、嵌套与截断（docs 03.14.1）。"""

from finharness.observability.redact import redact, redact_text, summarize_args


def test_secret_keys_are_redacted_recursively() -> None:
    payload = {
        "symbol": "600519",
        "api_key": "placeholder",
        "nested": {"access_token": "placeholder", "keep": "visible"},
        "items": [{"password": "placeholder"}],
    }

    cleaned = redact(payload)

    assert cleaned["symbol"] == "600519"
    assert cleaned["api_key"] == "<redacted>"
    assert cleaned["nested"]["access_token"] == "<redacted>"
    assert cleaned["nested"]["keep"] == "visible"
    assert cleaned["items"][0]["password"] == "<redacted>"


def test_token_count_fields_are_not_mistaken_for_secrets() -> None:
    """``max_tokens``/``budget_tokens`` 不是凭据，键名匹配必须是精确/后缀的。"""
    cleaned = redact({"max_tokens": 4096, "budget_tokens": 2048, "input_tokens": 100})

    assert cleaned == {"max_tokens": 4096, "budget_tokens": 2048, "input_tokens": 100}


def test_secret_value_patterns_are_scrubbed_from_free_text() -> None:
    text = "failed with Authorization: Bearer abcdef1234567890 and key sk-live-9876543210"

    cleaned = redact_text(text)

    assert "abcdef1234567890" not in cleaned
    assert "sk-live-9876543210" not in cleaned
    assert cleaned.count("<redacted>") == 2


def test_deep_structures_are_truncated_not_expanded() -> None:
    deep: dict = {}
    node = deep
    for _ in range(10):
        node["child"] = {}
        node = node["child"]

    cleaned = redact(deep)

    assert "<truncated>" in str(cleaned)


def test_summarize_args_truncates_and_redacts() -> None:
    secret_keys = ("api_key", "token")
    args = dict.fromkeys(secret_keys, "placeholder")
    args["symbol"] = "600519"
    args["note"] = "x" * 500

    text = summarize_args(args)

    assert "placeholder" not in text
    assert "symbol=600519" in text
    assert text.count("<redacted>") == 2
    assert "…" in text
    assert len(text) < 400


def test_summarize_args_scrubs_secret_shaped_values() -> None:
    text = summarize_args({"webhook": "tvly-dev-abcdefghijklmnop"})

    assert "tvly-dev-abcdefghijklmnop" not in text
    assert "<redacted>" in text

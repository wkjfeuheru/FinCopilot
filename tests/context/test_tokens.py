"""Token counting: exact path, and the documented fallback when unavailable."""

from finharness.context.tokens import CHARS_PER_TOKEN, TokenCounter


def test_empty_text_counts_zero_exactly():
    counted = TokenCounter().count("")

    assert counted.tokens == 0
    assert counted.exact is True


def test_exact_count_is_used_when_available():
    counter = TokenCounter()
    counter.warmup()

    counted = counter.count("贵州茅台最新股价是多少")

    # Whatever the vocabulary, an exact count is reported as exact.
    if counted.exact:
        assert counted.tokens > 0
    else:  # pragma: no cover - only when the vocabulary cannot be fetched
        assert counted.tokens == int(len("贵州茅台最新股价是多少") / CHARS_PER_TOKEN)


def test_fallback_estimates_when_the_encoder_is_unavailable():
    counter = TokenCounter()
    counter._failed = True  # simulate an unavailable vocabulary
    counter._encoder = None

    counted = counter.count("a" * 17)

    assert counted.exact is False
    assert counted.tokens == int(17 / CHARS_PER_TOKEN)


def test_count_many_sums_across_texts():
    counter = TokenCounter()

    total = counter.count_many(["你好", "世界"])

    assert total == counter.count("你好").tokens + counter.count("世界").tokens


def test_none_is_treated_as_empty():
    assert TokenCounter().count(None).tokens == 0

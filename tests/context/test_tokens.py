"""Token 计数：精确路径，以及编码器不可用时文档化的回退方案。"""

from finharness.context.tokens import CHARS_PER_TOKEN, TokenCounter


def test_empty_text_counts_zero_exactly():
    counted = TokenCounter().count("")

    assert counted.tokens == 0
    assert counted.exact is True


def test_exact_count_is_used_when_available():
    counter = TokenCounter()
    counter.warmup()

    counted = counter.count("贵州茅台最新股价是多少")

    # 无论词表如何，精确计数都会按精确上报。
    if counted.exact:
        assert counted.tokens > 0
    else:  # pragma: no cover - 仅在无法获取词表时才会走到
        assert counted.tokens == int(len("贵州茅台最新股价是多少") / CHARS_PER_TOKEN)


def test_fallback_estimates_when_the_encoder_is_unavailable():
    counter = TokenCounter()
    counter._failed = True  # 模拟词表不可用
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


def test_unwritable_cache_dir_degrades_instead_of_raising(tmp_path):
    """词表缓存目录不可写时必须降级为估算，而不是抛出。

    线上事故：容器以非 root 运行、仓库根不可写，``TokenCounter`` 初始化里的
    ``mkdir`` 抛出 ``PermissionError``，把首次对话整个打成 500。类的契约是
    "编码器不可用时改用字符近似"，因此构造期也不能因缓存目录失败而中断。
    用「父路径是文件」制造可移植的 OSError（Windows/ Linux 一致）。
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")

    counter = TokenCounter(cache_dir=blocker / "tiktoken")
    counted = counter.count("a" * 17)

    assert counted.exact is False
    assert counted.tokens == int(17 / CHARS_PER_TOKEN)

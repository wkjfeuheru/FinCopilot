"""评测 schema 的多对话形式（docs 03.13）：跨对话记忆用例的载体。"""

import pytest

from finharness.eval.schema import SchemaError, load_cases, load_cases_dir


def _write(tmp_path, text: str):
    path = tmp_path / "cases.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_multi_chat_case_is_flattened_in_execution_order(tmp_path):
    path = _write(
        tmp_path,
        """
- id: M-01
  tags: [core]
  chats:
    - conversation_id: c_a
      turns:
        - user: 第一个对话的问题
    - turns:
        - user: 第二个对话的第一个问题
        - user: 第二个对话的第二个问题
""",
    )

    case = load_cases(path)[0]

    groups = case.conversation_groups()
    assert [cid for cid, _turns in groups] == ["c_a", None]
    # 扁平化的轮次顺序与执行顺序一致：评分器据此逐轮比对 run.turns。
    assert [turn.user for turn in case.all_turns] == [
        "第一个对话的问题",
        "第二个对话的第一个问题",
        "第二个对话的第二个问题",
    ]


def test_single_turn_case_still_supported_and_flattened(tmp_path):
    path = _write(
        tmp_path,
        """
- id: S-01
  turns:
    - user: 单对话问题
    - user: 追问
""",
    )

    case = load_cases(path)[0]

    assert case.conversation_groups() == [(None, list(case.turns))]
    assert [turn.user for turn in case.all_turns] == ["单对话问题", "追问"]


def test_case_must_not_declare_both_turns_and_chats(tmp_path):
    path = _write(
        tmp_path,
        """
- id: X-01
  turns:
    - user: a
  chats:
    - turns:
        - user: b
""",
    )

    with pytest.raises(SchemaError):
        load_cases(path)


def test_case_must_declare_some_turns(tmp_path):
    path = _write(tmp_path, "- id: X-02\n  title: 空用例\n")

    with pytest.raises(SchemaError):
        load_cases(path)


def test_repo_memory_cases_load_cleanly():
    """M 系列（跨对话记忆）用例必须通过 schema 校验。"""
    cases = load_cases_dir("evals/cases")

    memory_cases = [case for case in cases if case.id.startswith("M1-")]
    assert len(memory_cases) >= 2
    assert all(case.chats for case in memory_cases)
    assert all(case.all_turns for case in memory_cases)

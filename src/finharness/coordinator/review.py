"""风险复核编排：运行复核者并汇合其裁决（docs 03.10）。

复核由 ``write_report`` 内部触发——而非由模型可自行选择的某个工具触发——因此
“每份研报都会被复核”是系统的一种属性，而不是对模型行为的一种期望。本模块负责
该触发之后发生的一切：避免为未变动正文重复付费的摘要闩锁、对 coordinator 的
调用、伴生文件（sidecar），以及工具结果所携带的措辞。

研报本身必须在任何复核失败中存活。``BaseTool.run`` 会把异常变成一个笼统的
``ok=False``，这会丢弃一份渲染得完好无损的研报，因此这里的每一次失败都降级为
一种结果，而不是抛出异常。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# 附录是一张引用表，而非正文：复核者评判的是正文与风险章节，因此把这张表一并
# 发送只会消耗 token，却不会改变任何一条意见。
APPENDIX_MARKER = "\n## 附录"
# 一次复核会记录它所评判正文的摘要，因此未改动的重写会复用该裁决，
# 而不会为同一次复核重复付费。
REVIEW_DIGEST_PREFIX = "<!-- reviewed-body:"

# 一次复核尝试的生命周期。"skipped" 表示不存在复核者（降级部署仍会撰写研报）；
# 每一个审计消费方都必须能够看出某份研报从未被检查过。
REVIEW_DONE = "done"
REVIEW_REUSED = "reused"
REVIEW_UNREVIEWED = "unreviewed"
REVIEW_SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    """一份研报被送往风险复核者后所发生的结果。"""

    status: str
    topic: str
    comments: str = ""
    review_path: str | None = None
    error: str | None = None

    def audit_metadata(self) -> dict:
        """审计行的载荷；复核本身必须是一次可审计的事件。"""
        return {
            "status": self.status,
            "topic": self.topic,
            "review_path": self.review_path,
            "error": self.error,
        }


async def review_report(coordinator, *, topic: str, markdown_path: str | Path) -> ReviewOutcome:
    """对一份已渲染的研报运行风险复核者；绝不抛出异常。

    正文与先前记录的复核相同 ⇒ 复用该结果；任何失败之处——读取、复核者、
    sidecar 写入——都降级为 ``unreviewed``，因此研报产物本身绝不会成为其自身
    复核的牺牲品。
    """
    if coordinator is None:
        return ReviewOutcome(status=REVIEW_SKIPPED, topic=topic)

    markdown_path = Path(markdown_path)
    review_path = markdown_path.with_suffix(".review.md")

    try:
        body = report_body(markdown_path)
    except OSError as exc:  # 正文不可读：标记为未复核，而非研报失败
        return ReviewOutcome(
            status=REVIEW_UNREVIEWED, topic=topic, error=f"{type(exc).__name__}: {exc}"
        )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()

    # 按内容而非按时间闩锁。重写同一主题会覆盖研报，因此比较 mtime 会把每一次
    # 复核都判为过期；比较已复核正文的摘要，则可在无变化时复用裁决，仅在研报
    # 确实被修订时才重新复核。
    if reviewed_digest(review_path) == digest:
        return ReviewOutcome(
            status=REVIEW_REUSED, topic=topic, review_path=str(review_path)
        )

    try:
        result = await coordinator.review_risk(topic=topic, markdown=body)
    except Exception as exc:  # noqa: BLE001 - 绝不让一份已渲染的研报失败
        return ReviewOutcome(
            status=REVIEW_UNREVIEWED, topic=topic, error=f"{type(exc).__name__}: {exc}"
        )
    if not result.ok:
        return ReviewOutcome(status=REVIEW_UNREVIEWED, topic=topic, error=result.error)

    comments = (result.summary or "").strip()
    try:
        review_path.write_text(
            f"{REVIEW_DIGEST_PREFIX}{digest} -->\n\n"
            f"# 风险终审意见：{topic}\n\n{_strip_redundant_heading(comments)}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        # 丢失 sidecar 文件不值得让研报失败；意见仍会随结果文本一起传递。
        return ReviewOutcome(
            status=REVIEW_DONE, topic=topic, comments=comments, error=f"OSError: {exc}"
        )
    return ReviewOutcome(
        status=REVIEW_DONE, topic=topic, comments=comments, review_path=str(review_path)
    )


def format_review_lines(outcome: ReviewOutcome) -> list[str]:
    """把一次结果渲染为模型下一轮读取的工具结果行。"""
    if outcome.status == REVIEW_DONE:
        lines = ["- 风险终审意见（独立复核，供你决定是否修订）："]
        lines.extend(
            f"  {line}"
            for line in (outcome.comments or "未发现实质性问题").splitlines()
        )
        if outcome.review_path:
            lines.append(f"- 完整意见：{outcome.review_path}")
        if outcome.error:  # sidecar 写入失败；意见仍然有效
            lines.append(f"- 终审意见落盘失败：{outcome.error}")
        return lines
    if outcome.status == REVIEW_REUSED:
        return [f"- 风险终审：已完成（复用 {outcome.review_path}）"]
    if outcome.status == REVIEW_SKIPPED:
        return []
    return [unreviewed_warning(outcome.error)]


def unreviewed_warning(error: str | None) -> str:
    """复核失败必须读作“未复核”，而不是一条中性的说明。

    研报正文被刻意保持原样（docs 03.10.5），因此工具结果是调用方得知复核未能
    发生的唯一地方。一条平淡的“未完成”很容易被略过；这里明确点出所缺失的保证。
    """
    detail = f"（{error}）" if error else ""
    return (
        f"- ⚠ 风险终审未完成{detail}：**本报告未经独立复核**，"
        "不得视为已复核交付；如需复核请重新成稿触发，或人工核对关键数字。"
    )


def report_body(markdown_path: Path) -> str:
    """去掉引用附录后的研报，供复核者阅读。"""
    text = markdown_path.read_text(encoding="utf-8")
    marker = text.find(APPENDIX_MARKER)
    return text if marker == -1 else text[:marker]


def _strip_redundant_heading(comments: str) -> str:
    """去掉复核者写下的开头标题，因为模板已经加了一个。

    sidecar 始终以 ``# 风险终审意见：{topic}`` 开头，但复核者常常把同一标题
    重复写为其第一行，这会渲染出重复的 H1。只移除复核者*自己*的开头标题——
    更靠下的章节标题（``## 一、…``）属于正文内容，予以保留。
    """
    lines = comments.splitlines()
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index < len(lines) and lines[index].lstrip().startswith("# 风险终审"):
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        return "\n".join(lines[index:]).strip()
    return comments


def reviewed_digest(review_path: Path) -> str | None:
    """先前的复核所依据的正文摘要（如果有记录的话）。"""
    try:
        head = review_path.read_text(encoding="utf-8")[:200]
    except OSError:
        return None
    marker = REVIEW_DIGEST_PREFIX
    start = head.find(marker)
    if start == -1:
        return None
    remainder = head[start + len(marker):]
    return remainder.split(" ", 1)[0].strip() or None

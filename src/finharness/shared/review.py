"""风险复核编排：运行复核者并汇合其裁决（docs 03.10）。

复核由 ``write_report`` 内部触发——而非由模型可自行选择的某个工具触发——因此
“每份研报都会被复核”是系统的一种属性，而不是对模型行为的一种期望。本模块负责
该触发之后发生的一切：避免为未变动正文重复付费的摘要闩锁、对 coordinator 的
调用、伴生文件（sidecar），以及工具结果所携带的措辞。

裁决的严重度也在这里解析。复核者的输出是自由文本，但“这份报告是否还留着未经
处理的高严重度问题”必须是**系统持有的事实**，而不是留给模型去读懂的印象：解析
结果既决定工具结果的措辞，也决定会话状态里被持续携带的未结事项（docs 03.10.7）。

研报本身必须在任何复核失败中存活。``BaseTool.run`` 会把异常变成一个笼统的
``ok=False``，这会丢弃一份渲染得完好无损的研报，因此这里的每一次失败都降级为
一种结果，而不是抛出异常。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# 附录是一张引用表，而非正文：复核者评判的是正文与风险章节，因此把这张表一并
# 发送只会消耗 token，却不会改变任何一条意见。
APPENDIX_MARKER = "\n## 附录"
# 一次复核会记录它所评判正文的摘要，因此未改动的重写会复用该裁决，
# 而不会为同一次复核重复付费。
REVIEW_DIGEST_PREFIX = "<!-- reviewed-body:"
# sidecar 模板加的那个标题。复核者自己写的同名标题会被去掉（见
# ``_strip_redundant_heading``），因此它既是写入的模板，也是读取时跳过模板行的依据。
REVIEW_HEADING_PREFIX = "# 风险终审意见："

# 复核者被要求"确实没有实质问题"时只输出这一行。整行精确匹配：它出现在散文中间
# 时不算通过，否则一句转述就能让一份没人真正复核过的报告看起来已经通过。
NO_FINDINGS = "未发现实质性问题"

# 只有高严重度构成未结事项；中/低是复核者行文里的分级，系统不为它们建档。
SEVERITY_HIGH = "高"

# 复核者的条目格式是提示词约定，而非接口契约：条目可能缺失，也可能写成别的样子。
# 因此只在行首识别约定形态（``### [高|中|低] 标题``），抽不到就交由调用方按“未分类”
# 处理（见 ``ReviewOutcome.unresolved_note``）——宁可多报一次，不可静默放行。
_FINDING_HEADING = re.compile(r"^#{2,4}\s*\[(高|中|低)\]\s*(.+?)\s*$")

# 未结事项每轮都会被注入会话状态，因此它的措辞要比 ``unreviewed_warning``
# （只出现一次的工具结果行）更短。
UNREVIEWED_NOTE = "终审未完成，本报告未经独立复核"
UNCLASSIFIED_NOTE = "终审返回了意见但未能解析严重度，按存在严重问题处理"

# 未消解高严重度项时，工具结果首行的阻断式措辞。
_BLOCKING_HEADER = "⛔ 风险终审发现 {count} 项高严重度问题，本报告在修订前不得视为已完成："
_UNCLASSIFIED_HEADER = (
    "⛔ 风险终审返回了 {count} 条意见但未能解析严重度，"
    "按存在严重问题处理，本报告在确认前不得视为已完成："
)
_REVISE_HIGH = (
    "请据意见修订后重新调用 write_report（会再次请用户确认）；"
    "若用户选择不修订，须按系统文案如实说明本报告存在未处理的高严重度问题。"
)
_REVISE_UNCLASSIFIED = (
    "请据意见修订后重新调用 write_report（会再次请用户确认）；"
    "若用户选择不修订，须按系统文案如实说明本报告存在未处理的风险终审问题。"
)
# 正文未变 ⇒ 没有重新复核，裁决沿用上一次；这句话必须让模型看出"这份意见不是
# 刚跑出来的新结论"，以免它把复用当成一次新的通过。
_REUSED_LINE = "- 风险终审：正文未变，复用上一次裁决（未重新复核）"

# 一次复核尝试的生命周期。"skipped" 表示不存在复核者（降级部署仍会撰写研报）；
# 每一个审计消费方都必须能够看出某份研报从未被检查过。
REVIEW_DONE = "done"
REVIEW_REUSED = "reused"
REVIEW_UNREVIEWED = "unreviewed"
REVIEW_SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    """复核者按约定格式写下的一条意见。"""

    severity: str
    title: str


def parse_findings(comments: str) -> tuple[ReviewFinding, ...]:
    """按约定格式抽取 ``### [严重度] 标题`` 条目。

    只做保守抽取：抽不到的条目不会凭空出现，调用方据“零条目”判定未分类
    （见 ``ReviewOutcome.unresolved_note``）。
    """
    findings: list[ReviewFinding] = []
    for line in comments.splitlines():
        match = _FINDING_HEADING.match(line.strip())
        if match:
            findings.append(ReviewFinding(severity=match.group(1), title=match.group(2)))
    return tuple(findings)


def declared_clear(comments: str) -> bool:
    """复核者是否**明确**声明了没有实质问题。

    比对的是"整行"，而不是"整行没有别的字"：行首的 markdown 装饰（``##``/``**``/``>``）
    与行尾的句号不算内容差异。这一点是必需的——复核者写成 ``**未发现实质性问题**`` 时若判为
    无法解析，系统会要求作者修订一份没有任何问题的报告，而正文未变又会命中内容闩锁复用同一条
    意见，形成一处走不出去的阻断。反过来，判据仍然拒绝"在散文里提到这句话"：那种段落整行
    归一化后并不等于这句话。
    """
    return any(_normalize_line(line) == NO_FINDINGS for line in comments.splitlines())


def _normalize_line(line: str) -> str:
    """去出行首/行尾的 markdown 装饰与句末标点，用于整行比对。"""
    text = line.strip().lstrip("#>*-+ \t").rstrip("*_` \t")
    return text[:-1] if text.endswith(("。", ".")) else text


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    """一份研报被送往风险复核者后所发生的结果。"""

    status: str
    topic: str
    comments: str = ""
    review_path: str | None = None
    error: str | None = None

    @property
    def findings(self) -> tuple[ReviewFinding, ...]:
        """从意见正文抽出的条目。

        由 ``comments`` 派生而非另存一份：派生值不会与它所描述的那段文本脱节，
        复用路径（只读得到 sidecar）也就自动得到与首次复核相同的判定。
        """
        return parse_findings(self.comments)

    def high_severity_titles(self) -> tuple[str, ...]:
        return tuple(f.title for f in self.findings if f.severity == SEVERITY_HIGH)

    def unclassified(self) -> bool:
        """有意见，但一条严重度都读不出来。

        空意见不在此列：``write_report`` 对它的既有语义是“未发现问题”，而这里要报
        的是“返回了意见却读不出严重度”——两者是不同的失败，不该合并。
        """
        return (
            not self.findings
            and bool(self.comments.strip())
            and not declared_clear(self.comments)
        )

    def unresolved_note(self) -> str | None:
        """未结事项的一行描述；已消解、或根本没有复核者时返回 None。

        这是“终审是否还留着问题”的唯一判定处：工具结果的措辞、会话状态的注入
        都从这里取，避免各自把严重度重新解释一遍。未消解只由两种情形触发——
        还留着高严重度条目，或意见无法分类；仅有中/低意见不构成未结事项。
        """
        if self.status == REVIEW_SKIPPED:
            return None
        if self.status == REVIEW_UNREVIEWED:
            return f"{self.topic}：{UNREVIEWED_NOTE}"
        highs = self.high_severity_titles()
        if highs:
            return f"{self.topic}：" + "；".join(
                f"[{SEVERITY_HIGH}] {title}" for title in highs
            )
        if self.findings:
            return None
        if self.unclassified():
            return f"{self.topic}：{UNCLASSIFIED_NOTE}"
        return None

    def audit_metadata(self) -> dict:
        """工具声明的复核载荷。

        ``AuditHook`` 只取 ``status``/``topic``/``review_path``/``error`` 四个键，
        因此后面两个引擎侧键不会改变审计行；它们承载的是会话状态所需的事实——
        这份报告是否还留着未消解的终审问题（docs 03.10.7）。
        """
        return {
            "status": self.status,
            "topic": self.topic,
            "review_path": self.review_path,
            "error": self.error,
            "unresolved_note": self.unresolved_note(),
            "high_severity_titles": list(self.high_severity_titles()),
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
        # 复用不等于通过：既有意见要重新读出来，未消解的高严重度问题不会因为
        # "这次没重跑复核"而消失，而调用方需要的正是这个判定。
        try:
            comments = sidecar_comments(review_path.read_text(encoding="utf-8"))
        except OSError:
            # 摘要读得到、意见读不到只可能发生在极窄的竞态里；此时按未分类处置
            # （未结），而不是默认通过。
            comments = ""
        return ReviewOutcome(
            status=REVIEW_REUSED,
            topic=topic,
            comments=comments,
            review_path=str(review_path),
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
            f"{REVIEW_HEADING_PREFIX}{topic}\n\n{_strip_redundant_heading(comments)}\n",
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
    """把一次结果渲染为模型下一轮读取的工具结果行。

    未消解高严重度项时首行是阻断式的：这是"必须先把问题改掉"得以成立的地方，
    而它必须来自解析（``ReviewOutcome``）而非复核者行文的语气。完整意见仍然
    照原样附在后面，模型需要其中的位置与依据才能修订。
    """
    if outcome.status == REVIEW_SKIPPED:
        return []
    if outcome.status == REVIEW_UNREVIEWED:
        return [unreviewed_warning(outcome.error)]

    highs = outcome.high_severity_titles()
    lines: list[str] = []
    if highs:
        lines.append(_BLOCKING_HEADER.format(count=len(highs)))
        lines.extend(f"  - [{SEVERITY_HIGH}] {title}" for title in highs)
        lines.append("- " + _REVISE_HIGH)
    elif outcome.unclassified():
        lines.append(_UNCLASSIFIED_HEADER.format(count=len(_comment_lines(outcome.comments))))
        lines.append("- " + _REVISE_UNCLASSIFIED)
    else:
        lines.append("- 风险终审意见（独立复核，供你决定是否修订）：")
    lines.extend(f"  {line}" for line in (outcome.comments or NO_FINDINGS).splitlines())
    if outcome.status == REVIEW_REUSED:
        lines.append(_REUSED_LINE)
    if outcome.error:  # sidecar 写入失败；意见仍然有效
        lines.append(f"- 终审意见落盘失败：{outcome.error}")
    return lines


def sidecar_comments(text: str) -> str:
    """从 sidecar 的全文中取回复核者写下的部分（跳过摘要闩锁行与模板标题）。"""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith(REVIEW_HEADING_PREFIX):
            return "\n".join(lines[index + 1 :]).strip()
    return text.strip()


def _comment_lines(comments: str) -> list[str]:
    """意见里的非空行；无法解析条目时用它给出一个规模提示。

    这是个近似值（硬折行会被算成多条），但没有更好的口径可用：既然解析不出条目，
    条数本身就是不可知的，而"有多少意见没被读懂"仍必须说个大概。
    """
    return [line for line in comments.splitlines() if line.strip()]


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

"""数字核对内核：找出摘要里"提了量却缺值"的位置，并按页定位其原文出处（docs 03.10）。

`reduce` 之后的最后一道关。全局摘要会在两处丢数字：分片摘要被裁剪（截断），或归并
时取舍掉具体值。前者由 ``tools/meta/summarize.py`` 的裁剪标记留下确证痕迹，后者常
留下一个悬空的量词（如"同比少增…"）。

本模块只做**纯逻辑**：识别缺口、给出定位锚文本、由锚文本回到原文页码。真正的"按页
精读补齐"由 ``summarize_document`` 通过 ``coordinator.spawn`` 派子代理执行 ``read_pdf``
——复用既有子代理及其只读读文件能力，不新造一条调用路径。

为什么只查"缺值"而不比对原文数字全集：一份研报有数百个数字，摘要本就不可能全收，
按全集比对会把正常的取舍成批误报为缺失，既刷屏又把子代理预算烧光。可判定为缺陷的是
**摘要自己引用了某个量却给不出值**——截断或漏抄在这里留下确定的痕迹。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 分片摘要被裁剪时留下的标记，与 ``tools/meta/summarize.py`` 同源：工具用它作为截断
# 后缀，本模块据此把它还原成一处缺口。两边引用同一个常量，因此"写标记"与"查标记"
# 不会各自漂移。
CLIP_MARKER = "…（本片摘要已截断）"

# 量词：出现它通常意味着后面应当跟一个具体数值。按长度倒序匹配，避免"同比少增"
# 被短词"少增"抢先匹配。
QUANTITY_CUES: tuple[str, ...] = (
    "同比少增",
    "同比多增",
    "环比少增",
    "环比多增",
    "分别",
    "同比",
    "环比",
    "少增",
    "多增",
    "增长",
    "减少",
    "增加",
    "下降",
    "上升",
    "降幅",
    "增幅",
    "扩大",
    "收窄",
    "回落",
    "增速",
    "累计",
    "单月",
    "占比",
)

# 一次核对最多识别的缺口数：用于封顶下游派发的子代理数量，避免异常输入导致膨胀。
MAX_GAPS = 12

# 定位锚文本的长度上限（字符，去空白后）。
_ANCHOR_CHARS = 40
# 锚文本最短可用长度；比这更短就没有定位价值，宁可放弃也不乱定位。
_MIN_ANCHOR_CHARS = 4
# 逐级缩短锚尾时的最短长度。取 4 个字符：截断处离缺值最近，四字仍足以在一页里
# 唯一命中（如"比少增""1.4"），再短就开始乱命中。
_MIN_NEEDLE_CHARS = 4

_ELLIPSIS_RE = re.compile(r"…+|\.{3,}")
# 省略号前这个窗口内若出现量词或数字，说明被省掉的是一个值，而不是行文里的停顿。
_CONTEXT_CHARS = 10


@dataclass(slots=True)
class Gap:
    """一处数字缺口：摘要在这里提到了量，却没有给出值。

    ``anchor`` 是缺口**之前**的原话片段，用作回到原文定位页码的锚——它通常是原文的
    逐字切片，因此比量词本身更能精确定位。
    """

    cue: str
    anchor: str


def _digest(text: str) -> str:
    """去掉所有空白。PDF 抽取会在数字与单位之间插入空格，逐字比对前必须先归一。"""
    return "".join(text.split())


def _label(anchor: str, *, fallback: str = "数值缺失") -> str:
    """给缺口一个人可读的标签：优先取锚里的量词连缀，其次取锚尾。"""
    for cue in QUANTITY_CUES:
        if cue in anchor:
            index = anchor.find(cue)
            return anchor[max(index - 4, 0) : index + len(cue)]
    return anchor[-12:] or fallback


def _is_value_context(before: str) -> bool:
    """省略号之前是否指向一个"值"：末尾是数字，或以量词收尾。

    真实截断有两种形态——"同比少增…"（量词后直接断）与"分别扩大 1.4…"（数值断在
    中间）。前者看量词，后者看数字；行文里的"……"两者都不满足，因此不会误报。
    """
    tail = before[-_CONTEXT_CHARS:]
    if tail and tail[-1].isdigit():
        return True
    return any(tail.endswith(cue) for cue in QUANTITY_CUES)


def _gaps_from_ellipsis(segment: str) -> list[Gap]:
    """段内省略号前跟着量词或半个数字：说明此处原本有值，被截掉了。"""
    gaps: list[Gap] = []
    for match in _ELLIPSIS_RE.finditer(segment):
        before = _digest(segment[: match.start()])
        if not _is_value_context(before):
            continue
        anchor = before[-_ANCHOR_CHARS:]
        if anchor:
            gaps.append(Gap(cue=_label(anchor), anchor=anchor))
    return gaps


def find_gaps(text: str) -> list[Gap]:
    """扫描文本，返回按出现顺序去重后的数字缺口。

    两类确证缺口：**裁剪标记**（本工具自己留下的，标记之前即被裁掉的值）与**值上下文
    里的省略号**（量词或数字之后被截断）。悬空量词后跟句号这类模糊情形不在此列——它
    可能只是正常行文（如小节标题），收进来只会制造噪声。

    锚按标记切段后再取尾，这样它不会跨过前一个缺口：否则相邻两句的锚会互相吞并，
    去重随之失效。
    """
    if not text:
        return []
    unique: list[Gap] = []
    seen: set[str] = set()
    segments = text.split(CLIP_MARKER)
    for index, segment in enumerate(segments):
        # 除最后一段外，每段都由一个标记收尾，故这段的尾部就是一处缺口。
        if index < len(segments) - 1:
            anchor = _digest(segment)[-_ANCHOR_CHARS:]
            if anchor:
                unique.append(Gap(cue=_label(anchor, fallback="分片摘要被截断"), anchor=anchor))
        unique.extend(_gaps_from_ellipsis(segment))

    deduped: list[Gap] = []
    for gap in unique:
        if gap.anchor in seen:
            continue
        seen.add(gap.anchor)
        deduped.append(gap)
        if len(deduped) >= MAX_GAPS:
            break
    return deduped


def pages_for_anchor(pages: list[str], anchor: str, *, limit: int = 3) -> list[int]:
    """在逐页原文里定位锚文本，返回命中页码（1 起）。

    先按整段锚匹配；失败则逐级缩短**锚的尾部**再试——缺口在锚之后，故锚尾离缺失数值
    最近，也最可能逐字出现在原文里（较短的前缀可能跨了页眉页脚）。定位不到就返回空
    列表：宁可说"未定位"，也不要指向一个猜测的页码。
    """
    digest_anchor = _digest(anchor)
    if len(digest_anchor) < _MIN_ANCHOR_CHARS or not pages:
        return []
    digests = [_digest(page) for page in pages]
    floor = min(_MIN_NEEDLE_CHARS, len(digest_anchor))
    for size in range(len(digest_anchor), floor - 1, -1):
        needle = digest_anchor[-size:]
        hits = [index + 1 for index, page in enumerate(digests) if needle in page]
        if hits:
            return hits[:limit]
    return []


def page_span(pages: list[int]) -> str:
    """把命中页码收敛为一次可读完的连续区间。

    ``read_pdf`` 单次上限 10 页，且每次调用只接受一个区间（``'2-5'``、``'2-'``），
    不接受逗号列表；因此定位到多个不连续页时也只能从首片起读，宁可多读几页。
    """
    first = min(pages)
    last = min(max(pages), first + 9)
    return f"{first}-{last}" if last > first else f"{first}"


def repair_task(*, path: str, pages: list[int], cue: str) -> str:
    """构造一个自包含的补齐任务，交子代理用 ``read_pdf`` 按页精读。

    任务文本必须自包含（谁、要什么、材料在哪），这是 ``spawn`` 的契约：worker 拿不到
    主上下文，只吃这里写下的东西。
    """
    span = page_span(pages)
    return (
        "你的任务：核对一份研报中的具体数值，并原样报告。\n"
        f"材料：本地 PDF，路径 {path}\n"
        f'用 read_pdf 读取第 {span} 页（pages="{span}"），只读这些页。\n'
        f"要找的内容：摘要里出现了「{cue}」，但具体数值缺失或被截断。\n"
        "请在上述页面的原文中找出与之对应的确切数字（保留原文口径与单位，如"
        "同比／环比、百分比或金额），逐条报告，并注明来自第几页。\n"
        "要求：\n"
        "1. 只报告原文真实出现的数字，原样抄录，不要改写、不要换算、不要推断；\n"
        "2. 页面上找不到就明确写“该页未见对应数值”，不得猜测；\n"
        "3. 不要复述任务，直接给结论。"
    )

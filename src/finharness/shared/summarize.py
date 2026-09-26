"""长文档的结构感知切分内核（docs 03.10）。

为什么不是"把全文塞给一次模型调用"：单篇研报、公告或报告主体动辄数万字，远超一次
请求的窗口，也远超单条工具结果的预算。因此把它拆成带稳定 id 的分片，由调用方按片
消化——``summarize_document`` 只返回索引，主 Agent 发现 ``spawn_agent`` 后再派 worker。

这里只放**纯逻辑**：分片、骨架、id 与乱序复原、归并分组。LLM 消化由主循环发现
spawn 驱动，本模块不再假定 ``Coordinator.spawn`` 是唯一调用路径。

两条必须正视的固有代价，本模块各有一条对策：

* **并行结果会乱序。** 分片并发消化时顺序不能作为契约。对策是给每片发布稳定
  id（``P01``…），要求每片结论以自身 id 开头，汇合前**按 id 重排**——顺序由数据
  决定，而不由返回次序决定。
* **分片会丢掉全局语义与逻辑关系。** 对策有三条：按结构（页/标题）切分而非定长
  截断；片间保留重叠以接住跨页的表格与结论；把一次抽取的**文档骨架**注入每一个
  分片任务，使每片都知道自己处在全文的哪一段。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 分片目标大小（token）。它要留出余量给骨架、任务说明与提示，因此明显小于模型窗口。
CHUNK_TOKENS = 2000
# 相邻分片的重叠（字符）。跨页的表格与"综上"式结论常落在接缝上，重叠让两侧都能
# 看到它；代价是少量重复，归并时按 id 归属，不重复计数。
CHUNK_OVERLAP_CHARS = 300
# 单次 spawn 至多派发的子任务数，与 Coordinator.MAX_SPAWN_TASKS 对齐。
MAX_TASKS_PER_BATCH = 8

# 结构行：中文公文/研报常见的各级标题。
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"第[一二三四五六七八九十百]+[章节部分]"
    r"|[一二三四五六七八九十]+[、.．]"
    r"|[（(][一二三四五六七八九十\d]+[）)]"
    r"|\d+(?:\.\d+)*[、.．\s]"
    r"|附[录表]"
    r")"
)
# 形如 "1.1 标题" 的编号行也算结构行。
_NUMBERED_RE = re.compile(r"^\s*\d+\.\d+")


@dataclass(slots=True)
class Chunk:
    """一个分片：稳定的 id、它在全文中的位置，以及它的文本。"""

    id: str
    seq: int
    text: str
    label: str = ""
    # 与前一片重叠的部分（若该片不是首片）。显式保留，使归并知道这部分已在前片出现。
    overlap: str = ""


@dataclass(slots=True)
class MapResult:
    """一片的摘要结果；``ok`` 为 False 时代表该片未能产出。"""

    chunk: Chunk
    summary: str = ""
    ok: bool = True
    error: str = ""


def chunk_id(seq: int) -> str:
    """分片 id：``P01``、``P02``…（1 起，两位补零）。

    这个 id 是全链路的主键：写进 map 任务、要求写进 map 输出、归并前据此重排。
    """
    return f"P{seq:02d}"


def _tokens(text: str, counter) -> int:
    """统计 token 数；无计数器时按文档约定的中文字符近似值回退。"""
    if not text:
        return 0
    if counter is None:
        from finharness.utils.text import CHARS_PER_TOKEN

        return int(len(text) / CHARS_PER_TOKEN)
    return counter.count(text).tokens


def split_document(
    text: str,
    *,
    counter=None,
    budget_tokens: int = CHUNK_TOKENS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """把文档切成带 id 与重叠的分片。

    切分是**结构感知**的：优先在标题行或空行处断开，使一节不被打断；只有在单节
    本身超过预算时，才退回按字符窗口硬切。定长切分会把"结论"与它依据的表格切到
    两片，这正是分片丢语义的主要来源。
    """
    if not text or not text.strip():
        return []
    segments = _structural_segments(text)
    chunks: list[Chunk] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        chunks.append(_make_chunk(len(chunks) + 1, "\n".join(buffer)))
        buffer.clear()

    for segment in segments:
        candidate = "\n".join([*buffer, segment]) if buffer else segment
        if buffer and _tokens(candidate, counter) > budget_tokens:
            flush()
            candidate = segment
        # 单节本身就超预算：硬切成若干片，但仍逐片编号。
        if _tokens(candidate, counter) > budget_tokens:
            for piece in _hard_split(candidate, counter, budget_tokens):
                chunks.append(_make_chunk(len(chunks) + 1, piece))
            buffer.clear()
            continue
        buffer = [candidate]
    flush()

    if overlap_chars > 0:
        _apply_overlap(chunks, overlap_chars)
    return chunks


def _make_chunk(seq: int, text: str) -> Chunk:
    return Chunk(id=chunk_id(seq), seq=seq, text=text, label=_label_of(text))


def _label_of(text: str) -> str:
    """该片的标签：优先取首个标题行，否则取首句。"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and (_HEADING_RE.match(stripped) or _NUMBERED_RE.match(stripped)):
            return stripped[:40]
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:40]


def _structural_segments(text: str) -> list[str]:
    """按标题行与空行把文本切成语义段落。"""
    segments: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        is_heading = bool(_HEADING_RE.match(stripped) or _NUMBERED_RE.match(stripped))
        if stripped and is_heading and current:
            segments.append("\n".join(current).strip())
            current = [line]
            continue
        if not stripped and current:
            segments.append("\n".join(current).strip())
            current = []
            continue
        current.append(line)
    if current:
        segments.append("\n".join(current).strip())
    return [segment for segment in segments if segment]


def _hard_split(text: str, counter, budget_tokens: int) -> list[str]:
    """单节超预算时按字符窗口硬切（此时结构无从依循）。"""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        low, high = start + 1, len(text)
        # 二分找能放进预算的最长前缀。
        while low < high:
            middle = (low + high + 1) // 2
            if _tokens(text[start:middle], counter) <= budget_tokens:
                low = middle
            else:
                high = middle - 1
        pieces.append(text[start:low])
        start = low
    return pieces


def _apply_overlap(chunks: list[Chunk], overlap_chars: int) -> None:
    """给每个（非首）分片接上前一片的尾部，作为显式的重叠段。"""
    for previous, current in zip(chunks, chunks[1:]):
        tail = previous.text[-overlap_chars:]
        if not tail:
            continue
        current.overlap = tail
        current.text = tail + "\n" + current.text


def document_outline(text: str, *, max_headings: int = 40) -> str:
    """抽取一份廉价的全文档骨架：首个非空行 + 各级标题清单。

    这份骨架会注入**每一个** map 任务。它是恢复全局语义最省成本的一招：单个分片
    因此知道自己处在全文的哪一段、全文在讲什么，而不必靠猜。
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    title = lines[0][:80]
    headings = [
        line[:60]
        for line in lines[1:]
        if _HEADING_RE.match(line) or _NUMBERED_RE.match(line)
    ]
    outline = [f"标题：{title}"]
    if headings:
        shown = headings[:max_headings]
        outline.append("目录/各级标题：")
        outline.extend(f"  - {item}" for item in shown)
        if len(headings) > len(shown):
            outline.append(f"  …（另有 {len(headings) - len(shown)} 个标题）")
    return "\n".join(outline)


def map_task(chunk: Chunk, *, outline: str = "", question: str | None = None) -> str:
    """构造一个分片的摘要任务文本（内联该片全文，不依赖 worker 读文件）。"""
    total_hint = f"（本片为 第 {chunk.seq} 片，id {chunk.id}）"
    parts = [
        f"请为下面这段文档分片生成摘要。{total_hint}",
    ]
    if outline:
        parts.append(
            "全文骨架（供你判断本片所处的上下文；不要复述它，也不要为它编造内容）：\n"
            + outline
        )
    if question:
        parts.append(f"聚焦问题：{question}\n优先保留与它相关的数据与结论。")
    parts.append(
        "输出要求（严格遵守）：\n"
        f"1. 第一行只写分片 id：{chunk.id}\n"
        "2. 然后给出本片所属章节（若有标题则引用它）\n"
        "3. 列出本片的关键事实与数据（保留原文数字，不要改写）\n"
        "4. 页眉页脚、图表标题与图注、免责声明与法律条款一律不列入清单，"
        "避免挤占篇幅\n"
        "5. 若本片出现指向其他章节的线索（如“如上文”“详见后文”），单独一行记录为"
        "「跨片线索：…」，使后续归并能接上逻辑\n"
        "6. 只写结论，不复述任务；材料中没有的内容不得编造"
    )
    if chunk.overlap:
        parts.append(
            "注意：本片开头有一段与上一片重叠的内容（用于接住跨页的表格/结论）；"
            "它在上一片已出现过，摘要时不必重复强调。"
        )
    parts.append("文档分片正文如下：\n<chunk>\n" + chunk.text + "\n</chunk>")
    return "\n\n".join(parts)


_ID_LINE_RE = re.compile(r"^\s*(?:\[)?(P\d{1,3})(?:\])?\s*[:：、.]?\s*", re.IGNORECASE)
_ID_ANY_RE = re.compile(r"(?:\[)?(P\d{1,3})(?:\])?", re.IGNORECASE)


def parse_map_summary(text: str, *, expected_id: str) -> tuple[str, str]:
    """从一片摘要中解析出 ``(id, 正文)``。

    模型可能漏写 id，也可能换个位置写。因此先看首行，再在全文里找第一个像 id 的
    记号；都没有时回退到该片应有的 id——归一会按位置把它放回去，而不是把它丢掉。
    """
    body = (text or "").strip()
    if not body:
        return expected_id, ""
    first_line, _, rest = body.partition("\n")
    match = _ID_LINE_RE.match(first_line)
    if match:
        return match.group(1).upper(), rest.strip()
    found = _ID_ANY_RE.search(body)
    if found:
        stripped = _ID_ANY_RE.sub("", body, count=1).strip()
        return found.group(1).upper(), stripped
    return expected_id, body


def order_by_id(results: list[MapResult], *, total: int) -> list[MapResult]:
    """按 id 把分片摘要重排成全文顺序。

    这是"并行乱序"的解法：返回次序不可信，id 可信。已知 id 按文档顺序排列；无法
    归属的 id 一律**排在末尾**（而不是按它恰好出现的位置插进中间）——这样下游要么
    看到正确的顺序，要么看到明确的"这几片来历不明"，而不会把一个错位的片段误读成
    正文的中间部分。同一 id 重复出现时保持它们彼此原有的相对次序。
    """
    rank = {chunk_id(seq): seq for seq in range(1, total + 1)}
    known: list[tuple[int, int, MapResult]] = []
    unknown: list[MapResult] = []
    for position, result in enumerate(results):
        value = rank.get(result.chunk.id.upper())
        if value is None:
            unknown.append(result)
        else:
            known.append((value, position, result))
    known.sort(key=lambda item: (item[0], item[1]))
    return [result for _, _, result in known] + unknown


def batch_chunks(chunks: list[Chunk], size: int = MAX_TASKS_PER_BATCH) -> list[list[Chunk]]:
    """把分片按每次 spawn 的上限分批。"""
    if size <= 0:
        size = MAX_TASKS_PER_BATCH
    return [chunks[index : index + size] for index in range(0, len(chunks), size)]


def needs_layered_reduce(items: list[str], *, counter=None, budget_tokens: int = CHUNK_TOKENS) -> bool:
    """分片摘要合起来是否已超出一次归并所能容纳的量。"""
    if not items:
        return False
    return _tokens("\n".join(items), counter) > budget_tokens


def reduce_groups(items: list[str], size: int = MAX_TASKS_PER_BATCH) -> list[list[str]]:
    """把待归并的材料按组切分，供分层 reduce 先分组归并再全局归并。"""
    if size <= 0:
        size = MAX_TASKS_PER_BATCH
    return [items[index : index + size] for index in range(0, len(items), size)]


def reduce_task(
    items: list[str],
    *,
    total: int,
    question: str | None = None,
    final: bool = True,
) -> str:
    """构造归并任务：按 id 与章节归属重建逻辑，而不是按到达顺序拼接。"""
    scope = f"全部 {total} 片" if final else "本组若干片"
    parts = [
        f"下面是同一份文档的分片摘要（{scope}）。请把它们归并为"
        + ("一份连贯的全局摘要。" if final else "一段阶段性归并结果。"),
        "要求：\n"
        "1. 按每片自报的分片 id（P01、P02…）与所属章节，重建全文的逻辑顺序，"
        "不要按它们在本消息中出现的顺序拼接；\n"
        "2. 合并同一主题的重复内容，但不得丢失任何数字或结论；\n"
        "3. 凡属跨片综合得出的判断，明确标注为“跨片综合”；\n"
        "4. 每条关键事实后保留其出处 id，形如 [P03]，使读者可回查；\n"
        "5. 若有分片标注自身失败或缺失，明确指出该部分内容缺失，不得假装完整；\n"
        "6. 不要编造材料中没有的信息。",
    ]
    if question:
        parts.append(f"聚焦问题：{question}\n围绕它组织摘要，但同时保留必要的背景。")
    if not final:
        parts.append("保留各自的分片出处 id，供下一级归并继续使用。")
    parts.append("分片摘要如下：\n\n" + "\n\n".join(items))
    return "\n\n".join(parts)

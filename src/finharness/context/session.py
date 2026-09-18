"""按会话维护的研究状态（docs 03.6.2）。

保存模型需要作为 *研究状态* 看到的内容：当前计划、已形成的结论、已加载的技能
与已激活的工具。原始对话记录与 token 核算仍留在 ``AgentLoop`` —— 同一事实只
放在一个地方。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from finharness.config.settings import Settings
from finharness.context.tokens import default_counter, truncate_to_tokens
from finharness.data.citation import CitationRegistry

STEP_STATUSES = ("pending", "done", "fail", "skipped")
_STATUS_MARK = {"pending": "○", "done": "✓", "fail": "✗", "skipped": "－"}

# 内存中保留的结论条数上限。结论按会话累积，且状态块只渲染最后几条，
# 因此保留最近这些即可；更早的已落库，由 ``prior_conclusions`` 召回。
CONCLUSION_MEMORY_LIMIT = 50


def _today_stamp() -> str:
    """当前日期（本地时区，ISO）。

    模型此前完全不知道「今天」：全仓的 ``date.today()`` 只服务于适配器的取数窗口与工具
    默认值，没有任何日期进入上下文。缺少这个锚点，模型无法判断某个数据期是否即当前
    最新已发布期，也无法把「月度指标尚未发布」与「系统给了旧数据」区分开——它只能
    把裸期间（如 2026-08）当成可能是过期的值。随状态块逐请求注入，遂使时效判断有据。
    """
    return datetime.now().astimezone().date().isoformat()


@dataclass(slots=True)
class PlanStep:
    seq: int
    action: str
    tool_hint: list[str] = field(default_factory=list)
    skill_hint: list[str] = field(default_factory=list)
    dep: list[int] = field(default_factory=list)
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.status not in STEP_STATUSES:
            raise ValueError(
                f"unknown step status: {self.status}; expected one of {STEP_STATUSES}"
            )

    def to_dict(self) -> dict:
        """序列化用于断点（docs 03.3）；字段名与工具 schema 保持可读一致。"""
        return {
            "seq": self.seq,
            "action": self.action,
            "tool_hint": list(self.tool_hint),
            "skill_hint": list(self.skill_hint),
            "dep": list(self.dep),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanStep":
        """从断点还原；未知/缺失字段按默认值处理，损坏的条目交给上层丢弃。"""
        status = str(data.get("status") or "pending")
        if status not in STEP_STATUSES:
            status = "pending"
        try:
            seq = int(data.get("seq") or 0)
        except (TypeError, ValueError):
            seq = 0
        return cls(
            seq=seq,
            action=str(data.get("action") or ""),
            tool_hint=[str(item) for item in data.get("tool_hint") or []],
            skill_hint=[str(item) for item in data.get("skill_hint") or []],
            dep=[int(item) for item in data.get("dep") or [] if str(item).lstrip("-").isdigit()],
            status=status,
        )


@dataclass(slots=True)
class Plan:
    plan_id: str
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    revision: int = 1

    def progress(self) -> tuple[int, int]:
        """返回计划各步骤的 ``(已完成, 总数)``。"""
        total = len(self.steps)
        done = sum(1 for step in self.steps if step.status == "done")
        return done, total

    def to_dict(self) -> dict:
        """序列化用于断点。

        ``plan_id`` 与 ``revision`` 原样保留：恢复的语义是"在同一份计划上
        继续"，重新分配 id 或把修订号归零都会让后续的修订/进度台账失真。
        """
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "revision": self.revision,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Plan | None":
        """从断点还原计划；无有效步骤时返回 ``None``（空计划没有恢复价值）。"""
        steps = [
            PlanStep.from_dict(item)
            for item in data.get("steps") or []
            if isinstance(item, dict)
        ]
        if not steps:
            return None
        try:
            revision = int(data.get("revision") or 1)
        except (TypeError, ValueError):
            revision = 1
        return cls(
            plan_id=str(data.get("plan_id") or "plan_001"),
            goal=str(data.get("goal") or ""),
            steps=steps,
            revision=max(revision, 1),
        )


@dataclass(slots=True)
class Conclusion:
    text: str
    cids: list[str] = field(default_factory=list)
    ts: str = ""


class ResearchContext:
    """会话级研究状态；是计划与引用视图的唯一持有者。"""

    def __init__(self, *, cite: CitationRegistry, settings: Settings) -> None:
        self.cite = cite
        self.settings = settings
        self.plan: Plan | None = None
        self.conclusions: list[Conclusion] = []
        # 累计条数（不受内存上限影响）：计划编号由它推导，必须单调，
        # 否则裁剪会让 plan_id 重复。
        self._conclusion_total = 0
        # 本轮新增、尚未落库的结论。持久化只写这些，既避免每轮重写整份
        # 列表（O(n²)），也保证被裁剪出内存的条目一定已经落过库。
        self.pending_conclusions: list[Conclusion] = []
        self.loaded_skills: list[str] = []
        self.loaded_tools: list[str] = []
        # 由路由注入的方法论正文，按目标键去重（docs 03.8）。与 ``loaded_skills`` 同源
        # 但保存内容：名字回答"有没有"，正文回答"是什么"。
        self.methodology: dict[str, str] = {}
        # 一旦 L1 记忆建立，就由 AgentLoop 设置；这些别名转发给它，使得
        # docs 03.6.2 的 append_* 接口可用，而无需让 ctx 持有对话记录。
        self.memory: object | None = None
        # 由 loop 在对话首轮注入的记忆接口。
        # 在这里持有，是为了让每一轮渲染出相同的文本：系统 prompt 每轮会被
        # 度量三次（构建、窗口检查、压缩），这些度量结果必须一致。
        self.summary: object | None = None          # SummaryLayer
        self.short_term: object | None = None       # ShortTermMemory
        self.recalled: list[object] = []
        self.prior_conclusions: list[object] = []
        self.notes: dict[str, str] = {}
        # 跨对话长期记忆（docs 03.6.4 LTM）：首轮被动注入的最近情节 + 标的
        # 驱动刷新的命中情节。与 notes 同源同注入路径（请求末尾状态消息），
        # 但按情节而非偏好渲染。
        self.ltm_recent: list[object] = []
        self.ltm_recalled: list[object] = []
        # 语义记忆（facts/concepts）中被向量召回的部分；偏好单独走 notes
        # （它是 kind='preference' 的 facts，渲染时与知识分开）。
        self.ltm_semantic: list[object] = []
        self._remembered_symbols: list[str] = []
        # 未消解的风险终审问题，按报告主题键控（docs 03.10.7）。终审是模型对模型
        # 的内部闸门，但"这份报告还留着未处理的问题"必须是一份有状态的事实，否则
        # 模型可以在拿到意见后的任意一轮里把它忘掉并照常交付。挂在 ctx 上，因而
        # 在本执行窗口的后续每一轮都被注入（见 ``state_block``）；同一主题的新终审
        # 结果整体替换旧条目，使修订真的能清账。
        self.review_findings: dict[str, str] = {}
        # 由 loop 在对话记忆可用时设置；让元工具无需了解存储的结构即可写入
        # 全局记忆（用户偏好）。
        self.store: object | None = None
        # 语义索引（docs 03.6.4 LTM）：元工具用它做向量召回，loop 用它刷新
        # 「相关知识」区块；未配置 embedding 时为 None。
        self.semantic_index: object | None = None
        # 会话所属用户（docs 03.13）：偏好写入按它隔离；由 loop 注入。
        self.user_id: str = ""

    # -- 对话记录转发（docs 03.6.2） -------------------------------------------
    def append_user(self, text: str) -> None:
        """转发给 L1 工作记忆；未挂载记忆时为空操作。"""
        memory = self.memory
        if memory is not None:
            memory.append_user(text)  # type: ignore[attr-defined]

    def append_tool_result(self, call_id: str, content: str) -> None:
        memory = self.memory
        if memory is not None:
            memory.append_tool_result(call_id, content)  # type: ignore[attr-defined]

    def refresh_recall(self) -> None:
        """根据对话自身的标的池重新计算召回事件。

        跳过任何已作为结论可见的内容，否则召回会浪费预算去重述屏幕上已有的事实。
        """
        short_term = self.short_term
        if short_term is None:
            return
        visible = [item.text for item in self.conclusions[-6:]]
        self.recalled = short_term.recall(  # type: ignore[attr-defined]
            self.symbols, exclude_summaries=visible
        )

    def refresh_ltm_recall(self) -> None:
        """按标的池重查跨对话情节（docs 03.6.4 LTM 的标的驱动召回）。

        与 ``refresh_recall`` 同一模式：新 symbol 进入对话时，历史上关于
        该标的的情节自动带出。排除已注入的最近情节，避免同一内容两处渲染。
        """
        store = self.store
        if store is None or not self.symbols:
            self.ltm_recalled = []
            return
        recent_ids = {
            getattr(item, "ep_uid", "") for item in self.ltm_recent
        }
        recalled: list[object] = []
        for symbol in self.symbols:
            for item in store.list_ltm_episodes(
                user_id=self.user_id, subject=symbol, limit=3
            ):
                if getattr(item, "ep_uid", "") not in recent_ids:
                    recalled.append(item)
        self.ltm_recalled = recalled

    def refresh_semantic_recall(self, query: str, *, index: object | None = None) -> None:
        """按当前问题做一次语义召回（docs 03.6.4 LTM 向量检索）。

        由 loop 每个 ``run()`` 调用一次：语义召回天然是"按问题找知识"，
        不是"按标的找事件"，因此它的驱动信号是问题文本而非标的池。
        未配置嵌入端点（``index`` 为 None 或未启用）时清空，让这一区块
        完全不出现在请求里——不留下一个空标题。
        """
        enabled = index is not None and getattr(index, "enabled", False)
        if not enabled or not query.strip():
            self.ltm_semantic = []
            return
        try:
            self.ltm_semantic = list(
                index.recall(  # type: ignore[attr-defined]
                    user_id=self.user_id,
                    query=query,
                    limit=int(self.settings.ltm.semantic_top_k),
                )
            )
        except Exception:  # noqa: BLE001 - 语义召回失败即视为无命中
            self.ltm_semantic = []

    # -- 标的 ------------------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        """已覆盖标的池：引用中的标的加上任何被显式记住的标的。"""
        seen = self.cite.resolve_symbols()
        for symbol in self._remembered_symbols:
            if symbol not in seen:
                seen.append(symbol)
        return seen

    def remember_symbol(self, symbol: str) -> None:
        """记录一个已覆盖标的（用于重新加载某段对话时）。"""
        if symbol and symbol not in self._remembered_symbols:
            self._remembered_symbols.append(symbol)

    # -- 计划 ------------------------------------------------------------------
    def set_plan(self, goal: str, steps: list[PlanStep]) -> Plan:
        """安装一份计划；重复调用会修订它并递增修订号。"""
        if self.plan is not None:
            self.plan = Plan(
                plan_id=self.plan.plan_id,
                goal=goal,
                steps=steps,
                revision=self.plan.revision + 1,
            )
        else:
            self.plan = Plan(
                plan_id=f"plan_{self._conclusion_total + 1:03d}", goal=goal, steps=steps
            )
        return self.plan

    def mark_plan_step(self, seq: int, status: str) -> bool:
        """更新某一步骤的状态；步骤不存在时返回 False。"""
        if self.plan is None:
            return False
        for step in self.plan.steps:
            if step.seq == seq:
                if status not in STEP_STATUSES:
                    raise ValueError(f"unknown step status: {status}")
                step.status = status
                return True
        return False

    def plan_digest(self) -> str:
        """渲染计划以供注入；没有计划时返回空字符串。

        每个步骤既带自身的标记，*也*带其依赖项的标记，因此模型一眼就能看出某步
        是否已解除阻塞。标题栏给出已完成/总数，形成一份紧凑的进度台账，而不是一份
        需要模型反复阅读才能弄清自己所处位置的列表。
        """
        if self.plan is None:
            return ""
        done, total = self.plan.progress()
        lines = [
            f"当前研究计划（{self.plan.plan_id}，第 {self.plan.revision} 版，"
            f"进度 {done}/{total}）：{self.plan.goal}"
        ]
        marks = {step.seq: _STATUS_MARK.get(step.status, "?") for step in self.plan.steps}
        for step in self.plan.steps:
            mark = marks.get(step.seq, "?")
            hints: list[str] = []
            if step.tool_hint:
                hints.append(f"建议工具：{', '.join(step.tool_hint)}")
            if step.skill_hint:
                hints.append(f"建议技能：{', '.join(step.skill_hint)}")
            if step.dep:
                # 渲染每个依赖项的当前标记，使未满足的前置条件无需模型
                # 交叉比对步骤编号即可看出。
                rendered = "，".join(
                    f"{dep}{marks.get(dep, '?')}" for dep in step.dep
                )
                hints.append(f"依赖：{rendered}")
            hint = f"（{'；'.join(hints)}）" if hints else ""
            lines.append(f"  {mark} {step.seq}. {step.action}{hint}")
        return "\n".join(lines)

    # -- 结论 ------------------------------------------------------------------
    def add_conclusion(self, text: str, cids: list[str]) -> Conclusion:
        """追加一条结论，附上时间戳与支撑引用的 cid。

        内存只保留最近 ``CONCLUSION_MEMORY_LIMIT`` 条：该列表按会话累积，
        而状态块与召回都只看最近几条。落库不受影响——新增的结论进入
        ``pending_conclusions`` 由本轮持久化写出，且 ``prior_conclusions``
        会带回更早的结论。
        """
        conclusion = Conclusion(
            text=text,
            cids=list(cids),
            ts=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self.conclusions.append(conclusion)
        self.pending_conclusions.append(conclusion)
        self._conclusion_total += 1
        if len(self.conclusions) > CONCLUSION_MEMORY_LIMIT:
            del self.conclusions[: len(self.conclusions) - CONCLUSION_MEMORY_LIMIT]
        return conclusion

    # -- 技能与工具 ------------------------------------------------------------
    def add_skill(self, name: str) -> bool:
        """记录一个已加载的技能；若此前已加载则返回 False。"""
        if name in self.loaded_skills:
            return False
        self.loaded_skills.append(name)
        return True

    def inject_methodology(self, key: str, body: str) -> bool:
        """记录一段由路由注入的方法论正文；已注入过则返回 False。

        ``loaded_skills`` 只记名字，用于向模型交代"你已经有了哪几份方法"；
        这里保存**正文**，因为注入的是一份要照着做的方法，而不是一条索引。
        两者分开，是因为状态块需要同时给出这两件事：哪几份是流程、内容是什么。
        """
        if key in self.methodology:
            return False
        self.methodology[key] = body
        self.add_skill(key)
        return True

    def activate_tool(self, name: str) -> bool:
        """记录一个已激活的工具；若此前已激活则返回 False。"""
        if name in self.loaded_tools:
            return False
        self.loaded_tools.append(name)
        return True

    # -- 系统注入 --------------------------------------------------------------
    def state_block(self) -> str:
        """追加到系统 prompt 的研究状态段落（docs 03.6.2）。

        按模型对其需求的紧迫程度排列，组合四个不同的面向：当前日期、当前计划、本对话的
        结论、召回的事件，然后是分段的历史摘要。每个面向的形态都刻意不同 —— 见
        记忆包的文档字符串。
        """
        parts: list[str] = [f"当前日期：{_today_stamp()}"]
        digest = self.plan_digest()
        if digest:
            parts.append(digest)
        if self.loaded_skills:
            parts.append("已加载方法论：" + "、".join(self.loaded_skills))
        # 正文紧随名字之后。它由引擎按意图注入，因此模型无需（也无法）请求加载；
        # 放在状态块里而不是静态 system prompt 里，是为了不把缓存边界移到对话最前端。
        for key, body in self.methodology.items():
            parts.append(f"【方法论：{key}】\n{body}")

        conclusion_lines = self._conclusion_lines()
        if conclusion_lines:
            parts.append("本对话已形成结论：\n" + "\n".join(conclusion_lines))

        recalled = self._recalled_lines()
        if recalled:
            parts.append("相关历史事件：\n" + "\n".join(recalled))

        prior = self._prior_conclusion_lines()
        if prior:
            parts.append(
                "历史结论（来自本对话更早的轮次，时间敏感数据请重新核对）：\n"
                + "\n".join(prior)
            )

        ltm = self._ltm_lines()
        if ltm:
            parts.append(ltm)

        semantic = self._semantic_lines()
        if semantic:
            parts.append(semantic)

        summary = self._summary_text()
        if summary:
            parts.append(summary)

        preferences = self._preference_lines()
        if preferences:
            parts.append("用户偏好（所有对话共享）：\n" + "\n".join(preferences))

        findings = self._review_finding_lines()
        if findings:
            parts.append(
                "未消解的风险终审问题（须先修订，不得声称已复核）：\n"
                + "\n".join(findings)
            )

        if not parts:
            return ""
        return "\n\n【会话研究状态】\n" + "\n".join(parts)

    # -- 注入辅助 --------------------------------------------------------------
    def note_review_finding(self, topic: str, note: str | None) -> None:
        """记录/清除一份报告的未结终审事项（docs 03.10.7）。

        ``note`` 为 None 表示该报告已消解（最新一次终审不再留有未处理的高严重度
        问题，或报告已被修订后重审通过）——此时必须删掉旧条目，否则一份改好的
        报告会永远背着旧账。以主题为键：一个会话可能产出多份报告，各自独立记账。
        """
        if not topic:
            return
        if note:
            self.review_findings[topic] = note
        else:
            self.review_findings.pop(topic, None)

    def _review_finding_lines(self) -> list[str]:
        return [f"  - {note}" for _, note in sorted(self.review_findings.items())]
    def _conclusion_lines(self) -> list[str]:
        return [
            f"  - {item.text}（依据 {'、'.join(item.cids) or '无'}）"
            for item in self.conclusions[-3:]
        ]

    def _recalled_lines(self) -> list[str]:
        """召回事件的注入文本，受 ``context.recall_max_tokens`` 约束。

        该设置此前虽存在于配置、环境变量与文档中，却没有任何消费者：L2 召回按事件
        条数上限注入，token 侧全无约束，于是一段召回可以挤占本应留给当前问题的窗口。
        这里把它接上，使文档承诺真正生效（docs 03.6.4）。
        """
        short_term = self.short_term
        if short_term is None or not self.recalled:
            return []
        text = short_term.render(  # type: ignore[attr-defined]
            self.recalled,
            counter=default_counter(),
            max_tokens=int(self.settings.context.recall_max_tokens),
            header=None,
        )
        return text.splitlines()

    def _prior_conclusion_lines(self) -> list[str]:
        """从存储中重新加载的结论；带上日期，使过期可被察觉。"""
        lines: list[str] = []
        for record in self.prior_conclusions:
            text = getattr(record, "text", "")
            if any(text == item.text for item in self.conclusions):
                continue
            formed = str(getattr(record, "ts", ""))[:10]
            subject = getattr(record, "subject", "")
            lines.append(f"  - {text}（{subject}，形成于 {formed}）")
        return lines[:5]

    def _summary_text(self) -> str:
        summary = self.summary
        if summary is None:
            return ""
        return summary.render(max_tokens=self.settings.context.summary_inject_max_tokens)  # type: ignore[attr-defined]

    # -- 跨对话长期记忆渲染 ----------------------------------------------------
    _LTM_KIND_LABELS = {
        "task_result": "任务结果",
        "decision": "关键决策",
        "excerpt": "对话片段",
    }

    def _ltm_lines(self) -> str:
        """【跨对话记忆】区块：最近情节 + 标的命中情节，受注入 token 上限约束。

        每条带类型、时间与来源对话标题，使模型能判断新旧与出处；结尾提示
        主动检索工具，超出被动注入之外的情节由 agent 按需查询。
        """
        episodes = list(self.ltm_recent) + [
            item
            for item in self.ltm_recalled
            if getattr(item, "ep_uid", "")
            not in {getattr(seen, "ep_uid", "") for seen in self.ltm_recent}
        ]
        if not episodes:
            return ""
        lines = ["跨对话记忆（来自此前其他对话，时间敏感数据请重新核对）："]
        for item in episodes:
            kind = self._LTM_KIND_LABELS.get(
                getattr(item, "kind", ""), getattr(item, "kind", "")
            )
            formed = str(getattr(item, "source_ts", "") or getattr(item, "created_at", ""))[:10]
            title = getattr(item, "source_title", "") or "未命名对话"
            subject = getattr(item, "subject", "")
            scope = f"{subject}，" if subject else ""
            cids = getattr(item, "cids", ()) or ()
            basis = f"，依据 {'、'.join(cids)}" if cids else ""
            lines.append(
                f"  - [{kind}] {scope}{getattr(item, 'summary', '')}"
                f"（{title}，{formed}{basis}）"
            )
        lines.append("  （如需更多历史研究，可调用 search_memory 检索跨对话记忆）")
        text = "\n".join(lines)
        limit = int(self.settings.ltm.inject_max_tokens)
        if limit > 0:
            text = truncate_to_tokens(text, default_counter(), limit)
        return text

    _FACT_KIND_LABELS = {
        "fact": "事实",
        "concept": "概念",
    }

    def _semantic_lines(self) -> str:
        """【相关知识】区块：按问题向量召回的语义记忆，受注入上限约束。

        与【跨对话记忆】（情节）刻意分开：情节是"我们做过什么"，语义是
        "我们知道什么"，模型对两者的用法不同——前者用于避免重复劳动，
        后者用于直接采用既定的口径。偏好不在这里：它由 ``notes`` 渲染为
        独立的"用户偏好"区块，因为那是**必须遵守**的指令而非背景知识。
        """
        facts = [
            item
            for item in self.ltm_semantic
            if getattr(item, "kind", "") != "preference"
        ]
        if not facts:
            return ""
        lines = ["相关知识（跨对话积累，与当前问题语义相近）："]
        for item in facts:
            kind = self._FACT_KIND_LABELS.get(
                getattr(item, "kind", ""), getattr(item, "kind", "")
            )
            subject = getattr(item, "subject", "")
            scope = f"{subject}：" if subject else ""
            lines.append(f"  - [{kind}] {scope}{getattr(item, 'statement', '')}")
        text = "\n".join(lines)
        limit = int(self.settings.ltm.semantic_inject_max_tokens)
        if limit > 0:
            text = truncate_to_tokens(text, default_counter(), limit)
        return text

    def _preference_lines(self) -> list[str]:
        return [f"  - {key}：{value}" for key, value in sorted(self.notes.items())]

export type PlanStepStatus = "pending" | "done" | "fail" | "skipped";

export type ResearchPlanStep = {
  seq: number;
  action: string;
  status: PlanStepStatus;
  dep: number[];
};

export type ResearchPlan = {
  plan_id: string;
  goal: string;
  revision: number;
  done: number;
  total: number;
  steps: ResearchPlanStep[];
};

export type PlanStatusMeta = {
  label: string;
  tone: PlanStepStatus;
  mark: string;
};

const TOOL_ACTIONS: Record<string, string> = {
  get_quote: "读取最新行情",
  get_kline: "读取历史走势",
  get_financials: "读取财务报表",
  get_indicators: "读取财务指标",
  get_valuation: "获取估值数据",
  get_peers: "比较同业公司",
  get_announcements: "查询公司公告",
  get_market_news: "检索市场资讯",
  get_macro_indicators: "读取宏观指标",
  get_industry_perf: "读取行业表现",
  get_industry_constituents: "读取行业成分股",
  get_research_reports: "检索券商研究报告",
  calc_metrics: "拆解财务指标",
  calc_valuation: "执行估值测算",
  run_backtest: "执行策略回测",
  make_chart: "生成研究图表",
  write_report: "生成研究报告并复核",
  web_search: "检索公开资料",
  read_file: "读取研究材料",
  read_pdf: "精读研报正文",
  summarize_document: "生成文档摘要",
  write_file: "保存研究文件",
  research_plan: "制定研究计划",
  update_plan_step: "更新计划进度",
  record_conclusion: "记录阶段结论",
  search_tools: "查找研究能力",
  spawn_agent: "分派独立研究任务",
  ask_user: "确认研究要求",
  remember_preference: "记住用户偏好",
};

const PLAN_STATUS: Record<PlanStepStatus, PlanStatusMeta> = {
  pending: { label: "待处理", tone: "pending", mark: "" },
  done: { label: "已完成", tone: "done", mark: "✓" },
  fail: { label: "未完成", tone: "fail", mark: "!" },
  skipped: { label: "已跳过", tone: "skipped", mark: "—" },
};

export type LiveTense = "running" | "done";

export type ToolSummary = {
  industry?: string;
  symbol?: string;
  query?: string;
  view?: string;
  period?: string;
  top?: number;
  years?: number;
  file?: string;
  tasks?: string[];
};

const SKILL_SCENES: Record<string, string> = {
  "industry-research": "行业研究",
  "equity-research": "个股研究",
  "macro-research": "宏观研究",
  "quant-factor": "量化因子",
};

export function presentToolAction(name: string): string {
  return TOOL_ACTIONS[name] ?? "执行研究步骤";
}

function uniqueStrings(values: Array<string | undefined>): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    if (!value || seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
}

function listedTargets(values: Array<string | undefined>): { text: string; total: number } {
  const items = uniqueStrings(values);
  return { text: items.slice(0, 3).join("、"), total: items.length };
}

function withTense(body: string, tense: LiveTense): string {
  return tense === "running" ? `正在${body}` : `已${body}`;
}

function asSummaryList(summary: ToolSummary | ToolSummary[] | undefined): ToolSummary[] {
  if (!summary) return [];
  return Array.isArray(summary) ? summary : [summary];
}

export function asToolSummary(value: unknown): ToolSummary | undefined {
  if (!value || typeof value !== "object") return undefined;
  const data = value as Record<string, unknown>;
  const out: ToolSummary = {};
  if (typeof data.industry === "string" && data.industry) out.industry = data.industry;
  if (typeof data.symbol === "string" && data.symbol) out.symbol = data.symbol;
  if (typeof data.query === "string" && data.query) out.query = data.query;
  if (typeof data.view === "string" && data.view) out.view = data.view;
  if (typeof data.period === "string" && data.period) out.period = data.period;
  if (typeof data.file === "string" && data.file) out.file = data.file;
  if (data.top !== undefined && data.top !== null && `${data.top}` !== "") {
    const top = Number(data.top);
    if (Number.isFinite(top)) out.top = top;
  }
  if (data.years !== undefined && data.years !== null && `${data.years}` !== "") {
    const years = Number(data.years);
    if (Number.isFinite(years)) out.years = years;
  }
  if (Array.isArray(data.tasks)) {
    const tasks = data.tasks.map((item) => String(item ?? "").trim()).filter(Boolean);
    if (tasks.length) out.tasks = tasks;
  }
  return Object.keys(out).length ? out : undefined;
}

export function presentLiveAction(
  name: string,
  summary: ToolSummary | ToolSummary[] | undefined,
  tense: LiveTense,
): string {
  const items = asSummaryList(summary);
  const industries = listedTargets(items.map((item) => item.industry));
  const symbols = listedTargets(items.map((item) => item.symbol));
  const queries = listedTargets(items.map((item) => item.query));
  const files = listedTargets(items.map((item) => item.file));

  if (name === "get_industry_perf") {
    if (industries.total === 0) {
      if (items.some((item) => item.view === "ranking")) return withTense("读取行业涨跌幅排行", tense);
      return withTense("读取行业表现", tense);
    }
    if (industries.total > 3) return withTense(`读取${industries.text}等 ${industries.total} 个行业表现`, tense);
    return withTense(`读取${industries.text}行业表现`, tense);
  }
  if (name === "get_industry_constituents") {
    if (industries.total === 0) return withTense("读取行业成分股", tense);
    if (industries.total > 3) return withTense(`读取${industries.text}等 ${industries.total} 个行业成分股`, tense);
    return withTense(`读取${industries.text}行业成分股`, tense);
  }
  if (name === "get_quote") {
    if (symbols.total === 0) return withTense("读取最新行情", tense);
    if (symbols.total > 3) return withTense(`读取 ${symbols.text} 等 ${symbols.total} 只最新行情`, tense);
    return withTense(`读取 ${symbols.text} 最新行情`, tense);
  }
  if (name === "get_kline") {
    return symbols.text ? withTense(`读取 ${symbols.text} 历史走势`, tense) : withTense("读取历史走势", tense);
  }
  if (name === "get_financials") {
    return symbols.text ? withTense(`读取 ${symbols.text} 财务报表`, tense) : withTense("读取财务报表", tense);
  }
  if (name === "get_indicators") {
    return symbols.text ? withTense(`读取 ${symbols.text} 财务指标`, tense) : withTense("读取财务指标", tense);
  }
  if (name === "get_valuation") {
    return symbols.text ? withTense(`获取 ${symbols.text} 估值数据`, tense) : withTense("获取估值数据", tense);
  }
  if (name === "get_announcements") {
    return symbols.text ? withTense(`查询 ${symbols.text} 公司公告`, tense) : withTense("查询公司公告", tense);
  }
  if (name === "web_search" || name === "get_market_news") {
    if (!queries.text) return withTense(presentToolAction(name), tense);
    return withTense(`检索「${queries.text}」`, tense);
  }
  if (name === "spawn_agent") {
    const task = items[0]?.tasks?.[0];
    if (task) return tense === "running" ? `正在研究：${task}` : `已完成研究：${task}`;
    return withTense("分派独立研究任务", tense);
  }
  if ((name === "read_file" || name === "read_pdf" || name === "write_file") && files.text) {
    return withTense(`${presentToolAction(name)} ${files.text}`, tense);
  }
  return withTense(presentToolAction(name), tense);
}

export function presentSkillAction(skills: string[], tense: LiveTense): string {
  const scenes = uniqueStrings(
    skills.map((id) => {
      const root = id.split("/")[0] ?? id;
      return SKILL_SCENES[root] ?? root;
    }),
  );
  const text = scenes.slice(0, 3).join("、");
  return withTense(text ? `加载${text}技能` : "加载研究技能", tense);
}

/**
 * 把长时间工具的中间进展事件渲染成一句可读文案。
 *
 * 这些事件的存在意义是让界面在长任务期间保持"在推进"的可信信号：服务端心跳是
 * SSE 注释帧、不会重置空闲看门狗，因此没有这类真实事件时，静默数分钟的会话会被
 * 误判为卡死。无法识别形态时返回 null，交由调用方退回默认文案。
 */
export function presentToolProgress(data: Record<string, unknown>): string | null {
  const phase = String(data.phase ?? "");
  const fetched = Number(data.fetched ?? 0);
  const total = Number(data.total ?? 0);
  if (phase === "pool") {
    const resolved = Number(data.resolved ?? total);
    if (data.capped === true) {
      return `解析到 ${resolved} 只成分，受上限限制纳入 ${total} 只`;
    }
    return `股票池 ${total} 只`;
  }
  if (phase === "panel" && total > 0) {
    const parts = [`已取数 ${fetched}/${total} 只`];
    if (data.eta_s !== null && data.eta_s !== undefined) {
      parts.push(`预计剩余 ${Math.max(0, Math.round(Number(data.eta_s)))} 秒`);
    }
    return parts.join(" · ");
  }
  if (phase === "spawn") {
    const task = String(data.task ?? "").trim();
    const status = String(data.status ?? "");
    if (status === "failed") return task ? `子任务未能完成：${task}` : "子任务未能完成";
    if (status === "completed") return task ? `已完成研究：${task}` : "已完成独立研究任务";
    return task ? `正在研究：${task}` : "正在派发独立研究任务";
  }
  return null;
}

export function planStatusMeta(status: string): PlanStatusMeta {
  return PLAN_STATUS[status as PlanStepStatus] ?? PLAN_STATUS.pending;
}

export function planFromEvent(data: Record<string, unknown>): ResearchPlan | null {
  if (!Array.isArray(data.steps) || typeof data.plan_id !== "string" || typeof data.goal !== "string") {
    return null;
  }
  return {
    plan_id: data.plan_id,
    goal: data.goal,
    revision: Number(data.revision ?? 1),
    done: Number(data.done ?? 0),
    total: Number(data.total ?? data.steps.length),
    steps: data.steps
      .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
      .map((item) => ({
        seq: Number(item.seq ?? 0),
        action: String(item.action ?? "研究步骤"),
        status: String(item.status ?? "pending") as PlanStepStatus,
        dep: Array.isArray(item.dep) ? item.dep.map(Number) : [],
      })),
  };
}

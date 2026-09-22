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

export function presentToolAction(name: string): string {
  return TOOL_ACTIONS[name] ?? "执行研究步骤";
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

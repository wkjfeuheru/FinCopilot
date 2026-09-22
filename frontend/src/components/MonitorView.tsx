import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Button, Card, Drawer, Empty, Select, Statistic, Table, Tabs, Tag, Tooltip } from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  fetchTraceMetrics,
  fetchTraceRun,
  fetchTraceRuns,
  fetchTraceStatus,
} from "../api/trace";
import type { TraceMetrics, TraceRun, TraceRunDetail, TraceFilters, TraceStatus } from "../api/trace";
import { AgentTrace } from "./AgentTrace";
import type { AgentStep, TurnTrace } from "./AgentTrace";
import {
  formatAverage,
  formatCount,
  formatDuration,
  formatRate,
  isDriftEvent,
  presentEventKind,
  presentReason,
  presentStatus,
} from "../lib/monitorPresentation";

/** trace 详情 → AgentTrace 可渲染的 TurnTrace（复用工作台时间线样式）。
 * 导出以便单测：跑偏点高亮与逐轮顺序是监控页的核心可读性。 */
export function detailToTrace(detail: TraceRunDetail): TurnTrace {
  const steps: AgentStep[] = [];
  for (const round of detail.rounds_trace ?? []) {
    if (round.thought) {
      steps.push({
        key: `thought-${round.turn}`,
        kind: "analysis",
        label: `第 ${round.turn} 轮 思考`,
        status: "info",
        detail: round.thought,
      });
    }
    for (const action of round.actions ?? []) {
      steps.push({
        key: `action-${action.call_id}`,
        kind: "tool",
        label: action.name,
        status: "done",
        detail: action.args,
      });
    }
    for (const obs of round.observations ?? []) {
      steps.push({
        key: `obs-${obs.call_id}`,
        kind: "tool",
        label: `${obs.name} 返回`,
        status: obs.ok ? "done" : "error",
        detail: obs.error ?? obs.preview,
        durationMs: obs.duration_ms,
      });
    }
  }
  // 事件里的跑偏点：单独高亮，因为"哪一步开始跑偏"是监控的核心问题。
  for (const event of detail.events ?? []) {
    if (isDriftEvent(event.kind, event.payload)) {
      steps.push({
        key: `event-${event.seq}`,
        kind: "system",
        label: `第 ${event.turn ?? "?"} 轮 跑偏点：${presentEventKind(event.kind)}`,
        status: "error",
        detail: JSON.stringify(event.payload ?? {}),
      });
    }
  }
  const status = detail.status === "done" ? "done" : detail.status === "stopped" ? "stopped" : detail.status === "running" ? "running" : "error";
  return {
    planned: Boolean(detail.plan),
    status,
    steps,
    metrics: {
      totalTokens: (detail.input_tokens ?? 0) + (detail.output_tokens ?? 0),
      inputTokens: detail.input_tokens ?? 0,
      outputTokens: detail.output_tokens ?? 0,
      steps: detail.rounds ?? 0,
      firstTokenMs: null,
      totalDurationMs: detail.duration_ms ?? 0,
      toolDurationMs: 0,
    },
  };
}

function MetricCard({ title, value, hint }: { title: string; value: string; hint?: string }) {
  return (
    <Card size="small" className="monitor-metric-card">
      <Statistic title={title} value={value} valueStyle={{ fontSize: 22 }} />
      {hint ? <p className="monitor-metric-hint">{hint}</p> : null}
    </Card>
  );
}

function MetricsPanel({ filters }: { filters: TraceFilters }) {
  const [metrics, setMetrics] = useState<TraceMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setMetrics(null);
    setError(null);
    void fetchTraceMetrics(filters)
      .then((data) => {
        if (!cancelled) setMetrics(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [filters.source, filters.since]);

  if (error) return <Empty description={error} />;
  if (!metrics) return <Empty description="加载中…" />;

  const reasonRows = Object.entries(metrics.completion.reason_counts).sort((a, b) => b[1] - a[1]);
  const toolRows = Object.entries(metrics.per_tool)
    .map(([name, value]) => ({ name, ...value }))
    .sort((a, b) => b.failed + b.blocked - (a.failed + a.blocked));

  return (
    <div className="monitor-metrics">
      <div className="monitor-metric-grid">
        <MetricCard
          title="任务完成率"
          value={formatRate(metrics.completion.task_completion_rate)}
          hint={`共 ${metrics.total_runs} 次运行`}
        />
        <MetricCard title="工具调用次数" value={formatCount(metrics.tool_calls_total)} />
        <MetricCard title="平均执行步数" value={formatAverage(metrics.avg_rounds)} hint="单位：轮" />
        <MetricCard title="工具失败率" value={formatRate(metrics.tool_failure_rate)} />
        <MetricCard
          title="重复调用率"
          value={formatRate(metrics.repeat_call_rate)}
          hint={`${metrics.loop_guard_events} 次拦截`}
        />
        <MetricCard title="安全拦截次数" value={formatCount(metrics.safety_blocks)} />
        <MetricCard title="超时率" value={formatRate(metrics.timeout_rate)} />
      </div>

      <div className="monitor-metric-columns">
        <Card size="small" title="终态与停止原因">
          {reasonRows.length === 0 ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="无数据" />
          ) : (
            <ul className="monitor-bars">
              {reasonRows.map(([reason, count]) => {
                const pct = metrics.total_runs ? (count / metrics.total_runs) * 100 : 0;
                return (
                  <li key={reason}>
                    <span className="monitor-bar-label">{presentReason(reason)}</span>
                    <span className="monitor-bar-track">
                      <span className="monitor-bar-fill" style={{ width: `${pct}%` }} />
                    </span>
                    <span className="monitor-bar-value">{count}</span>
                  </li>
                );
              })}
            </ul>
          )}
        </Card>
        <Card size="small" title="每工具失败与拦截">
          {toolRows.length === 0 ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="无数据" />
          ) : (
            <Table
              size="small"
              rowKey="name"
              pagination={false}
              dataSource={toolRows}
              columns={[
                { title: "工具", dataIndex: "name" },
                { title: "调用", dataIndex: "total", width: 70 },
                { title: "失败", dataIndex: "failed", width: 70 },
                { title: "拦截", dataIndex: "blocked", width: 70 },
                {
                  title: "失败率",
                  width: 90,
                  render: (_, row) => formatRate(row.failure_rate),
                },
              ]}
            />
          )}
        </Card>
      </div>
    </div>
  );
}

function RunsPanel({ filters }: { filters: TraceFilters }) {
  const [runs, setRuns] = useState<TraceRun[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [detail, setDetail] = useState<TraceRunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    void fetchTraceRuns(filters)
      .then((data) => {
        setRuns(data.runs);
        setTotal(data.total);
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [filters.source, filters.status]);

  useEffect(load, [load]);

  const openDetail = (runId: string) => {
    void fetchTraceRun(runId).then(setDetail).catch(() => setDetail(null));
  };

  const columns: ColumnsType<TraceRun> = [
    {
      title: "输入",
      dataIndex: "input",
      ellipsis: true,
      render: (text: string) => <span title={text}>{text.slice(0, 42) || "—"}</span>,
    },
    {
      title: "来源",
      dataIndex: "source",
      width: 80,
      render: (value: string) => <Tag>{value}</Tag>,
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 100,
      render: (_, row) => {
        const status = presentStatus(row.status);
        return <Tag color={status.tone === "ok" ? "green" : status.tone === "error" ? "red" : status.tone === "warn" ? "orange" : "default"}>{status.label}</Tag>;
      },
    },
    {
      title: "原因",
      dataIndex: "reason",
      width: 120,
      render: (value: string | null) => (
        <Tooltip title={value ?? ""}>{presentReason(value)}</Tooltip>
      ),
    },
    { title: "轮数", dataIndex: "rounds", width: 70 },
    { title: "工具", dataIndex: "tool_calls", width: 70 },
    {
      title: "耗时",
      dataIndex: "duration_ms",
      width: 100,
      render: (value: number | null) => formatDuration(value),
    },
  ];

  return (
    <div>
      <div className="monitor-toolbar">
        <span>{total} 条运行</span>
        <Button size="small" onClick={load} loading={loading}>
          刷新
        </Button>
      </div>
      {error ? (
        <Empty description={error} />
      ) : (
        <Table
          size="small"
          rowKey="run_id"
          loading={loading}
          dataSource={runs}
          columns={columns}
          pagination={{ pageSize: 20, total, showSizeChanger: false }}
          onRow={(row) => ({ onClick: () => openDetail(row.run_id), style: { cursor: "pointer" } })}
        />
      )}
      <Drawer
        open={detail !== null}
        width={720}
        onClose={() => setDetail(null)}
        title={detail ? `运行 ${detail.run_id}` : ""}
      >
        {detail ? (
          <div className="monitor-trace-detail">
            <p className="monitor-trace-input">
              <strong>输入：</strong>
              {detail.input || "—"}
            </p>
            <p className="monitor-trace-meta">
              <Tag>{presentStatus(detail.status).label}</Tag>
              <span>交付原因：{presentReason(detail.reason)}</span>
              <span>轮数：{detail.rounds ?? "—"}</span>
              <span>工具调用：{detail.tool_calls ?? "—"}</span>
              <span>耗时：{formatDuration(detail.duration_ms)}</span>
            </p>
            <AgentTrace trace={detailToTrace(detail)} />
            {detail.answer ? (
              <div className="monitor-trace-answer">
                <strong>交付内容</strong>
                <p>{detail.answer}</p>
              </div>
            ) : null}
          </div>
        ) : null}
      </Drawer>
    </div>
  );
}

/** 监控不可用时的引导：说清为什么、怎么开启，而不是一个"加载失败"。 */
function MonitorNotice({ title, lines }: { title: string; lines: string[] }) {
  return (
    <div className="monitor-notice">
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description={<span className="monitor-notice-title">{title}</span>}
      />
      <ul className="monitor-notice-lines">
        {lines.filter(Boolean).map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
    </div>
  );
}

export function MonitorView() {
  const [tab, setTab] = useState<"metrics" | "runs">("metrics");
  const [source, setSource] = useState<string>("");
  const [status, setStatus] = useState<TraceStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetchTraceStatus()
      .then((data) => {
        if (!cancelled) setStatus(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setStatusError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filters: TraceFilters = source ? { source } : {};

  const shell = (body: ReactNode) => (
    <div className="monitor-view">
      <div className="monitor-header">
        <div>
          <h2>运行监控</h2>
          <p>完整 trace 与七项指标：输入、每步思考、工具往返、跑偏点与停止原因</p>
        </div>
      </div>
      {body}
    </div>
  );

  if (statusError) return shell(<MonitorNotice title="无法加载监控状态" lines={[statusError]} />);
  if (!status) return shell(<Empty description="加载中…" />);

  if (!status.enabled) {
    return shell(
      <MonitorNotice
        title="运行监控未启用"
        lines={[
          "监控默认关闭；启用后即可记录完整 trace 并展示七项指标。",
          "在 settings.json 中设置 observability.trace_store.enabled = true，",
          "并把需要查看监控的用户名加入 observability.trace_store.admin_users，然后重启服务。",
          "也可用环境变量 FINH_OBSERVABILITY_TRACE_STORE_ENABLED=true 启用。",
        ]}
      />,
    );
  }

  if (!status.is_admin) {
    return shell(
      <MonitorNotice
        title="无监控访问权限"
        lines={[
          "当前用户不在监控白名单中。",
          status.admin_configured
            ? "请让运维方把你的用户名加入 observability.trace_store.admin_users。"
            : "当前白名单为空——即使监控已启用，也没有任何用户能看到，请在 settings.json 中补上 admin_users。",
        ]}
      />,
    );
  }

  return shell(
    <>
      <div className="monitor-toolbar">
        <span />
        <Select
          size="small"
          value={source}
          onChange={setSource}
          style={{ width: 140 }}
          options={[
            { value: "", label: "全部来源" },
            { value: "server", label: "线上对话" },
            { value: "eval", label: "评测运行" },
          ]}
        />
      </div>
      <Tabs
        activeKey={tab}
        onChange={(key) => setTab(key as "metrics" | "runs")}
        items={[
          { key: "metrics", label: "指标面板", children: <MetricsPanel filters={filters} /> },
          { key: "runs", label: "Trace 浏览器", children: <RunsPanel filters={filters} /> },
        ]}
      />
    </>,
  );
}

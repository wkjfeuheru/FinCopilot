import { useCallback, useEffect, useState } from "react";
import { Button, Card, Empty, Select, Statistic, Table, Tabs, Tag } from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  fetchAdminSummary,
  fetchAdminUsers,
} from "../api/admin";
import type { AdminUserRow, AdminUsageSummary, AdminWindow } from "../api/admin";
import { MetricsPanel, RunsPanel } from "./MonitorView";

/** 数字千分位；token 量级常用，空值显示 0。 */
function fmtTokens(value: number | undefined): string {
  return (value ?? 0).toLocaleString("zh-CN");
}

const WINDOW_OPTIONS: { value: AdminWindow; label: string }[] = [
  { value: "all", label: "全部时间" },
  { value: "24h", label: "近 24 小时" },
  { value: "7d", label: "近 7 天" },
  { value: "30d", label: "近 30 天" },
];

function SummaryCards({ window: win }: { window: AdminWindow }) {
  const [summary, setSummary] = useState<AdminUsageSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetchAdminSummary(win)
      .then((data) => {
        if (!cancelled) setSummary(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [win]);

  if (error) return <Empty description={error} />;
  if (!summary) return <Empty description="加载中…" />;

  return (
    <div className="monitor-metric-grid">
      <Card size="small">
        <Statistic title="注册用户" value={fmtTokens(summary.total_users)} valueStyle={{ fontSize: 22 }} />
      </Card>
      <Card size="small">
        <Statistic title="窗口活跃用户" value={fmtTokens(summary.active_users)} valueStyle={{ fontSize: 22 }} />
      </Card>
      <Card size="small">
        <Statistic title="窗口对话轮数" value={fmtTokens(summary.turns)} valueStyle={{ fontSize: 22 }} />
      </Card>
      <Card size="small">
        <Statistic
          title="窗口输入 token"
          value={fmtTokens(summary.input_tokens)}
          valueStyle={{ fontSize: 22 }}
        />
      </Card>
      <Card size="small">
        <Statistic
          title="窗口输出 token"
          value={fmtTokens(summary.output_tokens)}
          valueStyle={{ fontSize: 22 }}
        />
      </Card>
      <Card size="small">
        <Statistic
          title="窗口缓存命中 token"
          value={fmtTokens(summary.cache_hit_tokens)}
          valueStyle={{ fontSize: 22 }}
        />
      </Card>
    </div>
  );
}

function UsersPanel({ window: win }: { window: AdminWindow }) {
  const [users, setUsers] = useState<AdminUserRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    void fetchAdminUsers(win)
      .then((data) => setUsers(data.users))
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [win]);

  useEffect(load, [load]);

  const columns: ColumnsType<AdminUserRow> = [
    {
      title: "用户名",
      dataIndex: "username",
      width: 140,
      render: (value: string, row) => (
        <span>
          {value}
          {row.role === "admin" ? (
            <Tag color="gold" style={{ marginInlineStart: 8 }}>
              管理员
            </Tag>
          ) : null}
        </span>
      ),
    },
    { title: "注册时间", dataIndex: "created_at", width: 170, render: (v: string) => v.slice(0, 19).replace("T", " ") },
    { title: "对话数", dataIndex: "conversations", width: 90 },
    { title: `总轮数`, dataIndex: "turns", width: 90 },
    {
      title: "总输入 token",
      dataIndex: "input_tokens",
      width: 120,
      render: fmtTokens,
    },
    {
      title: "总输出 token",
      dataIndex: "output_tokens",
      width: 120,
      render: fmtTokens,
    },
    {
      title: "缓存命中",
      dataIndex: "cache_hit_tokens",
      width: 110,
      render: fmtTokens,
    },
    { title: "窗口轮数", dataIndex: "window_turns", width: 100 },
    {
      title: "窗口输入",
      dataIndex: "window_input_tokens",
      width: 110,
      render: fmtTokens,
    },
    {
      title: "窗口输出",
      dataIndex: "window_output_tokens",
      width: 110,
      render: fmtTokens,
    },
    {
      title: "最后活跃",
      dataIndex: "last_active_at",
      width: 170,
      render: (v: string) => (v ? v.slice(0, 19).replace("T", " ") : "—"),
    },
  ];

  return (
    <div>
      <div className="monitor-toolbar">
        <span>{users.length} 个账号</span>
        <Button size="small" onClick={load} loading={loading}>
          刷新
        </Button>
      </div>
      {error ? (
        <Empty description={error} />
      ) : (
        <Table
          size="small"
          rowKey="id"
          loading={loading}
          dataSource={users}
          columns={columns}
          pagination={users.length > 20 ? { pageSize: 20, showSizeChanger: false } : false}
          scroll={{ x: 1100 }}
        />
      )}
    </div>
  );
}

/** 统一管理页：用户总览 + 用量汇总 + 运行监控（原监控页并入为标签）。
 * 仅管理员可见：后端 /v1/admin/* 有 require_admin 兜底，此处按 role 隐藏入口。 */
export function AdminView() {
  const [window, setWindow] = useState<AdminWindow>("all");

  return (
    <div className="monitor-view">
      <div className="monitor-header">
        <div>
          <h2>管理</h2>
          <p>用户总览、token 用量与运行监控——仅管理员可见</p>
        </div>
      </div>
      <Tabs
        defaultActiveKey="users"
        items={[
          {
            key: "users",
            label: "用户总览",
            children: (
              <div>
                <div className="monitor-toolbar">
                  <span />
                  <Select
                    size="small"
                    value={window}
                    onChange={setWindow}
                    style={{ width: 140 }}
                    options={WINDOW_OPTIONS}
                  />
                </div>
                <SummaryCards window={window} />
                <div style={{ marginTop: 16 }}>
                  <UsersPanel window={window} />
                </div>
              </div>
            ),
          },
          {
            key: "monitor",
            label: "运行监控",
            children: <MonitorTab />,
          },
        ]}
      />
    </div>
  );
}

function MonitorTab() {
  const [source, setSource] = useState<string>("");
  const filters = source ? { source } : {};
  return (
    <div>
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
        defaultActiveKey="metrics"
        items={[
          { key: "metrics", label: "指标面板", children: <MetricsPanel filters={filters} /> },
          { key: "runs", label: "Trace 浏览器", children: <RunsPanel filters={filters} /> },
        ]}
      />
    </div>
  );
}

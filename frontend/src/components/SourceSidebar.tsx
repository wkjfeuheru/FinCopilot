import { useEffect, useMemo, useState } from "react";
import { Empty, Tag } from "antd";
import type { Citation } from "../api/client";
import { artifactUrl } from "../api/client";
import { presentToolAction } from "../lib/researchPresentation";
import { groupActivitiesByTool } from "../lib/liveStatus";

/** 智能体活动流中的一条实时记录。 */
export type Activity = {
  key: string;
  tool?: string;
  label: string;
  status: "running" | "done" | "error" | "info";
  detail?: string;
  attachments?: string[];
};

function statusMark(status: Activity["status"]): string {
  if (status === "running") return "…";
  if (status === "done") return "✓";
  if (status === "error") return "✕";
  return "•";
}

function fileName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function formatTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function ActivityChildren({ items }: { items: Activity[] }) {
  return (
    <ol className="activity-children">
      {items.map((item) => (
        <li key={item.key} className={`activity-item activity-${item.status}`}>
          <span className="activity-mark">{statusMark(item.status)}</span>
          <div className="activity-body">
            <span className="activity-label">{item.label}</span>
            {item.detail && <span className="activity-detail">{item.detail}</span>}
            {item.attachments && item.attachments.length > 0 && (
              <span className="activity-files">
                {item.attachments.map((path) => (
                  <a key={path} href={artifactUrl(path)} download>
                    {fileName(path)}
                  </a>
                ))}
              </span>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}

function ActivityGroupItem({
  tool,
  title,
  status,
  items,
  forcedOpen,
}: {
  tool: string;
  title: string;
  status: Activity["status"];
  items: Activity[];
  forcedOpen: boolean;
}) {
  const [open, setOpen] = useState(status === "running");
  useEffect(() => {
    if (forcedOpen) setOpen(true);
  }, [forcedOpen]);
  return (
    <li className={`activity-group activity-${status}`} id={`activity-group-${tool}`}>
      <details open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
        <summary className="activity-group-summary">
          <span className="activity-mark">{statusMark(status)}</span>
          <span className="activity-label">{title}</span>
        </summary>
        <ActivityChildren items={items} />
      </details>
    </li>
  );
}

export function SourceSidebar({
  activities,
  citations,
  focusedTool,
}: {
  activities: Activity[];
  citations: Citation[];
  focusedTool?: string | null;
}) {
  const groups = useMemo(
    () =>
      groupActivitiesByTool(
        activities.map((item) => ({
          ...item,
          tool: item.tool ?? "tool",
        })),
      ),
    [activities],
  );

  useEffect(() => {
    if (!focusedTool) return;
    document.getElementById(`activity-group-${focusedTool}`)?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }, [focusedTool]);

  return (
    <aside className="source-sidebar">
      <section className="sidebar-section" id="execution-activity">
        <span className="sidebar-title">执行动态</span>
        {groups.length === 0 ? (
          <p className="sidebar-empty">暂无执行记录</p>
        ) : (
          <ol className="activity-list">
            {groups.map((group) => (
              <ActivityGroupItem
                key={group.tool}
                tool={group.tool}
                title={group.title}
                status={group.status}
                forcedOpen={focusedTool === group.tool}
                items={group.children}
              />
            ))}
          </ol>
        )}
      </section>

      <section className="sidebar-section">
        <span className="sidebar-title">数据来源</span>
        {citations.length === 0 ? (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无数据来源" />
        ) : (
          <div className="source-list">
            {citations.map((item, index) => (
              <article className="source-card" id={`cite-${item.cid}`} key={item.cid}>
                <div className="source-card-head">
                  <span className="source-index">[{index + 1}]</span>
                  <span className="source-tool">{presentToolAction(item.tool)}</span>
                  <Tag color={item.from_cache ? "default" : "green"}>
                    {item.from_cache ? "缓存" : "实时"}
                  </Tag>
                </div>
                {item.symbol && <span className="source-symbol">标的 {item.symbol}</span>}
                <span className="source-endpoint">{item.endpoint}</span>
                <span className="source-meta">
                  {item.rows} 行 × {item.cols} 列
                </span>
                {Object.keys(item.params).length > 0 && (
                  <span className="source-params">
                    {Object.entries(item.params)
                      .map(([key, value]) => `${key}=${String(value)}`)
                      .join(" · ")}
                  </span>
                )}
                <span className="source-meta">{formatTime(item.ts)}</span>
                <span className="source-fingerprint" title={item.fingerprint}>
                  {item.cid} · {item.fingerprint.slice(0, 12)}
                </span>
              </article>
            ))}
          </div>
        )}
      </section>
    </aside>
  );
}

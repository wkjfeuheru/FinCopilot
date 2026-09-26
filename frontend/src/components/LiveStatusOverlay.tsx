import { ResearchPlanLedger } from "./ResearchPlanLedger";
import type { TurnTrace } from "./AgentTrace";
import { reduceLiveRows } from "../lib/liveStatus";
import type { LiveRow } from "../lib/liveStatus";

function rowKind(kind: LiveRow["kind"]): string {
  if (kind === "skill") return "技能";
  if (kind === "agent") return "子代理";
  if (kind === "analysis") return "分析";
  if (kind === "final") return "状态";
  return "工具";
}

export function LiveStatusOverlay({
  trace,
  onOpenProcess,
}: {
  trace: TurnTrace;
  onOpenProcess?: (toolName?: string) => void;
}) {
  const rows = reduceLiveRows(trace.steps, trace.status);
  return (
    <section className={`live-status live-status-${trace.status}`} aria-label="当前研究状态">
      {trace.plan && <ResearchPlanLedger plan={trace.plan} />}
      <ol className="live-status-rows">
        {rows.map((row) => (
          <li key={row.key}>
            <button
              type="button"
              className={`live-status-row live-status-row-${row.status}`}
              onClick={() => onOpenProcess?.(row.toolName)}
            >
              <span className={`live-status-pulse live-status-pulse-${row.status}`} aria-hidden="true" />
              <span className="live-status-kind">{rowKind(row.kind)}</span>
              <span className="live-status-body">
                <strong>{row.label}</strong>
                {row.detail && <small>{row.detail}</small>}
              </span>
              <span className="live-status-hint">完整过程</span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}

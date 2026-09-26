import { presentLiveAction, presentToolAction, type ToolSummary } from "./researchPresentation";

export const MAX_LIVE_ROWS = 3;

export type SpawnTaskProgress = {
  index: number;
  task: string;
  status: "running" | "done" | "error";
};

export type LiveStep = {
  key: string;
  kind: "analysis" | "plan" | "skill" | "tool" | "agent" | "final" | "system";
  label: string;
  status: "running" | "done" | "error" | "info" | "stopped";
  toolName?: string;
  summary?: ToolSummary;
  spawnTasks?: SpawnTaskProgress[];
  detail?: string;
};

export type LiveRow = {
  key: string;
  kind: "skill" | "tool" | "agent" | "analysis" | "final";
  label: string;
  status: LiveStep["status"];
  toolName?: string;
  detail?: string;
};

export type GroupableActivity = {
  key: string;
  tool: string;
  label: string;
  status: "running" | "done" | "error" | "info";
  detail?: string;
  attachments?: string[];
};

export type ActivityGroup = {
  tool: string;
  title: string;
  count: number;
  status: GroupableActivity["status"];
  children: GroupableActivity[];
};

function callCount(steps: LiveStep[]): number {
  return steps.filter((step) => step.kind === "tool" || step.kind === "skill" || step.kind === "agent").length;
}

function lastIndexOfTool(steps: LiveStep[], toolName: string): number {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index];
    if (step.status !== "running") continue;
    if ((step.toolName ?? step.key) === toolName) return index;
  }
  return -1;
}

function spawnRows(step: LiveStep): LiveRow[] {
  const tasks: SpawnTaskProgress[] = step.spawnTasks?.length
    ? step.spawnTasks
    : (step.summary?.tasks ?? []).map((task, index) => ({ index, task, status: "running" as const }));
  if (!tasks.length) return [];
  const visible = tasks.slice(0, MAX_LIVE_ROWS);
  const extra = tasks.length - visible.length;
  return visible.map((task, offset) => ({
    key: `${step.key}-${task.index}`,
    kind: "agent" as const,
    toolName: "spawn_agent",
    status: task.status,
    label:
      task.status === "error"
        ? `子任务未能完成：${task.task}`
        : task.status === "done"
          ? `已完成研究：${task.task}`
          : `正在研究：${task.task}`,
    detail: offset === visible.length - 1 && extra > 0 ? `另有 ${extra} 个` : undefined,
  }));
}

export function reduceLiveRows(
  steps: LiveStep[],
  traceStatus: "running" | "done" | "error" | "stopped",
): LiveRow[] {
  if (traceStatus !== "running") {
    const n = callCount(steps);
    const label =
      traceStatus === "stopped"
        ? `已停止 · ${n} 项调用`
        : traceStatus === "error"
          ? `研究未完成 · ${n} 项调用`
          : `研究完成 · ${n} 项调用`;
    return [
      {
        key: "summary",
        kind: "final",
        label,
        status: traceStatus === "error" ? "error" : traceStatus === "stopped" ? "stopped" : "done",
      },
    ];
  }

  const spawn = [...steps]
    .reverse()
    .find((step) => step.toolName === "spawn_agent" && step.status === "running");
  if (spawn) {
    const rows = spawnRows(spawn);
    if (rows.length) return rows;
  }

  const running = steps.filter((step) => step.status === "running");
  const toolRuns = running.filter((step) => step.kind === "tool" || step.kind === "plan");
  const grouped = new Map<string, LiveStep[]>();
  for (const step of toolRuns) {
    const name = step.toolName ?? "tool";
    const list = grouped.get(name) ?? [];
    list.push(step);
    grouped.set(name, list);
  }
  const ordered = [...grouped.keys()].sort(
    (left, right) => lastIndexOfTool(steps, left) - lastIndexOfTool(steps, right),
  );
  const toolRows: LiveRow[] = ordered.map((toolName) => {
    const group = grouped.get(toolName) ?? [];
    return {
      key: `tool-${toolName}`,
      kind: "tool",
      toolName,
      status: "running",
      label: presentLiveAction(
        toolName,
        group.map((item) => item.summary ?? {}),
        "running",
      ),
    };
  });
  const cappedTools = toolRows.slice(-MAX_LIVE_ROWS);

  const skill = running.find((step) => step.kind === "skill");
  const rows: LiveRow[] = [];
  if (skill && cappedTools.length < MAX_LIVE_ROWS) {
    rows.push({
      key: skill.key,
      kind: "skill",
      toolName: "skill",
      status: "running",
      label: skill.label,
    });
  }
  rows.push(...cappedTools);
  if (rows.length) return rows.slice(0, MAX_LIVE_ROWS);

  const analysis = running.find((step) => step.kind === "analysis") ?? running[0];
  if (analysis) {
    return [
      {
        key: analysis.key,
        kind: analysis.kind === "analysis" ? "analysis" : "tool",
        label: analysis.label,
        status: "running",
      },
    ];
  }
  return [{ key: "running", kind: "analysis", label: "研究任务执行中", status: "running" }];
}

function groupTitleBase(tool: string): string {
  if (tool === "skill") return "研究方法";
  if (tool === "compact") return "压缩上下文";
  if (tool === "ask") return "确认研究要求";
  return presentToolAction(tool);
}

export function groupActivitiesByTool(items: GroupableActivity[]): ActivityGroup[] {
  const groups: ActivityGroup[] = [];
  const index = new Map<string, ActivityGroup>();
  for (const item of items) {
    let group = index.get(item.tool);
    if (!group) {
      group = {
        tool: item.tool,
        title: `${groupTitleBase(item.tool)} · 1 次`,
        count: 0,
        status: item.status,
        children: [],
      };
      index.set(item.tool, group);
      groups.push(group);
    }
    group.children.push(item);
    group.count = group.children.length;
    group.title = `${groupTitleBase(item.tool)} · ${group.count} 次`;
    if (item.status === "running") group.status = "running";
    else if (group.status !== "running" && item.status === "error") group.status = "error";
  }
  return groups;
}

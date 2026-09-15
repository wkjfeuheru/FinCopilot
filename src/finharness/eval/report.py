"""写出一次运行的产物：逐用例 JSON、摘要 JSON 与 Markdown 报告。

报告供评审者阅读：四维度并列呈现，给出综合分与门禁结论，随后列出失败用例
及其足以看清原因的轨迹。与同一题集上一次运行的对比让回归问题一目了然。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finharness.eval import aggregate as aggregate_mod


def write_case_files(run_dir: Path, scores: list[aggregate_mod.CaseScore]) -> None:
    """把每个用例的评分结果以 JSON 写入运行目录的 cases/ 下。"""
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for score in scores:
        path = cases_dir / f"{score.case.id}.json"
        path.write_text(
            json.dumps(score.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def write_summary(
    run_dir: Path,
    summary: aggregate_mod.RunSummary,
    manifest: dict[str, Any],
) -> Path:
    """写出包含 manifest 与摘要的 summary.json，并返回其路径。"""
    payload = {"manifest": manifest, "summary": summary.to_dict()}
    path = run_dir / "summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def previous_summary(runs_root: Path, current_run_dir: Path, set_name: str) -> dict[str, Any] | None:
    """同一题集最近一次更早运行的摘要，若存在的话。"""
    if not runs_root.exists():
        return None
    candidates = sorted(
        (p for p in runs_root.iterdir() if p.is_dir() and p != current_run_dir),
        key=lambda p: p.name,
        reverse=True,
    )
    for candidate in candidates:
        summary_path = candidate / "summary.json"
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("manifest", {}).get("set") == set_name:
            return payload
    return None


def render_report(
    summary: aggregate_mod.RunSummary,
    manifest: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> str:
    """渲染 Markdown 报告文本；传入 ``previous`` 时附带与上次运行的对比。"""
    lines: list[str] = []
    lines.append(f"# FinHarness 评估报告 · {manifest.get('set', '')}")
    lines.append("")
    verdict = "✅ 通过" if summary.passed else "❌ 不通过"
    lines.append(f"**结论：{verdict}** ｜ 综合分 {summary.composite:.3f} ｜ "
                 f"通过 {summary.passed_count}/{summary.total}（{summary.pass_rate:.0%}）")
    lines.append("")
    lines.append(
        f"- 模型：`{manifest.get('model')}`（{manifest.get('provider_name')}，"
        f"temperature={manifest.get('temperature')}）"
    )
    lines.append(f"- 时间：{manifest.get('timestamp')} ｜ 离线自检：{manifest.get('offline')}"
                 f" ｜ 代码版本：{manifest.get('code_version')}")
    lines.append("")

    lines.append("## 四维度得分")
    lines.append("")
    lines.append("| 维度 | 权重 | 得分 |")
    lines.append("|---|---|---|")
    weights = _weights_from_manifest(manifest)
    dim_names = {
        "task": "任务完成率",
        "trajectory": "推理路径正确性",
        "efficiency": "效率",
        "safety": "安全性",
    }
    for name, label in dim_names.items():
        value = summary.dimensions.get(name, 0.0)
        lines.append(f"| {label} | {weights.get(name, '')} | {value:.3f} |")
    lines.append("")

    if summary.red_line_failures:
        lines.append("## 🚫 红线未通过")
        lines.append("")
        for case_id in summary.red_line_failures:
            lines.append(f"- {case_id}")
        lines.append("")

    if summary.failed:
        lines.append("## 失败用例")
        lines.append("")
        for score in summary.scores:
            if score.passed:
                continue
            lines.append(f"### {score.case.id} · {score.case.title}")
            lines.append("")
            lines.append(f"- 来源：{score.case.source}")
            lines.append(f"- 轨迹分 {score.dimensions['trajectory'].score:.2f} ｜ "
                         f"安全分 {score.dimensions['safety'].score:.2f} ｜ "
                         f"综合 {score.composite:.3f}")
            for name, dimension in score.dimensions.items():
                for check in dimension.checks:
                    if not check.passed:
                        lines.append(f"  - ❌ `{name}.{check.name}`：{check.detail}")
            for reason in score.reasons:
                lines.append(f"  - ⚠ {reason}")
            trace = score.run
            if trace.called_tools():
                lines.append(f"- 实际调用：{sorted(set(trace.called_tools()))}")
            if trace.refused_tools():
                lines.append(f"- 被拒调用：{sorted(set(trace.refused_tools()))}")
            lines.append("")
    else:
        lines.append("## 失败用例")
        lines.append("")
        lines.append("无。")
        lines.append("")

    if previous is not None:
        lines.append("## 与上次对比")
        lines.append("")
        prev = previous.get("summary", {})
        lines.append(f"- 上次综合分 {prev.get('composite')} → 本次 {summary.composite:.3f}")
        lines.append(f"- 上次通过 {prev.get('passed')}/{prev.get('total')} → "
                     f"本次 {summary.passed_count}/{summary.total}")
        prev_failed = set(prev.get("failed") or [])
        now_failed = set(summary.failed)
        newly_failed = sorted(now_failed - prev_failed)
        fixed = sorted(prev_failed - now_failed)
        if newly_failed:
            lines.append(f"- ⚠ 新失败：{newly_failed}")
        if fixed:
            lines.append(f"- ✅ 已修复：{fixed}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "> 判定看行为模式而非字面文本（README 已声明模型输出有方差）："
        "该规划时是否规划、该拒答时是否拒答、数字是否带 cid、是否重复取数。"
    )
    return "\n".join(lines)


def _weights_from_manifest(manifest: dict[str, Any]) -> dict[str, float]:
    return manifest.get("weights", {})


def write_report(run_dir: Path, text: str) -> Path:
    """把报告文本写入 report.md，并返回其路径。"""
    path = run_dir / "report.md"
    path.write_text(text, encoding="utf-8")
    return path


__all__ = [
    "previous_summary",
    "render_report",
    "write_case_files",
    "write_report",
    "write_summary",
]

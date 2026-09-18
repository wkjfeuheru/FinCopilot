"""评测工具的命令行入口（docs 03.13）。

    python -m finharness.eval list
    python -m finharness.eval check
    python -m finharness.eval run --set smoke
    python -m finharness.eval run --set core --offline
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from finharness.config.settings import Settings, SettingsError
from finharness.eval.aggregate import aggregate, score_case
from finharness.eval.config import ConfigError, EvalConfig, load_config
from finharness.eval.report import (
    previous_summary,
    render_report,
    write_case_files,
    write_report,
    write_summary,
)
from finharness.eval.runner import EvalRunner, build_manifest
from finharness.eval.schema import SchemaError, load_cases_dir

EVALS_DIR = Path("evals")


def _resolve_root(path: str | None) -> Path:
    """解析并校验评测资产根目录。"""
    root = Path(path) if path else EVALS_DIR
    if not root.exists():
        raise SystemExit(f"评估目录不存在：{root}（请在仓库根目录运行，或用 --evals-dir 指定）")
    return root


def _load(args) -> tuple[EvalConfig, list, Path]:
    """加载配置与用例，并按 ``--filter`` 过滤。"""
    root = _resolve_root(args.evals_dir)
    config = load_config(root / "config.yaml")
    cases = load_cases_dir(root / config.cases_dir)
    if args.filter:
        wanted = set(args.filter)
        cases = [
            case
            for case in cases
            if case.category in wanted
            or any(case.id.startswith(prefix) for prefix in wanted)
        ]
    return config, cases, root


def cmd_list(args) -> int:
    config, cases, _ = _load(args)
    spec = config.resolve_set(args.set or "full")
    selected = [case for case in cases if spec.matches(case)]
    print(f"题集 {args.set or 'full'}：{len(selected)}/{len(cases)} 题")
    for case in selected:
        tags = ",".join(case.tags)
        print(f"  {case.id:10s} [{case.category:3s}] {case.title}  ({tags})")
    return 0


def cmd_check(args) -> int:
    _, cases, root = _load(args)
    print(f"✅ {len(cases)} 个用例通过 schema 校验（{root}）")
    for case in cases:
        if case.judge == "todo":
            print(f"  · {case.id} 含 judge:todo（模糊标准，v1 结构近似）")
    return 0


def _build_provider(settings: Settings, args):
    """按 ``--offline`` 构造自检 Provider，否则从配置解析真实模型供应商。"""
    if args.offline:
        from finharness.eval.selfcheck import SelfCheckProvider

        return SelfCheckProvider()
    from finharness.config.crypto import SecretCipher
    from finharness.config.store import ConfigStore
    from finharness.provider.resolver import ProviderResolver

    store = ConfigStore(
        settings.paths.config_db,
        cipher=SecretCipher(settings.paths.secret_key),
    )
    resolver = ProviderResolver(store_factory=lambda: store, settings=settings)
    # eval CLI 是单用户运维工具：以空 user_id 查询（CLI 时代的配置即 user_id=''）。
    status = resolver.status("")
    if not status.configured:
        raise SystemExit(
            "未配置可用的模型供应商：请先在设置中配置，或设置对应的 API Key 环境变量；"
            "仅做管线自检可加 --offline。"
        )
    return resolver.current("")


def cmd_run(args) -> int:
    """执行选中的用例集、聚合打分并写出报告，返回进程退出码。"""
    config, cases, root = _load(args)
    spec = config.resolve_set(args.set or "full")
    selected = [case for case in cases if spec.matches(case)]
    if not selected:
        raise SystemExit(f"题集 {args.set} 未匹配到任何用例")

    try:
        settings = Settings.from_file(args.settings)
    except SettingsError as exc:
        raise SystemExit(str(exc))
    settings = settings.model_copy(
        update={"model": settings.model.model_copy(update={"max_turns": args.max_turns})}
    ) if args.max_turns else settings

    provider = _build_provider(settings, args)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{stamp}_{args.set or 'full'}"
    runs_root = Path(args.runs_dir) if args.runs_dir else root / "runs"
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"▶ 运行题集 {args.set or 'full'}：{len(selected)} 题 → {run_dir}")
    runner = EvalRunner(
        base_settings=settings,
        provider=provider,
        run_dir=run_dir,
        offline=args.offline,
        turn_timeout_s=args.timeout,
        verbose=True,
    )

    async def _drive():
        results = []
        for case in selected:
            results.append(await runner.run_case(case))
        return results

    runs = asyncio.run(_drive())
    scores = [score_case(run.case, run, config) for run in runs]
    summary = aggregate(scores, config)

    manifest = build_manifest(
        settings=settings,
        provider=provider,
        offline=args.offline,
        case_count=len(selected),
        set_name=args.set or "full",
    )
    manifest["weights"] = config.weights.as_dict()
    if args.offline:
        manifest["note"] = "离线自检运行，分数仅验证管线，不代表模型能力"

    write_case_files(run_dir, scores)
    write_summary(run_dir, summary, manifest)
    previous = previous_summary(runs_root, run_dir, args.set or "full")
    report_text = render_report(summary, manifest, previous)
    report_path = write_report(run_dir, report_text)

    print()
    print(f"综合分 {summary.composite:.3f} ｜ 通过 {summary.passed_count}/{summary.total}"
          f" ｜ 门禁 {'通过' if summary.passed else '不通过'}")
    if summary.red_line_failures:
        print(f"红线未通过：{summary.red_line_failures}")
    print(f"报告：{report_path}")
    return 0 if summary.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m finharness.eval",
        description="FinHarness agent 评估体系（任务完成率 / 轨迹 / 效率 / 安全）",
    )
    parser.add_argument("--evals-dir", default=None, help="评估资产目录（默认 evals）")
    parser.add_argument("--settings", default="settings.json", help="settings.json 路径")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出题集用例")
    p_list.add_argument("--set", default=None, help="题集名（smoke/core/full/selfcheck）")
    p_list.add_argument("--filter", nargs="*", default=None, help="按 category 或 id 前缀过滤")
    p_list.set_defaults(func=cmd_list)

    p_check = sub.add_parser("check", help="仅校验用例 YAML")
    p_check.add_argument("--filter", nargs="*", default=None)
    p_check.set_defaults(func=cmd_check)

    p_run = sub.add_parser("run", help="执行评估")
    p_run.add_argument("--set", default="core", help="题集名（默认 core）")
    p_run.add_argument("--filter", nargs="*", default=None)
    p_run.add_argument("--offline", action="store_true", help="用 FakeProvider 跑管线自检")
    p_run.add_argument("--max-turns", type=int, default=None, help="覆盖 context.max_turns")
    p_run.add_argument("--timeout", type=float, default=300.0, help="单轮超时秒数")
    p_run.add_argument("--run-name", default=None, help="本次运行目录名")
    p_run.add_argument("--runs-dir", default=None, help="运行产物根目录（默认 <evals>/runs）")
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (SchemaError, ConfigError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

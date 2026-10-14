"""路线① 统一命令行入口。

    python -m route1.cli selftest                     # 不依赖仿真器的自检（契约/总线/桥）
    python -m route1.cli parity    --run <RUN_DIR>    # 决策层等价性回归
    python -m route1.cli precision --run <RUN_DIR>    # P@1/P@3/P@5
    python -m route1.cli dataset   --run <RUN_DIR> --out <JSON>
    python -m route1.cli ingest-nav --run <RUN_DIR> --out <JSONL>
    python -m route1.cli merge     --input <JSONL> --input <JSONL> --root <BUS_DIR>
    python -m route1.cli summary   --root <BUS_DIR>

Python 3.9 兼容。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _ensure_importable() -> None:
    """把本仓库根目录放到 sys.path，保证 `python -m route1.cli` 与直接执行都能跑。"""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)


def cmd_selftest(args: argparse.Namespace) -> int:
    from .contracts import FailureEvent, Signals, validate_stream
    from .bridge import Bridge, demo_bridge

    root = args.root or os.path.join(os.getcwd(), "route1_selftest_bus")
    print("== 契约自检 ==")
    event = FailureEvent.create(
        source="nav", type="N7", scene_id="S", episode_id=1, step=5, action_step=5,
        signals=Signals(feedback_score=0.42, semantic_instance_margin=0.88,
                        semantic_verdict="LOST", semantic_confidence=0.88),
    )
    assert event.type == "N7"
    round_trip = FailureEvent.from_json(event.to_json())
    assert round_trip.to_dict() == event.to_dict(), "JSON 往返不一致"
    print("  契约往返 OK；schema =", event.schema)

    # 非法组合必须被拒绝
    bad_rejected = 0
    try:
        FailureEvent.create(source="nav", type="N7", recovery=None)
        FailureEvent.from_dict(dict(event.to_dict(), recovery={
            "reason_code": "x", "action": "retry_grasp"})).validate()
    except ValueError:
        bad_rejected += 1
    print("  非法恢复动作被拒绝:", bad_rejected == 1)

    print("== 桥自检 ==")
    summary = demo_bridge(root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    report = Bridge(root).write_report(os.path.join(root, "bridge_report.json"))
    print("  报告:", report)
    return 0


def cmd_parity(args: argparse.Namespace) -> int:
    from .nav.replay import main as replay_main
    argv = ["parity", "--run", args.run, "--samples", str(args.samples)]
    if args.episodes:
        argv += ["--episodes", str(args.episodes)]
    if args.out:
        argv += ["--out", args.out]
    return replay_main(argv)


def cmd_precision(args: argparse.Namespace) -> int:
    from .nav.replay import main as replay_main
    argv = ["precision"]
    for run in args.run:
        argv += ["--run", run]
    for predicate in (args.predicate or ["early_drift"]):
        argv += ["--predicate", predicate]
    if args.out:
        argv += ["--out", args.out]
    return replay_main(argv)


def cmd_dataset(args: argparse.Namespace) -> int:
    from .nav.replay import main as replay_main
    argv = ["dataset", "--run", args.run]
    if args.out:
        argv += ["--out", args.out]
    return replay_main(argv)


def cmd_ingest_nav(args: argparse.Namespace) -> int:
    from .nav.adapter import events_from_run
    run_dirs = []
    for pattern in args.run:
        matches = glob.glob(pattern)
        run_dirs.extend(matches or [pattern])
    events = []
    for run_dir in run_dirs:
        events.extend(events_from_run(run_dir, include_non_acting=args.include_non_acting))
    out = args.out or "nav_failure_events.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(event.to_json() + "\n")
    print("runs=%d events=%d -> %s" % (len(run_dirs), len(events), out))
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    from .bridge import Bridge
    bridge = Bridge(args.root)
    summary = bridge.merge_files(args.input, label=args.label)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("report:", bridge.write_report(os.path.join(args.root, "bridge_report.json")))
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    from .bridge import Bridge
    bridge = Bridge(args.root)
    print(json.dumps(bridge.summary(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="route1", description="路线① 统一入口")
    parser.add_argument("--version", action="version", version="route1 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("selftest", help="不依赖仿真器的自检")
    p.add_argument("--root", default=None)
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("parity", help="决策层等价性回归")
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_parity)

    p = sub.add_parser("precision", help="P@1/P@3/P@5")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--predicate", action="append", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_precision)

    p = sub.add_parser("dataset", help="导出 Monitor 候选集")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("ingest-nav", help="既有日志 → FailureEvent 流")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--include-non-acting", action="store_true")
    p.set_defaults(func=cmd_ingest_nav)

    p = sub.add_parser("merge", help="合流到事件总线")
    p.add_argument("--input", action="append", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--label", default=None)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("summary", help="总线汇总")
    p.add_argument("--root", required=True)
    p.set_defaults(func=cmd_summary)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_importable()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""按场景拆分的成绩单 —— 跨场景那一轮跑完后**立刻要用**的东西。

为什么需要它
-----------
现有 50 回合基准的全部 episode 来自**同一个场景**（根因见 `notes/route1-findings.md` 结论 9），
所以 `0.66 / 0.5734` 这类数字从来没有按场景拆开看过。
跨场景那一轮（4 场景 × 13 回合）跑完后，第一个要回答的问题就是：
**同一套配置在不同场景上的成绩差多少？** 如果不拆开看，池化平均会把场景间的巨大方差藏起来。

本模块只读日志，不跑仿真，Python 3.9 兼容。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..nav.replay import label_future_failures, load_run, trigger_precision

METRIC_KEYS = (
    ("success", "success"),
    ("spl", "spl"),
    ("ne", "ne"),
    ("collision", "collision_count"),
    ("steps", "steps"),
    ("system2_calls", "system2_calls"),
    ("replans", "replans"),
)


def per_scene(run_dir: str) -> Dict[str, Any]:
    run = load_run(run_dir)
    rows = run["progress"]
    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row.get("scene_id"))].append(row)

    decisions_by_scene: Dict[str, int] = Counter()
    for row in run["system1"]:
        decisions_by_scene[str(row.get("scene_id"))] += 1

    scenes: List[Dict[str, Any]] = []
    for scene, episodes in sorted(by_scene.items(), key=lambda kv: -len(kv[1])):
        entry: Dict[str, Any] = {
            "scene_id": scene,
            "episodes": len(episodes),
            "system1_decisions": decisions_by_scene.get(scene, 0),
        }
        for name, key in METRIC_KEYS:
            values = [float(row[key]) for row in episodes if row.get(key) is not None]
            entry[name] = float(np.mean(values)) if values else None
        scenes.append(entry)

    pooled: Dict[str, Any] = {"episodes": len(rows)}
    for name, key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        pooled[name] = float(np.mean(values)) if values else None

    # 场景间方差：只对"每个场景至少 5 个 episode"的场景算，避免小样本噪声
    eligible = [s for s in scenes if s["episodes"] >= 5]
    spread: Dict[str, Any] = {}
    for name, _ in METRIC_KEYS:
        values = [s[name] for s in eligible if s[name] is not None]
        if len(values) >= 2:
            spread[name] = {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values, ddof=1)),
                "pooled": pooled.get(name),
            }
    return {
        "run_dir": run_dir,
        "scenes_total": len(scenes),
        "scenes_with_5plus_episodes": len(eligible),
        "pooled": pooled,
        "per_scene": scenes,
        "cross_scene_spread": spread,
        "closed_loop_mode": (run["closed_loop_config"] or {}).get("mode"),
    }


def headline(run_dir: str) -> Dict[str, Any]:
    """池化成绩 + 几何触发 P@k（用日志自带标签）与无动作事件计数。"""
    from ..nav.replay import PRECURSOR_HORIZONS
    run = load_run(run_dir)
    rows = run["system1"]
    all_false = sum(1 for r in rows if r.get("has_executable_action") is False)
    true_fail = sum(1 for r in rows
                    if r.get("has_executable_action") is False and r.get("model_invoked"))
    out: Dict[str, Any] = {
        "run_dir": run_dir,
        "episodes": len(run["progress"]),
        "system1_decisions": len(rows),
        "no_executable_action_events": all_false,
        "official_label_events": true_fail,
        "official_label_coverage": (true_fail / all_false) if all_false else None,
        "replan_reasons": dict(Counter(r.get("replan_reason") for r in rows
                                       if r.get("should_replan"))),
    }
    for horizon in PRECURSOR_HORIZONS:
        r = trigger_precision(run_dir, "early_drift", require_model_invoked=True)
        out["early_drift_precision_at_%d" % horizon] = r["precision_at_%d" % horizon]
    return out


def _cmd_run(args: argparse.Namespace) -> int:
    report = per_scene(args.run)
    if not report["pooled"]["episodes"]:
        print("== %s ==" % os.path.basename(report["run_dir"].rstrip("/")))
        print("  这个 run 还没有回合（progress.json 为空）——跳过")
        return 0
    report["headline"] = headline(args.run)
    print("== %s ==" % os.path.basename(report["run_dir"].rstrip("/")))
    print("  mode=%s 场景数=%d 回合数=%d"
          % (report["closed_loop_mode"], report["scenes_total"], report["pooled"]["episodes"]))
    print()
    print("  %-26s %4s %6s %8s %8s %8s %8s %8s %6s"
          % ("scene", "ep", "decis", "success", "SPL", "NE", "coll", "steps", "S2"))
    for entry in report["per_scene"]:
        def f(value: Any, fmt: str = "%.4f") -> str:
            return "-" if value is None else fmt % float(value)
        print("  %-26s %4d %6d %8s %8s %8s %8s %8s %6s" % (
            entry["scene_id"][:26], entry["episodes"], entry["system1_decisions"],
            f(entry["success"]), f(entry["spl"]), f(entry["ne"], "%.3f"),
            f(entry["collision"], "%.2f"), f(entry["steps"], "%.1f"),
            f(entry["system2_calls"], "%.1f")))
    pooled = report["pooled"]
    print("  %-26s %4d %6s %8s %8s %8s %8s %8s %6s" % (
        "** 池化 **", pooled["episodes"], "-",
        "%.4f" % pooled["success"], "%.4f" % pooled["spl"],
        "%.3f" % pooled["ne"], "%.2f" % pooled["collision"],
        "%.1f" % pooled["steps"], "%.1f" % pooled["system2_calls"]))
    print()
    if report["cross_scene_spread"]:
        print("  场景间差异（只统计 ≥5 回合的场景，n=%d）：" % report["scenes_with_5plus_episodes"])
        for name, stat in report["cross_scene_spread"].items():
            print("    %-14s min=%.4f max=%.4f std=%.4f 池化=%.4f"
                  % (name, stat["min"], stat["max"], stat["std"], stat["pooled"]))
    else:
        print("  ⚠️ 只有一个场景 → **无法评估场景间方差**（这正是现有基准的问题）")
    h = report["headline"]
    print()
    print("  headline：无动作事件 %d（其中官方标签口径 %d，覆盖 %.1f%%）"
          % (h["no_executable_action_events"], h["official_label_events"],
             100 * (h["official_label_coverage"] or 0)))
    print("    重规划原因: %s" % json.dumps(h["replan_reasons"], ensure_ascii=False))
    print("    early_drift P@1/P@3/P@5 = %s / %s / %s" % (
        h.get("early_drift_precision_at_1"), h.get("early_drift_precision_at_3"),
        h.get("early_drift_precision_at_5")))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("\nwrote %s" % args.out)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """把多个 run 的按场景表并排（用于 单场景 vs 跨场景 对照）。"""
    reports = [per_scene(run_dir) for run_dir in args.run]
    names = [os.path.basename(r["run_dir"].rstrip("/"))[:34] for r in reports]
    keys = ("episodes", "scenes_total", "success", "spl", "ne", "collision", "steps",
            "system2_calls")
    print("%-24s %s" % ("指标", " ".join("%-36s" % n for n in names)))
    for key in keys:
        cells = []
        for report in reports:
            if key == "scenes_total":
                value: Any = report["scenes_total"]
            elif key == "episodes":
                value = report["pooled"]["episodes"]
            else:
                value = report["pooled"].get(key)
            cells.append("-" if value is None else ("%d" % value if isinstance(value, int)
                                                    else "%.4f" % value))
        print("%-24s %s" % (key, " ".join("%-36s" % c for c in cells)))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, ensure_ascii=False, indent=2)
        print("wrote %s" % args.out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按场景拆分的成绩单")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_run)
    p = sub.add_parser("compare")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_compare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

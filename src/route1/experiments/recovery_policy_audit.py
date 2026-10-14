"""实验③ 审计：把「恢复动作空间」放到真实决策流上，看它到底会做什么、能省什么。

动机（route1-bridge-plan.md §4-3 / failure-taxonomy.md §8）
--------------------------------------------------------
现状是**只有一种恢复动作**：重新调用 System-2 出计划。后果已经在你们自己的报表里：
208 次主动重规划里 70 次有害（P(Harmful) = 0.337），而触发那一刻的 P@1 = 0.0000。

本模块做的是**离线审计**（不需要跑仿真）：把 `DecisionLayerV3` 的恢复动作选择
重放到既有 50 回合日志上，回答三个具体问题：

1. 每种失败类型（N1/N2/N3/N4…）会被派到哪个动作？分布是什么？
2. 有多少次触发可以**不调 System-2**（`local_escape` 是短程闭环，不需要 S2）？
   —— 这是"同触发、少算力"的直接收益，可以直接换算成每秒/每回合的 S2 节省。
3. 派了动作之后，真失败有没有来？把 `(失败类型 × 动作)` 与 `no_action_within_k`
   交叉列表，看有没有"某类动作纯粹在浪费预算"。

诚实性声明
----------
这是**离线重放**，不是闭环实验：`local_escape` 实际能不能救回来，需要真跑仿真才知道。
所以本模块只报"决策层会怎么选"和"选完之后标签长什么样"，**不声称收益**。
真正闭环的收益表要等带帧/跨场景 run 出来后再做（那时才有"换了动作之后的分数"）。

只依赖 numpy 与既有日志，Python 3.9 兼容。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import RECOVERY_ACTION_LABEL
from ..nav.replay import (
    S2_INTERVAL_MODES,
    label_future_failures,
    load_run,
    reconstruct_scheduler_kwargs,
)
from ..nav.scheduler_v3 import DecisionLayerV3, SemanticConfig

#: 不需要调用 System-2 的恢复动作（省算力的那一类）
NO_S2_ACTIONS = ("local_escape", "retry_grasp", "change_grasp_point",
                 "reorder_subgoals", "ask_clarification", "none")


def audit_run(run_dir: str, horizons: Sequence[int] = (1, 3, 5)) -> Dict[str, Any]:
    run = load_run(run_dir)
    config = run["closed_loop_config"] or {}
    if not config.get("scheduler"):
        raise ValueError("run 缺少 closed_loop_config，无法重放：%s" % run_dir)
    kwargs = reconstruct_scheduler_kwargs(config)
    mode = str(config.get("mode") or "")
    allow_feedback_replan = bool(config.get("feedback_replan_enabled", False))
    tracks_s2 = mode in S2_INTERVAL_MODES
    temporal_cfg = dict(config.get("temporal_policy") or {})
    temporal_on = bool(temporal_cfg.get("enabled", False))
    temporal_threshold = float(temporal_cfg.get("threshold", 0.5))

    layer = DecisionLayerV3(
        scheduler_kwargs=kwargs,
        semantic=SemanticConfig(enabled=False),
        enable_recovery_actions=True,     # ← 本次审计的主角
        enable_semantic_channel=False,
    )
    labels = label_future_failures(run["system1"])
    label_by_key: Dict[Tuple[str, int, Any], Dict[str, Any]] = {}
    for row, label in zip(run["system1"], labels):
        key = (str(row.get("scene_id")), int(row.get("episode_id") or -1), row.get("action_step"))
        label_by_key[key] = label

    type_action: Dict[str, Counter] = defaultdict(Counter)
    action_type: Dict[str, Counter] = defaultdict(Counter)
    cross: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "hits": {h: 0 for h in horizons}, "observed": {h: 0 for h in horizons}}
    )
    #: 记录 N2（几何漂移）触发点上的 drift / plan_age，用来解释为什么
    #: `local_escape`（drift ≥ 0.20 且 plan_age ≤ 6）几乎从不命中
    n2_conditions: List[Dict[str, Any]] = []
    decisions = 0
    triggers = 0
    no_s2_triggers = 0
    escalation = 0
    episode_order: Dict[Tuple[str, int], int] = {}
    last_episode: Optional[Tuple[str, int]] = None

    for row in run["events"]:
        if row.get("event") == "system2":
            if tracks_s2:
                step = int(row.get("action_step", 0) or 0)
                if layer.scheduler.last_s2_call_step is None or \
                        step >= layer.scheduler.last_s2_call_step:
                    layer.record_system2_call(step)
            continue
        if row.get("event") != "system1":
            continue
        key = (str(row.get("scene_id")), int(row.get("episode_id") or -1))
        if key not in episode_order:
            episode_order[key] = len(episode_order)
        if last_episode is not None and key != last_episode:
            layer.reset_episode()
        last_episode = key
        current_step = int(row.get("action_step") or 0)
        if row.get("steps_since_last_s2") is not None:
            layer.scheduler.last_s2_call_step = current_step - int(row["steps_since_last_s2"])
        temporal_score = row.get("temporal_policy_score")
        external = bool(temporal_on and temporal_score is not None
                        and float(temporal_score) >= temporal_threshold)
        decision = layer.decide(
            feedback_score=float(row.get("feedback_score") or 0.0),
            plan_age=int(row.get("plan_age") or 0),
            current_step=current_step,
            has_executable_action=bool(row.get("has_executable_action", True)),
            allow_feedback_replan=allow_feedback_replan,
            trajectory_drift=row.get("trajectory_drift"),
            system2_calls_used=int(row.get("system2_call_count") or 0),
            external_trigger=external,
            external_reason="temporal_il",
            emit=False,
        )
        decisions += 1
        ftype = decision.get("failure_type")
        action = decision.get("recovery_action") or "none"
        if ftype is None:
            continue
        triggers += 1
        if action in NO_S2_ACTIONS:
            no_s2_triggers += 1
        if ftype == "N8":
            escalation += 1
        type_action[ftype][action] += 1
        action_type[action][ftype] += 1
        if ftype == "N2":
            drift = row.get("trajectory_drift")
            n2_conditions.append(
                {
                    "drift": None if drift is None else float(drift),
                    "plan_age": int(row.get("plan_age") or 0),
                    "trigger_kind": row.get("trigger_kind"),
                    "chosen_action": action,
                    # 与 `_select_recovery` 里的 N2 分支条件保持一致
                    "local_escape_condition_met": bool(
                        drift is not None and float(drift) >= 0.20
                        and int(row.get("plan_age") or 0) <= 6
                    ),
                }
            )

        label = label_by_key.get((key[0], key[1], row.get("action_step")))
        if label is None:
            continue
        bucket = cross[(ftype, action)]
        bucket["n"] += 1
        for horizon in horizons:
            value = label.get("no_action_within_%d" % horizon)
            if value is not None:
                bucket["observed"][horizon] += 1
                bucket["hits"][horizon] += 1 if value else 0

    cross_table = []
    for (ftype, action), bucket in sorted(cross.items()):
        entry: Dict[str, Any] = {"failure_type": ftype, "action": action,
                                 "action_label": RECOVERY_ACTION_LABEL.get(action, action),
                                 "triggers": bucket["n"]}
        for horizon in horizons:
            observed = bucket["observed"][horizon]
            entry["precision_at_%d" % horizon] = (
                bucket["hits"][horizon] / observed if observed else None
            )
            entry["observed_at_%d" % horizon] = observed
        cross_table.append(entry)

    return {
        "run_dir": run_dir,
        "mode": mode,
        "decisions": decisions,
        "triggers": triggers,
        "triggers_avoiding_system2": no_s2_triggers,
        "system2_avoid_rate": (no_s2_triggers / triggers) if triggers else None,
        "escalations": escalation,
        "failure_type_to_action": {k: dict(v) for k, v in sorted(type_action.items())},
        "action_to_failure_type": {k: dict(v) for k, v in sorted(action_type.items())},
        "n2_conditions": n2_conditions,
        "n2_local_escape_condition_rate": (
            sum(1 for item in n2_conditions if item["local_escape_condition_met"])
            / len(n2_conditions) if n2_conditions else None
        ),
        "cross_table": cross_table,
        "episode_summary": layer.episode_summary(),
    }


def _cmd_run(args: argparse.Namespace) -> int:
    report = audit_run(args.run)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("cross_table", "failure_type_to_action")},
                     ensure_ascii=False, indent=2))
    print("-- 失败类型 → 恢复动作 --")
    for ftype, actions in report["failure_type_to_action"].items():
        total = sum(actions.values())
        detail = ", ".join("%s(%s, %.0f%%)" % (
            RECOVERY_ACTION_LABEL.get(a, a), a, 100.0 * n / total
        ) for a, n in sorted(actions.items(), key=lambda kv: -kv[1]))
        print("  %-4s n=%-5d %s" % (ftype, total, detail))
    print("-- (失败类型 × 动作) → 真失败率 --")
    print("  %-6s %-20s %6s %8s %8s %8s" % ("type", "action", "n", "P@1", "P@3", "P@5"))
    for entry in report["cross_table"]:
        def f(v: Any) -> str:
            return "-" if v is None else "%.4f" % v
        print("  %-6s %-20s %6d %8s %8s %8s" % (
            entry["failure_type"], entry["action_label"], entry["triggers"],
            f(entry["precision_at_1"]), f(entry["precision_at_3"]),
            f(entry["precision_at_5"])))
    n2 = report.get("n2_conditions") or []
    if n2:
        print("-- N2（几何漂移）触发点上，local_escape 的条件满足情况 --")
        print("   触发 %d 次，其中 drift>=0.20 且 plan_age<=6 的只有 %d 次（%s）" % (
            len(n2), sum(1 for item in n2 if item["local_escape_condition_met"]),
            "-" if report["n2_local_escape_condition_rate"] is None
            else "%.1f%%" % (100 * report["n2_local_escape_condition_rate"])))
        drifts = [item["drift"] for item in n2 if item["drift"] is not None]
        ages = [item["plan_age"] for item in n2 if item["plan_age"] is not None]
        if drifts:
            print("   drift: min=%.3f p50=%.3f max=%.3f ｜ plan_age: min=%d p50=%d max=%d" % (
                min(drifts), sorted(drifts)[len(drifts) // 2], max(drifts),
                min(ages), sorted(ages)[len(ages) // 2], max(ages)))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


def _cmd_table(args: argparse.Namespace) -> int:
    rows = []
    skipped = []
    for run_dir in args.run:
        try:
            rows.append(audit_run(run_dir))
        except ValueError as exc:
            # 例如 mode=original 的 run 根本没有 closed_loop_config，跳过而不是崩掉
            skipped.append({"run_dir": run_dir, "reason": str(exc)})
    print("%-52s %8s %8s %12s %10s %8s"
          % ("run", "决策", "触发", "省S2触发", "省S2比例", "升级"))
    for row in rows:
        print("%-52s %8d %8d %12d %9s %8d" % (
            os.path.basename(row["run_dir"])[:52], row["decisions"], row["triggers"],
            row["triggers_avoiding_system2"],
            "-" if row["system2_avoid_rate"] is None else "%.1f%%" % (100 * row["system2_avoid_rate"]),
            row["escalations"]))
    if skipped:
        print()
        print("跳过 %d 个无法重放的 run：" % len(skipped))
        for item in skipped:
            print("  %s" % os.path.basename(item["run_dir"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"runs": rows, "skipped": skipped}, handle, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="恢复动作空间离线审计")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_run)
    p = sub.add_parser("table")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_table)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""离线验证两条新通道（碰撞密度 / 进度停滞）值不值得上：**先离线，再花 GPU**。

为什么必须先离线
--------------
`route1.nav.channels` 里的两条通道要在闭环里生效，一次跨场景 run 约 40–55 分钟 GPU，
而且闭环里"改了 A 会不会影响 B"没法归因。所以先用**日志里已有的逐决策观测量**
（`collision_count` / `distance_to_goal` / `counts_as_motion`）复算：
**这两条通道会在哪些决策点触发、覆盖多少个最终失败的回合、与现有 `early_drift` 比谁更准。**

口径（写清楚，避免又一次"标签口径"问题）
--------------------------------------
- 触发点 = 通道给出 reason 的那个决策点（`system1` 事件）。
- 标签 = **回合级**：该回合最终 success=0（并区分 `watchdog_stop` / `budget_exhausted_no_goal`，
  分类逻辑与 `scene_failure_diag` 完全一致，复用同一函数）。
  用回合级标签的理由：这两条通道的**目的**是"别让这个回合废掉"，而不是"预测下一个决策点出错"
  （后者是早漂移通道那套 P@k 口径，已经证明 P@1=0）。
- 因此指标是三条：
  1. `决策点精度` = 触发点里落在失败回合的比例（要和**基率**比：全体决策点里落在失败回合的比例）；
  2. `回合覆盖` = 失败回合里至少被触发一次的（按子类分开报，尤其看 `watchdog_stop` 那 5 个）；
  3. `提前量` = 首次触发到回合结束之间还剩多少个运动动作（越大越有用）。
  4. 代价 = 每回合触发次数（直接影响 System-2 调用数）。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..nav.channels import (ChannelConfig, REASON_COLLISION, REASON_PROGRESS, TriggerChannels)
from ..nav.replay import load_run
from .scene_failure_diag import classify_termination, episode_events_by_key


def _episode_stream(run: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """把每个回合的事件流压成"运动动作 + 决策点"序列（与在线推送顺序一致）。"""
    progress_by_key = {(str(row.get("scene_id")), int(row.get("episode_id"))): row
                       for row in run["progress"]}
    events_by_key = episode_events_by_key(run)
    horizon = max(int(row.get("steps") or 0) for row in run["progress"]) if run["progress"] else 0
    streams: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key, events in events_by_key.items():
        row = progress_by_key.get(key)
        if row is None:
            continue
        motion_steps: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        previous_collision = 0
        motion_step = 0
        for event in events:
            kind = event.get("event")
            if kind == "action":
                collision = event.get("collision_count")
                if collision is None:
                    continue
                if event.get("counts_as_motion"):
                    motion_step += 1
                    motion_steps.append({
                        "step": motion_step,
                        "increased": int(collision) > int(previous_collision),
                        "collision_count": int(collision),
                    })
                previous_collision = int(collision)
            elif kind == "system1":
                decisions.append({
                    "step": int(event.get("action_step") or motion_step),
                    "distance_to_goal": event.get("distance_to_goal"),
                    "replan_reason": event.get("replan_reason"),
                    "should_replan": event.get("should_replan"),
                    "collision_count": event.get("collision_count"),
                })
        termination, _evidence = classify_termination(row, events, horizon)
        streams[key] = {
            "scene_id": key[0],
            "episode_id": key[1],
            "success": float(row.get("success") or 0.0),
            "termination": termination,
            "motion_actions": motion_step,
            "motion_steps": motion_steps,
            "decisions": decisions,
        }
    return streams


def evaluate_channels(run_dir: str,
                      config: Optional[ChannelConfig] = None) -> Dict[str, Any]:
    run = load_run(run_dir)
    streams = _episode_stream(run)
    config = config or ChannelConfig()

    per_episode: List[Dict[str, Any]] = []
    for key, stream in sorted(streams.items()):
        channels = TriggerChannels(config)
        triggers: List[Dict[str, Any]] = []
        motion_index = 0
        motion_steps = stream["motion_steps"]
        for decision in stream["decisions"]:
            # 先把"这个决策点之前已经发生的运动动作"喂给碰撞窗口（与在线顺序一致）
            target_step = decision["step"]
            while motion_index < len(motion_steps) and motion_steps[motion_index]["step"] <= target_step:
                channels.push_collision(motion_steps[motion_index]["increased"])
                motion_index += 1
            reason = channels.evaluate(target_step, decision["distance_to_goal"])
            if reason:
                triggers.append({"step": target_step, "reason": reason})
        logged_early = sum(1 for d in stream["decisions"] if d.get("replan_reason") == "early_drift")
        logged_recovery = sum(1 for d in stream["decisions"] if d.get("replan_reason") == "recovery")
        per_episode.append({
            "scene_id": stream["scene_id"],
            "episode_id": stream["episode_id"],
            "success": stream["success"],
            "termination": stream["termination"],
            "motion_actions": stream["motion_actions"],
            "n_decisions": len(stream["decisions"]),
            "n_triggers": len(triggers),
            "n_collision_triggers": sum(1 for t in triggers if t["reason"] == REASON_COLLISION),
            "n_progress_triggers": sum(1 for t in triggers if t["reason"] == REASON_PROGRESS),
            "first_trigger_step": triggers[0]["step"] if triggers else None,
            "last_action_step": stream["motion_actions"],
            "logged_early_drift": logged_early,
            "logged_recovery": logged_recovery,
        })

    def summarise(name: str, key: str) -> Dict[str, Any]:
        total_decisions = sum(row["n_decisions"] for row in per_episode)
        failing = [row for row in per_episode if row["success"] < 0.5]
        failing_decisions = sum(row["n_decisions"] for row in failing)
        triggered_decisions = 0
        for row in per_episode:
            triggered_decisions += row[key]
        in_failing = sum(row[key] for row in failing)
        covered = [row for row in failing if row[key] > 0]
        covered_watchdog = [row for row in failing if row["termination"] == "watchdog_stop"
                            and row[key] > 0]
        watchdog = [row for row in failing if row["termination"] == "watchdog_stop"]
        lead = [row["last_action_step"] - row["first_trigger_step"] for row in covered
                if row["first_trigger_step"] is not None]
        return {
            "channel": name,
            "trigger_points": triggered_decisions,
            "triggers_per_episode": triggered_decisions / max(1, len(per_episode)),
            "decision_precision": (in_failing / triggered_decisions) if triggered_decisions else None,
            "decision_base_rate": (failing_decisions / total_decisions) if total_decisions else None,
            "episodes_covered": len(covered),
            "failing_episodes": len(failing),
            "coverage": (len(covered) / len(failing)) if failing else None,
            "watchdog_killed_covered": len(covered_watchdog),
            "watchdog_killed": len(watchdog),
            "median_lead_actions": (
                float(sorted(lead)[len(lead) // 2]) if lead else None
            ),
        }

    summaries = [
        summarise("collision_density", "n_collision_triggers"),
        summarise("progress_stall", "n_progress_triggers"),
        summarise("both", "n_triggers"),
        summarise("logged_early_drift(现有)", "logged_early_drift"),
    ]
    return {
        "run_dir": run_dir,
        "channel_config": config.__dict__,
        "episodes": len(per_episode),
        "pooled_success": (sum(row["success"] for row in per_episode) / max(1, len(per_episode))),
        "summaries": summaries,
        "per_episode": per_episode,
    }


def _cmd_eval(args: argparse.Namespace) -> int:
    config = ChannelConfig(
        collision_window_actions=args.collision_window,
        collision_min_in_window=args.collision_min,
        collision_cooldown_steps=args.collision_cooldown,
        progress_window_actions=args.progress_window,
        progress_min_improvement_m=args.progress_min_improvement,
        progress_regress_m=args.progress_regress,
        progress_cooldown_steps=args.progress_cooldown,
        collision_max_per_episode=args.collision_max_per_episode,
        progress_max_per_episode=args.progress_max_per_episode,
        collision_enabled=not args.disable_collision,
        progress_enabled=not args.disable_progress,
    )
    report = evaluate_channels(args.run, config)
    print("== %s ==" % os.path.basename(report["run_dir"].rstrip("/")))
    print("  回合 %d，池化 success %.4f" % (report["episodes"], report["pooled_success"]))
    print("  通道参数: %s" % json.dumps(report["channel_config"], ensure_ascii=False))
    print()
    print("  %-22s %8s %8s %10s %10s %8s %8s %8s"
          % ("通道", "触发点", "触发/回合", "决策点精度", "基率", "覆盖", "击杀覆盖", "提前量"))
    for row in report["summaries"]:
        def fmt(value: Any, pattern: str = "%.3f") -> str:
            return "-" if value is None else pattern % value
        print("  %-22s %8d %8s %10s %10s %8s %8s %8s"
              % (row["channel"], row["trigger_points"], fmt(row["triggers_per_episode"], "%.2f"),
                 fmt(row["decision_precision"]), fmt(row["decision_base_rate"]),
                 "%d/%d" % (row["episodes_covered"], row["failing_episodes"]),
                 "%d/%d" % (row["watchdog_killed_covered"], row["watchdog_killed"]),
                 fmt(row["median_lead_actions"], "%.1f")))
    print()
    print("  失败回合逐条（看门狗击杀的 5 个必须被覆盖）")
    for row in report["per_episode"]:
        if row["success"] >= 0.5:
            continue
        print("    %-14s ep%-4d %-24s motions=%3d decisions=%2d triggers=%d(coll %d/prog %d) "
              "first@%s"
              % (row["scene_id"][:14], row["episode_id"], row["termination"],
                 row["motion_actions"], row["n_decisions"], row["n_triggers"],
                 row["n_collision_triggers"], row["n_progress_triggers"],
                 row["first_trigger_step"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("\nwrote %s" % args.out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="离线验证碰撞/进度通道（只读日志）")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--collision-window", type=int, default=8)
    parser.add_argument("--collision-min", type=int, default=5)
    parser.add_argument("--collision-cooldown", type=int, default=6)
    parser.add_argument("--progress-window", type=int, default=12)
    parser.add_argument("--progress-min-improvement", type=float, default=0.3)
    parser.add_argument("--progress-regress", type=float, default=0.6)
    parser.add_argument("--progress-cooldown", type=int, default=12)
    parser.add_argument("--collision-max-per-episode", type=int, default=8)
    parser.add_argument("--progress-max-per-episode", type=int, default=8)
    parser.add_argument("--disable-collision", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    args = parser.parse_args(argv)
    return _cmd_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())

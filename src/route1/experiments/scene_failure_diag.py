"""跨场景失败诊断 —— 回答"同一套配置为什么在 A 场景 0.85、B 场景 0.15"。

为什么需要它
-----------
跨场景那一轮（`route1_r1multi4`，4 场景 × 13 回合）暴露出场景间巨大方差：
`2azQ1b91cZZ` 0.6923 / `TbHJrupSAjP` 0.6923 / `zsNo4HB9uLZ` 0.8462，而
`QUCTc6BB5sX` 只有 **0.1538**（池化 0.5962）。池化数字把这条裂隙藏起来了，
所以必须**按场景、按回合**把失败模式拆开——"成功率低"不是一条可行动的结论，
"这 5 个回合是碰撞看门狗把 episode 提前 STOP 掉的"才是。

本模块只读日志（不跑仿真），因此可以反复跑。它做三件事：

1. **给每个失败回合定"终止方式"**（用日志里的实证字段，不靠猜）：
   - `watchdog_stop`     碰撞看门狗触发 → 上游直接 `main_env_step(STOP)` 结束回合
   - `horizon_timeout`   走到 `max_steps_per_episode` 都没停（最后一个动作是移动）
   - `stop_wrong_place`  主动 STOP 了，但不在目标 1 m 内（"以为到了"）
   - `success`
2. **场景内对照**：成功 vs 失败回合在 collision / replans / no_action / steps 上的差异，
   以及 success 与这些量的点二列相关（看哪个量真的有判别力）。
3. **场景间方差**：min/max/std/spread，并指出"最差场景"。

`compare` 子命令把多个 run 的同 (scene, episode) 逐回合配对，输出胜/负/平与
每场景 success 表 —— 这是判断"某个改动是否真的让跨场景更稳"的唯一正确口径
（只看池化会把场景方差当成噪声）。

Python 3.9 兼容；numpy 只用于相关与统计。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..nav.replay import load_run

#: Habitat 的动作编码（`habitat_vln_evaluator.py:83`）
STOP_ACTION = 0
MOTION_ACTIONS = (1, 2, 3)


# --------------------------------------------------------------------------- #
# 统计小工具
# --------------------------------------------------------------------------- #


def point_biserial(labels: Sequence[float], values: Sequence[float]) -> Optional[float]:
    """二值标签与连续量的相关（= 两者的 Pearson 相关）。

    比"分组均值"多一个好处：它给出**方向**。这里最关键的诊断就是
    "success 与 collision_count 强负相关、与 replans 几乎无关"这一类判断。
    """
    pairs = [(float(a), float(b)) for a, b in zip(labels, values)
             if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    y = np.asarray([p[0] for p in pairs], dtype=float)
    x = np.asarray([p[1] for p in pairs], dtype=float)
    if float(np.std(y)) == 0.0 or float(np.std(x)) == 0.0:
        return None
    return float(np.corrcoef(y, x)[0, 1])


def _mean(values: Sequence[Any]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


# --------------------------------------------------------------------------- #
# 单 run 诊断
# --------------------------------------------------------------------------- #


def episode_events_by_key(run: Dict[str, Any]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """把事件流按回合切开。

    坑（第一次就踩了）：**`action` / `collision_watchdog_stop` / `episode_summary`
    这几类事件里没有 `scene_id`**（只有 `system1` / `system2` 有），而 `episode_id`
    在场景之间会重复（本次 4 场景里 `ep19` / `ep40` 等同时出现在两个场景）。
    所以不能按 `episode_id` 归并，必须利用"episode 是按 `progress.json` 的顺序
    串行跑"这一事实做**顺序切分**：遇到带 `scene_id` 的 `system1` 事件就把指针
    推到对应回合，其余无 `scene_id` 的事件归给当前回合。
    """
    keys: List[Tuple[str, int]] = [(str(row.get("scene_id")), int(row.get("episode_id")))
                                   for row in run["progress"]]
    table: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    if not keys:
        return table
    pointer = 0
    for row in run["events"]:
        scene = row.get("scene_id")
        episode = row.get("episode_id")
        if scene is not None and episode is not None:
            key = (str(scene), int(episode))
            if key in keys[pointer:]:
                pointer = keys.index(key, pointer)
            elif key in keys:                      # 兜底：万一乱序
                pointer = keys.index(key)
        table[keys[pointer]].append(row)
    return table


def classify_termination(progress_row: Dict[str, Any],
                 events: List[Dict[str, Any]],
                 horizon: int) -> Tuple[str, Dict[str, Any]]:
    """判定一个回合的终止方式（返回 类别 + 证据）。

    ⚠️ **日志的口径局限（必须和结论一起写）**：evaluator 不记录 Habitat 返回 `done` 的原因，
    所以"没有到达目标"的回合里，下面两种情形**在单 run 日志里无法区分**：

    - (a) episode 自己的步数预算用尽（VLN-CE 每个 episode 有自己的 `max_steps`，
      所以不同回合的 `steps` 差别很大：本 run 是 36–202）；
    - (b) 智能体主动执行了 STOP，但停在离目标 > 1 m 的地方。

    两者可行动的共同点是"**预算被烧完却没到目标**"，所以这里合成一类
    `budget_exhausted_no_goal`，并把 `n_stop_actions` 放进证据。
    要真正分开它们，用 `compare` 子命令：若同一回合在**多个 run 里 `steps` 逐位相同**，
    那它就是被 episode 自身预算截断的（决策层改不动），否则是决策层/策略决定的。
  """
    success = float(progress_row.get("success") or 0.0) >= 0.5
    steps = int(progress_row.get("steps") or 0)
    actions = [e for e in events if e.get("event") == "action" and e.get("action") is not None]
    codes = [int(e["action"]) for e in actions]
    last_action = codes[-1] if codes else None
    watchdog = int(progress_row.get("collision_watchdog_stops") or 0)
    evidence = {
        "steps": steps,
        "last_action": last_action,
        # 注：最后一个动作几乎总是 5（LOOK_DOWN）——那是每个决策周期内部的动作，
        # **不能**当成"智能体停下来了"的证据。
        "last_5_actions": codes[-5:],
        "n_stop_actions": sum(1 for code in codes if code == STOP_ACTION),
        "watchdog_stops": watchdog,
        "run_max_steps": horizon,
        "n_actions_logged": len(actions),
    }
    if success:
        return "success", evidence
    if watchdog > 0:
        return "watchdog_stop", evidence
    return "budget_exhausted_no_goal", evidence


def diagnose(run_dir: str) -> Dict[str, Any]:
    run = load_run(run_dir)
    progress = run["progress"]
    if not progress:
        raise ValueError("run 目录里没有 progress.json 记录: %s" % run_dir)
    events_by_key = episode_events_by_key(run)
    horizon = max(int(row.get("steps") or 0) for row in progress)

    episodes: List[Dict[str, Any]] = []
    for row in progress:
        scene = str(row.get("scene_id"))
        episode = int(row.get("episode_id"))
        key = (scene, episode)
        events = events_by_key.get(key, [])
        klass, evidence = classify_termination(row, events, horizon)
        reasons = Counter(e.get("replan_reason") for e in events if e.get("event") == "system1")
        episodes.append({
            "scene_id": scene,
            "episode_id": episode,
            "success": float(row.get("success") or 0.0),
            "spl": float(row.get("spl") or 0.0),
            "ne": float(row.get("ne") or 0.0),
            "steps": int(row.get("steps") or 0),
            "executed_motion_actions": int(row.get("executed_motion_actions") or 0),
            "collision_count": int(row.get("collision_count") or 0),
            "replans": int(row.get("replans") or 0),
            "no_action_count": int(row.get("no_action_count") or 0),
            "system2_calls": int(row.get("system2_calls") or 0),
            "watchdog_stops": int(row.get("collision_watchdog_stops") or 0),
            "early_drift_replans": int(row.get("early_drift_replans") or 0),
            "recovery_replans": int(row.get("recovery_replans") or 0),
            "termination": klass,
            "evidence": evidence,
            "replan_reasons": dict(reasons),
            "instruction": row.get("episode_instruction"),
        })

    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in episodes:
        by_scene[entry["scene_id"]].append(entry)

    scenes: List[Dict[str, Any]] = []
    for scene, rows in sorted(by_scene.items(), key=lambda kv: -len(kv[1])):
        terminals = Counter(row["termination"] for row in rows)
        ok = [row for row in rows if row["success"] >= 0.5]
        bad = [row for row in rows if row["success"] < 0.5]
        scenes.append({
            "scene_id": scene,
            "episodes": len(rows),
            "success": _mean([row["success"] for row in rows]),
            "spl": _mean([row["spl"] for row in rows]),
            "ne": _mean([row["ne"] for row in rows]),
            "collision": _mean([row["collision_count"] for row in rows]),
            "steps": _mean([row["steps"] for row in rows]),
            "replans": _mean([row["replans"] for row in rows]),
            "system2_calls": _mean([row["system2_calls"] for row in rows]),
            "watchdog_stops": int(sum(row["watchdog_stops"] for row in rows)),
            "terminations": dict(terminals),
            "failed_episodes": len(bad),
            "success_vs_failure": {
                "n_success": len(ok),
                "n_failure": len(bad),
                "collision": {"success": _mean([r["collision_count"] for r in ok]),
                              "failure": _mean([r["collision_count"] for r in bad])},
                "replans": {"success": _mean([r["replans"] for r in ok]),
                            "failure": _mean([r["replans"] for r in bad])},
                "no_action": {"success": _mean([r["no_action_count"] for r in ok]),
                              "failure": _mean([r["no_action_count"] for r in bad])},
                "steps": {"success": _mean([r["steps"] for r in ok]),
                          "failure": _mean([r["steps"] for r in bad])},
            },
            "correlation_with_success": {
                name: point_biserial([row["success"] for row in rows],
                                     [row[key] for row in rows])
                for name, key in (("collision_count", "collision_count"),
                                  ("replans", "replans"),
                                  ("no_action_count", "no_action_count"),
                                  ("system2_calls", "system2_calls"),
                                  ("steps", "steps"))
            },
        })

    successes = [scene["success"] for scene in scenes if scene["episodes"] >= 5]
    spread = None
    if len(successes) >= 2:
        spread = {
            "n_scenes": len(successes),
            "min": float(np.min(successes)),
            "max": float(np.max(successes)),
            "std": float(np.std(successes, ddof=1)),
            "range": float(np.max(successes) - np.min(successes)),
        }

    pooled = {
        "episodes": len(episodes),
        "success": _mean([row["success"] for row in episodes]),
        "spl": _mean([row["spl"] for row in episodes]),
        "ne": _mean([row["ne"] for row in episodes]),
        "collision": _mean([row["collision_count"] for row in episodes]),
        "steps": _mean([row["steps"] for row in episodes]),
        "replans": _mean([row["replans"] for row in episodes]),
        "terminations": dict(Counter(row["termination"] for row in episodes)),
    }

    mode = (run["closed_loop_config"] or {}).get("mode")
    safety = (run["closed_loop_config"] or {}).get("safety") or {}
    return {
        "run_dir": run_dir,
        "mode": mode,
        "safety": safety,
        "horizon_steps": horizon,
        "scenes_total": len(scenes),
        "pooled": pooled,
        "cross_scene_success_spread": spread,
        "per_scene": scenes,
        "episodes": episodes,
    }


# --------------------------------------------------------------------------- #
# 多 run 对照
# --------------------------------------------------------------------------- #


def _index(run_dir: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    report = diagnose(run_dir)
    return {(row["scene_id"], row["episode_id"]): row for row in report["episodes"]}


def compare(run_dirs: Sequence[str], baseline: Optional[str] = None) -> Dict[str, Any]:
    reports = [diagnose(path) for path in run_dirs]
    indexes = [_index(path) for path in run_dirs]
    names = [os.path.basename(r["run_dir"].rstrip("/")) for r in reports]

    base_index = None
    base_name = None
    if baseline:
        for name, index in zip(names, indexes):
            if baseline in name:
                base_index, base_name = index, name
                break
        if base_index is None:
            # 也允许用 mode 指定基线（前提是只有一个 run 是这个 mode）
            matches = [name for name, report in zip(names, reports)
                       if str(report.get("mode")) == str(baseline)]
            if len(matches) == 1:
                base_name = matches[0]
                base_index = indexes[names.index(base_name)]
            else:
                raise ValueError(
                    "baseline %r 既不是 run 名的一部分，也不是唯一的 mode（匹配到 %d 个 run）"
                    % (baseline, len(matches)))

    # 逐 (scene, episode) 配对：以第一个 run 的回合集合为准
    keys = sorted(indexes[0].keys())
    per_scene_pair: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "wins": 0, "losses": 0, "ties": 0}
    )
    # 同一回合在多个 run 里的 steps 是否逐位相同 → 判断"预算截断"还是"决策层提前结束"
    step_identical: Dict[str, int] = {"n_failed_episodes": 0, "identical_steps": 0}
    pairs: List[Dict[str, Any]] = []
    for key in keys:
        base_row = base_index.get(key) if base_index else None
        entry: Dict[str, Any] = {"scene_id": key[0], "episode_id": key[1]}
        steps_seen: List[int] = []
        for name, index in zip(names, indexes):
            row = index.get(key)
            entry[name] = None if row is None else row["success"]
            entry[name + "::steps"] = None if row is None else row["steps"]
            entry[name + "::termination"] = None if row is None else row["termination"]
            if row is not None:
                steps_seen.append(row["steps"])
        if base_row is not None and base_row["success"] < 0.5:
            step_identical["n_failed_episodes"] += 1
            if len(steps_seen) >= 2 and len(set(steps_seen)) == 1:
                step_identical["identical_steps"] += 1
        if base_row is not None:
            for name, index in zip(names, indexes):
                if name == base_name:
                    continue
                row = index.get(key)
                if row is None:
                    continue
                delta = row["success"] - base_row["success"]
                stats = per_scene_pair[key[0]]
                stats["n"] += 1
                if delta > 0.5:
                    stats["wins"] += 1
                elif delta < -0.5:
                    stats["losses"] += 1
                else:
                    stats["ties"] += 1
        pairs.append(entry)

    # 场景集合取所有 run 的并集（不同 run 理论上同一清单，但缺场景时不能崩）
    scene_ids: List[str] = []
    for report in reports:
        for scene in report["per_scene"]:
            if scene["scene_id"] not in scene_ids:
                scene_ids.append(scene["scene_id"])
    per_scene_success: Dict[str, Dict[str, Optional[float]]] = {}
    for scene_id in scene_ids:
        per_scene_success[scene_id] = {}
        for name, report in zip(names, reports):
            value = None
            for scene in report["per_scene"]:
                if scene["scene_id"] == scene_id:
                    value = scene["success"]
            per_scene_success[scene_id][name] = value

    return {
        "run_dirs": list(run_dirs),
        "names": names,
        "baseline": base_name,
        "per_scene_success": per_scene_success,
        "pooled_success": {name: report["pooled"]["success"]
                           for name, report in zip(names, reports)},
        "cross_scene_spread": {name: report["cross_scene_success_spread"]
                               for name, report in zip(names, reports)},
        "budget_clip_test": step_identical,
        "paired_vs_baseline": {scene: dict(stats) for scene, stats in per_scene_pair.items()},
        "episode_pairs": pairs,
    }


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #


def _print_run(report: Dict[str, Any]) -> None:
    print("== %s ==" % os.path.basename(report["run_dir"].rstrip("/")))
    print("  mode=%s  safety=%s  horizon(最大 steps)=%d  场景数=%d"
          % (report["mode"], json.dumps(report["safety"], ensure_ascii=False),
             report["horizon_steps"], report["scenes_total"]))
    pooled = report["pooled"]
    print("  池化：%d 回合 success=%.4f spl=%.4f ne=%.3f coll=%.2f steps=%.1f replans=%.1f"
          % (pooled["episodes"], pooled["success"], pooled["spl"], pooled["ne"],
             pooled["collision"], pooled["steps"], pooled["replans"]))
    print("  终止方式分布: %s" % json.dumps(pooled["terminations"], ensure_ascii=False))
    print()
    print("  %-14s %3s %7s %6s %6s %6s %6s %8s %s"
          % ("scene", "ep", "success", "coll", "steps", "repl", "S2", "wd_stop", "终止方式"))
    for scene in report["per_scene"]:
        print("  %-14s %3d %7.4f %6.2f %6.1f %6.1f %6.1f %8d %s"
              % (scene["scene_id"][:14], scene["episodes"], scene["success"],
                 scene["collision"], scene["steps"], scene["replans"],
                 scene["system2_calls"], scene["watchdog_stops"],
                 json.dumps(scene["terminations"], ensure_ascii=False)))
    spread = report["cross_scene_success_spread"]
    print()
    if spread:
        print("  场景间 success：min=%.4f max=%.4f std=%.4f range=%.4f（n=%d 场景）"
              % (spread["min"], spread["max"], spread["std"], spread["range"], spread["n_scenes"]))
    else:
        print("  ⚠️ 场景数不足，无法评估场景间方差")
    print()
    print("  场景内：成功 vs 失败 的均值差（collision / replans / no_action / steps）")
    for scene in report["per_scene"]:
        svf = scene["success_vs_failure"]
        print("    %-14s succ n=%2d fail n=%2d | coll %.2f vs %.2f | replans %.1f vs %.1f"
              " | no_action %.1f vs %.1f | steps %.1f vs %.1f"
              % (scene["scene_id"][:14], svf["n_success"], svf["n_failure"],
                 svf["collision"]["success"] or 0.0, svf["collision"]["failure"] or 0.0,
                 svf["replans"]["success"] or 0.0, svf["replans"]["failure"] or 0.0,
                 svf["no_action"]["success"] or 0.0, svf["no_action"]["failure"] or 0.0,
                 svf["steps"]["success"] or 0.0, svf["steps"]["failure"] or 0.0))
        corr = scene["correlation_with_success"]
        print("        相关(success, ·): %s"
              % ", ".join("%s=%.3f" % (k, v) for k, v in corr.items() if v is not None))
    print()
    print("  失败回合明细（含终止证据）")
    for row in report["episodes"]:
        if row["success"] >= 0.5:
            continue
        ev = row["evidence"]
        print("    %-14s ep%-4d %-16s steps=%3d last_action=%-4s stops=%2d tail=%s coll=%d wd=%d "
              "replans=%d no_action=%d ne=%.2f"
              % (row["scene_id"][:14], row["episode_id"], row["termination"], row["steps"],
                 ev["last_action"], ev["n_stop_actions"], ev["last_5_actions"],
                 row["collision_count"], row["watchdog_stops"],
                 row["replans"], row["no_action_count"], row["ne"]))


def _cmd_run(args: argparse.Namespace) -> int:
    report = diagnose(args.run)
    _print_run(report)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("\nwrote %s" % args.out)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    report = compare(args.run, baseline=args.baseline)
    names = report["names"]
    print("%-16s %s" % ("scene", " ".join("%-14s" % name[:14] for name in names)))
    for scene, values in report["per_scene_success"].items():
        print("%-16s %s" % (scene[:16], " ".join(
            "%-14s" % ("-" if values[name] is None else "%.4f" % values[name])
            for name in names)))
    print("%-16s %s" % ("** 池化 **", " ".join(
        "%-14s" % "%.4f" % report["pooled_success"][name] for name in names)))
    print()
    print("跨场景稳定性（success 的 min / std / range）")
    for name in names:
        spread = report["cross_scene_spread"][name]
        if spread:
            print("  %-58s min=%.4f std=%.4f range=%.4f"
                  % (name, spread["min"], spread["std"], spread["range"]))
    if report["baseline"]:
        print()
        print("逐回合配对 vs 基线 %s（win=该回合由失败变成功）" % report["baseline"])
        for scene, stats in sorted(report["paired_vs_baseline"].items()):
            print("  %-16s n=%2d win=%2d loss=%2d tie=%2d"
                  % (scene, stats["n"], stats["wins"], stats["losses"], stats["ties"]))
        clip = report["budget_clip_test"]
        if clip["n_failed_episodes"]:
            print()
            print("预算截断检验：基线里失败的 %d 个回合中，有 %d 个在**所有 run 里 steps 逐位相同**"
                  % (clip["n_failed_episodes"], clip["identical_steps"]))
            print("  → 这些回合是被 episode 自身步数预算截断的（决策层改不动）；"
                  "其余是决策层/策略决定的。")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("\nwrote %s" % args.out)
    return 0


def pool(run_dirs: Sequence[str]) -> Dict[str, Any]:
    """把**互补清单**上的多个 run 合并成一张更大的按场景表。

    用途：主清单 `val_unseen_multiscene` 只有 4 个场景，场景间 std 在 4 个点上估不准。
    扩场景那批 run 用的是另一份清单（新增 4 个场景、**不重叠**），合起来就是 8 个场景。
    合并口径：**按场景**取（同一场景只保留第一个出现的 run 的条目），
    池化 success / SPL 按回合数加权。

    前提（必须写进结论）：引擎每个 episode 都从 `env.reset()` 独立开始，
    所以同一配置在不同清单上跑出的同一 episode 可比；若某天"8 场景一起跑"的结果与
    这个拼接结果不一致，说明前提不成立——那本身是要报的发现。
    """
    reports = [diagnose(path) for path in run_dirs]
    per_scene: Dict[str, Dict[str, Any]] = {}
    duplicates: List[Tuple[str, str]] = []
    order: List[str] = []
    for report in reports:
        name = os.path.basename(report["run_dir"].rstrip("/"))
        for scene in report["per_scene"]:
            scene_id = scene["scene_id"]
            if scene_id in per_scene:
                duplicates.append((scene_id, name))
                continue
            entry = dict(scene)
            entry["source_run"] = name
            per_scene[scene_id] = entry
            order.append(scene_id)

    scenes = [per_scene[scene_id] for scene_id in order]
    total_episodes = sum(int(s["episodes"]) for s in scenes)
    pooled_success = (sum(float(s["success"]) * int(s["episodes"]) for s in scenes)
                      / total_episodes) if total_episodes else None
    pooled_spl = (sum(float(s["spl"]) * int(s["episodes"]) for s in scenes)
                  / total_episodes) if total_episodes else None
    eligible = [s["success"] for s in scenes if int(s["episodes"]) >= 5]
    spread = None
    if len(eligible) >= 2:
        spread = {
            "n_scenes": len(eligible),
            "min": float(np.min(eligible)),
            "max": float(np.max(eligible)),
            "std": float(np.std(eligible, ddof=1)),
            "range": float(np.max(eligible) - np.min(eligible)),
        }
    sources = [os.path.basename(r["run_dir"].rstrip("/")) for r in reports]
    return {
        "run_dirs": list(run_dirs),
        "sources": sources,
        "duplicated_scenes": duplicates,
        "scenes_total": len(scenes),
        "episodes_total": total_episodes,
        "pooled_success": pooled_success,
        "pooled_spl": pooled_spl,
        "cross_scene_success_spread": spread,
        "per_scene": scenes,
        "per_source": [
            {"run": name,
             "scenes": [s["scene_id"] for s in scenes if s["source_run"] == name],
             "episodes": sum(int(s["episodes"]) for s in scenes if s["source_run"] == name),
             "pooled_success": _mean([s["success"] for s in scenes
                                      if s["source_run"] == name])}
            for name in sources
        ],
    }


def _cmd_pool(args: argparse.Namespace) -> int:
    report = pool(args.run)
    print("== 合并 %d 个 run → %d 个场景 / %d 个回合 =="
          % (len(report["sources"]), report["scenes_total"], report["episodes_total"]))
    if report["duplicated_scenes"]:
        print("  ⚠️ 重复场景（已保留先出现的那个）: %s" % report["duplicated_scenes"])
    print("  %-14s %4s %8s %8s %7s %7s %7s %7s %s"
          % ("scene", "ep", "success", "SPL", "NE", "coll", "steps", "S2", "来源 run"))
    for scene in report["per_scene"]:
        print("  %-14s %4d %8.4f %8.4f %7.3f %7.2f %7.1f %7.1f %s"
              % (scene["scene_id"][:14], scene["episodes"], scene["success"], scene["spl"],
                 scene["ne"], scene["collision"], scene["steps"], scene["system2_calls"],
                 scene["source_run"]))
    print("  %-14s %4d %8.4f %8.4f"
          % ("** 池化 **", report["episodes_total"], report["pooled_success"],
             report["pooled_spl"]))
    spread = report["cross_scene_success_spread"]
    print()
    if spread:
        print("  场景间 success（n=%d 场景）：min=%.4f max=%.4f std=%.4f range=%.4f"
              % (spread["n_scenes"], spread["min"], spread["max"], spread["std"],
                 spread["range"]))
    print("  分来源:")
    for source in report["per_source"]:
        print("    %-40s %2d 场景 %3d 回合 池化 success=%s"
              % (source["run"], len(source["scenes"]), source["episodes"],
                 "%.4f" % source["pooled_success"]
                 if source["pooled_success"] is not None else "-"))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("\nwrote %s" % args.out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="跨场景失败诊断（只读日志）")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="单 run：按场景/回合拆失败模式")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_run)
    p = sub.add_parser("compare", help="多 run：逐回合配对 + 跨场景稳定性")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_compare)
    p = sub.add_parser("pool", help="互补清单的多个 run 合并成更多场景的稳定性表")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_pool)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

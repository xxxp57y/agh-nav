"""对已落盘的 trace/events 重放判定，验证 X1/X2/X3 判据（不需要 Isaac）。

用法（服务器上）：
    python3 route1/isaac/replay_judge.py <trace.json> [more.json ...]

trace JSON 是 `manip_task.py --record-trace` 写出的逐步数组，每行含
`subgoal_states / gripper_width_m / contact_force_n / object_dz_m / dist_to_goal_m /
joint_positions / joint_targets / phase`。本脚本用它重放 `subgoal_judge` 的
三个失败判据，并打印结论与证据——这样判据本身可以在没有仿真器时被审计。

Python 3.9 兼容。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from subgoal_judge import (  # noqa: E402
    StepObs,
    SubgoalJudge,
    SubgoalThresholds,
    detect_grasp_failure,
    detect_misplace_failure,
    detect_slip_failure,
)


def replay(path: str, goal_xy=(0.35, 0.0), verbose: bool = True) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        trace = json.load(handle)
    if not trace:
        return {"path": path, "error": "empty trace"}
    first = trace[0]
    init_pos = tuple(first["object_pose"]["pos"])
    judge = SubgoalJudge(SubgoalThresholds(), goal_xy=goal_xy, object_init_pos=init_pos)
    grasp_seen = False
    verdict: Optional[Any] = None
    verdict_step = None
    for entry in trace:
        obs = StepObs(
            step=int(entry["step"]),
            sim_time=float(entry.get("sim_time") or 0.0),
            gripper_width_m=float(entry["gripper_width_m"]),
            contact_force_n=float(entry["contact_force_n"]),
            contact_pairs=int(entry.get("contact_pairs") or 0),
            object_pos=tuple(entry["object_pose"]["pos"]),
            object_vel=tuple(entry["object_pose"].get("lin_vel") or (0, 0, 0)),
            gripper_pos=tuple(entry.get("gripper_pos") or (0, 0, 0)),
            joint_positions=tuple(entry.get("joint_positions") or (0, 0)),
            joint_targets=tuple(entry.get("joint_targets") or (0, 0)),
            phase=str(entry.get("phase") or ""))
        judge.update(obs)
        if judge.state.grasp:
            grasp_seen = True
        phase = obs.phase
        dist = entry.get("dist_to_goal_m")
        candidate = None
        if phase in ("lift", "transport", "lower"):
            candidate = detect_slip_failure(judge, grasp_seen=grasp_seen)
        if candidate is None or candidate.type is None:
            c2 = (detect_misplace_failure(judge, dist)
                  if phase in ("release", "retreat", "settle") else None)
            candidate = c2 if (c2 and c2.type) else candidate
        if candidate is None or candidate.type is None:
            c3 = (detect_grasp_failure(judge)
                  if phase in ("settle_grasp", "lift", "retreat", "settle") else None)
            candidate = c3 if (c3 and c3.type) else candidate
        if candidate is not None and candidate.type is not None:
            verdict = candidate
            verdict_step = int(entry["step"])
            break
    report = {
        "path": os.path.basename(path),
        "n_steps_replayed": len(judge.trace),
        "final_state": judge.state.as_dict(),
        "first_true_step": judge.summary()["first_true_step"],
        "min_gripper_width_m": judge.summary()["min_gripper_width_m"],
        "max_contact_force_n": judge.summary()["max_contact_force_n"],
        "max_object_dz_m": judge.summary()["max_object_dz_m"],
        "verdict": (verdict.as_dict() if verdict else None),
        "verdict_step": verdict_step,
        "transitions": judge.transitions[:12],
    }
    if verbose:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def main(argv: List[str]) -> int:
    paths = argv[1:]
    if not paths:
        print(__doc__)
        return 2
    for i, path in enumerate(paths):
        if i:
            print("-" * 70)
        replay(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

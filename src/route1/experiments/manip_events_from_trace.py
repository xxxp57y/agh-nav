"""从 Isaac 侧已落盘的逐步 trace 里，**独立**判定 X1/X2/X3 并产出契约事件。

为什么要有这个东西
----------------
Isaac 子代理的 `manip_task.py` 已经能产出**正确的逐步信号流**
（`*_events.jsonl`，588 行/次，含 `gripper_width_n` / `contact_force_n` /
`object_dz_m` / `dist_to_goal_m`），事件契约的字段实现是对的；
但**失败类型一直是 `null`**（X1 判定没接上，是硬 bug）。

本模块刻意与那个实现**解耦**：它只吃"逐步信号 + summary"，用**自己的一套、
显式写明的判据**重新判 X1/X2/X3，产出契约兼容的 `FailureEvent`。
这样：
- 操作侧的真实数据立刻能进总线，不再被一个 bug 卡住；
- 两套判据可以互相校对（如果子代理修好后结论一致，就是交叉验证）；
- 判据的阈值都在这里写清楚，可被质疑、可被复现。

判据阈值（我自己的，按实测量级标定，全部可改）
-------------------------------------------
实测锚点：`success` 模式下接触力恒为 **0.0 N**；
真正抓到的那次（smoke6）在 step 77 建立接触，力 **154–229 N**，夹爪合到 **0.0184 m**；
抓空时夹爪最小宽度是 0.0460 m（合上了但没碰到）或 0.1248 m（根本没合）。

    CONTACT_FORCE_N_MIN = 5.0     # 高于它才算"建立了接触"
    GRIPPER_CLOSED_M    = 0.06    # 低于它才算"夹爪合上了"
    LIFT_DZ_M_MIN       = 0.02    # 物体抬升超过它才算抬起来
    PLACE_DIST_M_MAX    = 0.10    # 水平距目标小于它才算"放到位"

Python 3.9 兼容，只依赖标准库。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import Attribution, FailureEvent, Outcome, Recovery, Signals

#: ---- 判据阈值（全部可调，改动请写进报告） ----
CONTACT_FORCE_N_MIN = 5.0
GRIPPER_CLOSED_M = 0.06
LIFT_DZ_M_MIN = 0.02
PLACE_DIST_M_MAX = 0.10

#: 与 `manip_task.py` 的阶段划分保持一致的"应当已经接触"的后半段阶段名
POST_GRASP_PHASES = ("settle_grasp", "lift", "transport", "place", "release", "retreat")


def _step_of(record: Dict[str, Any], fallback: int) -> int:
    match = re.search(r":(\d+)$", str(record.get("id", "")))
    return int(match.group(1)) if match else fallback


def load_run(run_dir: str, stem: str) -> Optional[Dict[str, Any]]:
    """读一对 `<stem>_events.jsonl` + `<stem>_summary.json`。

    注意：本模块**自己产出的契约事件文件**也叫 `*_events.jsonl`，
    如果输出落在同一个目录里，会被 glob 二次吃掉（我第一版就自吞了一次，
    凭空多出一个 X2）。这里用"有没有 `schema` 字段"把自家产物识别出来并跳过。
    """
    events_path = os.path.join(run_dir, stem + "_events.jsonl")
    summary_path = os.path.join(run_dir, stem + "_summary.json")
    if not os.path.exists(events_path) or os.path.getsize(events_path) == 0:
        return None
    steps: List[Dict[str, Any]] = []
    with open(events_path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if str(record.get("schema", "")).startswith("route1/failure-event"):
                # 这是我们自己产出的契约事件，不是仿真 trace
                return None
            signals = record.get("signals") or {}
            steps.append(
                {
                    "step": _step_of(record, index),
                    "gripper_width_m": signals.get("gripper_width_m"),
                    "contact_force_n": signals.get("contact_force_n"),
                    "object_dz_m": signals.get("object_dz_m"),
                    "dist_to_goal_m": signals.get("dist_to_goal_m"),
                    "collision_delta": signals.get("collision_delta"),
                }
            )
    summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        try:
            summary = json.load(open(summary_path, encoding="utf-8"))
        except ValueError:
            summary = {}
    if not steps:
        return None
    return {"stem": stem, "steps": steps, "summary": summary,
            "events_path": events_path, "summary_path": summary_path}


def _series(steps: Sequence[Dict[str, Any]], name: str) -> List[Optional[float]]:
    return [step.get(name) for step in steps]


def _max(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None]
    return max(finite) if finite else None


def _min(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None]
    return min(finite) if finite else None


def detect(run: Dict[str, Any]) -> Dict[str, Any]:
    """对一次运行的逐步信号做判定，返回诊断 dict（可能不含失败，也可能是 X1/X2/X3）。"""
    steps = run["steps"]
    summary = run["summary"]
    attempts = (summary.get("trials") or [{}])[0].get("attempts") if summary.get("trials") else None
    attempt = attempts[0] if attempts else {}
    subgoal = attempt.get("subgoal_summary") or {}
    injection = subgoal.get("injection") or {}
    phases = subgoal.get("phases_planned") or []
    phase_bounds: List[Tuple[str, int, int]] = []
    cursor = 0
    for phase in phases:
        count = int(phase.get("steps") or 0)
        phase_bounds.append((str(phase.get("name")), cursor, cursor + count))
        cursor += count

    def phase_at(step_index: int) -> Optional[str]:
        for name, start, end in phase_bounds:
            if start <= step_index < end:
                return name
        return None

    forces = _series(steps, "contact_force_n")
    widths = _series(steps, "gripper_width_m")
    dzs = _series(steps, "object_dz_m")
    dists = _series(steps, "dist_to_goal_m")

    contact_index: Optional[int] = None
    for index, value in enumerate(forces):
        if value is not None and float(value) >= CONTACT_FORCE_N_MIN:
            contact_index = index
            break
    contact_lost_index: Optional[int] = None
    if contact_index is not None:
        for index in range(contact_index + 1, len(forces)):
            value = forces[index]
            if value is not None and float(value) < CONTACT_FORCE_N_MIN:
                contact_lost_index = index
                break

    max_force = _max(forces)
    min_width = _min(widths)
    max_dz = _max(dzs)
    final_dist = dists[-1] if dists else None

    verdict: Dict[str, Any] = {
        "stem": run["stem"],
        "n_steps": len(steps),
        "injection_mode": injection.get("mode"),
        "max_contact_force_n": max_force,
        "min_gripper_width_m": min_width,
        "max_object_dz_m": max_dz,
        "final_dist_to_goal_m": final_dist,
        "contact_established_at": contact_index,
        "contact_lost_at": contact_lost_index,
        "contact_established_phase": phase_at(contact_index) if contact_index is not None else None,
        "contact_lost_phase": phase_at(contact_lost_index) if contact_lost_index is not None else None,
        "summary_final_state": subgoal.get("final_state"),
        "summary_final_success": attempt.get("final_success"),
        "type": None,
        "kind": "nominal",
        "summary_l1": "",
        "evidence": [],
    }

    gripper_closed = min_width is not None and float(min_width) <= GRIPPER_CLOSED_M
    lifted = max_dz is not None and float(max_dz) >= LIFT_DZ_M_MIN
    # 抓握阶段指的是"应当已经建立接触"的那些阶段
    reach = phase_at(len(steps) - 1)

    if contact_index is None:
        # 从未建立接触。这里要和子代理的判定器对齐一个**实质性的语义选择**：
        #
        #   我第一版把"夹爪根本没合上"也判成 X1（抓空）→ 恢复动作 retry_grasp。
        #   但用一个可重放的独立审计（`isaac/replay_judge.py`）核对后发现：
        #   这些 run 的 `min_gripper_width_m ≈ 0.125 m`，而闭合目标约 0.079 m
        #   （指头停在 ±0.0727/±0.0721），**误差 0.0327 m ≫ 容差**——
        #   这是**执行机构失效**（夹爪压根没动），不是"抓取没抓到"。
        #
        #   **重试同一个抓取动作对执行机构失效毫无帮助**，所以把它归成 X1/retry_grasp
        #   在恢复语义上是错的。子代理的判断在这里比我的更对。
        #
        # 处置：执行机构失效**仍然记录**，但 `type=None`（不进 X 分类）、
        # 恢复动作走 `ask_clarification`（升级给人/上层），而不是 retry_grasp。
        # 只有"夹爪确实合上了、但没有接触"才是真正的 X1 抓空。
        if not gripper_closed:
            verdict["type"] = None
            verdict["kind"] = "actuator_gripper_not_closed"
            verdict["summary_l1"] = (
                "夹爪未闭合到抓取宽度（最小宽度 %.4f m，闭合目标约 0.079 m）"
                "→ 执行机构失效，按恢复语义不应归成 X1/重试抓取"
                % float(min_width if min_width is not None else -1)
            )
            verdict["evidence"] = [
                "min_gripper_width_m=%s m（未达到闭合容差）"
                % ("n/a" if min_width is None else "%.4f" % float(min_width)),
                "max_contact_force_n=%.3f N" % float(max_force or 0.0),
                "max_object_dz_m=%.4f m → 物体从未被抬起" % float(max_dz or 0.0),
                "恢复动作应为升级/澄清，而非重试同一抓取",
            ]
            verdict["recovery_action"] = "ask_clarification"
            verdict["recovery_reason"] = "actuator_fault_escalate"
        else:
            verdict["type"] = "X1"
            verdict["kind"] = "grasp_closed_without_contact"
            verdict["summary_l1"] = (
                "夹爪已闭合（最小宽度 %.4f m）但全程未建立有效接触（最大接触力 %.1f N）"
                % (float(min_width or -1), float(max_force or 0.0))
            )
            verdict["evidence"] = [
                "max_contact_force_n=%.3f N (阈值 %.1f)"
                % (float(max_force or 0.0), CONTACT_FORCE_N_MIN),
                "min_gripper_width_m=%.4f m ≤ 闭合容差 %.2f"
                % (float(min_width or -1), GRIPPER_CLOSED_M),
                "max_object_dz_m=%.4f m → 物体从未被抬起" % float(max_dz or 0.0),
                "最后阶段=%s" % reach,
            ]
    elif contact_lost_index is not None and not lifted:
        # 建立过接触、又丢了、而且始终没抬起来 → X2 滑落
        verdict["type"] = "X2"
        verdict["kind"] = "grasp_slip_during_lift"
        verdict["summary_l1"] = (
            "step %d（%s）建立接触后，step %d（%s）接触丢失且物体未被抬起（最大抬升 %.4f m）"
            % (contact_index, verdict["contact_established_phase"],
               contact_lost_index, verdict["contact_lost_phase"], float(max_dz or 0.0))
        )
        verdict["evidence"] = [
            "接触力峰值 %.1f N @step %d" % (float(max_force or 0.0), contact_index),
            "step %d 起接触力 < %.1f N" % (contact_lost_index, CONTACT_FORCE_N_MIN),
            "max_object_dz_m=%.4f m < 抬升阈值 %.2f" % (float(max_dz or 0.0), LIFT_DZ_M_MIN),
            "夹爪最小宽度 %.4f m" % float(min_width if min_width is not None else -1),
        ]
    elif lifted and final_dist is not None and float(final_dist) > PLACE_DIST_M_MAX:
        # 抬起来了但最终离目标太远 → X3 放错
        verdict["type"] = "X3"
        verdict["kind"] = "place_off_target"
        verdict["summary_l1"] = (
            "物体已被抬起（最大抬升 %.4f m），但最终水平距目标 %.4f m 超过阈值 %.2f m"
            % (float(max_dz or 0.0), float(final_dist), PLACE_DIST_M_MAX)
        )
        verdict["evidence"] = [
            "max_object_dz_m=%.4f m" % float(max_dz or 0.0),
            "final_dist_to_goal_m=%.4f m (阈值 %.2f)" % (float(final_dist), PLACE_DIST_M_MAX),
            "max_contact_force_n=%.1f N" % float(max_force or 0.0),
        ]
    else:
        verdict["kind"] = "nominal"
        verdict["summary_l1"] = "未判定为 X1/X2/X3"
        verdict["evidence"] = [
            "max_contact_force_n=%.1f N" % float(max_force or 0.0),
            "max_object_dz_m=%.4f m" % float(max_dz or 0.0),
            "final_dist_to_goal_m=%s" % final_dist,
        ]
    return verdict


RECOVERY_FOR_TYPE = {
    "X1": ("retry_grasp", "grasp_empty"),
    "X2": ("change_grasp_point", "grasp_slip"),
    "X3": ("retry_grasp", "place_off_target"),
}


def to_event(verdict: Dict[str, Any], source_file: str) -> FailureEvent:
    failure_type = verdict["type"]
    # `type=None` 时允许带一个"非 X 分类"的恢复动作（契约只在 type 非空时校验菜单），
    # 用于记录执行机构失效这类**不该用 X 分类**、但仍需处置的事件。
    default_action, default_reason = RECOVERY_FOR_TYPE.get(failure_type or "", ("none", "nominal"))
    if failure_type is None and verdict.get("recovery_action"):
        default_action = str(verdict["recovery_action"])
        default_reason = str(verdict.get("recovery_reason") or default_reason)
    action, reason_code = default_action, default_reason
    # 每个 trace 文件是一次独立的尝试 → 用 stem 的稳定哈希当 episode_id，
    # 否则多个 X1 的 (scene, episode, step) 会撞成同一个事件 id（我第一版就是这个 bug）。
    stem = str(verdict.get("stem") or "unknown")
    episode_id = int(hashlib.sha1(stem.encode("utf-8")).hexdigest()[:6], 16) % 100000
    marker_step = verdict.get("contact_lost_at") or verdict.get("contact_established_at")
    signals = Signals(
        gripper_width_m=verdict.get("min_gripper_width_m"),
        contact_force_n=verdict.get("max_contact_force_n"),
        object_dz_m=verdict.get("max_object_dz_m"),
        dist_to_goal_m=verdict.get("final_dist_to_goal_m"),
        extra={
            "trace_stem": stem,
            "injection_mode": verdict.get("injection_mode"),
            "contact_established_at": verdict.get("contact_established_at"),
            "contact_lost_at": verdict.get("contact_lost_at"),
            "n_steps": verdict.get("n_steps"),
            "source_file": os.path.basename(source_file),
            "detector": "route1.experiments.manip_events_from_trace",
        },
    )
    return FailureEvent.create(
        source="manip",
        type=failure_type,
        scene_id="isaac_tabletop",
        episode_id=episode_id,
        step=int(marker_step if marker_step is not None else 0),
        action_step=int(verdict.get("n_steps") or 0),
        instruction="pick the object on the table and place it at the goal",
        signals=signals,
        attribution=Attribution(
            kind=verdict["kind"],
            summary_l1="[%s] %s" % (stem, verdict["summary_l1"]),
            summary_l2="judged from the per-step subgoal trace",
            evidence=list(verdict.get("evidence") or []),
            dominant_signal="contact_force_n" if failure_type in ("X1", "X2") else "dist_to_goal_m",
            source="trace",
        ),
        recovery=Recovery(reason_code=reason_code, action=action,
                          budget_used=0, budget_left=1),
        outcome=Outcome(recovered=None, post_score=None,
                        steps_to_recover=None),
    )


def scan(run_dir: str) -> Dict[str, Any]:
    stems = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*_events.jsonl"))):
        stem = os.path.basename(path)[: -len("_events.jsonl")]
        stems.append(stem)
    verdicts: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for stem in stems:
        run = load_run(run_dir, stem)
        if run is None:
            skipped.append(stem)
            continue
        verdicts.append(detect(run))
    return {"run_dir": run_dir, "stems": stems, "verdicts": verdicts, "skipped": skipped}


def _cmd_scan(args: argparse.Namespace) -> int:
    report = scan(args.run_dir)
    print("%-16s %-12s %-30s %9s %9s %9s %9s %6s"
          % ("stem", "type", "kind", "maxForce", "minWidth", "maxDz", "finalDist", "steps"))
    for v in report["verdicts"]:
        def f(value: Any, fmt: str = "%.4f") -> str:
            return "-" if value is None else fmt % float(value)
        print("%-16s %-12s %-30s %9s %9s %9s %9s %6d"
              % (v["stem"], v["type"] or "-", v["kind"],
                 f(v["max_contact_force_n"], "%.1f"), f(v["min_gripper_width_m"]),
                 f(v["max_object_dz_m"]), f(v["final_dist_to_goal_m"]), v["n_steps"]))
    if report["skipped"]:
        print("\n跳过空/坏文件:", ", ".join(report["skipped"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            for v in report["verdicts"]:
                # 写"判定为失败"的（type 非空），以及"被执行机构失效拦下、但需要处置"的
                if v["type"] or v.get("recovery_action"):
                    source = os.path.join(args.run_dir, v["stem"] + "_events.jsonl")
                    handle.write(to_event(v, source).to_json() + "\n")
        print("wrote %s（X 分类 + 需处置的非 X 事件）" % args.out)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("wrote %s" % args.report)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从 Isaac 逐步 trace 独立判定 X1/X2/X3")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None, help="契约事件 JSONL 输出路径")
    p.add_argument("--report", default=None, help="完整诊断 JSON 输出路径")
    p.set_defaults(func=_cmd_scan)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

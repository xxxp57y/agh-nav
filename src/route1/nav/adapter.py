"""导航侧适配器：把 InternNav 的既有日志**零改造**映射成 FailureEvent 流。

字段对照（`closed_loop_steps_rank*.jsonl` → `FailureEvent`）
----------------------------------------------------------
    feedback_score           → signals.feedback_score
    trajectory_drift         → signals.trajectory_drift
    plan_age                 → signals.plan_age
    collision_count 增量     → signals.collision_delta
    steps_since_last_s2      → signals.steps_since_last_s2
    replan_reason            → type (经 REASON_TO_FAILURE_TYPE) + recovery.reason_code
    trigger_kind             → trigger_kind
    system2_call_count       → recovery.budget_used
    progress.json 的指令     → instruction
    episode_summary.success  → outcome.recovered（回合级回填）

因为原日志里**没有**语义信号，`semantic_*` 一律为 None——**不补 0**。

Python 3.9 兼容。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..contracts import Attribution, FailureEvent, Outcome, Recovery, Signals
from ..bus import read_jsonl
from .replay import REASON_TO_FAILURE_TYPE, load_run


def _recovery_action_for(reason: Optional[str], failure_type: Optional[str]) -> str:
    if reason == "recovery":
        return "replan"
    if reason == "early_drift":
        return "replan"
    if reason == "max_plan_age":
        return "replan"
    if reason == "feedback_high":
        return "replan"
    if reason == "external":
        return "replan"
    return "none"


def events_from_run(
    run_dir: str,
    include_non_acting: bool = False,
    backfill_outcome: bool = True,
) -> List[FailureEvent]:
    """把一个 run 目录映射成 FailureEvent 列表。

    `include_non_acting=True` 时连"触发但没真正重规划"的决策（cooldown /
    proactive_budget_exhausted 等）也导出——做决策层分析时需要它们。
    """
    run = load_run(run_dir)
    instructions = run["instruction_by_episode"]
    rows = run["system1"]

    # 回合成功与否（回填 outcome.recovered）
    success_by_episode: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in run["progress"]:
        success_by_episode[
            (str(row.get("scene_id")), int(row.get("episode_id")))
        ] = row

    events: List[FailureEvent] = []
    last_collision: Dict[Tuple[str, int], int] = {}

    for index, row in enumerate(rows):
        key = (str(row.get("scene_id")), int(row.get("episode_id")))
        reason = row.get("replan_reason")
        failure_type = REASON_TO_FAILURE_TYPE.get(reason or "")
        acting = bool(row.get("should_replan"))
        if failure_type is None and not acting:
            if not include_non_acting:
                continue
        collision_count = int(row.get("collision_count") or 0)
        previous_collision = last_collision.get(key, collision_count)
        delta = max(0, collision_count - previous_collision)
        last_collision[key] = collision_count

        # 前瞻 outcome：后续 5 个决策内分数是否下降
        outcome = Outcome()
        if backfill_outcome:
            future_scores = [
                r.get("feedback_score") for r in rows[index + 1: index + 6]
                if r.get("feedback_score") is not None
            ]
            if future_scores:
                post = min(float(value) for value in future_scores)
                outcome.post_score = post
                current = row.get("feedback_score")
                if current is not None:
                    outcome.recovered = bool(post < float(current))
            episode_row = success_by_episode.get(key)
            if episode_row is not None:
                if outcome.recovered is None:
                    outcome.recovered = bool(episode_row.get("success"))

        event = FailureEvent.create(
            source="nav",
            type=failure_type,
            scene_id=key[0],
            episode_id=key[1],
            step=row.get("step"),
            action_step=row.get("action_step"),
            instruction=instructions.get(key),
            trigger_kind=row.get("trigger_kind"),
            signals=Signals(
                feedback_score=row.get("feedback_score"),
                trajectory_drift=row.get("trajectory_drift"),
                plan_age=row.get("plan_age"),
                collision_delta=delta,
                steps_since_last_s2=row.get("steps_since_last_s2"),
                semantic_instance_margin=None,
                semantic_verdict=None,
                semantic_confidence=None,
            ),
            attribution=Attribution(
                kind=(reason or "below_trigger"),
                summary_l1=_summary_for(reason, row),
                summary_l2="",
                evidence=_evidence_for(row),
                dominant_signal=(
                    reason.split(":", 1)[1] if reason and ":" in reason else None
                ),
                source="log",
            ),
            recovery=Recovery(
                reason_code=reason or "none",
                action=_recovery_action_for(reason, failure_type),
                budget_used=int(row.get("system2_call_count") or 0),
                budget_left=None if row.get("system2_call_count") is None else 0,
                bypassed_cooldown=bool(reason == "recovery"),
            ),
            outcome=outcome,
        )
        events.append(event)
    return events


def _summary_for(reason: Optional[str], row: Dict[str, Any]) -> str:
    trigger = row.get("trigger_kind")
    drift = row.get("trajectory_drift")
    if reason == "early_drift":
        return "早漂移触发（%s，drift=%s m）：计划轨迹与执行轨迹开始分离" % (
            trigger or "unknown",
            "%.3f" % drift if drift is not None else "n/a",
        )
    if reason == "recovery":
        return "无可用动作：System-2 计划里没有可执行的下一步"
    if reason == "feedback_high":
        return "加权反馈分数越过迟滞高阈（score=%s）" % (row.get("feedback_score"),)
    if reason == "max_plan_age":
        return "计划年龄达上限（plan_age=%s）" % (row.get("plan_age"),)
    if reason == "external":
        return "外部触发（temporal 策略或人在环）"
    if reason == "cooldown":
        return "被冷却抑制：距上次 System-2 调用不足 min_replan_interval"
    if reason == "proactive_budget_exhausted":
        return "早触发命中但 S2 预算已耗尽，被预算挡住"
    if reason == "no_action_replan_limit":
        return "连续无动作重规划达上限，刹车"
    return "未触发（%s）" % (reason or "below_trigger")


def _evidence_for(row: Dict[str, Any]) -> List[str]:
    evidence = []
    if row.get("trajectory_drift") is not None:
        evidence.append("trajectory_drift=%.4f m" % float(row["trajectory_drift"]))
    if row.get("plan_age") is not None:
        evidence.append("plan_age=%s" % row["plan_age"])
    if row.get("steps_since_last_s2") is not None:
        evidence.append("steps_since_last_s2=%s" % row["steps_since_last_s2"])
    if row.get("feedback_score") is not None:
        evidence.append("feedback_score=%.4f" % float(row["feedback_score"]))
    if row.get("distance_to_goal") is not None:
        evidence.append("distance_to_goal=%.3f m" % float(row["distance_to_goal"]))
    if row.get("rgb_path"):
        evidence.append("frame=%s" % os.path.basename(str(row["rgb_path"])))
    return evidence


def stream_from_run(run_dir: str, **kwargs: Any) -> Iterator[FailureEvent]:
    for event in events_from_run(run_dir, **kwargs):
        yield event

"""导航侧离线重放 / 回归验证 / 标签构造。

本模块回答三个问题，全部只用**已有日志**（不需要跑仿真、不需要 GPU）：

1. **等价性**：决策层 v3 在"语义关闭 + 恢复动作关闭"时，能否在真实 50 回合日志上
   逐点复现 InternNav 原调度器的输出？（`verify_scheduler_parity`）
2. **第 5 信号不污染核心**：`PlanFeedback5` 与 `PlanFeedback` 在同输入下是否逐位一致？
   （`verify_feedback_equivalence`）
3. **触发质量**：按 `analyze_budgeted_closed_loop.py:70-79` 的口径复算 P@1/P@3/P@5，
   并导出"几何 Monitor 的候选集 + 未来失败标签"，供语义 Monitor 做同口径对照。
   （`trigger_precision` / `build_monitor_dataset`）

Python 3.9 兼容。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .scheduler_v3 import REASON_TO_FAILURE_TYPE, DecisionLayerV3, SemanticConfig
from ..bus import read_jsonl

#: 与 `analyze_budgeted_closed_loop.py` 一致的前瞻窗口
PRECURSOR_HORIZONS: Tuple[int, ...] = (1, 3, 5)


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #


def run_event_paths(run_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(run_dir, "closed_loop_steps_rank*.jsonl")))


def load_run(run_dir: str) -> Dict[str, Any]:
    """把一个 run 目录读成结构化对象。"""
    events: List[Dict[str, Any]] = []
    for path in run_event_paths(run_dir):
        events.extend(read_jsonl(path))
    progress = read_jsonl(os.path.join(run_dir, "progress.json"))
    instruction_by_episode: Dict[Tuple[str, int], str] = {}
    for row in progress:
        key = (str(row.get("scene_id")), int(row.get("episode_id")))
        if row.get("episode_instruction"):
            instruction_by_episode[key] = str(row["episode_instruction"])
    config: Dict[str, Any] = {}
    for row in events:
        if row.get("event") == "episode_summary" and row.get("closed_loop_config"):
            config = row["closed_loop_config"]
            break
    return {
        "run_dir": run_dir,
        "events": events,
        "progress": progress,
        "instruction_by_episode": instruction_by_episode,
        "closed_loop_config": config,
        "system1": [row for row in events if row.get("event") == "system1"],
        "system2": [row for row in events if row.get("event") == "system2"],
        "early_outcomes": [row for row in events if row.get("event") == "early_drift_outcome"],
    }


def reconstruct_scheduler_kwargs(closed_loop_config: Dict[str, Any]) -> Dict[str, Any]:
    """从日志里落盘的 `closed_loop_config['scheduler']` 还原调度器构造参数。

    `habitat_vln_evaluator.py:442` 把 `self._closed_loop_scheduler_kwargs` 原样写进
    日志，因此这里**不需要**重推 `early_mode` 等推导逻辑。
    """
    scheduler = dict(closed_loop_config.get("scheduler") or {})
    if not scheduler:
        raise ValueError("closed_loop_config['scheduler'] 缺失，无法还原调度器参数")
    return scheduler


def reconstruct_feedback_kwargs(closed_loop_config: Dict[str, Any]) -> Dict[str, Any]:
    """对应 `habitat_vln_evaluator.py:395-407`。"""
    feedback = dict(closed_loop_config.get("feedback") or {})
    scheduler = dict(closed_loop_config.get("scheduler") or {})
    trajectory = dict(closed_loop_config.get("trajectory_consistency") or {})
    return {
        "w_uncertainty": feedback.get("w_uncertainty", 0.35),
        "w_drift": feedback.get("w_drift", 0.35),
        "w_age": feedback.get("w_age", 0.15),
        "w_risk": feedback.get("w_risk", 0.15),
        "renormalize_active_weights": feedback.get("renormalize_active_weights", True),
        "drift_scale_m": feedback.get("drift_scale_m", 0.50),
        "uncertainty_scale_m2": feedback.get("uncertainty_scale_m2", 0.25),
        "replan_threshold": scheduler.get("high_threshold", 0.70),
        "max_plan_age_steps": scheduler.get("max_plan_age_steps", 20),
        "num_points": trajectory.get("num_points", 33),
        "fallback_prefix_points": trajectory.get("fallback_prefix_points", 4),
    }


# --------------------------------------------------------------------------- #
# 1) 调度器等价性
# --------------------------------------------------------------------------- #

_COMPARED_FIELDS = (
    "should_replan",
    "reason",
    "plan_age",
    "current_step",
    "trigger_kind",
    "consecutive_drift_mid",
    "budget_exhausted",
)

#: 调度器字段名 → 日志里的字段名（日志把 `reason` 记成 `replan_reason`）
_LOG_KEY = {"reason": "replan_reason"}


#: 只有这两个模式会调用 `scheduler.record_system2_call()`
#: （`habitat_vln_evaluator.py:1577`），其它模式下 `steps_since_last_s2` 恒为 None，
#: 于是 `elapsed=None` → `in_cooldown=False`。这条差异曾让我的重放凭空多出 cooldown。
S2_INTERVAL_MODES = ("budgeted_drift", "temporal_il")


def _logged(row: Dict[str, Any], field: str) -> Any:
    return row.get(_LOG_KEY.get(field, field))


def verify_scheduler_parity(
    run_dir: str,
    resync_last_s2: bool = True,
    limit_episodes: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """用日志逐点重放，比较 InternNav 原调度器的输出与日志落盘值。

    为什么默认 `resync_last_s2=True`
    -------------------------------
    实测发现：`system2` 事件在日志里的写盘时机与**真实调用时机不同序**——严格按
    文件顺序回放 S2 调用会直接撞上 `scheduler.py:134 current_step must not move
    backwards`。因此正确做法是用每个 `system1` 事件自己记录的
    `steps_since_last_s2`（它就是调度器算出来的值）反推 `last_s2_call_step`。

    两种模式都会跑，并都统计 S2 事件的乱序次数——乱序次数本身就是一条值得报告的事实。
    """
    run = load_run(run_dir)
    if not run["system1"]:
        raise ValueError("run 目录里没有 system1 事件: %s" % run_dir)
    kwargs = reconstruct_scheduler_kwargs(run["closed_loop_config"])

    from internnav.model.closed_loop.scheduler import AdaptiveReplanScheduler  # type: ignore
    scheduler = AdaptiveReplanScheduler(**kwargs)

    allow_feedback_replan_base = bool(
        run["closed_loop_config"].get("feedback_replan_enabled", False)
    )
    mode = str(run["closed_loop_config"].get("mode") or "")
    temporal_cfg = dict(run["closed_loop_config"].get("temporal_policy") or {})
    temporal_enabled = bool(temporal_cfg.get("enabled", False))
    temporal_threshold = float(temporal_cfg.get("threshold", 0.5))
    tracks_s2_interval = mode in S2_INTERVAL_MODES

    seen_episodes: List[Tuple[str, int]] = []
    episode_order: Dict[Tuple[str, int], int] = {}
    compared_episodes: set = set()

    mismatches: List[Dict[str, Any]] = []
    compared = 0
    resync_count = 0
    out_of_order_s2 = 0
    temporal_triggers = 0
    last_episode_seen: Optional[Tuple[str, int]] = None

    for row in run["events"]:
        if row.get("event") == "system2":
            if tracks_s2_interval:
                step = int(row.get("action_step", 0) or 0)
                if scheduler.last_s2_call_step is not None and step < scheduler.last_s2_call_step:
                    out_of_order_s2 += 1
                    continue
                scheduler.record_system2_call(step)
            continue
        if row.get("event") != "system1":
            continue
        key = (str(row.get("scene_id")), int(row.get("episode_id")))
        if key not in episode_order:
            episode_order[key] = len(episode_order)
            seen_episodes.append(key)
        if limit_episodes is not None and episode_order[key] >= limit_episodes:
            continue
        if last_episode_seen is not None and key != last_episode_seen:
            scheduler.reset()
        last_episode_seen = key

        current_step = int(row.get("action_step") or 0)
        logged_since = row.get("steps_since_last_s2")
        if resync_last_s2 and logged_since is not None:
            target = current_step - int(logged_since)
            if scheduler.last_s2_call_step != target:
                scheduler.last_s2_call_step = target
                resync_count += 1

        allow_feedback_replan = allow_feedback_replan_base
        temporal_score = row.get("temporal_policy_score")
        external_trigger = bool(
            temporal_enabled and temporal_score is not None
            and float(temporal_score) >= temporal_threshold
        )
        if external_trigger:
            temporal_triggers += 1

        result = scheduler.step(
            feedback_score=float(row.get("feedback_score") or 0.0),
            plan_age=int(row.get("plan_age") or 0),
            current_step=current_step,
            has_executable_action=bool(row.get("has_executable_action", True)),
            allow_feedback_replan=allow_feedback_replan,
            trajectory_drift=row.get("trajectory_drift"),
            system2_calls_used=int(row.get("system2_call_count") or 0),
            external_trigger=external_trigger,
            external_reason="temporal_il",
        )
        compared += 1
        compared_episodes.add(key)
        logged = {name: _logged(row, name) for name in _COMPARED_FIELDS}
        for name in _COMPARED_FIELDS:
            got = result.get(name)
            want = logged[name]
            if name in ("plan_age", "current_step", "consecutive_drift_mid"):
                if want is None:
                    continue
                got, want = int(got), int(want)
            elif name == "budget_exhausted":
                got, want = bool(got), bool(want)
            if got != want:
                mismatches.append(
                    {
                        "scene_id": key[0],
                        "episode_id": key[1],
                        "action_step": current_step,
                        "field": name,
                        "logged": want,
                        "replayed": got,
                    }
                )
                if verbose and len(mismatches) <= 20:
                    print("  MISMATCH", mismatches[-1])

    by_field: Dict[str, int] = {}
    for item in mismatches:
        by_field[item["field"]] = by_field.get(item["field"], 0) + 1
    return {
        "run_dir": run_dir,
        "mode": "resync_last_s2" if resync_last_s2 else "strict_stream",
        "scheduler_kwargs": kwargs,
        "events_compared": compared,
        "episodes_in_stream": len(seen_episodes),
        "episodes_compared": len(compared_episodes),
        "mismatches": len(mismatches),
        "mismatch_rate": (len(mismatches) / compared) if compared else None,
        "mismatch_by_field": by_field,
        "resync_count": resync_count,
        "out_of_order_system2_events": out_of_order_s2,
        "temporal_external_triggers": temporal_triggers,
        "tracks_s2_interval": tracks_s2_interval,
        "examples": mismatches[:15],
    }


def verify_feedback_equivalence(
    closed_loop_config: Dict[str, Any],
    n_samples: int = 200,
    seed: int = 20261009,
) -> Dict[str, Any]:
    """`PlanFeedback5`（语义关闭） vs `PlanFeedback` 的逐位等价性测试。

    输入刻意覆盖：轨迹可用/不可用、漂移大小、年龄边界、碰撞风险有无、
    以及"只有 plan_age 可用"这种会触发单信号顶满的退化情形。
    """
    from internnav.model.closed_loop.feedback import PlanFeedback  # type: ignore
    from .feedback5 import PlanFeedback5

    kwargs = reconstruct_feedback_kwargs(closed_loop_config)
    base = PlanFeedback(**kwargs)
    semi = PlanFeedback5(semantic_enabled=False, **kwargs)

    rng = np.random.RandomState(seed)
    checked = 0
    diffs: List[Dict[str, Any]] = []
    for index in range(n_samples):
        case = index % 6
        plan_age = int(rng.randint(0, 25))
        collision_risk = None if case in (0, 4) else float(rng.rand())
        uncertainty = None if case in (1, 5) else float(rng.rand() * 0.4)
        if case in (2, 3):
            current: Any = None
            previous: Any = None
        else:
            n_pts = int(rng.randint(4, 40))
            previous = rng.rand(n_pts, 2) * 5.0
            jitter = rng.rand(n_pts, 2) * (0.6 if case == 1 else 0.05)
            current = previous + jitter
        got_a = base.evaluate(current, previous, plan_age, uncertainty, None, collision_risk)
        got_b = semi.evaluate(current, previous, plan_age, uncertainty, None, collision_risk)
        checked += 1
        for field in ("score", "should_replan", "reason", "replan_reason", "alignment_mode"):
            if got_a[field] != got_b[field]:
                diffs.append(
                    {"case": case, "field": field, "base": got_a[field], "v5": got_b[field]}
                )
    return {
        "samples": checked,
        "differences": len(diffs),
        "examples": diffs[:10],
        "verdict": "IDENTICAL" if not diffs else "DIFFERENT",
    }


def verify_decision_layer_parity(run_dir: str, limit_episodes: Optional[int] = None) -> Dict[str, Any]:
    """决策层 v3（全部新功能关闭） vs 日志落盘值。

    驱动方式与 `verify_scheduler_parity` 完全一致（含 `record_system2_call` 的模式
    限制、`temporal_policy_score` 外部触发），否则测的就不是同一个东西。
    """
    run = load_run(run_dir)
    kwargs = reconstruct_scheduler_kwargs(run["closed_loop_config"])
    config = run["closed_loop_config"]
    allow_feedback_replan = bool(config.get("feedback_replan_enabled", False))
    mode = str(config.get("mode") or "")
    tracks_s2_interval = mode in S2_INTERVAL_MODES
    temporal_cfg = dict(config.get("temporal_policy") or {})
    temporal_enabled = bool(temporal_cfg.get("enabled", False))
    temporal_threshold = float(temporal_cfg.get("threshold", 0.5))

    layer = DecisionLayerV3(
        scheduler_kwargs=kwargs,
        semantic=SemanticConfig(enabled=False),
        enable_recovery_actions=False,
        enable_semantic_channel=False,
    )

    episode_order: Dict[Tuple[str, int], int] = {}
    mismatches: List[Dict[str, Any]] = []
    compared = 0
    out_of_order_s2 = 0
    last_episode: Optional[Tuple[str, int]] = None

    for row in run["events"]:
        if row.get("event") == "system2":
            if not tracks_s2_interval:
                continue
            step = int(row.get("action_step", 0) or 0)
            if layer.scheduler.last_s2_call_step is not None and \
                    step < layer.scheduler.last_s2_call_step:
                out_of_order_s2 += 1
                continue
            layer.record_system2_call(step)
            continue
        if row.get("event") != "system1":
            continue
        key = (str(row.get("scene_id")), int(row.get("episode_id")))
        if key not in episode_order:
            episode_order[key] = len(episode_order)
        if limit_episodes is not None and episode_order[key] >= limit_episodes:
            continue
        if last_episode is not None and key != last_episode:
            layer.reset_episode()
        last_episode = key
        logged_since = row.get("steps_since_last_s2")
        current_step = int(row.get("action_step") or 0)
        if logged_since is not None:
            layer.scheduler.last_s2_call_step = current_step - int(logged_since)
        temporal_score = row.get("temporal_policy_score")
        external_trigger = bool(
            temporal_enabled and temporal_score is not None
            and float(temporal_score) >= temporal_threshold
        )
        decision = layer.decide(
            feedback_score=float(row.get("feedback_score") or 0.0),
            plan_age=int(row.get("plan_age") or 0),
            current_step=current_step,
            has_executable_action=bool(row.get("has_executable_action", True)),
            allow_feedback_replan=allow_feedback_replan,
            trajectory_drift=row.get("trajectory_drift"),
            system2_calls_used=int(row.get("system2_call_count") or 0),
            external_trigger=external_trigger,
            external_reason="temporal_il",
            emit=False,
        )
        compared += 1
        for name in _COMPARED_FIELDS:
            want = _logged(row, name)
            if want is None:
                continue
            got = decision.get(name)
            if name in ("plan_age", "current_step", "consecutive_drift_mid"):
                got, want = int(got), int(want)
            if got != want:
                mismatches.append(
                    {"episode_id": key[1], "action_step": row.get("action_step"),
                     "field": name, "logged": want, "replayed": got}
                )
    by_field: Dict[str, int] = {}
    for item in mismatches:
        by_field[item["field"]] = by_field.get(item["field"], 0) + 1
    return {
        "run_dir": run_dir,
        "events_compared": compared,
        "episodes_compared": len(episode_order),
        "mismatches": len(mismatches),
        "mismatch_by_field": by_field,
        "out_of_order_system2_events": out_of_order_s2,
        "examples": mismatches[:10],
        "verdict": "IDENTICAL" if not mismatches else "DIFFERENT",
    }


# --------------------------------------------------------------------------- #
# 2) 触发质量 P@k 与标签集
# --------------------------------------------------------------------------- #


def label_future_failures(
    system1_rows: Sequence[Dict[str, Any]],
    horizons: Sequence[int] = PRECURSOR_HORIZONS,
    require_model_invoked: bool = True,
) -> List[Dict[str, Any]]:
    """给每个决策点打「未来 k 个决策内是否出现无可执行动作」的标签。

    严格复刻 `habitat_vln_evaluator.py:1107-1119 + 2216-2227` 的口径，
    三个容易搞错的细节：
      1. `observed_future_decisions` **只统计 `feedback_result is not None` 的决策**
         （日志里表现为 `feedback_score is not None`），且**不跨回合**；
      2. 若回合在 k 个决策内结束且期间没有失败，标签是 **None（删失）而不是 False**
         —— `analyze_budgeted_closed_loop.py:74` 会把 None 从分母里剔除。
         把删失当 False 会系统性低估 P@k（我第一版就是这么错的）；
      3. `require_model_invoked`：上游条件是 `model_invoked and has_executable_action
         is False`。但实测发现**去掉 `model_invoked` 与日志自带标签吻合得多**
         （见 `label_agreement_with_log`），说明上游落盘的 `model_invoked` 与做判定时
         用的那个变量并不总是同一个。两种口径都保留，由数据决定用哪个。
    """
    n = len(system1_rows)
    labels: List[Dict[str, Any]] = []
    for index, row in enumerate(system1_rows):
        episode_key = (str(row.get("scene_id")), int(row.get("episode_id") or -1))
        first_failure: Optional[int] = None
        observed = 0
        for j in range(index + 1, n):
            nxt = system1_rows[j]
            if (str(nxt.get("scene_id")), int(nxt.get("episode_id") or -1)) != episode_key:
                break
            if nxt.get("feedback_score") is None:
                continue
            observed += 1
            if nxt.get("has_executable_action") is False and (
                (not require_model_invoked) or nxt.get("model_invoked")
            ):
                first_failure = observed
                break

        entry: Dict[str, Any] = {
            "index": index,
            "scene_id": row.get("scene_id"),
            "episode_id": row.get("episode_id"),
            "action_step": row.get("action_step"),
            "system1_decision_index": row.get("system1_decision_index"),
            "score": row.get("feedback_score"),
            "drift": row.get("trajectory_drift"),
            "plan_age": row.get("plan_age"),
            "collision_risk": row.get("collision_risk"),
            "uncertainty": row.get("uncertainty"),
            "has_executable_action": row.get("has_executable_action"),
            "model_invoked": row.get("model_invoked"),
            "reason": row.get("replan_reason"),
            "trigger_kind": row.get("trigger_kind"),
            "should_replan": row.get("should_replan"),
            "rgb_path": row.get("rgb_path"),
            "distance_to_goal": row.get("distance_to_goal"),
            "steps_since_last_s2": row.get("steps_since_last_s2"),
            "observed_future_decisions": observed,
            "first_no_action_offset": first_failure,
        }
        for horizon in horizons:
            if first_failure is not None:
                entry["no_action_within_%d" % horizon] = bool(first_failure <= horizon)
            elif observed >= horizon:
                entry["no_action_within_%d" % horizon] = False
            else:
                entry["no_action_within_%d" % horizon] = None
        labels.append(entry)
    return labels


def label_agreement_with_log(run_dir: str,
                             horizons: Sequence[int] = PRECURSOR_HORIZONS) -> Dict[str, Any]:
    """用日志里 `early_drift_outcome` 事件自带的标签，校验我复刻的标签计算。

    两种口径都测（`model_invoked` 必需 / 不必需），返回吻合度更高的那个，并把
    另一个的吻合度一起报告出来——这本身就是一条关于上游日志自洽性的发现。
    """
    run = load_run(run_dir)
    report: Dict[str, Any] = {"run_dir": run_dir, "logged_events": len(run["early_outcomes"])}
    for require in (True, False):
        labels = label_future_failures(run["system1"], horizons, require_model_invoked=require)
        index_by_identity: Dict[Tuple[str, int, Any], Dict[str, Any]] = {}
        for entry in labels:
            index_by_identity[
                (str(entry["scene_id"]), int(entry["episode_id"] or -1), entry["action_step"])
            ] = entry

        checked = 0
        agree = 0
        disagreements: List[Dict[str, Any]] = []
        for record in run["early_outcomes"]:
            key = (str(record.get("scene_id")), int(record.get("episode_id") or -1),
                   record.get("action_step"))
            mine = index_by_identity.get(key)
            if mine is None:
                disagreements.append({"missing": True, "key": key})
                continue
            checked += 1
            detail: Dict[str, Any] = {}
            for horizon in horizons:
                name = "no_action_within_%d" % horizon
                if record.get(name) != mine.get(name):
                    detail[name] = {"log": record.get(name), "ours": mine.get(name)}
            if mine.get("first_no_action_offset") != record.get("first_no_action_offset"):
                detail["first_no_action_offset"] = {
                    "log": record.get("first_no_action_offset"),
                    "ours": mine.get("first_no_action_offset"),
                }
            if not detail:
                agree += 1
            else:
                disagreements.append({"key": key, "detail": detail})
        variant = {
            "require_model_invoked": require,
            "checked": checked,
            "agree": agree,
            "agreement_rate": (agree / checked) if checked else None,
            "disagreements": disagreements[:10],
        }
        report["variant_model_invoked_required" if require else "variant_any_no_action"] = variant

    best_key = max(
        ("variant_model_invoked_required", "variant_any_no_action"),
        key=lambda name: (report[name]["agreement_rate"] or -1.0),
    )
    report["best_variant"] = best_key
    report["best_agreement_rate"] = report[best_key]["agreement_rate"]
    report["checked"] = report[best_key]["checked"]
    report["agree"] = report[best_key]["agree"]
    report["disagreements"] = report[best_key]["disagreements"]
    return report


def trigger_precision(
    run_dir: str,
    trigger_predicate: str = "early_drift",
    use_logged_labels: bool = False,
    require_model_invoked: Optional[bool] = None,
) -> Dict[str, Any]:
    """复算 P@1/P@3/P@5（删失样本从分母剔除，与上游 `analyze_*` 一致）。

    `trigger_predicate`:
      - `early_drift`：原报表口径（`replan_reason == 'early_drift'`）
      - `any_replan` ：任何真正重规划的决策
      - `feedback_high`：迟滞高阈触发
      - `cooldown` / `proactive_budget_exhausted` / … 直接写 reason 名
      - `semantic`：语义风险 ≥ 阈值（需要日志里有语义信号，当前为 0 条）

    `use_logged_labels=True` 时直接用 `early_drift_outcome` 自带的标签（仅对
    `early_drift` 有效），用于和上游 REPORT.md 对账。
    """
    run = load_run(run_dir)
    rows = run["system1"]

    if use_logged_labels:
        events = run["early_outcomes"]
        result: Dict[str, Any] = {
            "run_dir": run_dir,
            "predicate": trigger_predicate,
            "label_source": "logged_early_drift_outcome",
            "decisions": len(rows),
            "triggers": len(events),
        }
        for horizon in PRECURSOR_HORIZONS:
            name = "no_action_within_%d" % horizon
            observed = [bool(record[name]) for record in events if record.get(name) is not None]
            result["precision_at_%d" % horizon] = float(np.mean(observed)) if observed else None
            result["observed_at_%d" % horizon] = len(observed)
        return result

    if require_model_invoked is None:
        agreement = label_agreement_with_log(run_dir)
        require_model_invoked = agreement["best_variant"] == "variant_model_invoked_required"
    labels = label_future_failures(rows, require_model_invoked=require_model_invoked)

    def is_trigger(entry: Dict[str, Any]) -> bool:
        if trigger_predicate == "early_drift":
            return entry.get("reason") == "early_drift"
        if trigger_predicate == "any_replan":
            return bool(entry.get("should_replan"))
        if trigger_predicate == "feedback_high":
            return entry.get("reason") == "feedback_high"
        return entry.get("reason") == trigger_predicate

    triggered = [entry for entry in labels if is_trigger(entry)]
    result = {
        "run_dir": run_dir,
        "predicate": trigger_predicate,
        "label_source": "recomputed",
        "require_model_invoked": require_model_invoked,
        "decisions": len(labels),
        "triggers": len(triggered),
    }
    for horizon in PRECURSOR_HORIZONS:
        name = "no_action_within_%d" % horizon
        observed = [bool(entry[name]) for entry in triggered if entry.get(name) is not None]
        result["precision_at_%d" % horizon] = float(np.mean(observed)) if observed else None
        result["observed_at_%d" % horizon] = len(observed)
    base = [entry["no_action_within_1"] for entry in labels if entry.get("no_action_within_1") is not None]
    result["baseline_rate_at_1"] = float(np.mean([bool(v) for v in base])) if base else None
    result["baseline_observed_at_1"] = len(base)
    return result


def build_monitor_dataset(run_dir: str, out_path: Optional[str] = None) -> Dict[str, Any]:
    """导出「几何 Monitor」的候选决策集 + 未来失败标签（语义 Monitor 的同口径对照集）。

    每条记录包含：决策点身份、4 个几何原始信号与归一化分量、指令原文、帧路径（可能为
    None）、以及 `no_action_within_{1,3,5}` 标签。语义 Monitor 只要在同一批决策点上
    产出 `verdict/confidence`，就能算同一张 P@k 表。
    """
    run = load_run(run_dir)
    rows = run["system1"]
    labels = label_future_failures(rows)
    instructions = run["instruction_by_episode"]
    records: List[Dict[str, Any]] = []
    frames_available = 0
    for entry, row in zip(labels, rows):
        key = (str(row.get("scene_id")), int(row.get("episode_id")))
        record = dict(entry)
        record["instruction"] = instructions.get(key)
        record["normalized"] = row.get("normalized_components")
        record["availability"] = row.get("availability")
        record["mode"] = row.get("mode")
        record["run_dir"] = run["run_dir"]
        if record.get("rgb_path"):
            frames_available += 1
        records.append(record)
    dataset = {
        "run_dir": run_dir,
        "decisions": len(records),
        "episodes": len(set((str(r.get("scene_id")), int(r.get("episode_id"))) for r in rows)),
        "decisions_with_frames": frames_available,
        "horizons": list(PRECURSOR_HORIZONS),
        "records": records,
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(dataset, handle, ensure_ascii=False)
    return dataset


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cmd_parity(args: argparse.Namespace) -> int:
    run_dir = args.run
    run = load_run(run_dir)
    print("== %s ==" % run_dir)
    print("system1 事件 %d，progress 行 %d，closed_loop mode=%s"
          % (len(run["system1"]), len(run["progress"]),
             (run["closed_loop_config"] or {}).get("mode")))
    primary = verify_scheduler_parity(run_dir, resync_last_s2=True, limit_episodes=args.episodes)
    strict = verify_scheduler_parity(run_dir, resync_last_s2=False, limit_episodes=args.episodes)
    feed = verify_feedback_equivalence(run["closed_loop_config"], n_samples=args.samples)
    layer = verify_decision_layer_parity(run_dir, limit_episodes=args.episodes)
    agreement = label_agreement_with_log(run_dir)
    report = {
        "run_dir": run_dir,
        "closed_loop_mode": (run["closed_loop_config"] or {}).get("mode"),
        "scheduler_kwargs": primary["scheduler_kwargs"],
        "scheduler_parity_resync": {k: v for k, v in primary.items()
                                    if k not in ("scheduler_kwargs", "examples")},
        "scheduler_parity_strict": {k: v for k, v in strict.items()
                                    if k not in ("scheduler_kwargs", "examples")},
        "feedback_equivalence": feed,
        "decision_layer_parity": {k: v for k, v in layer.items() if k != "examples"},
        "label_agreement_with_log": agreement,
        "examples": {
            "scheduler_parity_resync": primary["examples"][:5],
            "decision_layer_parity": layer["examples"][:5],
        },
    }

    def line(name: str, item: Dict[str, Any], extra: str = "") -> None:
        print("  %-26s compared=%-5s episodes=%-3s mismatches=%-4s rate=%s %s"
              % (name, item.get("events_compared"), item.get("episodes_compared"),
                 item.get("mismatches"),
                 ("%.6f" % item["mismatch_rate"]) if item.get("mismatch_rate") is not None else "-",
                 extra))

    print("-- 回归结果 --")
    line("scheduler(resync, 主口径)", primary,
         "| S2 乱序 %d, 重同步 %d, temporal 外部触发 %d, 跟踪 S2 间隔=%s"
         % (primary["out_of_order_system2_events"], primary["resync_count"],
            primary["temporal_external_triggers"], primary["tracks_s2_interval"]))
    line("scheduler(strict 文件序)", strict,
         "| S2 乱序 %d 次" % strict["out_of_order_system2_events"])
    line("decision_layer_v3(全关)", layer)
    print("  feedback5 vs feedback4    samples=%s differences=%s => %s"
          % (feed["samples"], feed["differences"], feed["verdict"]))
    print("  标签复刻 vs 日志自带标签  logged=%s checked=%s agree=%s rate=%s (best=%s)"
          % (agreement["logged_events"], agreement["checked"], agreement["agree"],
             ("%.4f" % agreement["best_agreement_rate"])
             if agreement["best_agreement_rate"] is not None else "-",
             agreement["best_variant"]))
    if primary["mismatch_by_field"]:
        print("  scheduler 不一致字段分布:", primary["mismatch_by_field"])
    if layer["mismatch_by_field"]:
        print("  决策层不一致字段分布:", layer["mismatch_by_field"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("  wrote %s" % args.out)
    return 0


def _cmd_precision(args: argparse.Namespace) -> int:
    rows = []
    for run_dir in args.run:
        for predicate in args.predicate:
            rows.append(trigger_precision(run_dir, predicate,
                                          use_logged_labels=bool(args.logged_labels)))
    print("%-56s %-11s %-16s %6s %6s %8s %8s %8s"
          % ("run", "predicate", "label_source", "decis", "trig", "P@1", "P@3", "P@5"))
    def f(value: Any) -> str:
        return "-" if value is None else "%.4f" % value
    for row in rows:
        name = os.path.basename(row["run_dir"])
        print("%-56s %-11s %-16s %6d %6d %8s %8s %8s"
              % (name[:56], row["predicate"],
                 "logged" if args.logged_labels else "recomputed",
                 row["decisions"], row["triggers"],
                 f(row["precision_at_1"]), f(row["precision_at_3"]),
                 f(row["precision_at_5"])))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        print("wrote %s" % args.out)
    return 0


def _cmd_labelcheck(args: argparse.Namespace) -> int:
    print("%-56s %8s %8s %11s %11s %-24s"
          % ("run", "logged", "checked", "agree(req)", "agree(any)", "best"))
    totals = {}
    for run_dir in args.run:
        item = label_agreement_with_log(run_dir)
        a = item["variant_model_invoked_required"]
        b = item["variant_any_no_action"]
        def r(v: Any) -> str:
            return "-" if v is None else "%.4f" % v
        print("%-56s %8d %8d %11s %11s %-24s"
              % (os.path.basename(run_dir)[:56], item["logged_events"], item["checked"],
                 r(a["agreement_rate"]), r(b["agreement_rate"]), item["best_variant"]))
        for name in ("variant_model_invoked_required", "variant_any_no_action"):
            acc = totals.setdefault(name, [0, 0])
            acc[0] += item[name]["agree"]
            acc[1] += item[name]["checked"]
        if args.verbose and item["disagreements"]:
            print("   例:", json.dumps(item["disagreements"][:2], ensure_ascii=False)[:500])
    print("-- 汇总 --")
    for name, (agree, checked) in totals.items():
        print("  %-32s agree=%d checked=%d rate=%s"
              % (name, agree, checked, "-" if not checked else "%.4f" % (agree / checked)))
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    dataset = build_monitor_dataset(args.run, args.out)
    print(json.dumps({k: v for k, v in dataset.items() if k != "records"},
                     ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="导航侧离线重放与回归验证")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parity = sub.add_parser("parity", help="调度器/反馈等价性回归")
    p_parity.add_argument("--run", required=True)
    p_parity.add_argument("--episodes", type=int, default=None)
    p_parity.add_argument("--samples", type=int, default=200)
    p_parity.add_argument("--out", default=None)
    p_parity.set_defaults(func=_cmd_parity)

    p_prec = sub.add_parser("precision", help="P@1/P@3/P@5")
    p_prec.add_argument("--run", action="append", required=True)
    p_prec.add_argument("--predicate", action="append",
                        default=None, choices=["early_drift", "any_replan", "feedback_high"])
    p_prec.add_argument("--logged-labels", action="store_true",
                        help="直接用日志里 early_drift_outcome 自带的标签（仅 early_drift）")
    p_prec.add_argument("--out", default=None)
    p_prec.set_defaults(func=_cmd_precision)

    p_lab = sub.add_parser("labelcheck", help="校验标签复刻与日志自带标签是否一致")
    p_lab.add_argument("--run", action="append", required=True)
    p_lab.add_argument("--verbose", action="store_true")
    p_lab.set_defaults(func=_cmd_labelcheck)

    p_data = sub.add_parser("dataset", help="导出 Monitor 候选集")
    p_data.add_argument("--run", required=True)
    p_data.add_argument("--out", default=None)
    p_data.set_defaults(func=_cmd_dataset)

    args = parser.parse_args(argv)
    if getattr(args, "predicate", None) is None:
        args.predicate = ["early_drift"]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

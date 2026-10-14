"""决策层 v3：在不改动 InternNav 触发层的前提下，补齐归因、恢复动作空间与 N8 升级。

分层（对应 route1-bridge-plan.md §1）
------------------------------------
    第 5 信号  →  nav/feedback5.py        （semantic 不可用时逐位等价）
    触发层     →  InternNav scheduler.py  （**原样复用**，一行不改）
    本模块     →  归因 + 恢复动作选择 + 预算仲裁 + N8 升级 + FailureEvent 产出

**等价性契约**：当 `semantic_enabled=False` 且 `enable_recovery_actions=False` 时，
本模块的 `decide()` 必须与直接调用 `AdaptiveReplanScheduler.step()` 的输出在所有
原有字段上完全一致（`nav/replay.py` 会在真实 50 回合日志上验证这一点）。

Python 3.9 兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import (
    RECOVERY_ACTION_LABEL,
    RECOVERY_MENU,
    Attribution,
    FailureEvent,
    Recovery,
    Signals,
)

#: InternNav scheduler 的 12 个 reason → 失败分类学的 N 编号
REASON_TO_FAILURE_TYPE: Dict[str, Optional[str]] = {
    "recovery": "N1",
    "no_action_replan_limit": "N1",
    "recovery_budget_exhausted": "N1",
    # ⚠️ `local_horizon_replan` **不在调度器的 12 个 reason 里**，它是 evaluator 在
    # `action == STOP` 分支（`habitat_vln_evaluator.py:1996-2026`）调用
    # `scheduler.step(has_executable_action=False)` 之后，把落盘的 `replan_reason`
    # **改写**成的名字。它走的就是 `recovery`（N1）分支、也确实触发了重规划，
    # 只是被刻意与"真 System-1 失败"分开记录（原注释："recording this padded local STOP
    # separately from true System1 failure"）。不把它映射成 N1 会漏掉 87% 的无动作事件。
    "local_horizon_replan": "N1",
    "cooldown": None,                      # 抑制器，不是失败
    "early_drift": "N2",
    "external": "N4",
    "max_plan_age": "N3",
    "feedback_high": "N4",
    "below_trigger": None,                 # 正常巡航
    "proactive_budget_exhausted": "N4",    # 早触发命中但预算耗尽 → 决策层的问题
    "early_replan_observe": None,          # 消融观察档
    "recovery_only_observe": None,
    "recovery_adaptive_periodic": "N3",    # 恢复若干次后转周期性刷新
    "fixed_periodic": "N3",
    "collision_watchdog_stop": "N5",
}

#: 触发但**不**真正重规划的 reason（保留它们，才能和现有报表对账）
NON_ACTING_REASONS = (
    "cooldown",
    "below_trigger",
    "no_action_replan_limit",
    "recovery_budget_exhausted",
    "proactive_budget_exhausted",
    "early_replan_observe",
    "recovery_only_observe",
)


def _lazy_scheduler():
    try:
        from internnav.model.closed_loop.scheduler import AdaptiveReplanScheduler  # type: ignore
        return AdaptiveReplanScheduler
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "决策层 v3 需要 InternNav 在 PYTHONPATH 上（例如 "
            "PYTHONPATH=/srv/agh-demo/InternNav）。原始错误: %r" % (exc,)
        )


@dataclass
class SemanticConfig:
    """语义通道的配置。刻意与 S2 预算**分离**（独立小预算）。

    实测约束（`notes/monitor-agnes-report.md`）
    -----------------------------------------
    - **延迟**：Agnes `agnes-3.0-flash` p50 3.64 s / p90 **10.07 s** / max 24.41 s，
      >10 s 占 10% → **不能每个决策都调**。所以默认 `invoke_only_in_band=True`：
      只在几何分数落进迟滞中间带 [low, high) 时调 Monitor 做仲裁——
      这正好把"几何快但 P@1=0.0000"与"语义慢但更可信"互补起来。
    - **判决不稳**：含糊帧同帧同指令、temperature=0 时少数派可达 20–40%，
      而 confidence 挡不住 → 必须用**多数投票的一致度**做门控（`n_votes` / `min_agreement`），
      并且投票时要**绕开磁盘缓存**，否则重复调用会命中同一份缓存、把不稳定伪装成稳定。
    """

    enabled: bool = False
    weight: float = 0.30
    trigger_threshold: float = 0.70       # r_sem ≥ 该值 ⇒ 单独触发 N7
    max_recoveries_per_episode: int = 3   # 独立小预算，避免与几何触发抢额度
    min_interval_steps: int = 4           # 自己的冷却，避免连续抖动
    scale: float = 1.0
    # 语义信号怎么参与决策 —— 这是第五轮诊断逼出来的一个**必须显式选择**的设计点。
    #
    #   "independent"（默认）：语义**不混进**几何加权分（w_semantic 置 0），
    #       只走自己的阈值通道 + 自己的预算。理由：几何分实测 AUROC ≈ 0.44
    #       （见 §11），把新信号混进一个负相关的加权和里，等于让它被污染。
    #   "mixed"：语义按 `weight` 参与 4 信号加权（`feedback5.PlanFeedback5`），
    #       保留下来是为了能和 "independent" 做 A/B 对照。
    #
    # 注意：混不混是**构造 PlanFeedback5 时**决定的（`semantic_weight_for_mode`），
    # 决策层只负责通道一侧，所以两处要一致。
    mode: str = "independent"
    n_votes: int = 3                      # 每次判决的投票次数（>1 时取多数派）
    min_agreement: float = 0.6            # 多数派占比低于此值 ⇒ 判为"弃权"
    # 何时值得付一次 Monitor 调用（p90 ≈ 10 s，必须省着用）
    #
    # 实测把 `band` 判了死刑：几何分**几乎全部压在 0.40 以下**——
    # 参考 50 回合 run（1018 个决策）里只有 13 个落进 [0.40, 0.70)，**没有一个 ≥ 0.70**；
    # 带帧 run 前 166 个决策里只有 4 个。所以"只在迟滞中间带调 Monitor"这条设计
    # 实际上会 1.3% 触发，等于不调。
    #
    # 可选值：
    #   "trigger"  —— 只在几何**真正触发了恢复**的决策点上调（early_drift / recovery 等）。
    #                 这是默认：这些点就是"改变计划"的决策点，语义仲裁的边际价值最高。
    #   "periodic" —— 每 `periodic_interval` 个决策调一次（成本可预估，覆盖最均匀）。
    #   "band"     —— 迟滞中间带（**实测近乎为空，不推荐**，保留以便复现这个结论）。
    #   "always"   —— 每个决策都调（只用于离线批量实验，在线跑不起）。
    invoke_policy: str = "trigger"
    band_low: float = 0.40
    band_high: float = 0.70
    periodic_interval: int = 8


@dataclass
class RecoveryRecord:
    """一次已执行的恢复，用于 N8 二阶判定。"""

    action: str
    failure_type: str
    at_step: int
    score_before: float
    resolved: bool = False


class DecisionLayerV3(object):
    """把 InternNav 的调度器包成「触发 → 归因 → 恢复 → 记账」的决策层。"""

    def __init__(
        self,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        semantic: Optional[SemanticConfig] = None,
        enable_recovery_actions: bool = True,
        enable_semantic_channel: bool = True,
        low_threshold: float = 0.40,
        n8_patience: int = 1,
    ) -> None:
        scheduler_cls = _lazy_scheduler()
        self.scheduler_kwargs = dict(scheduler_kwargs or {})
        self.scheduler = scheduler_cls(**self.scheduler_kwargs)
        self.semantic = semantic or SemanticConfig()
        self.enable_recovery_actions = bool(enable_recovery_actions)
        self.enable_semantic_channel = bool(enable_semantic_channel)
        self.low_threshold = float(low_threshold)
        self.n8_patience = int(n8_patience)
        self.reset_episode()

    # ------------------------------------------------------------------ #
    def reset_episode(self) -> None:
        self.scheduler.reset()
        self.semantic_calls_used = 0
        self.last_semantic_step: Optional[int] = None
        self.semantic_events = 0
        self.escalations = 0
        self.pending: List[RecoveryRecord] = []
        self.history: List[Dict[str, Any]] = []
        self.attribution_counts: Dict[str, int] = {}
        self.action_counts: Dict[str, int] = {}
        self.type_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def record_system2_call(self, current_step: int) -> None:
        self.scheduler.record_system2_call(current_step)

    # ------------------------------------------------------------------ #
    def decide(
        self,
        feedback_score: float,
        plan_age: int,
        current_step: int,
        has_executable_action: bool = True,
        allow_feedback_replan: bool = True,
        trajectory_drift: Optional[float] = None,
        system2_calls_used: int = 0,
        external_trigger: bool = False,
        external_reason: str = "external",
        semantic: Optional[Dict[str, Any]] = None,
        semantic_risk: Optional[float] = None,
        scene_id: Optional[str] = None,
        episode_id: Optional[int] = None,
        step: Optional[int] = None,
        instruction: Optional[str] = None,
        collision_delta: Optional[int] = None,
        steps_since_last_s2: Optional[int] = None,
        emit: bool = True,
    ) -> Dict[str, Any]:
        """一次决策。

        返回：上游 `scheduler.step()` 的全部字段 + 决策层新增字段 + `failure_event`。
        """
        base = self.scheduler.step(
            feedback_score=feedback_score,
            plan_age=plan_age,
            current_step=current_step,
            has_executable_action=has_executable_action,
            allow_feedback_replan=allow_feedback_replan,
            trajectory_drift=trajectory_drift,
            system2_calls_used=system2_calls_used,
            external_trigger=external_trigger,
            external_reason=external_reason,
        )
        decision = dict(base)

        resolved_risk = semantic_risk
        verdict = None
        confidence = None
        agreement = None
        if semantic is not None:
            verdict = semantic.get("verdict")
            confidence = semantic.get("confidence")
            agreement = semantic.get("agreement")
        if resolved_risk is None and verdict is not None:
            from .feedback5 import semantic_risk as _risk
            resolved_risk = _risk(verdict, confidence, agreement, self.semantic.min_agreement)
        semantic_ambiguous = bool(
            semantic is not None and verdict is not None and resolved_risk is None
        )

        base_reason = base.get("reason")
        failure_type = REASON_TO_FAILURE_TYPE.get(base_reason)
        acting = bool(base.get("should_replan"))

        # ---- 1) N8：先看上一次恢复是否失败 --------------------------- #
        escalation = self._check_escalation(
            current_step=current_step,
            score=float(feedback_score),
            failure_type=failure_type,
        )

        # ---- 2) 语义通道（N7）：独立预算，独立冷却 ------------------- #
        semantic_fired = False
        semantic_abstained = False
        if self.enable_semantic_channel and self.semantic.enabled and not acting:
            if semantic_ambiguous and self._semantic_budget_available(current_step):
                # 模型有输出但一致度不够 ⇒ 这正是"弃权"该出场的时候
                semantic_fired = True
                semantic_abstained = True
                failure_type = "N7"
                acting = True
            elif (
                resolved_risk is not None
                and resolved_risk >= self.semantic.trigger_threshold
            ):
                semantic_fired = self._semantic_budget_available(current_step)
                if semantic_fired:
                    failure_type = "N7"
                    acting = True

        if escalation is not None:
            failure_type = "N8"
            acting = True

        # ---- 3) 归因 ------------------------------------------------ #
        attribution = self._attribute(
            failure_type=failure_type,
            reason=base_reason,
            dominant=base.get("reason", "").split(":")[-1] if base_reason else None,
            trigger_kind=base.get("trigger_kind"),
            drift=trajectory_drift,
            plan_age=plan_age,
            has_executable_action=has_executable_action,
            verdict=verdict,
            confidence=confidence,
            risk=resolved_risk,
            score=float(feedback_score),
        )

        # ---- 4) 恢复动作选择 ---------------------------------------- #
        recovery = self._select_recovery(
            failure_type=failure_type,
            reason=base_reason,
            acting=acting,
            drift=trajectory_drift,
            plan_age=plan_age,
            score=float(feedback_score),
            risk=resolved_risk,
            system2_calls_used=system2_calls_used,
            escalation=escalation,
            bypassed_cooldown=bool(base.get("reason") == "recovery"),
            semantic_abstained=semantic_abstained,
        )

        # ---- 5) 记账 ------------------------------------------------ #
        if failure_type is not None:
            self.type_counts[failure_type] = self.type_counts.get(failure_type, 0) + 1
        kind = attribution.kind
        self.attribution_counts[kind] = self.attribution_counts.get(kind, 0) + 1
        if recovery.action != "none":
            self.action_counts[recovery.action] = self.action_counts.get(recovery.action, 0) + 1
        if semantic_fired:
            self.semantic_calls_used += 1
            self.semantic_events += 1
            self.last_semantic_step = int(current_step)
        if acting and failure_type in ("N1", "N2", "N3", "N4", "N7", "N8"):
            self.pending.append(
                RecoveryRecord(
                    action=recovery.action,
                    failure_type=failure_type,
                    at_step=int(current_step),
                    score_before=float(feedback_score),
                )
            )

        decision.update(
            {
                "failure_type": failure_type,
                "acting": acting,
                "semantic_fired": semantic_fired,
                "semantic_risk": resolved_risk,
                "semantic_budget_used": self.semantic_calls_used,
                "semantic_budget_left": max(
                    0, self.semantic.max_recoveries_per_episode - self.semantic_calls_used
                ),
                "recovery_action": recovery.action,
                "recovery_action_label": RECOVERY_ACTION_LABEL.get(recovery.action, recovery.action),
                "attribution": attribution.to_dict(),
                "escalation": escalation,
                "base_reason": base_reason,
            }
        )

        if emit and failure_type is not None:
            event = self._to_event(
                failure_type=failure_type,
                attribution=attribution,
                recovery=recovery,
                signals=Signals(
                    feedback_score=float(feedback_score),
                    trajectory_drift=trajectory_drift,
                    plan_age=int(plan_age),
                    collision_delta=collision_delta,
                    steps_since_last_s2=steps_since_last_s2,
                    semantic_instance_margin=resolved_risk,
                    semantic_verdict=(None if verdict is None else str(verdict).upper()),
                    semantic_confidence=confidence,
                ),
                scene_id=scene_id,
                episode_id=episode_id,
                step=step,
                action_step=current_step,
                instruction=instruction,
                trigger_kind=base.get("trigger_kind"),
            )
            decision["failure_event"] = event
        else:
            decision["failure_event"] = None

        self.history.append(decision)
        return decision

    # ------------------------------------------------------------------ #
    def _semantic_budget_available(self, current_step: int) -> bool:
        if self.semantic_calls_used >= self.semantic.max_recoveries_per_episode:
            return False
        if (
            self.last_semantic_step is not None
            and current_step - self.last_semantic_step < self.semantic.min_interval_steps
        ):
            return False
        return True

    def _check_escalation(
        self, current_step: int, score: float, failure_type: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """N8：恢复后分数没降（或同类失败再次出现）⇒ 升级。

        判定口径（诚实版）：只看"恢复前后同状态下的分数没有下降"，不做同状态
        反事实（那需要冻结/恢复仿真状态，属 `counterfactual.py` 的能力，本轮不接）。
        """
        if not self.pending or self.escalations >= 1:
            return None
        record = self.pending[-1]
        if current_step - record.at_step < self.n8_patience:
            return None
        if record.resolved:
            return None
        improved = score <= record.score_before - 1e-9
        if improved:
            record.resolved = True
            self.pending = []
            return None
        if failure_type == record.failure_type or score >= self.scheduler.high_threshold:
            self.escalations += 1
            self.pending = []
            return {
                "from_action": record.action,
                "from_type": record.failure_type,
                "score_before": record.score_before,
                "score_after": float(score),
                "steps_since": int(current_step - record.at_step),
            }
        record.resolved = True
        self.pending = []
        return None

    def _select_recovery(
        self,
        failure_type: Optional[str],
        reason: Optional[str],
        acting: bool,
        drift: Optional[float],
        plan_age: int,
        score: float,
        risk: Optional[float],
        system2_calls_used: int,
        escalation: Optional[Dict[str, Any]],
        bypassed_cooldown: bool,
        semantic_abstained: bool = False,
    ) -> Recovery:
        menu = RECOVERY_MENU.get(failure_type or "", ("replan",))
        action = "none"
        reason_code = "no_action"

        if escalation is not None:
            action = "escalate"
            reason_code = "recovery_ineffective"
        elif acting and semantic_abstained:
            # 一致度不足 ⇒ 不假装看懂了，明确弃权（对齐 AVERT 的 abstention 动作，
            # 但出口是"请求澄清"而不是"转人工"）
            action = "ask_clarification"
            reason_code = "semantic_low_agreement_abstain"
        elif acting and failure_type == "N7":
            # 语义失败：先换实例假设；指令本身可疑才重新解析；都不确定就弃权
            if risk is not None and risk >= 0.90:
                action = "switch_instance"
                reason_code = "semantic_lost_high_confidence"
            elif risk is not None and risk >= 0.80:
                action = "reparse_instruction"
                reason_code = "semantic_lost_ambiguous_instruction"
            else:
                action = "ask_clarification"
                reason_code = "semantic_lost_low_confidence"
        elif acting and failure_type == "N1":
            action = "replan"
            reason_code = "no_executable_action"
        elif acting and failure_type == "N2":
            # 几何漂移：什么时候该"局部脱离"（短程闭环，不调 System-2）？
            #
            # ⚠️ 第一版的条件是 `drift >= 0.20 and plan_age <= 6`，**实测 0/15 命中**：
            # 在 15 个 N2 触发点上 plan_age 是 8–16（p50=12）——漂移本来就是**攒出来的**，
            # 等它越过 0.20 时计划早就不新了。也就是说这个条件与漂移的成因自相矛盾，
            # 是个结构性死条件（`recovery_policy_audit` 抓到的）。
            #
            # 改后的判据基于动机而不是年龄：**预算快用完时**才值得用省算力的局部脱离
            # （毕竟再重规划一次也未必有钱付）；计划确实新（plan_age 小）时同理。
            budget_left = max(
                0,
                int(self.scheduler.max_s2_calls_per_episode or 0) - int(system2_calls_used),
            )
            scarce = (
                self.scheduler.max_s2_calls_per_episode is not None and budget_left <= 1
            )
            if drift is not None and drift >= self.scheduler.early_drift_high_m and (
                plan_age <= 6 or scarce
            ):
                action = "local_escape"
                reason_code = (
                    "early_drift_high_plan_fresh" if plan_age <= 6
                    else "early_drift_high_budget_scarce"
                )
            else:
                action = "replan"
                reason_code = "early_drift"
        elif acting and failure_type == "N3":
            action = "replan"
            reason_code = "plan_age_limit"
        elif acting and failure_type == "N4":
            # 综合置信崩塌：主因不明。若预算已紧，用局部脱离换一次 S2 调用；
            # 否则老老实实重规划（这也是现有系统的行为，保持不变）。
            budget_left = max(
                0,
                int(self.scheduler.max_s2_calls_per_episode or 0) - int(system2_calls_used),
            )
            if self.scheduler.max_s2_calls_per_episode is not None and budget_left <= 1:
                action = "local_escape"
                reason_code = "feedback_high_budget_scarce"
            else:
                action = "replan"
                reason_code = "feedback_high"
        elif acting and failure_type == "N8":
            action = "escalate"
            reason_code = "recovery_ineffective"

        if action not in menu and menu:
            # 菜单校验：不允许的组合退回该类型的第一项（并在 kind 里留痕）
            action = menu[0]
        if action == "none":
            reason_code = reason or "no_action"

        return Recovery(
            reason_code=reason_code,
            action=action,
            budget_used=int(system2_calls_used),
            budget_left=max(0, int(self.scheduler.max_s2_calls_per_episode or 0) - int(system2_calls_used)),
            bypassed_cooldown=bool(bypassed_cooldown),
            escalated_from=(escalation or {}).get("from_action"),
        )

    def _attribute(
        self,
        failure_type: Optional[str],
        reason: Optional[str],
        dominant: Optional[str],
        trigger_kind: Optional[str],
        drift: Optional[float],
        plan_age: int,
        has_executable_action: bool,
        verdict: Optional[str],
        confidence: Optional[float],
        risk: Optional[float],
        score: float,
    ) -> Attribution:
        evidence: List[str] = []
        if drift is not None:
            evidence.append("trajectory_drift=%.3f m" % drift)
        evidence.append("plan_age=%d" % plan_age)
        evidence.append("feedback_score=%.3f" % score)
        if verdict is not None and confidence is not None:
            evidence.append(
                "Agnes Monitor: verdict=%s confidence=%.2f 语义风险=%.2f"
                % (str(verdict).upper(), float(confidence), float(risk or 0.0))
            )
        if not has_executable_action:
            evidence.append("has_executable_action=False")

        if failure_type == "N1":
            return Attribution(
                kind="no_executable_action",
                summary_l1="System-2 给出的计划里没有可执行的下一步动作",
                summary_l2="通常是子目标已达成但未触发 STOP，或像素目标落在不可达区域。",
                evidence=evidence, dominant_signal="has_executable_action", source="rule",
            )
        if failure_type == "N2":
            return Attribution(
                kind="feedback_high:trajectory_drift" if drift and drift >= 0.2 else "early_drift",
                summary_l1="实际轨迹开始偏离计划轨迹（几何判据，%s）" % (trigger_kind or "unknown"),
                summary_l2="几何漂移只说明「走得和计划不一样」，**不等于**任务语义出错。",
                evidence=evidence, dominant_signal="trajectory_drift", source="rule",
            )
        if failure_type == "N3":
            return Attribution(
                kind="plan_age_limit",
                summary_l1="计划执行步数达到硬上限，纯计时器触发刷新",
                summary_l2="与「该不该刷新」无关；这是保险丝，不是判断。",
                evidence=evidence, dominant_signal="plan_age", source="rule",
            )
        if failure_type == "N4":
            return Attribution(
                kind="feedback_high:%s" % (dominant or "unknown"),
                summary_l1="加权置信分数越过迟滞高阈，但主因按几何量判定为 %s" % (dominant or "未知"),
                summary_l2="现有系统在这里**盲选重规划**：这是 33.7% 有害重规划的来源之一。",
                evidence=evidence, dominant_signal=dominant, source="rule",
            )
        if failure_type == "N7":
            return Attribution(
                kind="semantic_instruction_mismatch",
                summary_l1="Agnes Monitor 判定当前执行与指令不一致（语义偏航）",
                summary_l2="几何量全部正常，因此 4 信号完全看不到这类失败——这是本项目补的第 5 个信号。",
                evidence=evidence, dominant_signal="semantic", source="rule",
            )
        if failure_type == "N8":
            return Attribution(
                kind="recovery_ineffective",
                summary_l1="上一次恢复动作执行后，失败判据仍未解除",
                summary_l2="按 route1-bridge-plan.md §3：每回合最多升级一次，之后才弃权。",
                evidence=evidence, dominant_signal="post_recovery_score", source="rule",
            )
        return Attribution(
            kind="none",
            summary_l1="未判定为失败",
            summary_l2="",
            evidence=evidence, dominant_signal=dominant, source="rule",
        )

    def _to_event(
        self,
        failure_type: str,
        attribution: Attribution,
        recovery: Recovery,
        signals: Signals,
        scene_id: Optional[str],
        episode_id: Optional[int],
        step: Optional[int],
        action_step: int,
        instruction: Optional[str],
        trigger_kind: Optional[str],
    ) -> FailureEvent:
        return FailureEvent.create(
            source="nav",
            type=failure_type,
            scene_id=scene_id,
            episode_id=episode_id,
            step=step if step is not None else action_step,
            action_step=action_step,
            instruction=instruction,
            signals=signals,
            attribution=attribution,
            recovery=recovery,
            outcome=None,
            trigger_kind=trigger_kind,
        )

    # ------------------------------------------------------------------ #
    def should_invoke_monitor(self, feedback_score: float, current_step: int,
                              geometric_trigger: bool = False,
                              precursor_score: Optional[float] = None,
                              precursor_top_rate: float = 0.05) -> bool:
        """是否值得为这个决策点付一次语义 Monitor 的调用（p90 ≈ 10 s，很贵）。

        这是"贵信号该在什么时候买"的仲裁口。默认 `trigger`：只在几何**真正触发恢复**
        的决策点上买（那才是改变计划的时刻）。

        `precursor_score` 是可选的高级口子：把便宜的前兆预测器（AUROC 0.55–0.72，
        可以每个决策都算）当**闸门**，只在它打分最高的 `precursor_top_rate` 比例里
        调贵 Monitor。这是目前推荐的上线形态——但需要在线加载预测器权重。
        """
        if not (self.enable_semantic_channel and self.semantic.enabled):
            return False
        if not self._semantic_budget_available(current_step):
            return False
        policy = self.semantic.invoke_policy
        if policy == "never":
            return False
        if policy == "always":
            return True
        if policy == "trigger":
            return bool(geometric_trigger)
        if policy == "band":
            return self.semantic.band_low <= float(feedback_score) < self.semantic.band_high
        if policy == "periodic":
            interval = max(1, int(self.semantic.periodic_interval))
            return int(current_step) % interval == 0
        if policy == "precursor":
            if precursor_score is None:
                return False
            return float(precursor_score) >= float(1.0 - precursor_top_rate)
        raise ValueError("unknown invoke_policy %r" % (policy,))

    def episode_summary(self) -> Dict[str, Any]:
        return {
            "semantic_events": self.semantic_events,
            "semantic_budget_used": self.semantic_calls_used,
            "escalations": self.escalations,
            "failure_type_counts": dict(self.type_counts),
            "attribution_counts": dict(self.attribution_counts),
            "recovery_action_counts": dict(self.action_counts),
            "decisions": len(self.history),
        }

"""导航侧第 5 个信号：语义失败风险，接进原有 4 信号加权。

为什么用「子类 + 后处理」而不是重写
----------------------------------
`failure-taxonomy.md` §6 的结论是：我们**已有**的触发层（4 信号加权 + 迟滞 +
预算）比 REFLECT/AVERT 都细，缺的是语义信号。所以本模块**不重写**触发层，而是
继承 `internnav.model.closed_loop.feedback.PlanFeedback`，在它的输出上追加第 5 个
信号的贡献。

由此得到一个**可证明的等价性**：
    semantic 不可用（或 w_semantic == 0）时，本类**原样返回父类的 dict**，
    因此与现有系统**逐位一致**（不是"误差很小"，是同一个对象）。
回归验证见 `nav/replay.py`。

语义风险的定义（这是本项目自己的判据，必须写清楚）
--------------------------------------------------
Agnes Monitor 只看「指令 + 视觉历史」，输出 `verdict ∈ {PASS, LOST}` 与
`confidence ∈ [0,1]`。我们把语义失败风险定义为：

    r_sem = confidence          if verdict == LOST
    r_sem = 1 - confidence      if verdict == PASS

于是 r_sem ∈ [0,1] 且 **0.5 就是判决边界**：模型越确信自己在指令上走对了，
风险越接近 0；越确信跑偏了，风险越接近 1。这与 AVERT-VLN 的 Monitor 输出口径
同构（它也是"判决 + 置信度"），但**我们把判决接到了预算化调度器上**，而 AVERT
是转人工——这正是论文里缺的那一层「接受规则」。

Python 3.9 兼容。本模块在**没有** InternNav 的环境里也能 import：只有实例化
`PlanFeedback5` 时才需要 `internnav` 在 `PYTHONPATH` 上。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: 与 InternNav 完全一致的 4 个信号名与顺序（顺序决定 dominant 的 tie-break，不能改）
BASE_SIGNALS: Tuple[str, ...] = (
    "uncertainty",
    "trajectory_drift",
    "plan_age",
    "collision_risk",
)

#: 第 5 个信号的名字（出现在 score 的 dominant 后缀里，如 `feedback_high:semantic`）
SEMANTIC_SIGNAL = "semantic"

#: 合并后的 5 个信号顺序（semantic 必须排最后，否则会改变 tie-break）
ALL_SIGNALS: Tuple[str, ...] = BASE_SIGNALS + (SEMANTIC_SIGNAL,)

#: 默认新增权重。刻意选 0.30：既要让语义信号能单独把分数推过 0.70，
#: 又不能让它一票否决 4 个几何信号（`r_sem=1` 时贡献 0.30/1.30 ≈ 0.23）。
DEFAULT_W_SEMANTIC = 0.30


def semantic_risk(verdict: Optional[str], confidence: Optional[float],
                  agreement: Optional[float] = None,
                  min_agreement: float = 0.6) -> Optional[float]:
    """把 Monitor 的判决转成 [0,1] 的语义失败风险。不可信时返回 None（**不要补 0**）。

    实测校准（`notes/monitor-agnes-report.md`）
    -----------------------------------------
    Agnes `agnes-3.0-flash` 在含糊帧上判决**不稳定**：同帧同指令、temperature=0，
    少数派比例可达 20–40%；而 **confidence 挡不住这个错误**（判 LOST 的那几次
    confidence 反而更高：0.85/0.85/0.90 vs PASS 的 0.85/0.60）。
    唯一有效的可靠性指标是**多次投票的一致度**，所以这里用 `agreement` 做门控：
    一致度低于 `min_agreement` 的判决**不进入分数**（返回 None），
    由决策层走"弃权/请求澄清"通道，而不是用一个抖动的信号去扰动几何分数。

    `agreement` 为 None 时视为"单次调用"，不做门控（保持向后兼容）。
    """
    if verdict is None or confidence is None:
        return None
    verdict = str(verdict).strip().upper()
    if verdict not in ("PASS", "LOST"):
        # "UNSURE" / "ABSTAIN" 等：模型自己说不确定 → 不构成信号
        return None
    if agreement is not None:
        agreement_value = float(agreement)
        if not np.isfinite(agreement_value) or agreement_value < min_agreement:
            return None
    value = float(confidence)
    if not np.isfinite(value):
        return None
    value = float(np.clip(value, 0.0, 1.0))
    return value if verdict == "LOST" else 1.0 - value


def normalize_semantic(risk: Optional[float], scale: float = 1.0) -> Optional[float]:
    """把风险归一化到 [0,1]；`scale < 1` 相当于锐化（小风险就顶满）。"""
    if risk is None:
        return None
    if scale <= 0:
        raise ValueError("semantic scale must be positive")
    return float(np.clip(float(risk) / float(scale), 0.0, 1.0))


def combine(
    weights: Mapping[str, float],
    normalized: Mapping[str, Optional[float]],
    availability: Mapping[str, bool],
    renormalize_active_weights: bool = True,
    signals: Sequence[str] = ALL_SIGNALS,
) -> Dict[str, Any]:
    """与 InternNav `feedback.py:205-226` **同构**的加权合并。

    这条式子必须与上游一字不差，否则第 5 信号会污染原有 4 信号的行为：
        score = Σ(wᵢ · normᵢ) / Σ(wᵢ | 信号可用)
    """
    contributions = {
        name: (float(weights.get(name, 0.0)) * float(normalized[name]))
        if availability.get(name) and normalized.get(name) is not None
        else 0.0
        for name in signals
    }
    active_weight = sum(float(weights.get(name, 0.0)) for name in signals if availability.get(name))
    denominator = active_weight if renormalize_active_weights else sum(
        float(weights.get(name, 0.0)) for name in signals
    )
    score = 0.0 if denominator <= 0 else float(sum(contributions.values()) / denominator)
    score = float(np.clip(score, 0.0, 1.0))
    active_names = [
        name for name in signals
        if availability.get(name) and float(weights.get(name, 0.0)) > 0
    ]
    if not active_names:
        dominant: Optional[str] = None
    else:
        dominant = max(active_names, key=lambda name: contributions[name])
    return {
        "score": score,
        "contributions": contributions,
        "active_weight": float(active_weight),
        "dominant": dominant,
        "active_names": active_names,
    }


def _import_base():
    """延迟导入 InternNav 的 PlanFeedback（只在实例化时需要）。"""
    try:
        from internnav.model.closed_loop.feedback import PlanFeedback  # type: ignore
        return PlanFeedback
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        raise ImportError(
            "PlanFeedback5 需要 InternNav 在 PYTHONPATH 上（例如 "
            "PYTHONPATH=/srv/agh-demo/InternNav）。原始错误: %r" % (exc,)
        )


class PlanFeedback5(object):
    """5 信号版反馈聚合。`semantic` 不可用时与 4 信号版**逐位一致**。"""

    def __init__(
        self,
        w_uncertainty: float = 0.35,
        w_drift: float = 0.35,
        w_age: float = 0.15,
        w_risk: float = 0.15,
        w_semantic: float = DEFAULT_W_SEMANTIC,
        renormalize_active_weights: bool = True,
        drift_scale_m: float = 0.50,
        uncertainty_scale_m2: float = 0.25,
        replan_threshold: float = 0.70,
        max_plan_age_steps: int = 20,
        num_points: int = 33,
        fallback_prefix_points: int = 4,
        semantic_scale: float = 1.0,
        semantic_trigger_threshold: float = 0.70,
        semantic_enabled: bool = True,
        semantic_min_agreement: float = 0.6,
    ) -> None:
        base = _import_base()
        self._base = base(
            w_uncertainty=w_uncertainty,
            w_drift=w_drift,
            w_age=w_age,
            w_risk=w_risk,
            renormalize_active_weights=renormalize_active_weights,
            drift_scale_m=drift_scale_m,
            uncertainty_scale_m2=uncertainty_scale_m2,
            replan_threshold=replan_threshold,
            max_plan_age_steps=max_plan_age_steps,
            num_points=num_points,
            fallback_prefix_points=fallback_prefix_points,
        )
        self.weights: Dict[str, float] = dict(self._base.weights)
        self.weights[SEMANTIC_SIGNAL] = float(w_semantic)
        self.renormalize_active_weights = bool(renormalize_active_weights)
        self.replan_threshold = float(replan_threshold)
        self.semantic_scale = float(semantic_scale)
        self.semantic_trigger_threshold = float(semantic_trigger_threshold)
        self.semantic_enabled = bool(semantic_enabled)
        self.semantic_min_agreement = float(semantic_min_agreement)

    # ---- 与父类同名的属性，方便替换 ---------------------------------- #
    @property
    def base(self):
        return self._base

    def evaluate(
        self,
        current_traj: Any,
        previous_traj: Any,
        plan_age: int,
        traj_uncertainty: Any = None,
        pose_delta: Any = None,
        collision_risk: Any = None,
        semantic: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """先跑上游 4 信号，再（可选地）追加语义信号。

        `semantic` 形如 `{"verdict": "LOST", "confidence": 0.82, ...}`；
        为 None / 关闭 / 权重为 0 时**原样返回上游结果**。
        """
        result = self._base.evaluate(
            current_traj=current_traj,
            previous_traj=previous_traj,
            plan_age=plan_age,
            traj_uncertainty=traj_uncertainty,
            pose_delta=pose_delta,
            collision_risk=collision_risk,
        )
        risk = None
        ambiguous = False
        if self.semantic_enabled and semantic is not None and self.weights[SEMANTIC_SIGNAL] > 0:
            risk = semantic_risk(
                semantic.get("verdict"), semantic.get("confidence"),
                semantic.get("agreement"), self.semantic_min_agreement,
            )
            # 模型有输出但被可靠性门控挡下 → 属于"该弃权"的情形，交由决策层处理
            ambiguous = risk is None and semantic.get("verdict") is not None
        if risk is None:
            # 不可用 → 逐位一致（连 dict 的构造都交给上游）
            result = dict(result)
            result["semantic_risk"] = None
            result["semantic_normalized"] = None
            result["semantic_available"] = False
            result["semantic_ambiguous"] = bool(ambiguous)
            result["semantic_trigger"] = False
            result["semantic_verdict"] = (
                None if semantic is None else semantic.get("verdict")
            )
            result["dominant"] = max(
                [name for name in BASE_SIGNALS
                 if result["availability"].get(name) and self.weights.get(name, 0.0) > 0],
                key=lambda name: result["contributions"][name],
                default=None,
            )
            result["four_signal_score"] = result["score"]
            return result

        norm_semantic = normalize_semantic(risk, self.semantic_scale)
        normalized = dict(result["normalized"])
        normalized[SEMANTIC_SIGNAL] = norm_semantic
        availability = dict(result["availability"])
        availability[SEMANTIC_SIGNAL] = True

        merged = combine(
            weights=self.weights,
            normalized=normalized,
            availability=availability,
            renormalize_active_weights=self.renormalize_active_weights,
        )
        score = merged["score"]
        should_replan = score >= self.replan_threshold
        dominant = merged["dominant"]
        reason = (
            "no_active_signals"
            if dominant is None
            else ("feedback_high:" if should_replan else "feedback_below_threshold:") + dominant
        )

        result = dict(result)
        four_signal_score = float(result["score"])
        result.update(
            {
                "score": score,
                "should_replan": should_replan,
                "reason": reason,
                "replan_reason": reason,
                "normalized": normalized,
                "availability": availability,
                "contributions": merged["contributions"],
                "active_weight": merged["active_weight"],
                "active_weights": {
                    name: float(self.weights[name] / merged["active_weight"])
                    for name in merged["active_names"]
                    if merged["active_weight"] > 0
                },
                "dominant": dominant,
                "semantic_risk": float(risk),
                "semantic_normalized": float(norm_semantic),
                "semantic_available": True,
                "semantic_verdict": str(semantic.get("verdict", "")).upper(),
                "four_signal_score": four_signal_score,
            }
        )
        # 语义触发（N7）独立于几何触发：见 nav/scheduler_v3.py
        result["semantic_trigger"] = bool(risk >= self.semantic_trigger_threshold)
        return result


def semantic_weight_for_mode(mode: str, w_semantic: float = DEFAULT_W_SEMANTIC) -> float:
    """根据语义通道的接入方式返回应使用的 `w_semantic`。

    `"independent"` → **0.0**：语义不混进几何加权分，只走自己的阈值通道。
        这是默认，依据是实测：几何加权分与真失败的 AUROC ≈ 0.44（**低于随机**，
        见 `notes/route1-progress.md` §11），把新信号混进一个负相关的加权和里
        会被污染。而且名义权重里有一半压在不可用信号上（`uncertainty` 关闭、
        `collision_risk` 从未提供），加权和的语义本身已经不成立。
    `"mixed"` → 原样返回：保留用于和 independent 做 A/B 对照。
    """
    normalized = (mode or "independent").strip().lower()
    if normalized == "independent":
        return 0.0
    if normalized == "mixed":
        return float(w_semantic)
    raise ValueError("semantic mode must be 'independent' or 'mixed', got %r" % (mode,))


def describe_signal_weights(weights: Mapping[str, float]) -> List[str]:
    """把权重渲染成人读的行，报告里直接用。"""
    return [
        "%-20s %.4f" % (name, float(weights[name]))
        for name in ALL_SIGNALS
        if name in weights
    ]

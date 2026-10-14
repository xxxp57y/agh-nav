"""路线① 跨仿真器失败事件契约（唯一契约）。

设计原则
--------
1. **只传事件，不传渲染帧**：导航侧（Habitat/MP3D）与操作侧（Isaac Sim）各自持有
   自己的物理世界，桥只传 `FailureEvent` 与任务状态，因此任一侧崩掉都能从事件流重放。
2. **信号字段沿用 InternNav 现有命名**（`feedback_score` / `trajectory_drift` /
   `plan_age` / `collision_delta` / `steps_since_last_s2`），操作侧追加物理量，
   新增语义信号 `semantic_instance_margin`。
3. **零改造接入**：导航侧的 `closed_loop_steps_rank*.jsonl` + `progress.json` 字段已齐，
   适配器只做映射（见 `nav/adapter.py`）。

本模块只依赖标准库，Python 3.9 兼容。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "route1/failure-event/v1"

# --------------------------------------------------------------------------- #
# 失败类型（沿用 failure-taxonomy.md §5 的 N/X 编号，全篇统一引用）
# --------------------------------------------------------------------------- #


class FailureType(str, Enum):
    """导航侧 N1..N8 + 操作侧 X1..X3。"""

    N1 = "N1"  # 无可执行动作
    N2 = "N2"  # 执行漂移（几何）
    N3 = "N3"  # 计划老化
    N4 = "N4"  # 综合置信崩塌（分不清主因）
    N5 = "N5"  # 碰撞 / 卡死
    N6 = "N6"  # 不确定性（当前配置未启用）
    N7 = "N7"  # 语义 / 指令级失败（**新增，本项目最大缺口**）
    N8 = "N8"  # 恢复本身失败（二阶）
    X1 = "X1"  # 抓空
    X2 = "X2"  # 滑落
    X3 = "X3"  # 放错


FAILURE_TYPE_DESCRIPTION: Dict[str, str] = {
    "N1": "无可执行动作：System-2 给出的计划里没有可执行的下一步",
    "N2": "执行漂移：实际轨迹与计划轨迹的几何偏离超阈值",
    "N3": "计划老化：计划执行步数达到硬上限，纯计时器触发",
    "N4": "综合置信崩塌：加权分数越过迟滞高阈，但主因不明确",
    "N5": "碰撞 / 卡死：Habitat 碰撞计数在窗口内超限",
    "N6": "不确定性：轨迹采样方差过大（当前配置 enabled=False）",
    "N7": "语义 / 指令级失败：走对了路但执行与指令不一致（选错实例、理解错、子目标顺序错）",
    "N8": "恢复失败：恢复动作执行后分数未下降，需要升级策略或弃权",
    "X1": "抓空：夹爪闭合时未建立有效接触",
    "X2": "滑落：抓稳后在抬起/搬运阶段物体脱离夹爪",
    "X3": "放错：物体被放在目标位置之外",
}


# --------------------------------------------------------------------------- #
# 恢复动作空间（把「只有 replan」扩成策略集，见 route1-bridge-plan.md §3）
# --------------------------------------------------------------------------- #


class RecoveryAction(str, Enum):
    NONE = "none"
    REPLAN = "replan"                          # 现有唯一动作：重新调用 System-2 出计划
    SWITCH_INSTANCE = "switch_instance"        # 换实例假设（语义失败专用）
    LOCAL_ESCAPE = "local_escape"              # 局部脱离：短程闭环，不走 System-2
    REPARSE_INSTRUCTION = "reparse_instruction"  # 重新解析指令
    ASK_CLARIFICATION = "ask_clarification"    # 弃权：向上层/人工请求澄清
    RETRY_GRASP = "retry_grasp"                # 操作侧：微调位姿后重试抓取
    CHANGE_GRASP_POINT = "change_grasp_point"  # 操作侧：换抓取点
    REORDER_SUBGOALS = "reorder_subgoals"      # 操作侧：改任务顺序
    ESCALATE = "escalate"                      # N8：升级到另一种动作类型


#: 每个失败类型允许的恢复动作（用于 `execute_recovery` 的合法性校验）
RECOVERY_MENU: Dict[str, Tuple[str, ...]] = {
    "N1": ("replan", "switch_instance", "ask_clarification"),
    "N2": ("replan", "local_escape"),
    "N3": ("replan",),
    "N4": ("replan", "local_escape", "switch_instance", "reparse_instruction", "ask_clarification"),
    "N5": ("replan", "local_escape"),
    "N6": ("replan", "local_escape"),
    "N7": ("switch_instance", "reparse_instruction", "ask_clarification"),
    "N8": ("escalate", "ask_clarification"),
    "X1": ("retry_grasp", "change_grasp_point", "reorder_subgoals"),
    "X2": ("change_grasp_point", "reorder_subgoals"),
    "X3": ("retry_grasp", "change_grasp_point"),
}

#: 恢复动作的中文标签（用于报告与前端展示）
RECOVERY_ACTION_LABEL: Dict[str, str] = {
    "none": "不动",
    "replan": "重新规划",
    "switch_instance": "换实例假设",
    "local_escape": "局部脱离",
    "reparse_instruction": "重新解析指令",
    "ask_clarification": "请求澄清",
    "retry_grasp": "重试抓取",
    "change_grasp_point": "换抓取点",
    "reorder_subgoals": "改任务顺序",
    "escalate": "升级策略",
}


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


def _finite_or_none(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("%s must be finite or None" % name)
    return number


@dataclass
class Signals:
    """跨仿真器统一的信号字典。

    导航侧五项沿用 InternNav 字段名；操作侧追加物理量。未产生的信号一律 `None`
    （**不要补 0**——导航侧加权分数的分母会剔除不可用信号，补 0 会改变语义）。
    """

    feedback_score: Optional[float] = None
    trajectory_drift: Optional[float] = None
    plan_age: Optional[int] = None
    collision_delta: Optional[int] = None
    steps_since_last_s2: Optional[int] = None
    # 新增：语义失败信号（N7）。定义为「语义失败风险」∈[0,1]，0.5 为判决边界，
    # confidence 越高、方向越明确，风险越极端。见 nav/feedback5.py。
    semantic_instance_margin: Optional[float] = None
    semantic_verdict: Optional[str] = None       # "PASS" | "LOST" | "UNSURE" | None
    semantic_confidence: Optional[float] = None
    # 实测发现 Agnes Monitor 在含糊帧上判决不稳定（同帧同指令、temperature=0，
    # 少数派可达 20–40%），而 **confidence 挡不住这个错误**（判 LOST 的那几次
    # confidence 反而更高）。唯一有效的可靠性指标是**多次投票的一致度**。
    semantic_agreement: Optional[float] = None   # 多数派占比 ∈ [0.5, 1]
    semantic_votes: Optional[int] = None         # 实际发起的投票次数
    semantic_source: Optional[str] = None        # "agnes:agnes-3.0-flash" 等
    # 操作侧物理量
    gripper_width_m: Optional[float] = None
    contact_force_n: Optional[float] = None
    object_dz_m: Optional[float] = None
    dist_to_goal_m: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN_FIELDS: Tuple[str, ...] = (
        "feedback_score",
        "trajectory_drift",
        "plan_age",
        "collision_delta",
        "steps_since_last_s2",
        "semantic_instance_margin",
        "semantic_verdict",
        "semantic_confidence",
        "semantic_agreement",
        "semantic_votes",
        "semantic_source",
        "gripper_width_m",
        "contact_force_n",
        "object_dz_m",
        "dist_to_goal_m",
    )

    def to_dict(self) -> Dict[str, Any]:
        out = {name: getattr(self, name) for name in self.KNOWN_FIELDS}
        if self.extra:
            out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Signals":
        data = dict(data or {})
        known = {name: data.pop(name, None) for name in cls.KNOWN_FIELDS}
        for name in ("plan_age", "collision_delta", "steps_since_last_s2", "semantic_votes"):
            if known[name] is not None:
                known[name] = int(known[name])
        for name in ("feedback_score", "trajectory_drift", "semantic_instance_margin",
                     "semantic_confidence", "semantic_agreement", "gripper_width_m",
                     "contact_force_n", "object_dz_m", "dist_to_goal_m"):
            known[name] = _finite_or_none(known[name], name)
        return cls(extra=data, **known)


@dataclass
class Attribution:
    """REFLECT 式归因：不只给 dominant signal，还给一句人话 + 证据。"""

    kind: str = "unknown"
    summary_l1: str = ""
    summary_l2: str = ""
    evidence: List[str] = field(default_factory=list)
    dominant_signal: Optional[str] = None
    source: str = "rule"  # rule | llm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "summary_l1": self.summary_l1,
            "summary_l2": self.summary_l2,
            "evidence": list(self.evidence),
            "dominant_signal": self.dominant_signal,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Attribution":
        data = dict(data or {})
        return cls(
            kind=str(data.get("kind", "unknown")),
            summary_l1=str(data.get("summary_l1", "")),
            summary_l2=str(data.get("summary_l2", "")),
            evidence=list(data.get("evidence") or []),
            dominant_signal=data.get("dominant_signal"),
            source=str(data.get("source", "rule")),
        )


@dataclass
class Recovery:
    """恢复动作 + 预算账本（预算意识是我们比 REFLECT/AVERT 都强的一层）。"""

    reason_code: str = "none"
    action: str = RecoveryAction.NONE.value
    budget_used: int = 0
    budget_left: int = 0
    bypassed_cooldown: bool = False
    escalated_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "action": self.action,
            "budget_used": int(self.budget_used),
            "budget_left": int(self.budget_left),
            "bypassed_cooldown": bool(self.bypassed_cooldown),
            "escalated_from": self.escalated_from,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Recovery":
        data = dict(data or {})
        return cls(
            reason_code=str(data.get("reason_code", "none")),
            action=str(data.get("action", RecoveryAction.NONE.value)),
            budget_used=int(data.get("budget_used", 0)),
            budget_left=int(data.get("budget_left", 0)),
            bypassed_cooldown=bool(data.get("bypassed_cooldown", False)),
            escalated_from=data.get("escalated_from"),
        )


@dataclass
class Outcome:
    recovered: Optional[bool] = None
    post_score: Optional[float] = None
    steps_to_recover: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recovered": self.recovered,
            "post_score": self.post_score,
            "steps_to_recover": self.steps_to_recover,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Outcome":
        data = dict(data or {})
        return cls(
            recovered=data.get("recovered"),
            post_score=_finite_or_none(data.get("post_score"), "post_score"),
            steps_to_recover=(
                None if data.get("steps_to_recover") is None else int(data["steps_to_recover"])
            ),
        )


@dataclass
class FailureEvent:
    """跨仿真器的唯一契约。字段与 route1-bridge-plan.md §6 一一对应。"""

    id: str
    t: float
    source: str                      # "nav" | "manip"
    type: Optional[str]              # N1..N8 | X1..X3 | None（仅观测、未判定失败）
    signals: Signals = field(default_factory=Signals)
    attribution: Attribution = field(default_factory=Attribution)
    recovery: Recovery = field(default_factory=Recovery)
    outcome: Outcome = field(default_factory=Outcome)
    scene_id: Optional[str] = None
    episode_id: Optional[int] = None
    step: Optional[int] = None
    action_step: Optional[int] = None
    instruction: Optional[str] = None
    trigger_kind: Optional[str] = None
    schema: str = SCHEMA_VERSION

    # ---- 构造 ---------------------------------------------------------- #
    @staticmethod
    def make_id(source: str, scene_id: Optional[str], episode_id: Optional[int],
                step: Optional[int], action_step: Optional[int]) -> str:
        raw = "%s|%s|%s|%s|%s" % (source, scene_id, episode_id, step, action_step)
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return "%s:%s:%s:%s" % (source, scene_id, episode_id, digest)

    @classmethod
    def create(
        cls,
        source: str,
        type: Optional[str] = None,
        scene_id: Optional[str] = None,
        episode_id: Optional[int] = None,
        step: Optional[int] = None,
        action_step: Optional[int] = None,
        instruction: Optional[str] = None,
        signals: Optional[Signals] = None,
        attribution: Optional[Attribution] = None,
        recovery: Optional[Recovery] = None,
        outcome: Optional[Outcome] = None,
        trigger_kind: Optional[str] = None,
        t: Optional[float] = None,
    ) -> "FailureEvent":
        event = cls(
            id=cls.make_id(source, scene_id, episode_id, step, action_step),
            t=float(time.time() if t is None else t),
            source=source,
            type=type,
            signals=signals or Signals(),
            attribution=attribution or Attribution(),
            recovery=recovery or Recovery(),
            outcome=outcome or Outcome(),
            scene_id=scene_id,
            episode_id=episode_id,
            step=step,
            action_step=action_step,
            instruction=instruction,
            trigger_kind=trigger_kind,
        )
        event.validate()
        return event

    # ---- 校验 ---------------------------------------------------------- #
    def validate(self) -> "FailureEvent":
        if self.source not in ("nav", "manip"):
            raise ValueError("source must be 'nav' or 'manip', got %r" % (self.source,))
        if self.type is not None and self.type not in FailureType.__members__:
            raise ValueError(
                "type must be one of %s or None, got %r"
                % (sorted(FailureType.__members__), self.type)
            )
        if self.recovery.action not in set(a.value for a in RecoveryAction):
            raise ValueError("unknown recovery action %r" % (self.recovery.action,))
        if self.type is not None and self.recovery.action != RecoveryAction.NONE.value:
            allowed = RECOVERY_MENU.get(self.type)
            if allowed is not None and self.recovery.action not in allowed:
                raise ValueError(
                    "action %r is not allowed for failure type %s (menu=%s)"
                    % (self.recovery.action, self.type, allowed)
                )
        score = self.signals.feedback_score
        if score is not None and not (0.0 <= score <= 1.0):
            raise ValueError("signals.feedback_score must be within [0,1]")
        margin = self.signals.semantic_instance_margin
        if margin is not None and not (0.0 <= margin <= 1.0):
            raise ValueError("signals.semantic_instance_margin must be within [0,1]")
        if not math.isfinite(float(self.t)):
            raise ValueError("t must be finite")
        return self

    # ---- 序列化 -------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "id": self.id,
            "t": self.t,
            "source": self.source,
            "type": self.type,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "action_step": self.action_step,
            "instruction": self.instruction,
            "trigger_kind": self.trigger_kind,
            "signals": self.signals.to_dict(),
            "attribution": self.attribution.to_dict(),
            "recovery": self.recovery.to_dict(),
            "outcome": self.outcome.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureEvent":
        schema = data.get("schema", SCHEMA_VERSION)
        if schema != SCHEMA_VERSION:
            raise ValueError("unsupported schema %r (expected %r)" % (schema, SCHEMA_VERSION))
        return cls(
            id=str(data["id"]),
            t=float(data["t"]),
            source=str(data["source"]),
            type=data.get("type"),
            signals=Signals.from_dict(data.get("signals")),
            attribution=Attribution.from_dict(data.get("attribution")),
            recovery=Recovery.from_dict(data.get("recovery")),
            outcome=Outcome.from_dict(data.get("outcome")),
            scene_id=data.get("scene_id"),
            episode_id=data.get("episode_id"),
            step=data.get("step"),
            action_step=data.get("action_step"),
            instruction=data.get("instruction"),
            trigger_kind=data.get("trigger_kind"),
            schema=schema,
        ).validate()

    @classmethod
    def from_json(cls, line: str) -> "FailureEvent":
        return cls.from_dict(json.loads(line))


def validate_stream(events: Iterable[FailureEvent]) -> Dict[str, Any]:
    """对一条事件流做完整性检查，返回可打印的报告。"""
    events = list(events)
    by_source: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    by_action: Dict[str, int] = {}
    ids = set()
    duplicates = 0
    for event in events:
        by_source[event.source] = by_source.get(event.source, 0) + 1
        key = event.type or "none"
        by_type[key] = by_type.get(key, 0) + 1
        by_action[event.recovery.action] = by_action.get(event.recovery.action, 0) + 1
        if event.id in ids:
            duplicates += 1
        ids.add(event.id)
    return {
        "count": len(events),
        "unique_ids": len(ids),
        "duplicate_ids": duplicates,
        "by_source": by_source,
        "by_type": by_type,
        "by_action": by_action,
    }

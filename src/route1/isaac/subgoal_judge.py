"""操作侧子目标判定器（contact → grasp → lift → in_place）。

设计原则
--------
1. **纯函数 + 状态机**：不 import Isaac / pxr / omni，只用标准库 + numpy，
   因此可以在普通 Python 下被单元测试（`python3 -m route1.isaac.subgoal_judge --selftest`），
   并且在 Isaac 的 kit python 3.12 里也照样跑。
2. **判定只用物理量**：接触力、关节位置、物体位姿/速度、相对位姿保持步数。
   没有任何"用了某个 mode 就返回某个答案"的硬编码——同一份判定代码同时服务
   success / X1 / X2 / X3 四组实验，失败与否完全由物理决定。
3. **阈值有实测依据**：`SubgoalThresholds` 的默认值是先跑 smoke、看 trace 数值分布
   再定的，注释里写了实测区间（见 `notes/isaac-manip-report.md`）。

Python 3.9 兼容（服务器导航环境是 3.9）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # numpy 在两边都有；缺失时退化成纯 Python 实现
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


# --------------------------------------------------------------------------- #
# 阈值
# --------------------------------------------------------------------------- #


@dataclass
class SubgoalThresholds:
    """子目标判定阈值。单位一律 SI（m / N / s）。"""

    # -- contact ---------------------------------------------------------- #
    #: 接触力上限噪声地板。实测：完全无接触时 contact_force_n 恒为 0.0；
    #: 手指刚贴上但还没夹紧时约 0.2~1.0 N；夹紧后 5~40 N。
    contact_force_min_n: float = 0.5
    #: 需要在窗口内连续多少步有接触才算 contact 成立（去抖）
    contact_window: int = 3

    # -- grasp ------------------------------------------------------------ #
    #: 关节位置与闭合目标的误差容限。全闭合（抓空）时误差≈0 也会"到位"，
    #: 所以 grasp 不能只看关节到位——必须叠加"相对位姿保持"。
    grasp_joint_tol_m: float = 0.004
    #: 物体相对夹爪位姿需要在最近 N 步内保持
    grasp_hold_steps: int = 20
    #: 相对位姿漂移超过该值即视为没夹住
    grasp_rel_drift_m: float = 0.006
    #: 物体线速度超过该值即视为在滑/没夹住
    grasp_vel_max_mps: float = 0.02

    # -- lift ------------------------------------------------------------- #
    #: 物体 z 相对初始 z 的抬升阈值
    lift_dz_m: float = 0.05

    # -- in_place --------------------------------------------------------- #
    #: 物体中心到目标点的水平距离阈值
    in_place_xy_m: float = 0.05
    #: 已经放开：夹爪张开到该宽度以上
    release_width_m: float = 0.02
    #: 已经放开：接触力低于该值
    release_force_n: float = 0.5


# --------------------------------------------------------------------------- #
# 单步观测
# --------------------------------------------------------------------------- #


@dataclass
class StepObs:
    """判定器需要的**全部**输入，全部来自仿真器实测。"""

    step: int
    #: 世界时间（仿真秒），用于核对步长
    sim_time: float = 0.0
    #: 夹爪两指内侧间距（m）
    gripper_width_m: float = 0.0
    #: 夹爪两指与物体之间的接触力之和（N）
    contact_force_n: float = 0.0
    #: 参与接触的接触对数量
    contact_pairs: int = 0
    #: 物体中心世界坐标
    object_pos: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    #: 物体线速度（m/s）
    object_vel: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    #: 夹爪基座（手掌）世界位置
    gripper_pos: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    #: 夹爪朝向的四元数 (w, x, y, z)
    gripper_quat: Sequence[float] = field(default_factory=lambda: (1.0, 0.0, 0.0, 0.0))
    #: 关节位置（两指），用于核对是否真的闭合到目标
    joint_positions: Sequence[float] = field(default_factory=lambda: (0.0, 0.0))
    #: 关节驱动目标
    joint_targets: Sequence[float] = field(default_factory=lambda: (0.0, 0.0))
    #: 任务相位名（仅用于 trace 展示，不参与判定）
    phase: str = ""


@dataclass
class SubgoalState:
    contact: bool = False
    grasp: bool = False
    lift: bool = False
    in_place: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return {
            "contact": self.contact,
            "grasp": self.grasp,
            "lift": self.lift,
            "in_place": self.in_place,
        }

    @property
    def count(self) -> int:
        return int(self.contact) + int(self.grasp) + int(self.lift) + int(self.in_place)


# --------------------------------------------------------------------------- #
# 判定器
# --------------------------------------------------------------------------- #


class SubgoalJudge:
    """把逐步观测流变成子目标状态流，并给出**基于物理量**的失败判定。"""

    def __init__(self, thresholds: Optional[SubgoalThresholds] = None,
                 goal_xy: Optional[Sequence[float]] = None,
                 object_init_pos: Optional[Sequence[float]] = None) -> None:
        self.th = thresholds or SubgoalThresholds()
        self.goal_xy: Optional[Tuple[float, float]] = (
            None if goal_xy is None else (float(goal_xy[0]), float(goal_xy[1]))
        )
        self.object_init_pos: Optional[Tuple[float, float, float]] = (
            None if object_init_pos is None
            else (float(object_init_pos[0]), float(object_init_pos[1]), float(object_init_pos[2]))
        )
        self.state = SubgoalState()
        self.step_index = 0
        self._contact_streak = 0
        # 相对位姿窗口：[(object_pos - gripper_pos)]，同时记录物体速度
        self._rel_window: List[Tuple[float, float, float]] = []
        self._vel_window: List[float] = []
        # 事件（首次达成 / 首次违反）
        self.trace: List[Dict[str, Any]] = []
        self.transitions: List[Dict[str, Any]] = []
        # 关键历史，用于事后归因
        self.min_width_seen = float("inf")
        self.max_force_seen = 0.0
        self.max_dz_seen = 0.0
        self.dz_at_grasp: Optional[float] = None

    # ---- 内部工具 ------------------------------------------------------- #

    @staticmethod
    def _dist(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _dist_xy(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)

    # ---- 主入口 --------------------------------------------------------- #

    def update(self, obs: StepObs) -> SubgoalState:
        th = self.th
        self.step_index = obs.step
        if self.object_init_pos is None:
            self.object_init_pos = (float(obs.object_pos[0]), float(obs.object_pos[1]),
                                    float(obs.object_pos[2]))

        prev = SubgoalState(**asdict(self.state))

        # ---------------- contact：接触力窗口 ---------------- #
        if obs.contact_force_n >= th.contact_force_min_n:
            self._contact_streak += 1
        else:
            self._contact_streak = 0
        has_contact = self._contact_streak >= th.contact_window

        # ---------------- 相对位姿窗口（grasp 用） ---------------- #
        rel = (float(obs.object_pos[0]) - float(obs.gripper_pos[0]),
               float(obs.object_pos[1]) - float(obs.gripper_pos[1]),
               float(obs.object_pos[2]) - float(obs.gripper_pos[2]))
        self._rel_window.append(rel)
        if len(self._rel_window) > th.grasp_hold_steps:
            self._rel_window.pop(0)
        vel = math.sqrt(sum(float(v) ** 2 for v in obs.object_vel))
        self._vel_window.append(vel)
        if len(self._vel_window) > th.grasp_hold_steps:
            self._vel_window.pop(0)

        # ---------------- grasp ---------------- #
        # 条件 1：两指关节都接近"夹住东西"的位置——注意这里**不是**判断关节是否
        # 到了驱动目标（抓空时目标=全闭合也到位），而是判断关节被物体挡住、
        # 停在物体宽度上；抓空时 width→0，会触发 X1 判定。
        joint_err = 0.0
        if obs.joint_targets and len(obs.joint_targets) == len(obs.joint_positions):
            joint_err = max(abs(float(p) - float(t))
                            for p, t in zip(obs.joint_positions, obs.joint_targets))
        joints_at_target = joint_err <= th.grasp_joint_tol_m
        # 条件 2：窗口内相对位姿漂移小
        rel_drift = 0.0
        if len(self._rel_window) >= 2:
            first = self._rel_window[0]
            rel_drift = max(self._dist(p, first) for p in self._rel_window)
        pose_held = (len(self._rel_window) >= th.grasp_hold_steps
                     and rel_drift <= th.grasp_rel_drift_m)
        # 条件 3：物体速度≈0
        vel_small = bool(self._vel_window) and max(self._vel_window) <= th.grasp_vel_max_mps
        # 条件 4：必须有接触（抓空时没有接触，不该判 grasp）
        has_grasp = bool(has_contact and joints_at_target and pose_held and vel_small
                         and obs.contact_pairs > 0)

        # ---------------- lift ---------------- #
        dz = float(obs.object_pos[2]) - float(self.object_init_pos[2])
        self.max_dz_seen = max(self.max_dz_seen, dz)
        has_lift = dz >= th.lift_dz_m

        # ---------------- in_place ---------------- #
        dist_goal = None
        if self.goal_xy is not None:
            dist_goal = self._dist_xy(obs.object_pos, (self.goal_xy[0], self.goal_xy[1], 0.0))
        released = (obs.gripper_width_m >= th.release_width_m
                    and obs.contact_force_n <= th.release_force_n)
        has_in_place = bool(dist_goal is not None and dist_goal <= th.in_place_xy_m and released)

        self.state = SubgoalState(contact=bool(has_contact), grasp=has_grasp,
                                  lift=bool(has_lift), in_place=has_in_place)

        # ---------------- 统计 ---------------- #
        self.min_width_seen = min(self.min_width_seen, float(obs.gripper_width_m))
        self.max_force_seen = max(self.max_force_seen, float(obs.contact_force_n))
        if has_grasp and self.dz_at_grasp is None:
            self.dz_at_grasp = dz

        # ---------------- trace ---------------- #
        for name in ("contact", "grasp", "lift", "in_place"):
            before = bool(getattr(prev, name))
            after = bool(getattr(self.state, name))
            if before != after:
                self.transitions.append({
                    "step": int(obs.step),
                    "subgoal": name,
                    "value": after,
                    "phase": obs.phase,
                    "gripper_width_m": round(float(obs.gripper_width_m), 6),
                    "contact_force_n": round(float(obs.contact_force_n), 6),
                    "object_dz_m": round(dz, 6),
                    "dist_to_goal_m": None if dist_goal is None else round(dist_goal, 6),
                    "rel_drift_m": round(rel_drift, 6),
                    "object_speed_mps": round(vel, 6),
                })

        self.trace.append({
            "step": int(obs.step),
            "sim_time": round(float(obs.sim_time), 6),
            "phase": obs.phase,
            "subgoal_states": self.state.as_dict(),
            "object_pose": {
                "pos": [round(float(v), 6) for v in obs.object_pos],
                "quat": None,
                "lin_vel": [round(float(v), 6) for v in obs.object_vel],
            },
            "gripper_width_m": round(float(obs.gripper_width_m), 6),
            "gripper_pos": [round(float(v), 6) for v in obs.gripper_pos],
            "contact_force_n": round(float(obs.contact_force_n), 6),
            "contact_pairs": int(obs.contact_pairs),
            "joint_positions": [round(float(v), 6) for v in obs.joint_positions],
            "joint_targets": [round(float(v), 6) for v in obs.joint_targets],
            "object_dz_m": round(dz, 6),
            "dist_to_goal_m": None if dist_goal is None else round(dist_goal, 6),
            "object_speed_mps": round(vel, 6),
            "rel_drift_m": round(rel_drift, 6),
        })
        return self.state

    # ---- 诊断输出 ------------------------------------------------------- #

    def last_close_error(self) -> float:
        """最近若干步里「指头位置 vs 闭合指令」的最大误差（m）。"""
        window = self.trace[-self.th.grasp_hold_steps:]
        if not window:
            return float("inf")
        worst = 0.0
        for entry in window:
            positions = entry.get("joint_positions") or []
            targets = entry.get("joint_targets") or []
            if not positions or len(positions) != len(targets):
                return float("inf")
            worst = max(worst, max(abs(float(p) - float(t))
                                   for p, t in zip(positions, targets)))
        return worst

    def fingers_at_close_target(self) -> bool:
        """最近 N 步指头是否都停在闭合指令附近。

        抓空时弹簧/驱动会把指头推到指令位置；抓稳时指头被物体挡住，误差大。
        """
        if len(self.trace) < self.th.grasp_hold_steps:
            return False
        return self.last_close_error() <= self.th.grasp_joint_tol_m

    def first_true_step(self, subgoal: str) -> Optional[int]:
        for entry in self.transitions:
            if entry["subgoal"] == subgoal and entry["value"]:
                return int(entry["step"])
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "final_state": self.state.as_dict(),
            "first_true_step": {name: self.first_true_step(name)
                                for name in ("contact", "grasp", "lift", "in_place")},
            "min_gripper_width_m": None if self.min_width_seen == float("inf")
            else round(self.min_width_seen, 6),
            "max_contact_force_n": round(self.max_force_seen, 6),
            "max_object_dz_m": round(self.max_dz_seen, 6),
            "transitions": self.transitions,
            "n_steps": len(self.trace),
        }

    def write_trace_jsonl(self, path: str) -> int:
        """把逐步 trace 写成一整个 JSON 数组文件（--record-trace）。"""
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.trace, handle, ensure_ascii=False, indent=1)
        return len(self.trace)


# --------------------------------------------------------------------------- #
# 失败判定（同样只吃物理量）
# --------------------------------------------------------------------------- #


@dataclass
class FailureVerdict:
    """一条失败判定结果；`type` 为 None 表示未判失败。"""

    type: Optional[str] = None
    step: Optional[int] = None
    kind: str = "none"
    summary_l1: str = ""
    evidence: List[str] = field(default_factory=list)
    recovery_action: str = "none"
    reason_code: str = "none"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "step": self.step,
            "kind": self.kind,
            "summary_l1": self.summary_l1,
            "evidence": list(self.evidence),
            "recovery_action": self.recovery_action,
            "reason_code": self.reason_code,
        }


def detect_grasp_failure(judge: SubgoalJudge,
                         width_closed_air_m: float = 0.02) -> FailureVerdict:
    """X1 抓空判定：进入了闭合相位、却没有建立接触 / grasp 不成立。

    物理依据
    --------
    * 抓空时两指可以把驱动目标走满 → 指头位置收敛到"闭合指令"，
      `min_gripper_width_m` 落到（接近）目标闭合宽度；
    * 抓稳时指头被物体挡住 → 宽度停在**物体实际宽度**上（0.06 m 物体 → ≈0.08 m
      指心间距口径下明显大于闭合目标），且接触力非零。
    * 这里的判据是**"两指都收敛到闭合指令" ∧ "整段没有任何有效接触"**，
      不依赖绝对宽度阈值有多漂亮：只要夹爪到位而力恒 0，就是抓空。
    """
    if judge.state.grasp:
        return FailureVerdict()
    if judge.max_force_seen >= judge.th.contact_force_min_n:
        # 有接触但没夹稳 → 不是"抓空"，交给 X2 / 其他判据
        return FailureVerdict()
    if not judge.fingers_at_close_target():
        return FailureVerdict()
    closed_air = judge.min_width_seen <= width_closed_air_m
    evidence = [
        "两指位置在最近 %d 步内都收敛到闭合指令（max|pos-target|=%.6f m ≤ %.6f m）"
        % (judge.th.grasp_hold_steps, judge.last_close_error(),
           judge.th.grasp_joint_tol_m),
        "min_gripper_width_m=%.6f（两指之间是空的，宽度走到闭合量级）"
        % (judge.min_width_seen,),
        "max_contact_force_n=%.6f < %.3f（整个尝试里没有出现有效接触）"
        % (judge.max_force_seen, judge.th.contact_force_min_n),
        "contact_pairs 从未出现；grasp 未成立：contact=%s lift=%s"
        % (judge.state.contact, judge.state.lift),
    ]
    if not closed_air:
        evidence.append(
            '注：宽度未到 %.3f m 以下（可能夹爪被物体或障碍挡住），'
            '但"到位且零接触"依然只可能解释为抓空' % (width_closed_air_m,))
    return FailureVerdict(
        type="X1", step=judge.step_index,
        kind="grasp_closed_without_contact", recovery_action="retry_grasp",
        reason_code="gripper_closed_on_air",
        summary_l1="夹爪闭合到位但两指之间没有建立任何接触，抓取失败（抓空）",
        evidence=evidence,
    )


def detect_slip_failure(judge: SubgoalJudge,
                        grasp_seen: bool = True) -> FailureVerdict:
    """X2 滑落判定：抓稳并抬起过，随后物体脱离夹爪、掉回初始水平。

    必须排除"正常放下"：正常放置时物体也在下降，但会停在**目标点附近**，
    且最后一次处于 lifted 状态之后物体不再被重新抓起。因此判据是
    「抬起过 + 现在 dz 掉回 → 但水平上并不在目标点容差内」。
    """
    if not grasp_seen:
        return FailureVerdict()
    if judge.first_true_step("lift") is None:
        return FailureVerdict()
    if judge.state.in_place or judge.state.grasp:
        return FailureVerdict()
    peak = judge.max_dz_seen
    last_dz = float(judge.trace[-1]["object_dz_m"]) if judge.trace else 0.0
    dropped = peak >= judge.th.lift_dz_m and last_dz <= peak * 0.35
    if not dropped:
        return FailureVerdict()
    # 落在目标点容差内 → 是正常放置，不是滑落
    if judge.goal_xy is not None and judge.trace:
        pos = judge.trace[-1]["object_pose"]["pos"]
        dist = math.sqrt((float(pos[0]) - judge.goal_xy[0]) ** 2
                         + (float(pos[1]) - judge.goal_xy[1]) ** 2)
        if dist <= judge.th.in_place_xy_m:
            return FailureVerdict()
    mean = _mean_of_trace(judge.trace, "contact_force_n", last_n=20)
    return FailureVerdict(
        type="X2", step=judge.step_index, kind="object_slip",
        recovery_action="change_grasp_point", reason_code="object_dropped_while_holding",
        summary_l1="物体在抬起/搬运阶段脱离夹爪掉落，物体高度回落到初始水平（滑落）",
        evidence=[
            "peak_object_dz_m=%.6f（抬到过 %.3f m 以上）" % (prompt(peak), judge.th.lift_dz_m),
            "last_object_dz_m=%.6f（掉回接近初始高度）" % (last_dz,),
            "最近 20 步平均接触力=%.6f N，当前 contact=%s grasp=%s"
            % (mean, judge.state.contact, judge.state.grasp),
            "min_gripper_width_m=%.6f（抓空时两指走到全闭合宽度）"
            % (judge.min_width_seen,),
        ],
    )


def detect_misplace_failure(judge: SubgoalJudge,
                            dist_to_goal_m: Optional[float]) -> FailureVerdict:
    """X3 放错判定：已经放开物体，但物体水平位置离目标点太远。"""
    if judge.state.in_place or dist_to_goal_m is None:
        return FailureVerdict()
    if judge.trace is None or not judge.trace:
        return FailureVerdict()
    released = (judge.trace[-1]["gripper_width_m"] >= judge.th.release_width_m
                and judge.trace[-1]["contact_force_n"] <= judge.th.release_force_n)
    # 必须"已经放下"（抬起来过、且当前是松开的）才算放错，而不是"还没放"
    if not (released and judge.max_dz_seen >= judge.th.lift_dz_m):
        return FailureVerdict()
    return FailureVerdict(
        type="X3", step=judge.step_index, kind="wrong_placement",
        recovery_action="retry_grasp", reason_code="placed_outside_goal_tolerance",
        summary_l1="物体已放开但落点偏离目标位置超过容差，放置失败（放错）",
        evidence=[
            "dist_to_goal_m=%.6f > %.3f（水平距离超容差）"
            % (dist_to_goal_m, judge.th.in_place_xy_m),
            "gripper_width_m=%.6f >= %.3f 且 contact_force_n=%.6f <= %.3f（确认已放开）"
            % (judge.trace[-1]["gripper_width_m"], judge.th.release_width_m,
               judge.trace[-1]["contact_force_n"], judge.th.release_force_n),
            "max_object_dz_m=%.6f（确实搬运过）" % (judge.max_dz_seen,),
        ],
    )


def prompt(value: float) -> float:
    return round(float(value), 6)


def _mean_of_trace(trace: List[Dict[str, Any]], key: str, last_n: int = 20) -> float:
    window = trace[-last_n:]
    if not window:
        return 0.0
    return sum(float(entry.get(key) or 0.0) for entry in window) / float(len(window))


# --------------------------------------------------------------------------- #
# 离线自检（不需要 Isaac）
# --------------------------------------------------------------------------- #


def _selftest() -> int:
    """用合成的物理量序列验证判定器行为（不依赖仿真器）。"""
    failures: List[str] = []

    def make_judge() -> SubgoalJudge:
        return SubgoalJudge(goal_xy=(0.35, 0.0), object_init_pos=(0.0, 0.0, 0.78))

    def base_obs(step: int, **kw: Any) -> StepObs:
        data = dict(step=step, sim_time=step / 60.0, gripper_width_m=0.041,
                    contact_force_n=0.0, contact_pairs=0, object_pos=(0.0, 0.0, 0.78),
                    object_vel=(0.0, 0.0, 0.0), gripper_pos=(0.0, 0.0, 0.78),
                    joint_positions=(0.045, 0.045), joint_targets=(0.0015, 0.0015),
                    phase="close")
        data.update(kw)
        return StepObs(**data)

    # --- 抓稳 → 抬起 → 放到目标点上方 → 放开，四子目标应依次为 True ---
    judge = make_judge()
    for step in range(60):
        judge.update(base_obs(step, gripper_width_m=0.040, contact_force_n=12.0,
                              contact_pairs=2, object_vel=(0.0, 0.0, 0.0),
                              gripper_pos=(0.0, 0.0, 0.78),
                              object_pos=(0.0, 0.0, 0.78),
                              joint_positions=(0.0018, 0.0018)))
        if step == 59:
            if not judge.state.contact:
                failures.append("success-case: contact 未在 60 步内成立")
            if not judge.state.grasp:
                failures.append("success-case: grasp 未在 60 步内成立")
    for step in range(60, 140):
        judge.update(base_obs(step, gripper_width_m=0.040, contact_force_n=11.0,
                              contact_pairs=2, gripper_pos=(0.0, 0.0, 0.90),
                              object_pos=(0.0, 0.0, 0.90),
                              joint_positions=(0.0018, 0.0018)))
    if not judge.state.lift:
        failures.append("success-case: lift 未成立")
    for step in range(140, 220):
        judge.update(base_obs(step, gripper_width_m=0.041, contact_force_n=0.0,
                              contact_pairs=0, gripper_pos=(0.35, 0.0, 0.79),
                              object_pos=(0.35, 0.0, 0.78), phase="release",
                              joint_positions=(0.045, 0.045)))
    if not judge.state.in_place:
        failures.append("success-case: in_place 未成立")
    if judge.first_true_step("contact") is None or judge.first_true_step("grasp") is None:
        failures.append("success-case: transitions 未记录首次达成")

    # --- X1 抓空：两指走到全闭合宽度、无接触 ---
    judge_x1 = make_judge()
    for step in range(80):
        judge_x1.update(base_obs(step, gripper_width_m=0.004, contact_force_n=0.0,
                                 contact_pairs=0, joint_positions=(0.0005, 0.0005),
                                 object_pos=(0.0, 0.0, 0.78)))
    verdict_x1 = detect_grasp_failure(judge_x1)
    if verdict_x1.type != "X1":
        failures.append("X1-case: 期望 X1，实得 %r" % (verdict_x1.type,))

    # --- X2 滑落：先抓住抬起，再掉回 ---
    judge_x2 = make_judge()
    for step in range(60):
        judge_x2.update(base_obs(step, gripper_width_m=0.040, contact_force_n=12.0,
                                 contact_pairs=2, joint_positions=(0.0018, 0.0018)))
    grasp_seen = judge_x2.state.grasp
    if not grasp_seen:
        failures.append("X2-case: 前置条件 grasp 未成立")
    for step in range(60, 120):
        judge_x2.update(base_obs(step, gripper_width_m=0.040, contact_force_n=11.0,
                                 contact_pairs=2, gripper_pos=(0.0, 0.0, 0.90),
                                 object_pos=(0.0, 0.0, 0.90),
                                 joint_positions=(0.0018, 0.0018)))
    for step in range(120, 200):
        judge_x2.update(base_obs(step, gripper_width_m=0.041, contact_force_n=0.0,
                                 contact_pairs=0, gripper_pos=(0.05, 0.0, 0.90),
                                 object_pos=(0.05, 0.0, 0.779),
                                 joint_positions=(0.045, 0.045), phase="transport"))
    verdict_x2 = detect_slip_failure(judge_x2, grasp_seen=grasp_seen)
    if verdict_x2.type != "X2":
        failures.append("X2-case: 期望 X2，实得 %r" % (verdict_x2.type,))

    # --- X3 放错：搬运到偏移点后放开 ---
    judge_x3 = make_judge()
    for step in range(60):
        judge_x3.update(base_obs(step, gripper_width_m=0.040, contact_force_n=12.0,
                                 contact_pairs=2, joint_positions=(0.0018, 0.0018)))
    for step in range(60, 140):
        judge_x3.update(base_obs(step, gripper_width_m=0.040, contact_force_n=11.0,
                                 contact_pairs=2, gripper_pos=(0.0, 0.0, 0.90),
                                 object_pos=(0.0, 0.0, 0.90),
                                 joint_positions=(0.0018, 0.0018)))
    for step in range(140, 220):
        judge_x3.update(base_obs(step, gripper_width_m=0.041, contact_force_n=0.0,
                                 contact_pairs=0, gripper_pos=(0.20, 0.0, 0.79),
                                 object_pos=(0.20, 0.0, 0.78), phase="release",
                                 joint_positions=(0.045, 0.045)))
    verdict_x3 = detect_misplace_failure(judge_x3, dist_to_goal_m=0.15)
    if verdict_x3.type != "X3":
        failures.append("X3-case: 期望 X3，实得 %r" % (verdict_x3.type,))
    # 不能误报：成功那组不该被判 X1/X2/X3
    if detect_grasp_failure(judge).type is not None:
        failures.append("success-case: 被误判为 X1")
    if detect_slip_failure(judge, grasp_seen=True).type is not None:
        failures.append("success-case: 被误判为 X2")
    if detect_misplace_failure(judge, dist_to_goal_m=0.0).type is not None:
        failures.append("success-case: 被误判为 X3")

    report = {
        "success_case": judge.summary(),
        "x1": verdict_x1.as_dict(),
        "x2": verdict_x2.as_dict(),
        "x3": verdict_x3.as_dict(),
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("SELFTEST %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="操作侧子目标判定器（可离线自检）")
    parser.add_argument("--selftest", action="store_true", help="用合成物理量序列自检")
    parser.add_argument("--dump-defaults", action="store_true", help="打印默认阈值")
    args = parser.parse_args(argv)
    if args.dump_defaults:
        print(json.dumps(asdict(SubgoalThresholds()), ensure_ascii=False, indent=1))
        return 0
    return _selftest()


if __name__ == "__main__":
    sys.exit(main())

"""两条**场景无关**的触发通道：碰撞密度 + 进度停滞。

为什么是这两条（而不是再调一个漂移阈值）
--------------------------------------
跨场景实测（`notes/route1-nav-stability-plan.md` §1）：
- **碰撞**是唯一能把最差场景与其它场景分开的观测量（`QUCTc6BB5sX` 3.00 次/回合，
  另外三个场景 0.00 / 0.00 / 0.62）；但上游把这个信号用在**判死**上
  （`collision_watchdog_stop` → 直接 `STOP` 结束 episode，最差场景 6/13 次触发、5 次保证失败）。
  这里改成"要求重规划"，用的是同一个信号、但用在该用的地方。
- **进度**是目标相对量，天生不需要按场景调阈值。上游的几何通道里**没有**它：
  加权分只有 uncertainty（关）/ drift / plan_age（纯计时器）/ collision_risk（从未提供），
  实测 AUROC ≈ 0.44（低于随机）——"一路走远"这种失败它测不出来
  （本 run 里有 NE = 12.87 / 25.39 的回合）。

设计约束
-------
- **离线与在线共用这一份实现**（`route1.experiments.channel_eval` 与
  `route1.nav.overlay_v4` 都调它），避免"离线一个样、在线另一个样"。
- 只依赖**已经存在**的两个量：`collision_count`（Habitat metrics）与
  `distance_to_goal`（Habitat metrics），因此 upstream 一行都不用改。
- 每条通道**自己带冷却**（独立预算的轻量版），互不压制，也不压制上游的
  `recovery` / `no_action_replan_limit` 硬刹车。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

#: 通道触发的 reason 字符串（会写进日志的 `replan_reason`，便于事后归因）
REASON_COLLISION = "collision_density_replan"
REASON_PROGRESS = "progress_stall_replan"


@dataclass
class ChannelConfig:
    """两条通道的参数。默认值的标定依据写在每个字段后面。"""

    # ---- 碰撞通道 ----
    #: 观察窗口（**运动动作**数）。与上游看门狗同口径（8），便于对照。
    collision_window_actions: int = 8
    #: 窗口内碰撞次数达到多少就要求重规划。上游看门狗用的是 6/8 才"判死"；
    #: 这里只要求重规划（不是结束回合），所以取比它低一档的 5。
    collision_min_in_window: int = 5
    #: 碰撞通道自己的冷却（运动动作数），避免贴墙时反复重规划。
    collision_cooldown_steps: int = 6

    # ---- 进度通道 ----
    #: 进度比较窗口（运动动作数）。约等于一次自主重规划周期（PERIODIC_INTERVAL=12）。
    progress_window_actions: int = 12
    #: 窗口内朝目标的改善量小于这个值 ⇒ 判"停滞"（米）。
    #: 12 个运动动作 ≈ 3 m 位移；走对方向时改善远大于 0.3 m。
    progress_min_improvement_m: float = 0.3
    #: 窗口内**远离**目标的量超过这个值 ⇒ 判"走反了"（米）。
    progress_regress_m: float = 0.6
    #: 进度通道的冷却（运动动作数）。比碰撞长，避免一路都"停滞"时反复触发。
    progress_cooldown_steps: int = 12

    # ---- 防"重规划抖动"（thrash）的护栏 ----
    #: 每条通道每个 episode 最多触发多少次（<=0 表示不限制）。
    #: 标定依据：`r1ovl4` 里 `QUCTc6BB5sX ep50` 一个回合触发了 **76 次**重规划、
    #: 走了 501 步、撞了 325 次，最后仍然失败——而同一回合在 `r1ro4` 里 117 步就成功了。
    #: 所以"卡住时不停地重规划"是有害的，必须封顶。
    collision_max_per_episode: int = 8
    progress_max_per_episode: int = 8
    #: 冷却递增：每触发一次，下一次的冷却 ×growth（封顶 cooldown_max_multiple × 基础冷却）。
    #: 语义：同一回合里连续触发说明"重规划没解决卡住问题"，应该越来越保守。
    cooldown_growth: float = 2.0
    cooldown_max_multiple: float = 4.0

    #: 两条通道的开关（离线做单通道消融用）
    collision_enabled: bool = True
    progress_enabled: bool = True

    def __post_init__(self) -> None:
        if self.collision_min_in_window > self.collision_window_actions:
            raise ValueError("collision_min_in_window 不能大于 collision_window_actions")
        if self.progress_window_actions < 1:
            raise ValueError("progress_window_actions 必须 >= 1")


@dataclass
class TriggerChannels:
    """一个 episode 一个实例；`reset()` 清状态。

    使用顺序（与在线一致）：
    1. 每个**运动动作**之后：`push_collision(碰撞计数是否增加)`；
    2. 每个**决策点**：`evaluate(current_motion_step, distance_to_goal)` → 返回 reason 或 None。
    """

    config: ChannelConfig = field(default_factory=ChannelConfig)

    def __post_init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._collision_flags: Deque[bool] = deque(maxlen=self.config.collision_window_actions)
        self._distance_samples: List[Tuple[int, float]] = []
        self._last_collision_trigger_step: Optional[int] = None
        self._last_progress_trigger_step: Optional[int] = None
        self.collision_triggers = 0
        self.progress_triggers = 0

    # ------------------------------------------------------------------ #
    def push_collision(self, increased: bool) -> None:
        """记录一个运动动作是否新增碰撞（LOOK_DOWN 等非运动动作不要调用）。"""
        self._collision_flags.append(bool(increased))

    def collision_count_in_window(self) -> int:
        return int(sum(1 for flag in self._collision_flags if flag))

    # ------------------------------------------------------------------ #
    def _effective_cooldown(self, base: int, triggers: int) -> float:
        """同一回合里触发得越多，冷却越长（防抖动）。"""
        if triggers <= 0:
            return float(base)
        growth = self.config.cooldown_growth ** (triggers - 1)
        return float(base) * min(growth, self.config.cooldown_max_multiple)

    def _collision_trigger(self, current_step: int) -> bool:
        if not self.config.collision_enabled:
            return False
        if not self._collision_flags:
            return False
        if self.config.collision_max_per_episode > 0 \
                and self.collision_triggers >= self.config.collision_max_per_episode:
            return False
        if self.collision_count_in_window() < self.config.collision_min_in_window:
            return False
        if len(self._collision_flags) < self.config.collision_window_actions:
            return False
        last = self._last_collision_trigger_step
        cooldown = self._effective_cooldown(self.config.collision_cooldown_steps,
                                           self.collision_triggers)
        if last is not None and current_step - last < cooldown:
            return False
        self._last_collision_trigger_step = current_step
        self.collision_triggers += 1
        return True

    # ------------------------------------------------------------------ #
    def _progress_trigger(self, current_step: int, distance: float) -> Optional[str]:
        """返回 'progress_stall' / 'progress_regress' / None。"""
        if not self.config.progress_enabled:
            return None
        if self.config.progress_max_per_episode > 0 \
                and self.progress_triggers >= self.config.progress_max_per_episode:
            return None
        samples = self._distance_samples
        if not samples:
            return None
        reference: Optional[Tuple[int, float]] = None
        for step, value in samples:
            if current_step - step >= self.config.progress_window_actions:
                reference = (step, value)
            else:
                break
        if reference is None:
            return None
        last = self._last_progress_trigger_step
        cooldown = self._effective_cooldown(self.config.progress_cooldown_steps,
                                            self.progress_triggers)
        if last is not None and current_step - last < cooldown:
            return None
        improvement = reference[1] - distance           # >0 = 靠近目标
        if improvement < -self.config.progress_regress_m:
            reason = "progress_regress"
        elif improvement < self.config.progress_min_improvement_m:
            reason = "progress_stall"
        else:
            return None
        self._last_progress_trigger_step = current_step
        self.progress_triggers += 1
        return reason

    # ------------------------------------------------------------------ #
    def evaluate(self, current_step: int, distance_to_goal: Optional[float]) -> Optional[str]:
        """在决策点评估两条通道；返回触发的 reason（都没有则 None）。"""
        collision_hit = self._collision_trigger(current_step)
        progress_reason: Optional[str] = None
        if distance_to_goal is not None:
            distance = float(distance_to_goal)
            progress_reason = self._progress_trigger(current_step, distance)
            self._distance_samples.append((int(current_step), distance))
            # 只保留足够回溯一个窗口的样本，控制内存
            keep = self.config.progress_window_actions * 3 + 4
            if len(self._distance_samples) > keep:
                del self._distance_samples[:-keep]
        if collision_hit:
            return REASON_COLLISION
        if progress_reason is not None:
            return REASON_PROGRESS
        return None

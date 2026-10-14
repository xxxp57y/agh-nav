"""out-of-tree 覆盖：把 `channels.py` 的两条通道接进上游闭环——**不改 InternNav 一行**。

为什么要用覆盖而不是改上游
------------------------
工程约束（`notes/route1-prep-brief.md` §2.2）：`InternNav/` 只读，只允许在
`scripts/eval/configs/` 下**新增**配置。所以我们不去动
`internnav/habitat_extensions/vln/habitat_vln_evaluator.py`，而是在自己的进程里
**monkeypatch 三个符号**——这既满足约束，也让"改了什么"一目了然：

| 上游符号 | 我们做什么 | 为什么需要它 |
|---|---|---|
| `habitat_vln_evaluator.update_collision_streak` | 包一层，把"这个运动动作有没有新增碰撞"推给通道状态 | 这是唯一能拿到**逐运动动作**碰撞增量的地方（`:2122`） |
| `HabitatVLNEvaluator._collision_count`（staticmethod） | 包一层，把 `metrics['distance_to_goal']` 发布到共享状态 | 它在每个运动动作后被调用（`:2121`），是逐动作的进度观测点 |
| `habitat_vln_evaluator.AdaptiveReplanScheduler` | 换成 `V4Scheduler`（子类） | 每条 episode 在 `:910` 用这个模块级名字构造调度器 |

`V4Scheduler.step()` 只做一件事：**调用上游 `super().step()` 之后，如果上游说"不用重规划"，
而我们的通道说"要求重规划"，就把 `should_replan` 改成 True**。上游的硬刹车一律尊重：
`no_action_replan_limit`（防死循环）、`recovery`（无可执行动作，本来就重规划）、
`budget_exhausted`、`in_cooldown` 都不会被我们覆盖。

副作用（必须知道）
----------------
1. **`.pyc` 陷阱**：这套覆盖是进程内生效的，但部署时如果 `route1/` 下留着旧的
   `__pycache__`，跑的可能还是旧逻辑（本项目已经踩过一次，见 `notes/route1-findings.md` §10.1）。
   部署脚本里必须清 `__pycache__`。
2. 改的是**调度器**，不是反馈分：几何加权分一个比特都不变，因此
   `route1.nav.replay parity` 的等价性回归仍然成立（新通道只在 `should_replan=False` 时加动作）。
3. 日志里会出现新的 `replan_reason`（`collision_density_replan` / `progress_stall_replan`），
   归因和统计都靠它。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .channels import ChannelConfig, TriggerChannels

#: 进程级共享状态：传感器钩子写、调度器读
_BRIDGE: Dict[str, Any] = {
    "channels": None,
    "config": None,
    "distance_to_goal": None,
    "last_step": -1,
    "installed": False,
    "collision_enabled": True,
    "progress_enabled": True,
    "forced_replans": 0,
    "forced_reasons": {},
}


def _bridge_channels() -> TriggerChannels:
    channels = _BRIDGE.get("channels")
    if channels is None:
        channels = TriggerChannels(_BRIDGE["config"])
        _BRIDGE["channels"] = channels
    return channels


def _reset_episode() -> None:
    _BRIDGE["channels"] = None
    _BRIDGE["distance_to_goal"] = None


def reset_state() -> None:
    """清掉跨 episode 的共享状态（自检/单元测试用，也给新 episode 用）。"""
    _reset_episode()
    _BRIDGE["last_step"] = -1


def install(config: Optional[ChannelConfig] = None,
            enable_collision: bool = True,
            enable_progress: bool = True,
            verbose: bool = True) -> Dict[str, Any]:
    """打上覆盖。**必须在构造 Evaluator 之前调用**（见 `run_eval_v4.py`）。

    可重复调用：已经装过时只更新通道开关（自检里用它来验证
    `V4_CHANNELS=none` 真的把两条通道都关掉）。
    """
    if _BRIDGE["installed"]:
        if _BRIDGE["config"] is not None:
            _BRIDGE["config"].collision_enabled = bool(enable_collision)
            _BRIDGE["config"].progress_enabled = bool(enable_progress)
            _BRIDGE["collision_enabled"] = bool(enable_collision)
            _BRIDGE["progress_enabled"] = bool(enable_progress)
        return _BRIDGE
    config = config or ChannelConfig(
        collision_window_actions=int(os.environ.get("V4_COLLISION_WINDOW", "8")),
        collision_min_in_window=int(os.environ.get("V4_COLLISION_MIN", "5")),
        collision_cooldown_steps=int(os.environ.get("V4_COLLISION_COOLDOWN", "6")),
        progress_window_actions=int(os.environ.get("V4_PROGRESS_WINDOW", "12")),
        progress_min_improvement_m=float(os.environ.get("V4_PROGRESS_MIN_IMPROVEMENT", "0.3")),
        progress_regress_m=float(os.environ.get("V4_PROGRESS_REGRESS", "0.6")),
        progress_cooldown_steps=int(os.environ.get("V4_PROGRESS_COOLDOWN", "12")),
        collision_max_per_episode=int(os.environ.get("V4_COLLISION_MAX", "8")),
        progress_max_per_episode=int(os.environ.get("V4_PROGRESS_MAX", "8")),
        cooldown_growth=float(os.environ.get("V4_COOLDOWN_GROWTH", "2.0")),
        cooldown_max_multiple=float(os.environ.get("V4_COOLDOWN_MAX_MULTIPLE", "4.0")),
    )
    _BRIDGE["config"] = config
    _BRIDGE["collision_enabled"] = bool(enable_collision)
    _BRIDGE["progress_enabled"] = bool(enable_progress)
    # ⚠️ 坑（2026-10-11 实测踩到）：`enable_*` 原来只 gate **传感器推送**，
    # 而 `V4Scheduler.step()` 里的 `evaluate()` 是无条件调用的 —— 于是
    # `V4_CHANNELS=none` 的那一轮**进度通道仍然在工作**（`r1ro4b` 里出现了 67 次
    # `progress_stall_replan`），把"无通道"这一臂污染了。
    # 修法：把开关写进 ChannelConfig，让通道自己在 evaluate 里短路。
    config.collision_enabled = bool(enable_collision)
    config.progress_enabled = bool(enable_progress)

    import internnav.habitat_extensions.vln.habitat_vln_evaluator as evaluator_module  # type: ignore
    from internnav.evaluator import Evaluator  # type: ignore
    from internnav.model.closed_loop.scheduler import AdaptiveReplanScheduler  # type: ignore

    # ⚠️ 上游用 `@Evaluator.register('habitat_vln')` 装饰这个类，而那个装饰器**没有 return**，
    # 所以模块级名字 `HabitatVLNEvaluator` 其实是 **None**（`base.py:22-32` 可核对）。
    # 真正要 patch 的类必须从注册表里取。
    target_class = Evaluator.evaluators.get("habitat_vln") or evaluator_module.HabitatVLNEvaluator
    if target_class is None:
        raise RuntimeError("找不到 habitat_vln 评估器类（注册表为空）")

    original_update_streak = evaluator_module.update_collision_streak
    original_collision_count = target_class._collision_count

    def update_collision_streak(previous_count, current_count, previous_streak, counts_as_motion):
        result = original_update_streak(previous_count, current_count, previous_streak,
                                        counts_as_motion)
        if counts_as_motion and _BRIDGE["collision_enabled"]:
            _bridge_channels().push_collision(int(current_count) > int(previous_count))
        return result

    def _collision_count(metrics):
        value = original_collision_count(metrics)
        if isinstance(metrics, dict) and metrics.get("distance_to_goal") is not None:
            _BRIDGE["distance_to_goal"] = float(metrics["distance_to_goal"])
        return value

    class V4Scheduler(AdaptiveReplanScheduler):
        """上游调度器 + 两条独立通道（只在"上游说别重规划"时加一次重规划）。"""

        def step(self, *args, **kwargs):  # type: ignore[override]
            current_step = kwargs.get("current_step")
            if current_step is None and len(args) >= 3:
                current_step = args[2]
            has_executable_action = kwargs.get("has_executable_action", True)
            if has_executable_action is None:
                has_executable_action = True
            result = super().step(*args, **kwargs)
            if current_step is None:
                return result
            current_step = int(current_step)
            if current_step < int(_BRIDGE["last_step"]):     # 新 episode
                _reset_episode()
            _BRIDGE["last_step"] = current_step
            if not has_executable_action:                    # recovery 路径本来就重规划
                return result
            if result.get("should_replan"):
                return result
            if result.get("reason") == "no_action_replan_limit" or result.get("budget_exhausted"):
                return result
            if result.get("in_cooldown"):
                return result
            reason = _bridge_channels().evaluate(current_step, _BRIDGE["distance_to_goal"])
            if not reason:
                return result
            forced = dict(result)
            forced["should_replan"] = True
            forced["reason"] = reason
            forced["trigger_kind"] = "v4:%s" % reason
            forced["v4_forced"] = True
            _BRIDGE["forced_replans"] += 1
            _BRIDGE["forced_reasons"][reason] = _BRIDGE["forced_reasons"].get(reason, 0) + 1
            return forced

    evaluator_module.update_collision_streak = update_collision_streak
    target_class._collision_count = staticmethod(_collision_count)
    evaluator_module.AdaptiveReplanScheduler = V4Scheduler
    _BRIDGE["installed"] = True
    _BRIDGE["scheduler_cls"] = V4Scheduler
    _BRIDGE["upstream_scheduler_cls"] = AdaptiveReplanScheduler
    _BRIDGE["evaluator_cls"] = target_class
    _BRIDGE["evaluator_module"] = evaluator_module
    if verbose:
        print("[overlay_v4] installed: config=%s collision=%s progress=%s"
              % (config.__dict__, enable_collision, enable_progress))
    return _BRIDGE


def stats() -> Dict[str, Any]:
    return {
        "forced_replans": _BRIDGE["forced_replans"],
        "forced_reasons": dict(_BRIDGE["forced_reasons"]),
    }

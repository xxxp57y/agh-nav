"""v4 覆盖的自检（在服务器上、有 InternNav 的 python 里跑）。

要验证五件事：
1. **零侵入**：通道都不触发时，`V4Scheduler.step()` 与上游 `AdaptiveReplanScheduler.step()`
   返回**逐位相同**的 dict（这才是"不改上游行为"的证明，而不是"误差很小"）。
2. **通道真的能加一次重规划**：8 个运动动作里 ≥5 次碰撞 → `should_replan=True`、
   `reason=collision_density_replan`。
3. **硬刹车不被覆盖**：`has_executable_action=False`（recovery 路径）与 `in_cooldown`
   两种情况下，即使通道触发也不改上游结果。
4. **冷却生效**：两次触发之间必须隔够 `collision_cooldown_steps`。
5. **传感器钩子真的在发数据**：直接调上游被包装的两个符号，看共享状态有没有更新。

用法（在 InternNav 根目录下）::

    cd /srv/agh-demo/InternNav
    PYTHONPATH=/srv/agh-demo/src \
      /srv/agh-demo/habitat-env/bin/python \
      -m route1.experiments.selftest_overlay
"""

from __future__ import annotations

from ..nav.overlay_v4 import _BRIDGE, _bridge_channels, install, reset_state

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print("  [%s] %s%s" % ("OK " if condition else "FAIL", name,
                           ("  <- " + detail) if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


KWARGS = dict(
    high_threshold=0.70, low_threshold=0.40, min_cooldown_steps=4,
    max_plan_age_steps=20, max_consecutive_replans_without_action=3,
    early_mode="drift_only", early_age_min=8, early_drift_mid_m=0.10,
    early_drift_high_m=0.20, early_consecutive_steps=2,
    min_replan_interval=4, max_s2_calls_per_episode=None,
    recovery_can_exceed_budget=True,
)
CALL = dict(feedback_score=0.10, plan_age=2, current_step=10,
            has_executable_action=True, allow_feedback_replan=False,
            trajectory_drift=0.02, system2_calls_used=1)


def _fill_collisions(count: int = 8, hits: int = 5) -> None:
    """通过"传感器等价路径"喂碰撞（与 overlay 钩子调用的是同一个通道对象）。"""
    channels = _bridge_channels()
    for index in range(count):
        channels.push_collision(index < hits)


def main() -> int:
    print("== 安装覆盖 ==")
    install(verbose=True)
    v4_cls = _BRIDGE["scheduler_cls"]
    upstream_cls = _BRIDGE["upstream_scheduler_cls"]

    print("== 1. 零侵入（通道未触发时逐位相同） ==")
    reset_state()
    upstream = upstream_cls(**KWARGS)
    v4 = v4_cls(**KWARGS)
    same = True
    for step in range(0, 60, 2):
        call_step = dict(CALL, current_step=step)
        left = upstream.step(**call_step)
        right = v4.step(**call_step)
        if left != right:
            same = False
            print("    step=%d 差异: %s vs %s" % (step, left, right))
            break
    check("未触发时 V4Scheduler == 上游调度器（逐位）", same)

    print("== 2. 碰撞通道能要求重规划 ==")
    reset_state()
    v4 = v4_cls(**KWARGS)
    v4.step(**dict(CALL, current_step=0))
    _fill_collisions(8, 5)
    out = v4.step(**dict(CALL, current_step=8))
    check("8 动作内 5 次碰撞 → should_replan=True", bool(out.get("should_replan")), str(out))
    check("reason == collision_density_replan",
          out.get("reason") == "collision_density_replan", str(out.get("reason")))

    print("== 3. 硬刹车不被覆盖 ==")
    reset_state()
    v4 = v4_cls(**KWARGS)
    v4.step(**dict(CALL, current_step=0))
    _fill_collisions(8, 8)
    no_action = v4.step(**dict(CALL, current_step=8, has_executable_action=False))
    check("has_executable_action=False → 走上游 recovery 分支",
          no_action.get("reason") == "recovery", str(no_action.get("reason")))

    reset_state()
    v4 = v4_cls(**dict(KWARGS, max_s2_calls_per_episode=1))
    v4.step(**dict(CALL, current_step=0, system2_calls_used=5))
    _fill_collisions(8, 8)
    exhausted = v4.step(**dict(CALL, current_step=8, system2_calls_used=5))
    check("budget_exhausted 时不被覆盖",
          exhausted.get("v4_forced") is not True,
          "%s / budget_exhausted=%s" % (exhausted.get("reason"),
                                        exhausted.get("budget_exhausted")))

    # 上游冷却：先让它自己重规划一次（feedback_high），再在 min_cooldown_steps 内问一次。
    # 注：R2A 这类模式不调用 record_system2_call，`in_cooldown` 可能恒为 False
    #（见 `notes/route1-findings.md` 结论与 `replay.py` 的 S2_INTERVAL_MODES），
    # 那种情况下真正的防抖来自通道自己的冷却（测试 4），这里如实标注 SKIP。
    reset_state()
    v4 = v4_cls(**KWARGS)
    first_replan = v4.step(**dict(CALL, feedback_score=0.9, current_step=0,
                                  allow_feedback_replan=True))
    _fill_collisions(8, 8)
    cooled = v4.step(**dict(CALL, feedback_score=0.9, current_step=2,
                            allow_feedback_replan=True))
    if cooled.get("in_cooldown"):
        check("上游 in_cooldown 时不被覆盖", cooled.get("v4_forced") is not True, str(cooled))
    else:
        print("  [SKIP] 上游在这个配置下不会进冷却（in_cooldown 恒 False）；"
              "防抖由通道自身冷却保证。上游自身重规划=%s" % first_replan.get("should_replan"))

    print("== 4. 冷却生效 ==")
    reset_state()
    v4 = v4_cls(**KWARGS)
    v4.step(**dict(CALL, current_step=0))
    _fill_collisions(8, 8)
    first = v4.step(**dict(CALL, current_step=8))
    _fill_collisions(8, 8)
    second = v4.step(**dict(CALL, current_step=10))     # 距上次触发 2 步 < 冷却 6
    _fill_collisions(8, 8)
    third = v4.step(**dict(CALL, current_step=20))      # 距上次触发 12 步 > 冷却 6
    check("首次触发生效", first.get("v4_forced") is True, str(first.get("reason")))
    check("冷却期内（+2 步）不重复触发", second.get("v4_forced") is not True, str(second.get("reason")))
    check("冷却期外（+12 步）可再次触发", third.get("v4_forced") is True, str(third.get("reason")))

    print("== 5. 每回合触发次数封顶（防重规划抖动） ==")
    reset_state()
    v4 = v4_cls(**KWARGS)
    v4.step(**dict(CALL, current_step=0))
    forced = 0
    step = 0
    for _ in range(40):
        step += 8
        _fill_collisions(8, 8)
        if v4.step(**dict(CALL, current_step=step)).get("v4_forced"):
            forced += 1
    limit = _BRIDGE["config"].collision_max_per_episode
    check("碰撞通道触发次数 <= collision_max_per_episode(%d)" % limit,
          forced <= limit, "forced=%d" % forced)

    print("== 6. V4_CHANNELS=none 时两条通道都必须真的关闭 ==")
    # 补的回归测试：2026-10-11 发现 enable_* 原来只 gate 了传感器推送，
    # 于是"无通道"那一臂里进度通道仍在触发（`r1ro4b` 里 67 次 progress_stall_replan）。
    # 这里用 install() 的"重复调用只更新开关"能力端到端验证。
    install(enable_collision=False, enable_progress=False, verbose=False)
    reset_state()
    v4 = v4_cls(**KWARGS)
    v4.step(**dict(CALL, current_step=0))
    _BRIDGE["distance_to_goal"] = 6.0
    forced_off = 0
    step = 0
    for _ in range(30):
        step += 8
        _fill_collisions(8, 8)                      # 碰撞喂满
        _BRIDGE["distance_to_goal"] = 6.0           # 距离恒定 = 进度"停滞"
        if v4.step(**dict(CALL, current_step=step)).get("v4_forced"):
            forced_off += 1
    check("通道全关时零触发", forced_off == 0, "forced=%d" % forced_off)
    install(enable_collision=True, enable_progress=True, verbose=False)   # 恢复

    print("== 7. 传感器钩子 ==")
    reset_state()
    evaluator_module = _BRIDGE["evaluator_module"]
    evaluator_module.update_collision_streak(0, 1, 0, True)      # 一个运动动作新增碰撞
    _BRIDGE["evaluator_cls"]._collision_count(
        {"collisions": {"count": 3}, "distance_to_goal": 4.25})
    check("_collision_count 发布了 distance_to_goal",
          _BRIDGE["distance_to_goal"] == 4.25, str(_BRIDGE["distance_to_goal"]))
    channels = _BRIDGE["channels"]
    check("update_collision_streak 推了一个碰撞事件",
          channels is not None and channels.collision_count_in_window() == 1,
          str(None if channels is None else channels.collision_count_in_window()))

    print()
    if FAILURES:
        print("SELFTEST_FAILED: %s" % ", ".join(FAILURES))
        return 1
    print("SELFTEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

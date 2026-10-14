#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""turn_probe.py —— 为什么"转身"改不了朝向？

`apply_action_probe` 实测：move_forward 有效（位置每次 +0.25 m），
但 turn_left/right **朝向纹丝不动**（一直 180°），导致机器人朝墙走、规划器反复让它转。

这个脚本把每一步的值都打出来：
  quat_from_angle_axis 的返回类型/数值 → set_agent_state 的入参 → 读回来的旋转
用来分清是"四元数造错了"还是"set_agent_state 不接受这个对象"。
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

REPO = os.environ.get("NAVLIVE_REPO", "/srv/agh-demo/InternNav")
CONFIG = os.environ.get(
    "NAVLIVE_CONFIG", os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py")
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")


def build_env():
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.path.insert(0, "/srv/agh-demo/src")
    import habitat as _h
    from habitat_baselines.config.default import get_config as get_habitat_config

    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401

    spec = importlib.util.spec_from_file_location("navlive_eval_config", CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["navlive_eval_config"] = module
    spec.loader.exec_module(module)
    config = get_habitat_config(module.eval_cfg.env.env_settings["config_path"])
    with _h.config.read_write(config):
        config.habitat.dataset.split = SPLIT
    return _h.Env(config=config)


def describe(tag, q):
    # 注意：`quaternion.quaternion` 迭代出来的不是标量（每个分量还是 quaternion 对象），
    # 所以必须用 .w/.x/.y/.z 取值（踩过）。
    print("    %-30s type=%-22s wxyz=[%.4f, %.4f, %.4f, %.4f]" % (
        tag, type(q).__name__,
        float(q.w), float(q.x), float(q.y), float(q.z)))


def main() -> int:
    import numpy as np
    from habitat_sim.utils import common as sim_common

    env = build_env()
    env.reset()
    sim = env.sim

    st = sim.get_agent_state()
    print("初始旋转: type=%s x=%.4f y=%.4f z=%.4f w=%.4f" % (
        type(st.rotation).__name__, st.rotation.x, st.rotation.y, st.rotation.z, st.rotation.w))

    print()
    print("=== 1) 造一个 +15° 绕 y 的四元数，看它长什么样 ===")
    q_new = sim_common.quat_from_angle_axis(
        math.radians(15.0), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    describe("quat_from_angle_axis 返回", q_new)

    print()
    print("=== 2) 直接把 q_new 交给 set_agent_state，看读回来变没变 ===")
    pos = [float(v) for v in st.position]
    for label, rot in (
        ("q_new 对象本身", q_new),
        ("list → [w,x,y,z] 顺序", [float(q_new.w), float(q_new.x), float(q_new.y), float(q_new.z)]),
        ("[x,y,z,w] 顺序", [float(q_new.x), float(q_new.y), float(q_new.z), float(q_new.w)]),
        ("numpy [x,y,z,w]", np.array([float(q_new.x), float(q_new.y), float(q_new.z), float(q_new.w)], dtype=np.float32)),
        ("numpy [w,x,y,z]", np.array([float(q_new.w), float(q_new.x), float(q_new.y), float(q_new.z)], dtype=np.float32)),
    ):
        try:
            sim.set_agent_state(pos, rot)
            after = sim.get_agent_state().rotation
            angle, _ = sim_common.quat_to_angle_axis(
                sim_common.quat_from_coeffs([float(after.w), float(after.x), float(after.y), float(after.z)]))
            print("    %-24s → 读回角度 %.2f°  (x=%.3f y=%.3f z=%.3f w=%.3f)" % (
                label, math.degrees(angle), after.x, after.y, after.z, after.w))
        except Exception as exc:  # noqa: BLE001
            print("    %-24s → %s: %s" % (label, type(exc).__name__, str(exc)[:80]))

    print()
    print("=== 3) 用 action 空间正常步进来转身（对照组） ===")
    before = sim.get_agent_state().rotation
    for _ in range(3):
        env.step(2)  # turn_left
    after = sim.get_agent_state().rotation
    angle_before, _ = sim_common.quat_to_angle_axis(
        sim_common.quat_from_coeffs([float(before.w), float(before.x), float(before.y), float(before.z)]))
    angle_after, _ = sim_common.quat_to_angle_axis(
        sim_common.quat_from_coeffs([float(after.w), float(after.x), float(after.y), float(after.z)]))
    print("    env.step(2) 三次: %.2f° → %.2f°（差 %.1f°）" % (
        math.degrees(angle_before), math.degrees(angle_after),
        math.degrees(angle_after) - math.degrees(angle_before)))

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

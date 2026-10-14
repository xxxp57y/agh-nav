#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apply_action_probe.py —— 把 `Session._apply_action` 的代码原样抽出来测。

为什么单独测：会话里它表现为"动作算出来了（last_action=3）但位姿不变"，
而单独写一遍又像是能走。所以要**用同一个函数体**在真实 env 上连续跑，
并把每次的位姿变化打出来，才能分清是"代码错"还是"仿真拒绝"。
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


STEP_FORWARD_M = 0.25
TURN_RIGHT_DEG = 15.0


def apply_action(env, action):
    """与 Session._apply_action 完全一致的实现（复制过来，便于对照）。"""
    import habitat_sim
    import numpy as np
    from habitat_sim.utils import common as sim_common

    action = int(action)
    state = env.sim.get_agent_state()
    position = np.asarray(state.position, dtype=np.float32).copy()
    rotation = state.rotation

    def _as_quat(rot):
        return sim_common.quat_from_coeffs([float(rot.w), float(rot.x), float(rot.y), float(rot.z)])

    if action == 1:
        heading = np.asarray(
            sim_common.quat_rotate_vector(_as_quat(rotation), habitat_sim.geo.FRONT), dtype=np.float32)
        heading[1] = 0.0
        n = float(np.linalg.norm(heading))
        if n > 1e-6:
            position = position + (heading / n) * STEP_FORWARD_M
    elif action in (2, 3):
        current_angle, _axis = sim_common.quat_to_angle_axis(_as_quat(rotation))
        delta = -TURN_RIGHT_DEG if action == 2 else TURN_RIGHT_DEG
        rotation = sim_common.quat_from_angle_axis(
            current_angle + math.radians(delta), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    env.sim.set_agent_state(position, rotation)
    obs = env.sim.get_observations_at(
        position=[float(v) for v in position],
        rotation=[float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)],
        keep_agent_at_new_pose=True,
    )
    return obs


def snapshot(env):
    import numpy as np
    from habitat_sim.utils import common as sim_common

    s = env.sim.get_agent_state()
    q = sim_common.quat_from_coeffs([float(s.rotation.w), float(s.rotation.x),
                                    float(s.rotation.y), float(s.rotation.z)])
    angle, _ = sim_common.quat_to_angle_axis(q)
    return [float(s.position[0]), float(s.position[2])], math.degrees(angle)


def main() -> int:
    env = build_env()
    env.reset()
    pos, ang = snapshot(env)
    print("起始: 位置=(%.2f,%.2f) 朝向=%.1f°" % (pos[0], pos[1], ang))

    print()
    print("=== 连续 5 次 turn_right（action=3），每次应有 +15° ===")
    for i in range(5):
        apply_action(env, 3)
        p, a = snapshot(env)
        print("  第%d次: 位置=(%.2f,%.2f) 朝向=%.1f°" % (i + 1, p[0], p[1], a))

    print()
    print("=== 连续 5 次 move_forward（action=1），每次应走 0.25 m ===")
    for i in range(5):
        apply_action(env, 1)
        p, a = snapshot(env)
        print("  第%d次: 位置=(%.2f,%.2f) 朝向=%.1f°" % (i + 1, p[0], p[1], a))

    print()
    print("=== 用 ShortestPathFollower 驱动 20 步，看它会不会靠近目标 ===")
    import numpy as np
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    follower = ShortestPathFollower(env.sim, 0.5, return_one_hot=False)
    start, _ = snapshot(env)
    goal = [start[0] + 4.0, 0.17, start[1]]
    print("  目标 = (%.2f,%.2f)" % (goal[0], goal[2]))
    actions = []
    for i in range(20):
        action = follower.get_next_action(np.asarray(goal, dtype=np.float32))
        actions.append(int(action) if action is not None else -1)
        if action is not None:
            apply_action(env, action)
    p, a = snapshot(env)
    print("  动作序列: %s" % actions)
    print("  结束位置=(%.2f,%.2f) 距目标 %.2f m" % (
        p[0], p[1], math.dist((p[0], p[1]), (goal[0], goal[2]))))

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""action_probe.py —— 验证"自己执行一步动作"到底能不能让机器人移动。

背景：为了让自由导览不受 habitat 回合语义干扰，`Session._apply_action` 改成
直接改 `sim` 的位姿（`set_agent_state` + `get_observations_at`），不再走 `env.step`。
这里单独把这段逻辑跑一遍，量一下"走一步到底移动了多少米"。

用法：CUDA_VISIBLE_DEVICES=0 python3 action_probe.py [场景]
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


def main() -> int:
    import habitat_sim
    import numpy as np
    from habitat_sim.utils import common as sim_common
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    env = build_env()
    env.reset()
    sim = env.sim

    def pose():
        s = sim.get_agent_state()
        return [float(s.position[0]), float(s.position[1]), float(s.position[2])], s.rotation

    pos0, rot0 = pose()
    print("起始位置:", [round(v, 3) for v in pos0])

    print()
    print("=== 1) 直接 set_agent_state 往前挪 0.25 m ===")
    target = np.array([pos0[0], pos0[1], pos0[2] - 0.25], dtype=np.float32)
    sim.set_agent_state(target, rot0)
    pos1, _ = pose()
    print("  挪动后位置:", [round(v, 3) for v in pos1], "位移 %.3f m" % (
        math.dist((pos0[0], pos0[2]), (pos1[0], pos1[2]))))

    print()
    print("=== 2) 用 quat_rotate_vector 求朝向并走一步（_apply_action 的做法） ===")
    state = sim.get_agent_state()
    position = np.asarray(state.position, dtype=np.float32).copy()
    rotation = state.rotation
    quat = sim_common.quat_from_coeffs(
        float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w))
    heading = np.asarray(sim_common.quat_rotate_vector(quat, habitat_sim.geo.FRONT), dtype=np.float32)
    print("  朝向向量 (原始):", [round(float(v), 3) for v in heading])
    heading[1] = 0.0
    norm = float(np.linalg.norm(heading))
    print("  水平分量模长:", round(norm, 4), "（应接近 1，若为 0 说明朝向算法不对）")
    if norm > 1e-6:
        position = position + (heading / norm) * 0.25
        sim.set_agent_state(position, rotation)
        pos2, _ = pose()
        print("  走一步后:", [round(v, 3) for v in pos2], "位移 %.3f m" % (
            math.dist((pos1[0], pos1[2]), (pos2[0], pos2[2]))))

    print()
    print("=== 3) quat_to_angle_axis / quat_from_angle_axis 是否可用 ===")
    try:
        angle, axis = sim_common.quat_to_angle_axis(rotation)
        print("  当前朝向角 = %.1f°  轴=%s" % (math.degrees(angle), [round(float(v), 2) for v in axis]))
        new_rot = sim_common.quat_from_angle_axis(
            math.radians(math.degrees(angle) + 15.0), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        sim.set_agent_state(np.asarray(sim.get_agent_state().position, dtype=np.float32), new_rot)
        angle2, _ = sim_common.quat_to_angle_axis(sim.get_agent_state().rotation)
        print("  转 +15° 后朝向角 = %.1f°  （差 %.1f°）" % (
            math.degrees(angle2), math.degrees(angle2) - math.degrees(angle)))
    except Exception as exc:  # noqa: BLE001
        print("  失败:", exc)

    print()
    print("=== 4) ShortestPathFollower.get_next_action 返回什么 ===")
    follower = ShortestPathFollower(sim, 0.5, return_one_hot=False)
    goal = [pos0[0] + 3.0, pos0[1], pos0[2]]
    for i in range(3):
        action = follower.get_next_action(np.asarray(goal, dtype=np.float32))
        print("  第 %d 次 get_next_action(goal=%s) = %s" % (
            i + 1, [round(v, 2) for v in goal], action))
        # 自己执行
        state = sim.get_agent_state()
        position = np.asarray(state.position, dtype=np.float32).copy()
        rotation = state.rotation
        if int(action) == 1:
            quat = sim_common.quat_from_coeffs(
                float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w))
            heading = np.asarray(
                sim_common.quat_rotate_vector(quat, habitat_sim.geo.FRONT), dtype=np.float32)
            heading[1] = 0.0
            n = float(np.linalg.norm(heading))
            if n > 1e-6:
                position = position + (heading / n) * 0.25
        elif int(action) in (2, 3):
            cur, _ = sim_common.quat_to_angle_axis(rotation)
            delta = -15.0 if int(action) == 2 else 15.0
            rotation = sim_common.quat_from_angle_axis(
                math.radians(math.degrees(cur) + delta),
                np.array([0.0, 1.0, 0.0], dtype=np.float32))
        sim.set_agent_state(position, rotation)

    pos3, _ = pose()
    print("  三歩后位置:", [round(v, 3) for v in pos3])
    print("  距目标 %.2f m" % math.dist((pos3[0], pos3[2]), (goal[0], goal[2])))

    print()
    print("=== 5) get_observations_at 能否刷新画面 ===")
    obs = sim.get_observations_at(position=[float(v) for v in pos3],
                                 rotation=sim.get_agent_state().rotation,
                                 keep_agent_at_new_pose=True)
    print("  返回键:", sorted(obs.keys()) if obs else None)
    if obs and obs.get("rgb") is not None:
        arr = np.asarray(obs["rgb"])
        print("  rgb shape=%s 均值=%.1f" % (arr.shape, float(arr.mean())))

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

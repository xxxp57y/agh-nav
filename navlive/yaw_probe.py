#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""yaw_probe.py —— 直接量"朝向角"，逐个试 set_agent_state 的写法。

前面绕了太久，原因是靠 `quat_to_angle_axis` 反推角度不直观。这里换一个**不依赖四元数**的
度量：用 `quat_rotate_vector(FRONT)` 得到朝向向量，再算它与世界 +x 轴的夹角（yaw）。

然后逐种写法试 set_agent_state，看哪种真的让 yaw 变了。
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


def yaw_of(sim, sim_common, habitat_sim, np):
    """朝向向量与 +x 轴的夹角（水平面），单位度。不依赖 quat_to_angle_axis。"""
    r = sim.get_agent_state().rotation
    q = sim_common.quat_from_coeffs([float(r.w), float(r.x), float(r.y), float(r.z)])
    v = np.asarray(sim_common.quat_rotate_vector(q, habitat_sim.geo.FRONT), dtype=np.float32)
    return math.degrees(math.atan2(float(v[2]), float(v[0]))), [round(float(x), 3) for x in v]


def main() -> int:
    import numpy as np
    from habitat_sim.utils import common as sim_common
    import habitat_sim

    env = build_env()
    env.reset()
    sim = env.sim
    pos = [float(v) for v in sim.get_agent_state().position]

    y0, v0 = yaw_of(sim, sim_common, habitat_sim, np)
    print("起始 yaw=%.1f°  朝向向量=%s" % (y0, v0))

    # 造一个"当前朝向 + 30°"的四元数（绕 up 轴）。这里把三个候选 up 轴都试一遍。
    print()
    print("=== 逐个候选写法，每次都用「当前朝向 + 30°」重建 ===")
    for axis_label, axis in (("y 轴 (0,1,0)", [0.0, 1.0, 0.0]),
                             ("z 轴 (0,0,1)", [0.0, 0.0, 1.0])):
        for order_label in ("quaternion 对象", "list [x,y,z,w]"):
            # 从当前状态出发
            env.reset()
            sim.set_agent_state(pos, sim.get_agent_state().rotation)
            before, _ = yaw_of(sim, sim_common, habitat_sim, np)
            cur = sim.get_agent_state().rotation
            cq = sim_common.quat_from_coeffs([float(cur.w), float(cur.x), float(cur.y), float(cur.z)])
            angle, _ax = sim_common.quat_to_angle_axis(cq)
            q_new = sim_common.quat_from_angle_axis(
                angle + math.radians(30.0), np.array(axis, dtype=np.float32))
            payload = q_new if order_label == "quaternion 对象" else [
                float(q_new.x), float(q_new.y), float(q_new.z), float(q_new.w)]
            try:
                sim.set_agent_state(pos, payload)
                after, v = yaw_of(sim, sim_common, habitat_sim, np)
                delta = after - before
                while delta > 180:
                    delta -= 360
                while delta < -180:
                    delta += 360
                print("  %-14s %-16s → yaw %.1f° → %.1f°  (Δ%+.1f°)  向量=%s" % (
                    axis_label, order_label, before, after, delta, v))
            except Exception as exc:  # noqa: BLE001
                print("  %-14s %-16s → %s: %s" % (
                    axis_label, order_label, type(exc).__name__, str(exc)[:70]))

    print()
    print("=== 对照：用 env.step(2) turn_left 三次 ===")
    env.reset()
    sim.set_agent_state(pos, sim.get_agent_state().rotation)
    before, _ = yaw_of(sim, sim_common, habitat_sim, np)
    for _ in range(3):
        env.step(2)
    after, _ = yaw_of(sim, sim_common, habitat_sim, np)
    print("  yaw %.1f° → %.1f°  (Δ%+.1f°)" % (before, after, after - before))

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

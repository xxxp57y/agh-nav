#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sim_api_probe.py —— 查清"无副作用地刷新渲染"到底该用哪个 API。

背景（踩过两次）：
  * `sim.step("move_forward", 0.0)` → TypeError: step() takes 2 positional arguments
  * `sim.step("move_forward")`     → 运行时报 "No action move_forward in action space"，
                                     异常被 focus() 吞掉，导致**焦点切换整段失效**

所以要一次问清：这个环境里
  1. `sim.step()` 接受什么
  2. 动作空间里到底有哪些动作（是不是 "MOVE_FORWARD" 大写？）
  3. 有没有"只刷新渲染、不推进回合"的正路（get_observations_at）
"""

from __future__ import annotations

import importlib.util
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
    env = build_env()
    env.reset()
    sim = env.sim

    print("=== 1) 动作空间里有什么 ===")
    try:
        space = env.action_space
        print("  env.action_space =", space)
        if hasattr(space, "spaces"):
            for key, sub in space.spaces.items():
                print("    %-16s %s" % (key, sub))
    except Exception as exc:  # noqa: BLE001
        print("  读 env.action_space 失败:", exc)

    print()
    print("=== 2) sim 上与取观测/步进相关的方法 ===")
    for name in dir(sim):
        low = name.lower()
        if any(k in low for k in ("observ", "step", "render", "frame", "sensor")):
            print("   ", name)

    print()
    print("=== 3) get_observations_at 能不能用来刷新渲染（不改回合状态） ===")
    import numpy as np

    state = sim.get_agent_state()
    pos = [float(v) for v in state.position]
    print("  当前位姿:", [round(v, 2) for v in pos])
    try:
        obs = sim.get_observations_at(position=pos, rotation=state.rotation, keep_agent_at_new_pose=True)
        if obs:
            rgb = obs.get("rgb")
            print("  get_observations_at → OK，传感器键: %s" % sorted(obs.keys()))
            if rgb is not None:
                arr = np.asarray(rgb)
                print("    rgb shape=%s dtype=%s 均值=%.1f" % (arr.shape, arr.dtype, float(arr.mean())))
        else:
            print("  返回空/None")
    except Exception as exc:  # noqa: BLE001
        print("  调用失败:", exc)

    print()
    print("=== 4) 试几种 step 写法（只试一次，观察是否报错） ===")
    trials = [
        ('sim.step("move_forward")', lambda: sim.step("move_forward")),
        ('sim.step("MOVE_FORWARD")', lambda: sim.step("MOVE_FORWARD")),
        ('sim.step({"action": "move_forward"})', lambda: sim.step({"action": "move_forward"})),
        ('sim.step("move_forward", 0.0)', lambda: sim.step("move_forward", 0.0)),
    ]
    for label, fn in trials:
        try:
            out = fn()
            keys = sorted(out.keys()) if isinstance(out, dict) else type(out).__name__
            print("  %-42s → OK  (%s)" % (label, keys))
        except Exception as exc:  # noqa: BLE001
            print("  %-42s → %s: %s" % (label, type(exc).__name__, str(exc)[:90]))

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

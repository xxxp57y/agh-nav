#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reach_probe.py —— 为什么 ShortestPathFollower 对某个地标直接 STOP？

排查思路（实测用）：
  1. 用 navmesh 的 `ShortestPath` 从**机器人当前位置**到目标点找路 —— 有没有路
  2. 看两个点是不是在**同一个 navmesh 岛**上（跨岛＝走不过去）
  3. 看 `snap_point` 后目标是否仍然是"可走点"
  4. 对比 `ShortestPathFollower.get_next_action` 的返回

用法：python3 reach_probe.py <场景> <起点x> <起点z> <目标x> <目标z>
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


def scene_key(raw):
    name = str(raw).replace("\\", "/").rstrip("/").split("/")[-1]
    for suffix in (".glb", ".navmesh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def main() -> int:
    if len(sys.argv) < 6:
        print(__doc__)
        return 2
    scene = sys.argv[1]
    sx, sz, tx, tz = (float(v) for v in sys.argv[2:6])

    import habitat_sim
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    env = build_env()
    episode = next(e for e in env.episodes if scene_key(e.scene_id) == scene)
    env.current_episode = episode
    env.reset()
    pf = env.sim.pathfinder

    start = [sx, 0.17, sz]
    target = [tx, 0.17, tz]
    print("场景 %s" % scene)
    print("  起点 (%6.2f,%6.2f)  目标 (%6.2f,%6.2f)" % (sx, sz, tx, tz))
    print("  navmesh: 岛数量=%d  面积=%.1f m2" % (pf.num_islands, pf.navigable_area))

    snap_start = pf.snap_point(start)
    snap_target = pf.snap_point(target)
    print("  吸附后起点 (%6.2f,%6.2f)  目标 (%6.2f,%6.2f)" % (
        snap_start[0], snap_start[2], snap_target[0], snap_target[2]))
    print("  目标是否可走 is_navigable = %s" % pf.is_navigable(snap_target))

    island_start = pf.get_island(snap_start)
    island_target = pf.get_island(snap_target)
    print("  起点岛=%s  目标岛=%s  同一岛=%s" % (
        island_start, island_target, island_start == island_target))

    query = habitat_sim.ShortestPath()
    query.requested_start = [float(snap_start[0]), float(snap_start[1]), float(snap_start[2])]
    query.requested_end = [float(snap_target[0]), float(snap_target[1]), float(snap_target[2])]
    found = pf.find_path(query)
    points = list(query.points or [])
    print("  ShortestPath.find_path = %s，航点 %d 个" % (found, len(points)))
    if points:
        length = 0.0
        for i in range(1, len(points)):
            a, b = points[i - 1], points[i]
            length += ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5
        print("  路径长度 %.2f m" % length)

    # 真正的执行者：ShortestPathFollower
    follower = ShortestPathFollower(env.sim, 0.5, return_one_hot=False)
    import numpy as np

    action = follower.get_next_action(np.asarray(target, dtype=np.float32))
    print("  ShortestPathFollower.get_next_action = %s （0=STOP,1=FORWARD,2=LEFT,3=RIGHT）" % action)

    # 再试从"吸附后的目标"出发
    action2 = follower.get_next_action(np.asarray(
        [float(snap_target[0]), float(snap_target[1]), float(snap_target[2])], dtype=np.float32))
    print("  用吸附后的坐标再试 = %s" % action2)

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

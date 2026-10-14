#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""navlive_rendercheck.py —— 用定稿的 navlive_map 渲染楼层图，肉眼验收用。

跑一个场景 + 一次 navmesh 导航，把"走过的轨迹"画到楼层图上，
输出：<scene>_render_start.png / <scene>_render_end.png / <scene>_rgb_end.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (os.path.dirname(HERE), REPO := os.environ.get("ROUTE1_REPO", "/srv/agh-demo/InternNav")):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from navlive import navlive_map  # noqa: E402

CONFIG = os.environ.get(
    "NAVLIVE_CONFIG", os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py")
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")


def log(msg: str) -> None:
    sys.stderr.write("[render] %s\n" % msg)
    sys.stderr.flush()


def scene_key_of(raw: Any) -> str:
    name = str(raw).replace("\\", "/").rstrip("/").split("/")[-1]
    for suffix in (".glb", ".navmesh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/navlive-render")
    parser.add_argument("--scenes", default="2azQ1b91cZZ,QUCTc6BB5sX")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=220)
    args = parser.parse_args()

    os.chdir(REPO)
    os.makedirs(args.out, exist_ok=True)
    import numpy as np
    from PIL import Image

    import habitat
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat_baselines.config.default import get_config as get_habitat_config
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    import importlib.util

    spec = importlib.util.spec_from_file_location("navlive_render_config", CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["navlive_render_config"] = module
    spec.loader.exec_module(module)
    cfg = module.eval_cfg

    config = get_habitat_config(cfg.env.env_settings["config_path"])
    with habitat.config.read_write(config):
        config.habitat.dataset.split = SPLIT
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=False,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(draw=False, visibility_dist=100.0, fov=180),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = habitat.Env(config)
    by_scene: Dict[str, List[Any]] = {}
    for episode in env.episodes:
        by_scene.setdefault(scene_key_of(episode.scene_id), []).append(episode)

    summary = {}
    for scene in [s for s in args.scenes.split(",") if s]:
        episode = by_scene[scene][args.episode_index]
        env.current_episode = episode
        env.reset()
        follower = ShortestPathFollower(env.sim, 0.5, return_one_hot=False)
        goal = np.asarray(episode.goals[0].position, dtype=np.float32)

        def agent_rc(map_info) -> tuple:
            """agent_map_coord 可能是 (2,) / (2,1) 的形状，统一压成 (row, col) 二元组。"""
            return tuple(float(v) for v in np.asarray(map_info["agent_map_coord"]).reshape(-1)[:2])

        info = env.get_metrics()["top_down_map"]
        path_rc: List[tuple] = [agent_rc(info)]
        goal_rc = None
        if "goal_map_coord" in info:
            goal_rc = tuple(float(v) for v in np.asarray(info["goal_map_coord"]).reshape(-1)[:2])

        img = navlive_map.render(info, path_rc, goal_rc=goal_rc, use_fog=False)
        origin = (0, 0)  # crop 原点在 render 内部，这里只做展示用图
        img.save(os.path.join(args.out, "%s_render_start.png" % scene))

        t0 = time.time()
        for _ in range(args.max_steps):
            action = follower.get_next_action(goal)
            if action is None:
                break
            env.step(int(action))
            info = env.get_metrics()["top_down_map"]
            path_rc.append(agent_rc(info))
            if env.episode_over:
                break
        nav_s = time.time() - t0

        img = navlive_map.render(info, path_rc, goal_rc=goal_rc, use_fog=False)
        from PIL import ImageDraw

        navlive_map.draw_robot(
            img,
            agent_rc(info),
            float(np.asarray(info["agent_angle"]).reshape(-1)[0]),
        )
        ImageDraw.Draw(img)
        img.save(os.path.join(args.out, "%s_render_end.png" % scene))
        Image.fromarray(np.asarray(env.sim.get_observations()["rgb"])).save(
            os.path.join(args.out, "%s_rgb_end.jpg" % scene)
        )

        metrics = env.get_metrics()
        summary[scene] = {
            "episode_id": str(episode.episode_id),
            "instruction": str(getattr(episode.instruction, "instruction_text", episode.instruction)),
            "steps": len(path_rc) - 1,
            "nav_seconds": round(nav_s, 2),
            "success": metrics.get("success"),
            "spl": metrics.get("spl"),
            "map_size": list(img.size),
        }
        log("%s: %s" % (scene, json.dumps(summary[scene], ensure_ascii=False)))

    with open(os.path.join(args.out, "render-report.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    env.close()
    log("images in %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

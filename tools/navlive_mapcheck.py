#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""navlive_mapcheck.py —— 钉死"楼层图怎么渲染才像作品里那样"。

现状：默认 TopDownMap 带 fog_of_war，整张图几乎全黑（只有走过的一条亮线）。
真正的原因（实测）：在线评估里 `draw_shortest_path=True`，而 habitat 的 TopDownMap 在
**没有 shortest_path 可用时会把整张图乘 0.5（变暗）**。所以"黑"不是雾，是变暗。

这个脚本一次性试几种口径，各出一张图，比一比哪种能当工作台底图：

  A 默认（照抄现有评估）
  B fog 关掉、shortest_path 关掉（预期：正常亮度的整层楼底图）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

REPO = os.environ.get("ROUTE1_REPO", "/srv/agh-demo/InternNav")
CONFIG = os.environ.get(
    "NAVLIVE_CONFIG", os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py")
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")
if os.path.isdir(REPO) and REPO not in sys.path:
    sys.path.insert(0, REPO)


def log(msg: str) -> None:
    sys.stderr.write("[mapcheck] %s\n" % msg)
    sys.stderr.flush()


def scene_key_of(raw: Any) -> str:
    name = str(raw).replace("\\", "/").rstrip("/").split("/")[-1]
    for suffix in (".glb", ".navmesh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def make_env(variant: str):
    import habitat
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat_baselines.config.default import get_config as get_habitat_config

    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    import importlib.util

    spec = importlib.util.spec_from_file_location("navlive_mapcheck_config", CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["navlive_mapcheck_config"] = module
    spec.loader.exec_module(module)
    cfg = module.eval_cfg

    config = get_habitat_config(cfg.env.env_settings["config_path"])
    if variant == "A":
        fog = FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90)
        draw_path = True
    else:
        fog = FogOfWarConfig(draw=False, visibility_dist=100.0, fov=180)
        draw_path = False
    with habitat.config.read_write(config):
        config.habitat.dataset.split = SPLIT
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=draw_path,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=fog,
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    return habitat.Env(config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/navlive-mapcheck")
    parser.add_argument("--scene", default="QUCTc6BB5sX")
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    os.chdir(REPO)
    os.makedirs(args.out, exist_ok=True)
    import numpy as np
    from PIL import Image

    report = {}
    for variant in ("A", "B"):
        t0 = time.time()
        env = make_env(variant)
        by_scene: dict = {}
        for episode in env.episodes:
            by_scene.setdefault(scene_key_of(episode.scene_id), []).append(episode)
        episode = by_scene[args.scene][args.episode_index]
        env.current_episode = episode
        obs = env.reset()

        node = env._task._config.measurements.top_down_map
        log("variant %s: fog node = %s" % (variant, dict(node.fog_of_war)))
        arr = np.asarray(env.get_metrics()["top_down_map"]["map"])
        Image.fromarray(arr).save(os.path.join(args.out, "%s_map_reset.png" % variant))
        report[variant] = {
            "fog_node": dict(node.fog_of_war),
            "map_shape": list(arr.shape),
            "map_max": int(arr.max()),
            "map_mean": round(float(arr.mean()), 2),
            "bright_pixels": int((arr > 40).sum()),
            "seconds": round(time.time() - t0, 1),
        }

        for _ in range(args.steps):
            env.step(1)
        arr2 = np.asarray(env.get_metrics()["top_down_map"]["map"])
        Image.fromarray(arr2).save(os.path.join(args.out, "%s_map_after%d.png" % (variant, args.steps)))
        report[variant]["after_bright_pixels"] = int((arr2 > 40).sum())
        report[variant]["after_max"] = int(arr2.max())
        Image.fromarray(np.asarray(obs["rgb"])).save(os.path.join(args.out, "%s_rgb.jpg" % variant))
        env.close()
        log("variant %s: %s" % (variant, json.dumps(report[variant], ensure_ascii=False)))

    with open(os.path.join(args.out, "mapcheck-report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    log("images + report in %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""navlive_spike.py —— 第二个钉子：**楼层图（关雾）+ 真实寻路**能不能一次跑通 4 个场景。

navlive_probe.py 证明了"建环境 + 出画面 + 出楼层图"可行；
这一步再验证两件事，它们决定工作台能不能做出图上那种效果：

1. **楼层图关掉 fog_of_war**：默认 fog 只显示走过的地方，整张图几乎全黑（probe 的 map_*.png
   就是这个现象）。关掉之后应该能拿到一张"整层楼"的底图。
2. **navmesh 真寻路**：用 `habitat.tasks.nav.shortest_path_follower.ShortestPathFollower`
   真的一步一步走到目标点 —— 这是工作台的"planner 之一"（另一个才是 VLA）。
   顺便量一下"每步多少毫秒"，这决定实时循环能跑多快。

用法::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/srv/agh-demo/src \\
      /srv/agh-demo/habitat-env/bin/python -u navlive_spike.py \\
      --out /tmp/navlive-spike --max-steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

REPO = os.environ.get("ROUTE1_REPO", "/srv/agh-demo/InternNav")
CONFIG = os.environ.get(
    "NAVLIVE_CONFIG",
    os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py"),
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")

if os.path.isdir(REPO) and REPO not in sys.path:
    sys.path.insert(0, REPO)


def log(msg: str) -> None:
    sys.stderr.write("[spike] %s\n" % msg)
    sys.stderr.flush()


def scene_key_of(raw: Any) -> str:
    text = str(raw).replace("\\", "/").rstrip("/")
    name = text.split("/")[-1]
    for suffix in (".glb", ".navmesh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def instruction_text(episode) -> Any:
    value = getattr(episode, "instruction", None)
    if value is None:
        return None
    return getattr(value, "instruction_text", None) or str(value)


def build_env(fog: bool):
    """建 habitat.Env；fog=False 时关掉迷雾，拿到整层楼底图。"""
    import habitat
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat_baselines.config.default import get_config as get_habitat_config

    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401 触发注册

    import importlib.util

    spec = importlib.util.spec_from_file_location("navlive_spike_config", CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["navlive_spike_config"] = module
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
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(draw=bool(fog), visibility_dist=5.0, fov=90),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    return habitat.Env(config)


def crop_to_content(image, pad: int = 24):
    """把楼层图裁到"有内容"的范围，避免大片空白（工作台里用同样逻辑做自适应缩放）。"""
    import numpy as np

    arr = np.asarray(image)
    gray = arr.max(axis=2) if arr.ndim == 3 else arr
    rows = np.where(gray.max(axis=1) > 8)[0]
    cols = np.where(gray.max(axis=0) > 8)[0]
    if len(rows) == 0 or len(cols) == 0:
        return image, (0, 0)
    r0, r1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, arr.shape[0])
    c0, c1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad + 1, arr.shape[1])
    return image.crop((c0, r0, c1, r1)), (c0, r0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/navlive-spike")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fog", action="store_true", help="保留迷雾（默认关）")
    parser.add_argument("--scenes", default="", help="逗号分隔；默认跑清单里全部场景")
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()

    os.chdir(REPO)
    os.makedirs(args.out, exist_ok=True)
    import numpy as np
    from PIL import Image

    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    t0 = time.time()
    env = build_env(fog=args.fog)
    log("env built in %.1fs, episodes=%d" % (time.time() - t0, len(env.episodes)))

    by_scene: Dict[str, List[Any]] = {}
    for episode in env.episodes:
        by_scene.setdefault(scene_key_of(episode.scene_id), []).append(episode)

    wanted = [s for s in args.scenes.split(",") if s] or sorted(by_scene)
    report: Dict[str, Any] = {"scenes": []}

    for scene in wanted:
        if scene not in by_scene:
            log("跳过未知场景 %s" % scene)
            continue
        episode = by_scene[scene][args.episode_index]

        t1 = time.time()
        env.current_episode = episode
        obs = env.reset()
        reset_s = time.time() - t1

        # ShortestPathFollower 每次切 episode 后要重新建（它绑在 sim 与 episode 上）
        follower = ShortestPathFollower(env.sim, 0.5, return_one_hot=False)
        goal = np.asarray(episode.goals[0].position, dtype=np.float32)

        metrics = env.get_metrics()
        tdm = metrics.get("top_down_map")
        base_map = np.asarray(tdm["map"]).copy()
        map_image = Image.fromarray(base_map)
        cropped, origin = crop_to_content(map_image)
        cropped.save(os.path.join(args.out, "%s_map_fog%s.png" % (scene, int(args.fog))))
        Image.fromarray(np.asarray(obs["rgb"])).save(os.path.join(args.out, "%s_rgb_start.jpg" % scene))

        start = env.sim.get_agent_state().position
        geodesic = env.sim.geodesic_distance(np.asarray(start, dtype=np.float32), goal)

        actions_used: List[int] = []
        t2 = time.time()
        done = False
        for _ in range(args.max_steps):
            action = follower.get_next_action(goal)
            if action is None:
                log("%s: follower 返回 None（可能不在 navmesh 上）" % scene)
                break
            actions_used.append(int(action))
            result = env.step(int(action))
            metrics = env.get_metrics() if not isinstance(result, dict) else (result.get("info") or env.get_metrics())
            done = bool(metrics.get("success", 0.0) >= 1.0) or bool(env.episode_over)
            if done:
                break
        nav_s = time.time() - t2

        end = env.sim.get_agent_state().position
        remaining = env.sim.geodesic_distance(np.asarray(end, dtype=np.float32), goal)
        final_metrics = env.get_metrics()
        tdm = final_metrics.get("top_down_map")
        if tdm is not None:
            Image.fromarray(np.asarray(tdm["map"])).save(
                os.path.join(args.out, "%s_map_end_fog%s.png" % (scene, int(args.fog)))
            )
        Image.fromarray(np.asarray(obs["rgb"])).save(os.path.join(args.out, "%s_rgb_end.jpg" % scene))

        entry = {
            "scene_id": scene,
            "episode_id": str(episode.episode_id),
            "instruction": instruction_text(episode),
            "reset_seconds": round(reset_s, 2),
            "geodesic_start_m": None if geodesic is None else round(float(geodesic), 2),
            "geodesic_end_m": None if remaining is None else round(float(remaining), 2),
            "steps": len(actions_used),
            "nav_seconds": round(nav_s, 2),
            "ms_per_step": round(1000.0 * nav_s / max(len(actions_used), 1), 2),
            "success": final_metrics.get("success"),
            "spl": final_metrics.get("spl"),
            "map_shape": list(base_map.shape),
            "map_crop_origin": list(origin),
        }
        report["scenes"].append(entry)
        log(
            "%s ep=%s: %d steps, %.2fs (%.1f ms/step), geodesic %.2f→%.2f m, success=%s"
            % (
                scene,
                episode.episode_id,
                entry["steps"],
                nav_s,
                entry["ms_per_step"],
                entry["geodesic_start_m"] or -1,
                entry["geodesic_end_m"] if entry["geodesic_end_m"] is not None else -1,
                entry["success"],
            )
        )

    with open(os.path.join(args.out, "spike-report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    log("report + images in %s" % args.out)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""landmark_probe.py —— 找出"场景里有哪些可以叫得出名字的地方"。

为什么需要它
------------
工作台能走到任意坐标，但**自然语言说的是地方名**（"客厅沙发""阳台下的入口"）。
要把"带客人去客厅"变成一次真实的 goto，中间缺一层"名字 → 坐标"的对照表。
这份脚本负责把这张表**从数据里挖出来**，而不是靠人拍脑袋：

1. **语义区域**（首选）：Matterport3D 的场景带 region/room 语义标注。如果 habitat 能读到，
   就能拿到"厨房""卧室"这类**真实房间名 + 质心坐标**。
2. **episode 端点**（兜底）：4 个场景 × 13 个回合的起点/终点，本身就是 52 个"真人写指令时
   关心的位置"。它们不算房间名，但坐标是真的、可复现的，适合当"地标锚点"。

输出：`navlive/data/landmarks.json`（场景 → 地标列表），并打印每个场景的构成。

用法::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/srv/agh-demo/src \\
      /srv/agh-demo/habitat-env/bin/python -u landmark_probe.py \\
      --out /srv/agh-demo/agh-nav/navlive/data/landmarks.json
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
    "NAVLIVE_CONFIG", os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py")
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")
if os.path.isdir(REPO) and REPO not in sys.path:
    sys.path.insert(0, REPO)

#: 房间类别 → 中文名（MP3D 的 region category 是英文小写）
ROOM_ZH = {
    "kitchen": "厨房", "living room": "客厅", "livingroom": "客厅", "dining room": "餐厅",
    "bedroom": "卧室", "bathroom": "卫生间", "toilet": "卫生间", "hallway": "走廊",
    "corridor": "走廊", "entrance": "入口", "entry": "入口", "foyer": "门厅",
    "balcony": "阳台", "stairs": "楼梯", "staircase": "楼梯", "garage": "车库",
    "office": "书房", "study": "书房", "closet": "衣帽间", "laundry": "洗衣房",
    "storage": "储藏间", "porch": "门廊", "patio": "露台", "garden": "花园",
    "gym": "健身房", "bar": "吧台", "lounge": "休息区", "utility": "设备间",
    "room": "房间", "area": "区域", "space": "空间",
}


def log(msg: str) -> None:
    sys.stderr.write("[landmark] %s\n" % msg)
    sys.stderr.flush()


def scene_key_of(raw: Any) -> str:
    name = str(raw).replace("\\", "/").rstrip("/").split("/")[-1]
    for suffix in (".glb", ".navmesh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def instruction_text(episode) -> str:
    value = getattr(episode, "instruction", None)
    if value is None:
        return ""
    return (getattr(value, "instruction_text", None) or str(value)).strip()


def build_env():
    """与 navlive_server 完全一致的构建口径（关雾、不开最短路径）。"""
    import habitat
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat_baselines.config.default import get_config as get_habitat_config

    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401

    import importlib.util

    spec = importlib.util.spec_from_file_location("landmark_cfg", CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["landmark_cfg"] = module
    spec.loader.exec_module(module)
    cfg = module.eval_cfg

    config = get_habitat_config(cfg.env.env_settings["config_path"])
    with habitat.config.read_write(config):
        config.habitat.dataset.split = SPLIT
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3, map_resolution=1024, draw_source=True, draw_border=True,
                    draw_shortest_path=False, draw_view_points=False,
                    draw_goal_positions=True, draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(draw=False, visibility_dist=100.0, fov=180),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    return habitat.Env(config)


def semantic_regions(env) -> List[Dict[str, Any]]:
    """尝试从 habitat 的语义场景里读出房间区域。读不到就返回空表（不编造）。"""
    import numpy as np

    regions: List[Dict[str, Any]] = []
    try:
        semantic = env.sim.semantic_scene
    except Exception as exc:  # noqa: BLE001
        log("semantic_scene 不可用：%s" % exc)
        return regions
    if semantic is None:
        log("semantic_scene 是 None")
        return regions
    try:
        levels = list(getattr(semantic, "levels", []) or [])
    except Exception:  # noqa: BLE001
        levels = []
    if not levels:
        # 有些版本直接挂在 scene 上
        levels = [semantic]
    for level in levels:
        for region in list(getattr(level, "regions", []) or []):
            label = getattr(getattr(region, "category", None), "name", None)
            label = str(label) if label is not None else "unknown"
            centroid = getattr(region, "centroid", None)
            if centroid is None:
                continue
            position = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
            regions.append({
                "label_en": label,
                "label": ROOM_ZH.get(label.lower().strip(), label),
                "position": position,
                "source": "semantic-region",
                "id": str(getattr(region, "id", "")),
            })
    log("语义区域 %d 个" % len(regions))
    return regions


def episode_points(episodes: List[Any]) -> List[Dict[str, Any]]:
    """把 13 个回合的起点/终点做成坐标锚点——它们是"真人在指令里关心的位置"。"""
    out: List[Dict[str, Any]] = []
    for episode in episodes:
        eid = str(getattr(episode, "episode_id", "?"))
        start = [float(v) for v in episode.start_position]
        goals = []
        for goal in getattr(episode, "goals", []) or []:
            position = getattr(goal, "position", None)
            if position is not None:
                goals.append([float(v) for v in position])
        out.append({
            "episode_id": eid,
            "instruction": instruction_text(episode),
            "start": start,
            "goal": goals[0] if goals else None,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "landmarks.json"))
    parser.add_argument("--probe-only", action="store_true", help="只打印，不写文件")
    args = parser.parse_args()

    os.chdir(REPO)
    t0 = time.time()
    env = build_env()
    log("env 就绪 %.1fs" % (time.time() - t0))

    by_scene: Dict[str, List[Any]] = {}
    for episode in env.episodes:
        by_scene.setdefault(scene_key_of(episode.scene_id), []).append(episode)

    payload: Dict[str, Any] = {
        "schema": "agh-navlive/landmarks/1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": SPLIT,
        "source_note": (
            "room 名称来自 Matterport3D 语义区域（habitat semantic_scene.regions）；"
            "ep-X 锚点来自该场景 13 个回合的起点/终点，坐标真实可复现。"
        ),
        "scenes": {},
    }

    for scene in sorted(by_scene):
        episodes = by_scene[scene]
        log("-- %s" % scene)
        env.current_episode = episodes[0]
        env.reset()
        regions = semantic_regions(env)
        points = episode_points(episodes)
        payload["scenes"][scene] = {
            "regions": regions,
            "episodes": points,
            "counts": {"regions": len(regions), "episodes": len(points)},
        }
        log("   区域 %d / 回合锚点 %d" % (len(regions), len(points)))

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.probe_only:
        print(text[:4000])
    else:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        log("已写出 %s（%d 字节）" % (args.out, len(text)))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

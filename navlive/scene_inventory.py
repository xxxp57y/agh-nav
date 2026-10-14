#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scene_inventory.py —— 挖场景里**能用来做地标的东西**。

背景：现在的语义地标只来自 R2R 指令里提到的终点（每场景 6–8 个），
而"客人想去哪就去哪"需要覆盖整个场景。这台 habitat 又读不到语义区域标注
（之前的探测是空的）。所以先如实清点：场景里到底有什么可挖。

清点内容：
  1. `semantic_annotations()`：objects / regions 各有多少、category 名叫什么
     —— 如果 object 数量可观且类别是 "chair/door/window/…"，那就能按物体做地标
  2. navmesh 规模（面积、随机可走点）—— 决定"自由导航"的空间有多大
  3. 现有 R2R 指令提到的地点短语全集 —— 看看靠指令还能榨出多少名字

用法：CUDA_VISIBLE_DEVICES=0 python3 scene_inventory.py [场景id]
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, List

REPO = os.environ.get("NAVLIVE_REPO", "/srv/agh-demo/InternNav")
CONFIG = os.environ.get(
    "NAVLIVE_CONFIG", os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py")
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")


def build_env():
    """和 navlive 完全同一套建环境方式（否则探到的不是同一个环境，结论没用）。"""
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.path.insert(0, "/srv/agh-demo/src")
    import habitat
    from habitat_baselines.config.default import get_config as get_habitat_config

    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401

    spec = importlib.util.spec_from_file_location("navlive_eval_config", CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["navlive_eval_config"] = module
    spec.loader.exec_module(module)
    cfg = module.eval_cfg
    config = get_habitat_config(cfg.env.env_settings["config_path"])
    with habitat.config.read_write(config):
        config.habitat.dataset.split = SPLIT
    return habitat.Env(config=config)


def scene_key(raw: Any) -> str:
    name = str(raw).replace("\\", "/").rstrip("/").split("/")[-1]
    for suffix in (".glb", ".navmesh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    env = build_env()
    print("episodes=%d" % len(env.episodes))

    # 选一个场景
    target = None
    for episode in env.episodes:
        key = scene_key(episode.scene_id)
        if wanted is None or key == wanted:
            target = episode
            break
    if target is None:
        print("没找到场景 %r" % wanted)
        return 1
    key = scene_key(target.scene_id)
    print("清点场景: %s" % key)

    # 切到该 episode 以便 sim 加载对应场景
    env.current_episode = target
    env.reset()

    # ---- 1) 语义标注 ----
    print()
    print("=== 1) semantic_annotations ===")
    try:
        sem = env.sim.semantic_annotations()
    except Exception as exc:  # noqa: BLE001
        sem = None
        print("  调用失败: %s" % exc)
    if sem is None:
        print("  返回 None（这个构建没有语义标注）")
    else:
        regions = list(getattr(sem, "regions", None) or [])
        objects = list(getattr(sem, "objects", None) or [])
        print("  regions=%d  objects=%d" % (len(regions), len(objects)))

        def cat_name(item: Any) -> str:
            category = getattr(item, "category", None)
            return str(getattr(category, "name", None) or category or "?")

        if regions:
            print("  region 类别（前 20）: %s" % ", ".join(sorted({cat_name(r) for r in regions})[:20]))
            for region in regions[:5]:
                print("    %s | centroid=%s" % (cat_name(region),
                      [round(float(v), 2) for v in getattr(region, "centroid", [])]))
        if objects:
            from collections import Counter

            counter = Counter(cat_name(o) for o in objects)
            print("  object 类别 Top20: %s" % counter.most_common(20))
            for obj in objects[:5]:
                aabb = getattr(obj, "aabb", None)
                center = aabb.center if aabb is not None else None
                print("    %s | center=%s" % (cat_name(obj),
                      [round(float(v), 2) for v in center] if center is not None else None))

    # ---- 2) navmesh ----
    print()
    print("=== 2) navmesh ===")
    pf = env.sim.pathfinder
    print("  navigable_area = %.1f m2" % pf.navigable_area)
    pts: List[List[float]] = []
    for _ in range(8):
        p = pf.get_random_navigable_point()
        pts.append([round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2)])
    print("  随机可走点 8 个: %s" % pts)

    # ---- 3) 该场景所有指令里提到的地点短语 ----
    print()
    print("=== 3) 全部指令里出现过的地点短语（现在的做法能榨出的上限） ===")
    import re

    PLACE = re.compile(
        r"\b(living room|bedroom|kitchen|bathroom|hallway|hall|dining room|office|balcony|"
        r"stairs|staircase|doorway|entrance|archway|closet|pantry|garage|porch|foyer|"
        r"table|couch|sofa|bed|chair|fireplace|sink|toilet|shower|window|door|rug|carpet|"
        r"counter|shelf|stairs|room)\b",
        re.I,
    )
    found: Dict[str, int] = {}
    scene_episodes = [ep for ep in env.episodes if scene_key(ep.scene_id) == key]
    for episode in scene_episodes:
        text = getattr(getattr(episode, "instruction", None), "instruction_text", "") or ""
        for match in PLACE.finditer(text.lower()):
            found[match.group(0)] = found.get(match.group(0), 0) + 1
    print("  该场景 episode 数 = %d，指令里出现的地点词 = %d 种" % (len(scene_episodes), len(found)))
    print("  %s" % sorted(found.items(), key=lambda kv: -kv[1]))

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

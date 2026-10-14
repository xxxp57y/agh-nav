#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""landmark_mine_probe.py —— 验证"从语义标注生成地标"这条路通不通（最小验证）。

思路：
  1. region 给出房间（bedroom / kitchen / bathroom …）
  2. object 给出真实家具与**所属房间**（bed = [x,y,z] in bedroom）
  3. 只取**可停靠**的类别（床/桌/水槽/沙发/马桶/浴缸/柜子…），
     丢掉 wall / ceiling / floor / void / objects / misc 这些不该当地标的
  4. 每个候选点用 `pathfinder.snap_point` 吸到 navmesh 上，并用
     `geodesic_distance` 确认**真的走得到**（不是隔墙对望）

这个脚本只**报告**结果，不写文件 —— 先确认数量与可达率，再决定要不要全量生成。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

REPO = os.environ.get("NAVLIVE_REPO", "/srv/agh-demo/InternNav")
CONFIG = os.environ.get(
    "NAVLIVE_CONFIG", os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py")
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")

#: 可以当"客人想去的地方"的类别。刻意排除建筑构件（wall/ceiling/floor/void）
#: 与无意义的杂项（objects/misc/lighting），因为没人会说"带我去墙那里"。
PLACEABLE = {
    "bed", "table", "sink", "toilet", "bathtub", "couch", "sofa", "chair", "seating",
    "cabinet", "shelving", "shelf", "counter", "chest_of_drawers", "desk", "bench",
    "stove", "oven", "refrigerator", "microwave", "dishwasher", "washer", "dryer",
    "fireplace", "mirror", "plant", "tv", "television", "pool", "stairs", "door",
    "window", "picture", "bed", "dresser", "nightstand", "bookcase", "wardrobe",
    "shower", "bath", "towel", "curtain", "rug", "carpet", "cushion",
}

#: 中文名（人工表；没有的就用英文原名，不硬翻）
ZH = {
    "bed": "床", "table": "桌子", "sink": "水槽", "toilet": "马桶", "bathtub": "浴缸",
    "couch": "沙发", "sofa": "沙发", "chair": "椅子", "seating": "座位",
    "cabinet": "柜子", "shelving": "置物架", "shelf": "架子", "counter": "台面",
    "chest_of_drawers": "五斗柜", "desk": "书桌", "bench": "长凳",
    "stove": "灶台", "oven": "烤箱", "refrigerator": "冰箱", "microwave": "微波炉",
    "dishwasher": "洗碗机", "washer": "洗衣机", "dryer": "烘干机",
    "fireplace": "壁炉", "mirror": "镜子", "plant": "植物", "tv": "电视",
    "stairs": "楼梯", "door": "门", "window": "窗", "picture": "画", "dresser": "梳妆台",
    "shower": "淋浴", "bath": "浴缸", "towel": "毛巾", "curtain": "窗帘",
    "rug": "地毯", "carpet": "地毯", "cushion": "靠垫", "wardrobe": "衣柜",
    "bookcase": "书架", "nightstand": "床头柜",
    # 房间
    "bedroom": "卧室", "bathroom": "卫生间", "kitchen": "厨房", "living room": "客厅",
    "dining room": "餐厅", "hallway": "走廊", "office": "书房", "closet": "衣帽间",
    "familyroom/lounge": "家庭厅", "laundryroom/mudroom": "洗衣间",
    "porch/terrace/deck": "阳台", "entryway/foyer/lobby": "门厅", "toilet": "卫生间",
    "junk": "杂物间", "room": "房间", "stairs": "楼梯",
}


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


def scene_key(raw: Any) -> str:
    name = str(raw).replace("\\", "/").rstrip("/").split("/")[-1]
    for suffix in (".glb", ".navmesh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def cat_name(item: Any) -> str:
    category = getattr(item, "category", None)
    if category is None:
        return "?"
    name = getattr(category, "name", None)
    if callable(name):
        try:
            name = name()
        except Exception:  # noqa: BLE001
            return "?"
    return str(name)


def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else "zsNo4HB9uLZ"
    env = build_env()
    episode = next(e for e in env.episodes if scene_key(e.scene_id) == wanted)
    env.current_episode = episode
    env.reset()
    sim = env.sim
    pathfinder = sim.pathfinder
    sem = sim.semantic_annotations()

    regions = list(getattr(sem, "regions", None) or [])
    objects = list(getattr(sem, "objects", None) or [])
    print("场景 %s：regions=%d objects=%d" % (wanted, len(regions), len(objects)))

    # ---- 1) 按 region 归拢物体，选出可停靠的候选 ----
    per_region: Dict[str, List[Tuple[str, List[float]]]] = defaultdict(list)
    region_of_object: Dict[str, str] = {}
    for obj in objects:
        region = getattr(obj, "region", None)
        region_name = cat_name(region) if region is not None else "unknown"
        name = cat_name(obj)
        if name not in PLACEABLE:
            continue
        aabb = getattr(obj, "aabb", None)
        if aabb is None:
            continue
        try:
            center = [float(v) for v in aabb.center]
        except Exception:  # noqa: BLE001
            continue
        per_region[region_name].append((name, center))

    print()
    print("=== 候选（可停靠物体，按房间） ===")
    total = 0
    for region_name, items in sorted(per_region.items(), key=lambda kv: -len(kv[1])):
        counter = Counter(name for name, _ in items)
        total += len(items)
        print("  %-22s %3d 个  %s" % (
            region_name, len(items),
            dict(sorted(counter.items(), key=lambda kv: -kv[1])[:6])))
    print("  合计候选 = %d" % total)

    # ---- 2) 吸附到 navmesh + 测可达性 ----
    print()
    print("=== 可达性检查（snap_point + ShortestPath） ===")
    import habitat_sim

    start = episode.start_position
    ok = 0
    fail_snap = 0
    fail_geo = 0
    examples: List[str] = []
    for region_name, items in per_region.items():
        for name, center in items:
            snapped = pathfinder.snap_point(center)
            if snapped is None or not pathfinder.is_navigable(snapped):
                fail_snap += 1
                continue
            # 这个构建的 PathFinder 只接受 ShortestPath / MultiGoalShortestPath 对象；
            # 找不到路（返回 False）就说明**隔墙对望、走不过去**。
            query = habitat_sim.ShortestPath()
            query.requested_start = list(start)
            query.requested_end = [float(snapped[0]), float(snapped[1]), float(snapped[2])]
            if not pathfinder.find_path(query):
                fail_geo += 1
                continue
            points = list(query.points or [])
            length = 0.0
            for index in range(1, len(points)):
                a, b = points[index - 1], points[index]
                length += ((float(b[0]) - float(a[0])) ** 2
                           + (float(b[1]) - float(a[1])) ** 2
                           + (float(b[2]) - float(a[2])) ** 2) ** 0.5
            if length <= 0.01:
                fail_geo += 1
                continue
            ok += 1
            if len(examples) < 14:
                examples.append("%-22s %-16s → 吸附 (%6.2f,%6.2f) 从起点 %5.1f m（%d 个航点）" % (
                    "%s的%s" % (ZH.get(region_name, region_name), ZH.get(name, name)),
                    name, float(snapped[0]), float(snapped[2]), length, len(points)))
    print("  可走（吸附成功且测地可达）= %d" % ok)
    print("  吸附失败 = %d ；测地不可达 = %d" % (fail_snap, fail_geo))
    print()
    for line in examples:
        print("    " + line)

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

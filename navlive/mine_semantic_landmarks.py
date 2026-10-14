#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mine_semantic_landmarks.py —— 从 MP3D **语义标注**生成"客人想去哪就去哪"的地标表。

和旧做法（`landmark_probe.py`：只从 R2R 指令里抽地点短语，每场景 6–8 个）的区别：

| | 旧 | 新 |
| --- | --- | --- |
| 来源 | 指令文本里提到过的地点 | 场景自带的 **region + object 标注** |
| 数量 | 6–8 / 场景 | 25–40 / 场景（只取大件） |
| 名字 | 靠人工别名表猜 | **region 就是房间名**，object 就是家具名 |
| 坐标 | 回合端点（可能与指令含义有偏差） | 物体 AABB 中心 **吸附到 navmesh** 后的真实可走点 |
| 可达性 | 未验证 | **用 `ShortestPath` 在真实 navmesh 上验证过** |

为什么这件事现在才做对：之前的探测把 `region.category.name`（**方法**）当属性读，
并且只看了空的 `region.centroid`，于是得出"语义标注是空的"这个**错误结论**。
实际每个场景有 20+ region、400+ object，而且每个 object 都带所属 region。

只取"大件"：客人真会点名的东西（床、沙发、餐桌、灶台、冰箱、马桶、浴缸、楼梯…），
不含靠垫/画/毛巾这类小件，也不含 wall/ceiling/floor/void 这些建筑构件。

用法（每个场景约 30–60 秒）::

    CUDA_VISIBLE_DEVICES=0 python3 mine_semantic_landmarks.py --out data/landmarks_semantic.json
    CUDA_VISIBLE_DEVICES=0 python3 mine_semantic_landmarks.py --scenes zsNo4HB9uLZ --out /tmp/one.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

REPO = os.environ.get("NAVLIVE_REPO", "/srv/agh-demo/InternNav")
CONFIG = os.environ.get(
    "NAVLIVE_CONFIG", os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py")
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")

#: 只留"大件"——客人真会点名的家具/设施。分两档，因为并不是所有"大"的都值得当地标。
#:
#: 收的时候按"客人会不会这么说"来判断：
#:   · 会说"带我去卧室的床" → 收
#:   · 不会说"带我去卧室的柜子（第 2 个）" → 同房间同类只留一个，名字也不带编号
#:
#: 刻意排除：cushion / picture / towel / curtain / mirror / plant / lighting / misc /
#: objects / void / wall / ceiling / floor（小件或建筑构件）。
MAJOR = {
    # 一定会点名的
    "bed", "sofa", "couch", "dining table", "table", "desk",
    "sink", "stove", "oven", "refrigerator", "microwave", "dishwasher",
    "washer", "dryer", "toilet", "bathtub", "shower", "fireplace",
    "stairs", "staircase", "wardrobe", "dresser", "kitchen island", "pool",
    # 备选（同类只有一个时才收）
    "cabinet", "counter", "shelving", "shelf", "bookcase",
    "chest_of_drawers", "nightstand", "chair", "bench", "seating",
}

#: 优先收的类别：这些在同一个房间里即使有多个也算不同地标（例如卧室有两张床）
PREFERRED = {
    "bed", "sofa", "couch", "table", "desk", "dining table",
    "sink", "stove", "refrigerator", "toilet", "bathtub", "shower",
    "fireplace", "stairs", "staircase", "pool",
}

#: region 类别名 → 房间名（MP3D 原始词表就这些，见 sem_probe 的输出）。
#: 中文由 `translate_landmarks.py` 用大模型批量翻，这里只做可读性整理。
REGION_ALIAS = {
    "familyroom/lounge": "family room",
    "laundryroom/mudroom": "laundry room",
    "porch/terrace/deck": "porch",
    "entryway/foyer/lobby": "entryway",
    "junk": "storage",
}

#: 同一个 region 里同类物体最多保留几个：优先类别最多两个（两张床确实是两个地方），
#: 其余只留一个 —— 名字里就不必再出现 "(2)" 这种编号。
#: 同一房间里两点距离小于这个值就认为是**同一件东西**，只留优先级高的那个名字。
#: 实测例子：`bathroom的cabinet` 与 `bathroom的counter` 坐标差 0.02 m ——
#: 同一个台盆柜被 cabinet 和 counter 两个类别各标了一次，留着会让客人困惑。
MERGE_RADIUS_M = 0.6

#: 合并时的优先级（数字越大越愿意留）。房间级判断"这是个啥"时，
#: "水槽/灶台/浴缸"比"柜子/台面"更具体，所以优先留具体的那个名字。
#: 房间级判断"这是个啥"时，具体的家具名比泛指更有用。
PRIORITY = {
    "bathtub": 90, "shower": 88, "toilet": 86, "sink": 84, "stove": 82,
    "refrigerator": 80, "oven": 78, "dishwasher": 76, "washer": 74, "dryer": 72,
    "fireplace": 70, "stairs": 68, "staircase": 68, "pool": 66,
    "bed": 64, "sofa": 62, "couch": 62, "dining table": 60, "table": 58,
    "desk": 56, "wardrobe": 54, "dresser": 52, "chest_of_drawers": 50,
    "nightstand": 48, "bookcase": 46, "shelving": 40, "shelf": 38,
    "cabinet": 30, "counter": 28, "chair": 26, "bench": 24, "seating": 22,
}


def merge_nearby(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把同一房间里**挨在一起的**候选合并成一个，保留优先级最高的名字。

    不合并的话会出现"卧室的床"和"卧室的桌子"坐标几乎相同（床边的桌子贴着床），
    客人点哪个都走到同一处 —— 不如按"这件东西最像什么"给一个名字。
    """
    kept: List[Dict[str, Any]] = []
    for item in sorted(items, key=lambda i: -PRIORITY.get(i["object"], 0)):
        duplicate = False
        for existing in kept:
            dx = item["position"][0] - existing["position"][0]
            dz = item["position"][2] - existing["position"][2]
            if (dx * dx + dz * dz) ** 0.5 < MERGE_RADIUS_M:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def cap_for(object_name: str) -> int:
    return 2 if object_name in PREFERRED else 1


def build_env():
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.environ.get("ROUTE1_SRC", os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")))
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
    """`category.name` 在这个构建里是**方法**，不是属性（踩过）。"""
    category = getattr(item, "category", None)
    if category is None:
        return "?"
    name = getattr(category, "name", None)
    if callable(name):
        try:
            name = name()
        except Exception:  # noqa: BLE001
            return "?"
    return str(name or "?")


def extract_scene(env: Any, episode: Any) -> List[Dict[str, Any]]:
    """把一个场景的语义标注变成地标列表。"""
    import habitat_sim

    env.current_episode = episode
    env.reset()
    sim = env.sim
    pathfinder = sim.pathfinder
    sem = sim.semantic_annotations()
    if sem is None:
        return []

    regions = list(getattr(sem, "regions", None) or [])
    objects = list(getattr(sem, "objects", None) or [])

    # 每个 region 用它的 id 建索引，方便按 region 归拢
    region_name_by_id: Dict[str, str] = {}
    for region in regions:
        rid = str(getattr(region, "id", "") or "")
        region_name_by_id[rid] = cat_name(region)

    # ---- 1) 收集候选（大件 + 有 AABB + 能吸附到 navmesh） ----
    candidates: List[Tuple[str, str, List[float], Any]] = []
    for obj in objects:
        name = cat_name(obj)
        if name not in MAJOR:
            continue
        region = getattr(obj, "region", None)
        rid = str(getattr(region, "id", "") or "") if region is not None else ""
        region_name = region_name_by_id.get(rid) or (cat_name(region) if region is not None else "unknown")
        aabb = getattr(obj, "aabb", None)
        if aabb is None:
            continue
        try:
            center = [float(v) for v in aabb.center]
        except Exception:  # noqa: BLE001
            continue
        candidates.append((region_name, name, center, obj))

    # ---- 2) 吸附 + 可达性验证，并在同一 region 内按类别去重 ----
    start = [float(v) for v in episode.start_position]
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for region_name, name, center, _obj in candidates:
        snapped = pathfinder.snap_point(center)
        if snapped is None or not pathfinder.is_navigable(snapped):
            continue
        end = [float(snapped[0]), float(snapped[1]), float(snapped[2])]
        query = habitat_sim.ShortestPath()
        query.requested_start = start
        query.requested_end = end
        if not pathfinder.find_path(query):
            continue  # 隔墙对望，走不过去
        points = list(query.points or [])
        length = 0.0
        for index in range(1, len(points)):
            a, b = points[index - 1], points[index]
            length += ((float(b[0]) - float(a[0])) ** 2
                       + (float(b[1]) - float(a[1])) ** 2
                       + (float(b[2]) - float(a[2])) ** 2) ** 0.5
        if length <= 0.01:
            continue
        grouped[(region_name, name)].append({
            "region": region_name,
            "object": name,
            "position": [round(end[0], 3), round(end[1], 3), round(end[2], 3)],
            "geodesic_from_start_m": round(length, 2),
            "snapped_from": [round(center[0], 3), round(center[1], 3), round(center[2], 3)],
        })

    # ---- 3) 生成地标：名字 = 「房间的物体」 ----
    landmarks: List[Dict[str, Any]] = []
    for (region_name, name), items in sorted(grouped.items()):
        # 离起点近的排前面，保证"第一张床"是最容易到达的那张
        items.sort(key=lambda item: item["geodesic_from_start_m"])
        room = REGION_ALIAS.get(region_name, region_name)
        for index, item in enumerate(items[:cap_for(name)]):
            suffix = "" if index == 0 else " (%d)" % (index + 1)
            # 房间名和物体名一样时（toilet 的 toilet）不写成"卫生间的卫生间"
            label = name if name.lower() == room.lower() else "%s的%s" % (room, name)
            landmarks.append({
                "name": label + suffix,
                "name_en": "%s in the %s" % (name, room),
                "region": room,
                "object": name,
                "position": item["position"],
                "geodesic_from_start_m": item["geodesic_from_start_m"],
                "source": "semantic-region-object",
                "how": "object AABB center snapped to navmesh; reachability verified with ShortestPath",
            })

    # ---- 4) 同房间内空间去重（同一件东西被两个类别标了名） ----
    by_region: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in landmarks:
        by_region[item["region"]].append(item)
    deduped: List[Dict[str, Any]] = []
    for _region, items in by_region.items():
        deduped.extend(merge_nearby(items))
    landmarks = deduped
    landmarks.sort(key=lambda item: (item["region"], item["object"], item["name"]))
    return landmarks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "landmarks_semantic.json"))
    parser.add_argument("--scenes", nargs="*", default=None, help="只挖这些场景；不填=全部")
    args = parser.parse_args()

    env = build_env()
    wanted = set(args.scenes) if args.scenes else None
    # 每个场景取一个 episode 来加载它（加载场景需要 episode）
    per_scene: Dict[str, Any] = {}
    for episode in env.episodes:
        key = scene_key(episode.scene_id)
        if wanted and key not in wanted:
            continue
        per_scene.setdefault(key, episode)

    out: Dict[str, Any] = {
        "schema": "agh-navlive/landmarks/2",
        "source_note": (
            "地标来自 Matterport3D 场景自带的语义标注（region=房间，object=家具），"
            "坐标是物体 AABB 中心吸附到 navmesh 后的**可走点**，"
            "并已用 ShortestPath 在真实 navmesh 上验证可达。只收录大件家具/设施。"
        ),
        "scenes": {},
    }

    for key in sorted(per_scene):
        episode = per_scene[key]
        landmarks = extract_scene(env, episode)
        counter = Counter(item["region"] for item in landmarks)
        out["scenes"][key] = {
            "scene_id": key,
            "landmark_count": len(landmarks),
            "region_count": len(counter),
            "landmarks": landmarks,
        }
        print("[mine] %-14s %3d 个地标 / %2d 个房间  %s" % (
            key, len(landmarks), len(counter),
            dict(sorted(counter.items(), key=lambda kv: -kv[1])[:6])))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
    total = sum(v["landmark_count"] for v in out["scenes"].values())
    print("[mine] 写出 %s：%d 个场景 / %d 个地标" % (args.out, len(out["scenes"]), total))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

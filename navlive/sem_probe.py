#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sem_probe.py —— 把 MP3D 语义标注的 API 摸清楚（region / object 到底能读出什么）。

之前的结论"语义区域是空的"是**错的**，原因是调用方式不对：
  · `category.name` 是**方法**，不是属性 → 直接 print 出来是一串 bound method
  · `region.centroid` 是空的 → 得从 `region.navigable_locations` 或 volume 里取点
这个脚本把每条属性/方法都实际调一遍，列出**真正可用**的字段。

用法：CUDA_VISIBLE_DEVICES=0 python3 sem_probe.py <场景id>
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, List

REPO = os.environ.get("NAVLIVE_REPO", "/srv/agh-demo/InternNav")
CONFIG = os.environ.get(
    "NAVLIVE_CONFIG", os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py")
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")


def build_env():
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
    config = get_habitat_config(module.eval_cfg.env.env_settings["config_path"])
    import habitat as _h

    with _h.config.read_write(config):
        config.habitat.dataset.split = SPLIT
    return _h.Env(config=config)


def call(obj: Any, name: str) -> Any:
    """属性或方法都试一下。"""
    value = getattr(obj, name, None)
    if value is None:
        return None
    if callable(value):
        try:
            return value()
        except Exception as exc:  # noqa: BLE001
            return "<调用失败: %s>" % exc
    return value


def describe(tag: str, obj: Any, names: List[str]) -> None:
    print("  %s:" % tag)
    for name in names:
        value = call(obj, name)
        if value is None:
            continue
        text = repr(value)
        if len(text) > 150:
            text = text[:150] + "…"
        print("    %-22s = %s" % (name, text))


def scene_key(raw: Any) -> str:
    name = str(raw).replace("\\", "/").rstrip("/").split("/")[-1]
    for suffix in (".glb", ".navmesh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else "zsNo4HB9uLZ"
    env = build_env()
    episode = next(e for e in env.episodes if scene_key(e.scene_id) == wanted)
    env.current_episode = episode
    env.reset()
    sem = env.sim.semantic_annotations()
    print("场景 %s" % wanted)

    print()
    print("=== REGION（房间级） ===")
    regions = list(getattr(sem, "regions", None) or [])
    print("  数量 = %d" % len(regions))
    if regions:
        describe("region[0]", regions[0], [
            "id", "category", "centroid", "aabb", "volume", "navigable_locations",
            "max_distance", "floor_height",
        ])
        category = getattr(regions[0], "category", None)
        if category is not None:
            describe("region[0].category", category, ["name", "id", "name_original"])
        print()
        print("  所有 region 的类别名（去掉重复）:")
        names = {}
        for region in regions:
            name = call(getattr(region, "category", None), "name")
            names[str(name)] = names.get(str(name), 0) + 1
        print("    %s" % sorted(names.items(), key=lambda kv: -kv[1]))

    print()
    print("=== OBJECT（物体级，共 %d 个） ===" % len(getattr(sem, "objects", None) or []))
    objects = list(getattr(sem, "objects", None) or [])
    if objects:
        describe("object[0]", objects[0], [
            "id", "category", "aabb", "position", "rotation", "region",
        ])
        category = getattr(objects[0], "category", None)
        if category is not None:
            describe("object[0].category", category, ["name", "id", "name_original"])
        print()
        from collections import Counter

        counter = Counter(str(call(getattr(o, "category", None), "name")) for o in objects)
        print("  object 类别 Top25: %s" % counter.most_common(25))
        print()
        print("  %-22s %-28s %s" % ("物体类别", "中心 (x,y,z)", "所属 region"))
        shown = 0
        seen = set()
        for obj in objects:
            name = str(call(getattr(obj, "category", None), "name"))
            if name in seen:
                continue
            seen.add(name)
            aabb = getattr(obj, "aabb", None)
            center = None
            if aabb is not None:
                try:
                    center = [round(float(v), 2) for v in aabb.center]
                except Exception:  # noqa: BLE001
                    center = None
            region = getattr(obj, "region", None)
            region_name = call(getattr(region, "category", None), "name") if region else None
            print("  %-22s %-28s %s" % (name, center, region_name))
            shown += 1
            if shown >= 25:
                break

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

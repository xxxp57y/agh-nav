#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""landmarks.py —— 「名字 → 坐标」这一层。

工作台能走到任意坐标，但**自然语言说的是地方名**。这一层负责把"客厅""阳台下的入口""厨房"
这类说法解析成一个真实可达的坐标点，于是 Agent 可以把用户的话变成一次真正的 goto。

数据来源（见 `data/landmarks.json` 里的 source_note）：
* 场景自带 R2R 指令里提到的地点短语；
* 坐标是该短语真正指代的回合端点（指令里最后一个地点短语 → 该回合 goal；第一个 → start）；
* 中文名来自 `build_landmarks.py` 里的别名表。

**这里不猜坐标**：表里没有的名字就明确报"没这个地标"，并给出可选项。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "landmarks.json")


def _norm(text: str) -> str:
    """归一化：小写、去标点、抹平空白。中英文都能比。"""
    text = str(text).strip().lower()
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"[.,;:!?，。；：！？'\"()（）\-_/\\]", "", text)
    return text


class LandmarkBook(object):
    """一个场景一本"地名册"。"""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self.scenes: Dict[str, List[Dict[str, Any]]] = {}
        self.source_note = ""
        self.error: Optional[str] = None
        self._index: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.reload()

    def reload(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            self.error = "地标表还不存在：%s（先跑 navlive/landmark_probe.py + build_landmarks.py）" % self.path
            return
        except ValueError as exc:
            self.error = "地标表不是合法 JSON：%s" % exc
            return
        self.error = None
        self.source_note = payload.get("source_note") or ""
        self.scenes = {}
        self._index = {}
        for scene, block in (payload.get("scenes") or {}).items():
            items = list(block.get("landmarks") or [])
            self.scenes[scene] = items
            index: Dict[str, Dict[str, Any]] = {}
            for item in items:
                keys = set()
                for value in (item.get("name"), item.get("name_en"), item.get("name_zh")):
                    if value:
                        keys.add(_norm(value))
                for alias in item.get("aliases") or []:
                    keys.add(_norm(alias))
                for key in keys:
                    if key and key not in index:
                        index[key] = item
            self._index[scene] = index

    # ---- 查询 ------------------------------------------------------------- #

    def has_scene(self, scene: str) -> bool:
        return scene in self.scenes

    def list_for(self, scene: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = self.scenes.get(scene) or []
        return items[:limit] if limit else list(items)

    def scene_summary(self) -> List[Dict[str, Any]]:
        out = []
        for scene, items in sorted(self.scenes.items()):
            out.append({
                "scene_id": scene,
                "count": len(items),
                "samples": [item.get("name") for item in items[:6]],
            })
        return out

    def resolve(self, scene: str, query: str) -> Dict[str, Any]:
        """把一句话/一个名字解析成一个地标。返回 {ok, match?, candidates?}。"""
        if not self.scenes:
            return {"ok": False, "error": self.error or "地标表是空的"}
        if scene not in self.scenes:
            return {
                "ok": False,
                "error": "场景 %s 没有地标表；可用：%s" % (scene, ", ".join(sorted(self.scenes))),
            }
        index = self._index.get(scene) or {}
        if not index:
            return {"ok": False, "error": "场景 %s 的地标表是空的" % scene}
        key = _norm(query)
        if not key:
            return {"ok": False, "error": "要查的名字是空的"}

        # 1) 精确命中（含中英文与别名）
        if key in index:
            return {"ok": True, "match": index[key], "how": "exact"}
        # 2) 包含命中：优先最长的键，避免"客厅"命中到"客厅尽头"这种更具体的地标
        hits = [(k, v) for k, v in index.items() if k and (k in key or key in k)]
        if hits:
            hits.sort(key=lambda pair: -len(pair[0]))
            return {"ok": True, "match": hits[0][1], "how": "contains", "matched_key": hits[0][0]}
        # 3) 没命中：给最接近的几个候选（按共同字符数排序，纯提示用）
        scored = []
        for k, v in index.items():
            overlap = len(set(k) & set(key))
            if overlap:
                scored.append((overlap, k, v))
        scored.sort(key=lambda row: -row[0])
        return {
            "ok": False,
            "error": "地标表里没有 %r" % query,
            "candidates": [
                {"name": v.get("name"), "name_en": v.get("name_en"), "position": v.get("position")}
                for _, _, v in scored[:5]
            ] or [
                {"name": item.get("name"), "name_en": item.get("name_en")}
                for item in self.list_for(scene, 5)
            ],
        }

    def resolve_many(self, scene: str, queries: List[str]) -> Dict[str, Any]:
        """把一串名字都解出来；任何一个解不出就整体失败（不猜、不跳过）。"""
        resolved: List[Dict[str, Any]] = []
        for query in queries:
            result = self.resolve(scene, query)
            if not result.get("ok"):
                return {
                    "ok": False,
                    "error": "解析 %r 失败：%s" % (query, result.get("error")),
                    "failed": query,
                    "candidates": result.get("candidates") or [],
                    "resolved_so_far": resolved,
                }
            item = result["match"]
            resolved.append({
                "label": item.get("name") or query,
                "position": item.get("position"),
                "name_en": item.get("name_en"),
                "source": item.get("source"),
                "episode_id": item.get("episode_id"),
                "how": result.get("how"),
            })
        return {"ok": True, "waypoints": resolved}

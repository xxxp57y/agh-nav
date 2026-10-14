#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_landmarks.py —— 从指令里挖"能叫得出名字的地方"，做成名字 → 坐标的对照表。

为什么这么做（而不是靠人拍脑袋写坐标）
--------------------------------------
1. 这台 habitat 构建里 **没有** 语义区域标注（`semantic_scene.regions` 为空，实测），
   拿不到"厨房/卧室"这种官方房间名。
2. 场景自带的 52 条 R2R 指令**本身就是人写的**，里面到处是具体地点。
   每条指令都配一对已验证可达的坐标：起点（start）与终点（goal）。
3. 关键在**怎么锚定**：一条指令里会出现多个地点短语，但只有**最后**提到的那个才是终点
   （"Walk into the living room … Then walk into the entrance under the balcony.
   **Wait in the entrance** to the other room." → 终点 = entrance）。
   所以规则是：
     · **最后一个**地点短语 → 锚到该回合的 goal
     · **第一个**地点短语（且与最后不同）→ 锚到该回合的 start
   这样每个坐标都对得上它真正指代的地方；短语重复出现时沿用第一次的锚点。

产物：`navlive/data/landmarks.json`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

#: 地点短语里允许出现的"地理名词"
PLACE_PATTERN = re.compile(
    r"\b(living room|dining room|bedroom|bathroom|kitchen|hallway|corridor|hall|foyer|"
    r"entrance|entryway|entry|doorway|door|archway|arches|arch|stairs|staircase|stairwell|"
    r"steps|balcony|family room|office|study|closet|laundry|garage|patio|porch|deck|"
    r"table|counter|sofa|couch|bed|fireplace|island|bar|room)\b",
    re.IGNORECASE,
)
#: 这些词单独出现时信息量为零，丢掉（"room" 不是地点名）
WEAK_ALONE = {"room", "door", "area", "space", "bar"}
#: 中文别名（按键做包含匹配，长键优先）
ZH_ALIASES: List[Tuple[str, str]] = [
    ("living room", "客厅"),
    ("dining room", "餐厅"),
    ("family room", "家庭厅"),
    ("bedroom", "卧室"),
    ("bathroom", "卫生间"),
    ("kitchen", "厨房"),
    ("hallway", "走廊"),
    ("hall", "走廊"),
    ("foyer", "门厅"),
    ("entrance", "入口"),
    ("entryway", "入口"),
    ("entry", "入口"),
    ("archway", "拱门"),
    ("arch", "拱门"),
    ("doorway", "门口"),
    ("door", "门"),
    ("stairwell", "楼梯间"),
    ("staircase", "楼梯"),
    ("stairs", "楼梯"),
    ("steps", "台阶"),
    ("balcony", "阳台"),
    ("office", "书房"),
    ("study", "书房"),
    ("closet", "衣帽间"),
    ("laundry", "洗衣房"),
    ("garage", "车库"),
    ("patio", "露台"),
    ("porch", "门廊"),
    ("deck", "露台"),
    ("table", "桌子"),
    ("counter", "台面"),
    ("sofa", "沙发"),
    ("couch", "沙发"),
    ("bed", "床"),
    ("fireplace", "壁炉"),
    ("island", "中岛"),
]


def log(msg: str) -> None:
    sys.stderr.write("[landmark] %s\n" % msg)
    sys.stderr.flush()


def mentions(instruction: str) -> List[str]:
    """按出现顺序返回规范化后的地点短语（保留最长的名词核心）。"""
    text = instruction.lower()
    out: List[str] = []
    for match in PLACE_PATTERN.finditer(text):
        phrase = re.sub(r"\s+", " ", match.group(0).strip())
        if phrase in WEAK_ALONE and not re.search(r"\b(the|a|an)\s+" + re.escape(phrase), text):
            continue
        out.append(phrase)
    return out


def zh_for(phrase: str) -> Optional[str]:
    for key, value in ZH_ALIASES:  # 已按长键优先排列
        if key in phrase:
            return value
    return None


def alias_key(phrase: str) -> str:
    """把短语压成可比较的键：去掉冠词与方位词，便于"同一个地方"归并。"""
    text = re.sub(
        r"\b(the|a|an|into|in|to|towards|toward|through|past|under|on|at|by|near|inside|"
        r"outside|of|from|down|up|along|across|around|beside|next)\b",
        " ",
        phrase,
    )
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", default=os.path.join(os.path.dirname(__file__), "data", "landmarks_raw.json"))
    parser.add_argument("--out", dest="dst", default=os.path.join(os.path.dirname(__file__), "data", "landmarks.json"))
    args = parser.parse_args()

    with open(args.src, encoding="utf-8") as handle:
        raw = json.load(handle)

    scenes: Dict[str, Any] = {}
    for scene, block in (raw.get("scenes") or {}).items():
        #: alias_key → 地标（同一地点只保留第一次锚定，避免坐标自相矛盾）
        table: Dict[str, Dict[str, Any]] = {}
        for episode in block.get("episodes") or []:
            goal = episode.get("goal")
            start = episode.get("start")
            if not goal:
                continue
            found = mentions(episode.get("instruction") or "")
            if not found:
                continue
            # 只锚 goal：指令里**最后**提到的地点就是"要去的地方"，
            # 而 start 是"机器人出生的位置"——把它也做成地标，会出现
            # "让机器人去客厅，它已经到了"这种没有位移的目标（实测踩到）。
            pairs: List[Tuple[str, List[float], str]] = [(found[-1], goal, "goal")]
            del start  # 保留变量语义，但不再用它做地标锚点
            for phrase, position, which in pairs:
                key = alias_key(phrase)
                if not key or key in table:
                    continue
                zh = zh_for(phrase)
                table[key] = {
                    "name": zh or phrase,
                    "name_en": phrase,
                    "name_zh": zh,
                    "aliases": sorted({a for a in (phrase, zh, key) if a}),
                    "position": [round(float(v), 3) for v in position],
                    "source": f"episode-{which}",
                    "episode_id": episode.get("episode_id"),
                    "instruction": (episode.get("instruction") or "").strip(),
                }
        scenes[scene] = {"landmarks": list(table.values())}

    payload = {
        "schema": "agh-navlive/landmarks/1",
        "source_note": (
            "地标 = 场景自带 R2R 指令里提到的地点短语；坐标 = 该短语真正指代的回合端点"
            "（指令最后一个地点短语 → goal；第一个 → start）。中文名为人工别名表。"
            "这台 habitat 构建没有语义区域标注，所以没有官方房间名。"
        ),
        "scenes": scenes,
    }
    with open(args.dst, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    total = sum(len(v["landmarks"]) for v in scenes.values())
    log("写出 %s：%d 个场景 / %d 个地标" % (args.dst, len(scenes), total))
    for scene, block in scenes.items():
        names = [f"{item['name']}({item['name_en']})" for item in block["landmarks"]]
        log("  %s: %s" % (scene, ", ".join(names[:10]) + ("…" if len(names) > 10 else "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

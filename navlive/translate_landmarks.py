#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""translate_landmarks.py —— 用大模型把地标名批量翻成中文。

为什么单独做这一步（而不是挖的时候就翻）：
  * 翻译是**一次性的**：场景固定的那几个词，翻一次缓存起来；
  * 挖地标要 GPU（habitat），翻译只要网络 —— 分开就能各自重跑；
  * 用**缓存文件**而不是重翻，避免同一批词每次结果不一样（命名要稳定）。

做法：把地标表里所有 `region` 与 `object` 的英文词（去重后通常 40–80 个）一次性
发给模型，要求返回**严格 JSON**，然后写回每个地标的 `name` / `name_en`。
模型返回不合法就重试；最终失败则保留英文原名（不编）。

用法::

    python3 translate_landmarks.py --in data/landmarks_semantic.json \\
        --cache data/zh_glossary.json [--key-file agent/.llm-key] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

DEFAULT_BASE = os.environ.get("AGNES_LLM_BASE", "https://api.agnes-ai.cn/v1")
DEFAULT_MODEL = os.environ.get("AGNES_LLM_MODEL", "agnes-3.0-flash")

PROMPT = """你是家居空间领域的翻译。下面是从 Matterport3D 真实户型里提取的英文词，
分成"房间"和"家具/设施"两类。请给出每个词的**简体中文常用名**。

要求：
1. 只输出 JSON，不要任何解释、不要 markdown 代码块。
2. 格式：{{"rooms": {{"<英文>": "<中文>"}}, "objects": {{"<英文>": "<中文>"}}}}
3. 用**家庭日常口语**里会说的词，不要学术译名：
   - chest_of_drawers → 五斗柜（不要"抽屉柜式家具"）
   - familyroom/lounge → 家庭厅；laundryroom/mudroom → 洗衣间
   - porch/terrace/deck → 阳台；entryway/foyer/lobby → 门厅
   - shelving → 置物架；counter → 台面；seating → 座位
   - bathtub → 浴缸；shower → 淋浴；toilet → 马桶
   - rec/game → 娱乐室；workout/gym/exercise → 健身房；utilityroom/toolroom → 杂物间
   - "other room" → 其他房间；bar → 吧台区；lounge → 休息厅
4. 拿不准的词，保留英文原样（不要硬翻、不要编造）。

房间：{rooms}
家具/设施：{objects}
"""


def read_key(explicit_key: str | None, key_file: str | None) -> str:
    """密钥来源优先级：显式 --key > --key-file > 环境变量 AGNES_LLM_KEY > 默认文件。

    这里踩过：把 `--key-file` 的**路径**当成密钥本身传进来，
    于是鉴权头变成 `Bearer /mnt/.../.llm-key`，网关回 401。
    """
    if explicit_key:
        return explicit_key.strip()
    candidates = [p for p in (key_file, os.environ.get("AGNES_LLM_KEY_FILE")) if p]
    candidates.append("/srv/agh-demo/agh-nav/agent/.llm-key")
    for candidate in candidates:
        try:
            with open(candidate, encoding="utf-8") as handle:
                value = handle.read().strip()
            if value:
                return value
        except OSError:
            continue
    env_key = os.environ.get("AGNES_LLM_KEY")
    if env_key:
        return env_key.strip()
    raise SystemExit("找不到模型密钥（用 --key / --key-file，或设 AGNES_LLM_KEY）")


def call_model(base: str, model: str, key: str, prompt: str, timeout: float = 120.0) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你只输出严格 JSON，不输出任何多余字符。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 2048,
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError("模型网关返回 %s：%s" % (exc.code, exc.read().decode("utf-8", "replace")[:300]))
    return ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def parse_json_loose(text: str) -> dict:
    """模型有时会裹一层 ```json，这里宽松地抠出第一个 JSON 对象。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("模型返回里没有 JSON：%s" % text[:200])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "landmarks_semantic.json"))
    parser.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "zh_glossary.json"))
    parser.add_argument("--key", default=None, help="直接给密钥")
    parser.add_argument("--key-file", default=None, help="从文件读密钥（默认取 agent/.llm-key）")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="只列要翻的词，不调模型")
    args = parser.parse_args()

    with open(args.src, encoding="utf-8") as handle:
        data = json.load(handle)

    # ---- 收集需要翻的词 ----
    rooms: set[str] = set()
    objects: set[str] = set()
    for block in (data.get("scenes") or {}).values():
        for item in block.get("landmarks") or []:
            if item.get("region"):
                rooms.add(str(item["region"]))
            if item.get("object"):
                objects.add(str(item["object"]))

    cache: dict = {"rooms": {}, "objects": {}}
    if os.path.isfile(args.cache):
        try:
            with open(args.cache, encoding="utf-8") as handle:
                cached = json.load(handle)
            cache["rooms"] = cached.get("rooms") or {}
            cache["objects"] = cached.get("objects") or {}
        except (OSError, ValueError):
            pass

    todo_rooms = sorted(r for r in rooms if r not in cache["rooms"])
    todo_objects = sorted(o for o in objects if o not in cache["objects"])
    print("房间 %d 个（待翻 %d）  家具 %d 个（待翻 %d）" % (
        len(rooms), len(todo_rooms), len(objects), len(todo_objects)))

    if args.dry_run:
        print("  待翻房间: %s" % todo_rooms)
        print("  待翻家具: %s" % todo_objects)
        return 0

    if todo_rooms or todo_objects:
        key = read_key(args.key, args.key_file)
        prompt = PROMPT.format(
            rooms=json.dumps(todo_rooms, ensure_ascii=False),
            objects=json.dumps(todo_objects, ensure_ascii=False),
        )
        last_error = None
        for attempt in range(1, 4):
            try:
                raw = call_model(args.base, args.model, key, prompt)
                parsed = parse_json_loose(raw)
                for name, zh in (parsed.get("rooms") or {}).items():
                    if name in todo_rooms and isinstance(zh, str) and zh.strip():
                        cache["rooms"][name] = zh.strip()
                for name, zh in (parsed.get("objects") or {}).items():
                    if name in todo_objects and isinstance(zh, str) and zh.strip():
                        cache["objects"][name] = zh.strip()
                print("  第 %d 次调用成功：补翻房间 %d、家具 %d" % (
                    attempt, len(todo_rooms) - len([r for r in todo_rooms if r not in cache['rooms']]),
                    len(todo_objects) - len([o for o in todo_objects if o not in cache['objects']])))
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print("  第 %d 次调用失败：%s" % (attempt, str(exc)[:160]))
        if last_error and not cache["rooms"] and not cache["objects"]:
            raise SystemExit("翻译全部失败：%s" % last_error)

        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        with open(args.cache, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print("  词表已写入 %s" % args.cache)

    missing_rooms = sorted(r for r in rooms if r not in cache["rooms"])
    missing_objects = sorted(o for o in objects if o not in cache["objects"])
    if missing_rooms or missing_objects:
        print("  ⚠️ 仍缺译名（这些将保留英文原名）：房间 %s 家具 %s" % (missing_rooms, missing_objects))

    # ---- 写回地标表 ----
    for block in (data.get("scenes") or {}).values():
        for item in block.get("landmarks") or []:
            room_en = str(item.get("region") or "")
            obj_en = str(item.get("object") or "")
            room_zh = cache["rooms"].get(room_en) or room_en
            obj_zh = cache["objects"].get(obj_en) or obj_en
            # 先用不带编号的名字，编号在下一个循环里按"同场景内是否重名"统一加，
            # 否则会出现两个都叫「卫生间的洗手池」的地标（踩过）。
            item["name"] = obj_zh if obj_zh.lower() == room_zh.lower() else "%s的%s" % (room_zh, obj_zh)
            item["name_en"] = "%s in the %s" % (obj_en, room_en)
            item["region_zh"] = room_zh
            item["object_zh"] = obj_zh
            item["translated"] = bool(cache["rooms"].get(room_en) and cache["objects"].get(obj_en))

        # 同场景内重名才加序号；能按到起点的距离排序（近的在前，编号有意义）
        from collections import Counter

        landmarks = block.get("landmarks") or []
        counts = Counter(item["name"] for item in landmarks)
        seen: Dict[str, int] = {}
        ordered = sorted(landmarks, key=lambda i: i.get("geodesic_from_start_m") or 0)
        for item in ordered:
            if counts[item["name"]] > 1:
                seen[item["name"]] = seen.get(item["name"], 0) + 1
                item["name"] = "%s (%d)" % (item["name"], seen[item["name"]])
        block["landmarks"] = ordered

    with open(args.src, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    sample = []
    for block in (data.get("scenes") or {}).values():
        for item in (block.get("landmarks") or [])[:6]:
            sample.append(item["name"])
        break
    print("  已写回 %s；示例：%s" % (args.src, "、".join(sample)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

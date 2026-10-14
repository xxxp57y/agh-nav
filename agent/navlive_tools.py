#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""navlive_tools.py —— 给大模型用的**导航工具集**（模型能调的那几个动作）。

设计原则（这几条是从"模型会不会瞎调"倒推出来的）：

1. **工具是薄的**：每个工具只做一件事，把 navlive 的真实 HTTP 接口包一层，
   返回**结构化的、可核对的**结果（世界坐标、像素坐标、步数、剩余距离）。
   不做"智能"加工，免得模型分不清"我说的"和"实际发生的"。
2. **写操作要能被看见**：会动机器人的工具（去某地、重开本轮）返回里带
   `now_at` / `target` / `distance_to_goal`，模型能据此判断"到底动没动"。
3. **不许猜地名**：地名解析失败时，工具返回明确的失败 + 候选列表，
   模型必须照实转述"没有这个地方"，而不是编一个。
4. 工具的 `description` 是给模型看的**说明书**，写清楚"什么时候该用、什么时候别用"。

坐标系一律是 Habitat 的右手系：x 向右、z 向前、y 向上（俯视图只关心 x/z）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

DEFAULT_NAVLIVE = "http://127.0.0.1:43121"


def _http(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Any:
    """极简 HTTP 客户端：只发 JSON、只收 JSON。失败抛出带原文的异常。"""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        raise RuntimeError(_explain_http(exc.code, raw)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "连不上导航服务（%s）：%s\n排查：服务器上 `tmux ls` 看 navlive 会话；本机到 43121 的端口转发。"
            % (url, exc.reason)
        ) from exc
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise RuntimeError("导航服务返回的不是 JSON：%s" % raw[:300]) from exc


def _explain_http(code: int, raw: str) -> str:
    """把 HTTP 错误翻成**模型能照实转述**的说法。"""
    detail = raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            detail = parsed.get("detail") or parsed.get("error") or raw
            if isinstance(detail, dict):
                detail = detail.get("error") or json.dumps(detail, ensure_ascii=False)
    except ValueError:
        pass
    return "导航服务返回 %s：%s" % (code, detail)


class NavTools:
    """把 navlive 的接口包成一堆"模型可调"的工具。"""

    def __init__(self, base: str = DEFAULT_NAVLIVE, session: Optional[str] = None):
        self.base = base.rstrip("/")
        #: 当前会话 id —— 由 Agent 在 /open 之后写回，后续工具都用它
        self.session = session
        self.last_state: Optional[Dict[str, Any]] = None

    # ---- 底层 --------------------------------------------------------------- #

    def _state(self, session: Optional[str] = None) -> Dict[str, Any]:
        sid = session or self.session
        query = ("?session=" + urllib.parse.quote(sid)) if sid else ""
        state = _http(self.base + "/state" + query, timeout=15)
        self.last_state = state
        return state

    def _action(self, payload: Dict[str, Any], timeout: float = 90.0) -> Dict[str, Any]:
        body = dict(payload)
        if self.session and "session" not in body:
            body["session"] = self.session
        result = _http(self.base + "/action", method="POST", payload=body, timeout=timeout)
        if isinstance(result, dict) and isinstance(result.get("state"), dict):
            self.session = result["state"].get("session_id") or self.session
            self.last_state = result["state"]
        return result

    # ---- 给模型看的工具清单 ------------------------------------------------- #

    def catalog(self) -> List[Dict[str, Any]]:
        """OpenAI 风格的 tools 定义。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "nav_status",
                    "description": (
                        "看仿真服务的健康状态与当前会话。**先调这个**确认环境可用、拿到 session_id，"
                        "再决定下一步。不需要参数。"
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "nav_list_scenes",
                    "description": (
                        "列出可用的 Matterport3D 场景与每个场景里的自然语言任务（episode）。"
                        "用户说'换个房子/换到某个场景/做第几条任务'时用它找 id。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "with_tasks": {
                                "type": "boolean",
                                "description": "true 时连每个场景的任务指令一起返回（内容较长）",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "nav_open",
                    "description": (
                        "载入一个场景（可选指定某条任务）。**换场景慢（8–20 秒），同场景换任务约 3 秒**。"
                        "调用后当前会话就切到这个场景，后续工具都作用在它上面。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scene_id": {"type": "string", "description": "场景 id，如 zsNo4HB9uLZ"},
                            "episode_id": {
                                "type": "string",
                                "description": "任务 id（可选）。不填就用该场景第一条任务。",
                            },
                        },
                        "required": ["scene_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "nav_list_places",
                    "description": (
                        "列出当前场景里**可以去的地方**（地名册）：中文名、所属房间、世界坐标。"
                        "每个场景有 40–60 个：房间（卧室/厨房/客厅/卫生间/餐厅/走廊/书房/衣帽间/"
                        "洗衣间/门厅/家庭厅/休息厅/娱乐室/健身房/阳台…）里的家具与设施"
                        "（床、沙发、桌子、洗手池、马桶、浴缸、淋浴、灶台、壁炉、楼梯…）。"
                        "**客人用地方名下指令时，先看这里有没有这个名字。**"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "region": {
                                "type": "string",
                                "description": "只看某个房间里的地方（可选），例如「卧室」「厨房」",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "nav_go_to_place",
                    "description": (
                        "**让机器人去一个地方：客人想去哪就去哪，不依赖任何预置指令。** "
                        "names 可以给多个，按顺序当成航点（先 A 再去 B）。"
                        "名字必须是 `nav_list_places` 里有的；没有就**照实说没有**并给候选，不要自己编坐标。 "
                        "走完一程机器人会停在原地等下一个目的地，**可以直接接着派下一个地方，不需要重开本轮**。"
                        "客人中途改主意也用它重派。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "地方名，按顺序；例如 ['厨房的桌子'] 或 ['卧室的床','卫生间的洗手池']",
                            },
                            "reason": {"type": "string", "description": "为什么派它去（会记进事件流）"},
                        },
                        "required": ["names"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "nav_observe",
                    "description": (
                        "读当前状态：机器人在哪、朝哪、当前目标、剩余直线距离、已走步数、碰撞次数、"
                        "计划队列、以及事件流（改派/到达/卡住）。回答'它到哪了/还剩多远'这类问题时用它。"
                        "可视化面板会同步更新。"
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "nav_pause",
                    "description": "暂停机器人（它停下但保留计划）。用户说'停一下/先别走'时用。",
                    "parameters": {
                        "type": "object",
                        "properties": {"reason": {"type": "string", "description": "暂停原因"}},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "nav_resume",
                    "description": "恢复被暂停的机器人，继续执行原计划。",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "nav_restart_episode",
                    "description": (
                        "**把本回合重置回起点。** 什么时候必须用：机器人走到 Habitat 判定成功点后"
                        "本回合就结束了（状态里 episode_finished=true），此时再派远处的地标不会执行，"
                        "表现为'派了活但不动'。想接着派下一个地方就先调它。"
                        "注意它会重载环境，可能要几十秒。"
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

    # ---- 工具实现 ----------------------------------------------------------- #

    def nav_status(self) -> Dict[str, Any]:
        health = _http(self.base + "/health", timeout=15)
        return {
            "ok": bool(health.get("ok")),
            "env_loaded": health.get("env_loaded"),
            "scene_count": health.get("scenes"),
            "episode_count": health.get("episodes"),
            "current_session": health.get("session_id"),
            "tick_hz": health.get("tick_hz"),
            "last_error": health.get("last_error"),
            "接口": self.base,
        }

    def nav_list_scenes(self, with_tasks: bool = False) -> Dict[str, Any]:
        data = _http(self.base + "/scenes", timeout=20)
        scenes = data.get("scenes") or []
        out: List[Dict[str, Any]] = []
        for scene in scenes:
            item: Dict[str, Any] = {
                "scene_id": scene.get("scene_id"),
                "episode_count": scene.get("episode_count"),
            }
            if with_tasks:
                item["tasks"] = [
                    {"episode_id": str(e.get("episode_id")), "instruction": (e.get("instruction") or "").strip()}
                    for e in (scene.get("episodes") or [])
                ]
            out.append(item)
        return {"scene_count": len(out), "scenes": out}

    def nav_open(self, scene_id: str, episode_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"scene": scene_id}
        if episode_id:
            payload["episode"] = str(episode_id)
        data = _http(self.base + "/open", method="POST", payload=payload, timeout=180)
        state = data.get("state") or data
        self.session = state.get("session_id") or self.session
        self.last_state = state
        return {
            "ok": True,
            "session_id": self.session,
            "scene_id": state.get("scene_id"),
            "episode_id": state.get("episode_id"),
            "task_instruction": (state.get("instruction") or "").strip(),
            "start_position": state.get("start_position"),
            "episode_goal": state.get("episode_goal"),
            "places_count": len(state.get("landmarks") or []),
        }

    def nav_list_places(self, region: Optional[str] = None) -> Dict[str, Any]:
        state = self._state()
        places = []
        overlay = {m["name"]: m for m in ((state.get("overlay") or {}).get("landmarks_px") or [])}
        for item in state.get("landmarks") or []:
            position = item.get("position") or []
            name = item.get("name")
            room = item.get("region")
            if region and region not in (room or "") and region not in (name or ""):
                continue
            entry = {
                "name": name,
                "room": room,
                "name_en": item.get("name_en"),
                "world": {"x": round(position[0], 2), "z": round(position[2], 2)} if len(position) >= 3 else None,
            }
            px = overlay.get(name)
            if px:
                entry["map_pixel"] = {"col": round(px["px"][0]), "row": round(px["px"][1])}
            places.append(entry)
        rooms: Dict[str, int] = {}
        for item in places:
            rooms[item["room"]] = rooms.get(item["room"], 0) + 1
        return {
            "scene_id": state.get("scene_id"),
            "place_count": len(places),
            "rooms": rooms,
            "places": places,
        }

    def nav_go_to_place(self, names: List[str], reason: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(names, str):
            names = [names]

        # —— 先看回合是不是已经结束 ——
        # 这一步不能省。Habitat 的 VLN 回合在判定成功后就结束了，之后 `env.step()`
        # 一律被拒。而 navlive 的 `goto` 会把 `episode_finished` 清成 False 并接受这次
        # 派活，**但机器人一步都不会走** —— 现象就是"模型说出发了，用户看到没动"。
        # 所以：回合已结束时先自动重开，再派活。用户不该为仿真器的回合语义买单。
        #
        # 注意：**自由导览模式下（默认）走完一程不会结束回合**（navlive 报
        # `tour_completed` 而不是 `episode_over`），所以正常连续导览不会走到这个分支；
        # 它只在"用户手动重开过/上一轮确实是回合制"时兜底。
        pre = self._state()
        pre_finished = bool(pre.get("episode_finished"))
        restart_info: Optional[Dict[str, Any]] = None
        if pre_finished:
            restart_info = self.nav_restart_episode()
            self._state()

        data = self._action({"kind": "goto_landmark", "names": list(names), "note": reason})
        result = data.get("result") or {}
        state = data.get("state") or {}
        resolved = result.get("resolved") or []
        metrics = state.get("metrics") or {}
        distance = metrics.get("distance_to_goal")
        payload: Dict[str, Any] = {
            "ok": bool(result.get("ok")),
            "dispatched": [item.get("label") for item in resolved],
            "resolved": [
                {
                    "label": item.get("label"),
                    "name_en": item.get("name_en"),
                    "world": {"x": round(item["position"][0], 2), "z": round(item["position"][2], 2)},
                }
                for item in resolved
                if item.get("position")
            ],
            "now_targeting": result.get("label"),
            "robot_at": self._pose_brief(),
            "distance_from_episode_goal_m": round(distance, 2) if isinstance(distance, (int, float)) else None,
            "auto_restarted_episode": bool(restart_info),
        }
        if restart_info:
            payload["note"] = (
                "上一回合已经结束，我**先重开了本轮**（机器人回到起点）再派的活，所以现在是真的在走。"
                "如果要跟用户解释，就说'上一轮已经结束，我重开了一轮再出发'。"
            )
        else:
            payload["note"] = "已下发，机器人开始走。想确认进展用 nav_observe。"

        # 派活后隔一小会儿读一次真实状态：如果它还停在原地，说明这次没生效，
        # 要如实告诉模型别报"正在前往"。
        import time as _time

        _time.sleep(1.5)
        after = self._state()
        moved = self._pose_brief()
        payload["robot_after_1_5s"] = moved
        if after.get("step_count") == pre.get("step_count") and not restart_info:
            payload["warning"] = (
                "派活之后 1.5 秒内机器人**一步都没走**（步数与派活前相同）。"
                "可能原因：目标已在 3 米内、或回合状态异常。"
                "请用 nav_observe 核实后再回答用户，**不要说'正在前往'**。"
            )
        return payload

    def nav_observe(self) -> Dict[str, Any]:
        state = self._state()
        metrics = state.get("metrics") or {}
        pose = (state.get("pose") or {}).get("position") or []
        events = state.get("events_tail") or []
        return {
            "scene_id": state.get("scene_id"),
            "episode_id": str(state.get("episode_id")),
            "task_instruction": (state.get("instruction") or "").strip(),
            "robot": {
                "x": round(pose[0], 2) if len(pose) > 0 else None,
                "z": round(pose[2], 2) if len(pose) > 2 else None,
            },
            "current_target": state.get("target_label"),
            "plan_queue": [p.get("label") for p in (state.get("plan") or [])],
            "distance_to_goal_m": metrics.get("distance_to_goal"),
            "steps": state.get("step_count"),
            "collisions": metrics.get("collisions"),
            "paused": state.get("paused"),
            "episode_finished": state.get("episode_finished"),
            "success": metrics.get("success"),
            "recent_events": [e.get("kind") for e in events[-8:]],
            "hint": (
                "本回合已结束 —— 想继续派活要先调 nav_restart_episode"
                if state.get("episode_finished")
                else None
            ),
        }

    def nav_pause(self, reason: Optional[str] = None) -> Dict[str, Any]:
        data = self._action({"kind": "pause", "reason": reason or "模型请求暂停"})
        return {"ok": True, "paused": (data.get("state") or {}).get("paused"), "reason": reason}

    def nav_resume(self) -> Dict[str, Any]:
        data = self._action({"kind": "resume"})
        return {"ok": True, "paused": (data.get("state") or {}).get("paused")}

    def nav_restart_episode(self) -> Dict[str, Any]:
        data = self._action({"kind": "restart"}, timeout=180)
        result = data.get("result") or {}
        state = data.get("state") or {}
        pose = (state.get("pose") or {}).get("position") or []
        return {
            "ok": bool(result.get("ok")),
            "back_at_start": {
                "x": round(pose[0], 2) if len(pose) > 0 else None,
                "z": round(pose[2], 2) if len(pose) > 2 else None,
            },
            "steps_reset_to": state.get("step_count"),
            "note": "回合已重置，现在可以继续派活。",
        }

    # ---- 调度 --------------------------------------------------------------- #

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """按名字调工具。未知名字要明确报错（模型偶发会编工具名）。"""
        handler: Optional[Callable[..., Dict[str, Any]]] = getattr(self, name, None)
        if handler is None or not name.startswith("nav_") or name not in {
            "nav_status", "nav_list_scenes", "nav_open", "nav_list_places",
            "nav_go_to_place", "nav_observe", "nav_pause", "nav_resume", "nav_restart_episode",
        }:
            raise RuntimeError(
                "没有这个工具：%r。可用工具：nav_status / nav_list_scenes / nav_open / "
                "nav_list_places / nav_go_to_place / nav_observe / nav_pause / nav_resume / "
                "nav_restart_episode" % name
            )
        return handler(**(arguments or {}))

    def _pose_brief(self) -> Optional[Dict[str, Any]]:
        state = self.last_state
        if not state:
            return None
        pose = (state.get("pose") or {}).get("position") or []
        if len(pose) < 3:
            return None
        return {"x": round(pose[0], 2), "z": round(pose[2], 2)}

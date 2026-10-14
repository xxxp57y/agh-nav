#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agnes_agent.py —— 导航 harness 的**大脑**：真大模型 + 我们的导航工具。

它做什么
--------
用户用大白话说一句话 → 这里把「系统提示 + 历史对话 + 工具清单」发给模型 →
模型决定调哪个导航工具 → 我们**真的执行**它（打到 navlive 服务上） →
把结果回喂给模型 → 模型接着说人话。循环到模型不再调工具为止。

架构
----
* **工作区（workspace）**：一个工作区 = 一个目录，放它的对话历史与元数据。
  目前只有一个 `default` 工作区，但结构留好了，以后一个场景/一个项目一个工作区。
* **会话（thread）**：工作区里的一条对话线。每个 thread 一个 JSON 文件，逐条追加重写。
* **事件流（/api/stream）**：模型推理、工具开始/结束、状态变化都写成事件，
  前端用轮询取增量，右侧可视化面板据此刷新。
* **历史即知识库**：每轮的模型回复、工具调用与其返回都落盘（`trace`），
  既是"历史对话"，也是以后检索/复盘的素材。

配置
----
从环境变量或 `F:\\AGH\\.secrets\\agnes-key.txt` 读密钥：

    AGNES_LLM_BASE   默认 https://api.agnes-ai.cn/v1
    AGNES_LLM_MODEL  默认 agnes-3.0-flash（这个令牌只能用它，已实测）
    AGNES_LLM_KEY    直接给密钥；不给就读 AGNES_LLM_KEY_FILE
    AGNES_LLM_KEY_FILE 默认 /srv/agh-demo/.secrets/agnes-key.txt

用法::

    python3 agnes_agent.py --port 43123 --navlive http://127.0.0.1:43121 \\
        --workspaces /srv/agh-demo/agh-nav/agent/workspaces
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from navlive_tools import NavTools  # noqa: E402

DEFAULT_BASE = os.environ.get("AGNES_LLM_BASE", "https://api.agnes-ai.cn/v1")
DEFAULT_MODEL = os.environ.get("AGNES_LLM_MODEL", "agnes-3.0-flash")
DEFAULT_KEY_FILE = os.environ.get("AGNES_LLM_KEY_FILE", "/srv/agh-demo/.secrets/agnes-key.txt")
DEFAULT_NAVLIVE = os.environ.get("NAVLIVE_URL", "http://127.0.0.1:43121")

#: 系统提示 —— 这是"模型的说明书"。写清楚：能做什么、不许做什么、怎么说话。
SYSTEM_PROMPT = """你是 Agnes 导航 harness 里的**导览助理**，替客人操控一台仿真机器人
（Habitat + Matterport3D 真实户型）。

## 工作方式：客人想去哪就去哪（地标驱动，不是指令驱动）
这台机器人**不靠预置指令做事**。每个场景有一份"可以去的地方"清单（40–60 个），
来自场景自带的语义标注：**房间**（卧室/厨房/客厅/卫生间/餐厅/走廊/书房/衣帽间/
洗衣间/门厅/家庭厅/休息厅/娱乐室/健身房/阳台…）里的**家具与设施**
（床、沙发、桌子、洗手池、马桶、浴缸、淋浴、灶台、壁炉、楼梯…），
名字形如「卧室的床」「厨房的桌子」「卫生间的洗手池 (2)」。
**客人指哪就去哪**，不要拿某个固定任务去套他的话。

## 你的能力
- `nav_status` 看服务是否可用
- `nav_list_scenes` 看有哪些场景
- `nav_open` 载入某个场景（换场景慢，8–20 秒）
- `nav_list_places` **看可以去的地方**（可按 `region` 过滤，例如只看"卧室"）
- `nav_go_to_place` **让机器人去某地（最常用）**，可一次给多个地方当航点
- `nav_observe` 读当前状态（它在哪、还剩多远、走到哪一步）
- `nav_pause` / `nav_resume` 暂停/恢复
- `nav_restart_episode` 把机器人送回本场景起点（**一般用不到**）

## 铁律
1. **绝不低于凭记忆编造地名或坐标。** 客人说的地方必须先在 `nav_list_places` 的结果里存在；
   不存在就如实说"这个场景里没有这个地方"，并把**同一房间或同类的候选**念给他听。
2. **走完一程不用重开。** 机器人到地方就停下等下一个目的地；客人接着说下一个地方，
   直接再调 `nav_go_to_place`。**不要**动不动就建议"重开本轮"。
3. **别假装机器人动了。** 派活之后如果客人问进展，用 `nav_observe` 读真实状态再回答。
4. **坐标只用来解释，不用来下指令。** 下指令一律用地方名。
5. 没把握就先 `nav_list_places` 看一眼，不要猜。
6. **客人说得明确就直接派**；只在"名字对不上 / 有多个候选"时才回头问。

## 说话方式
- 用中文，简洁，像导游汇报：先说结果（去了哪、到了没），再给一两个关键数字
  （还剩多少米、走了多少步）。
- 不要复述工具返回的 JSON；把里面的信息说成人话。
- 客人一次说了多个地方（"先去厨房再去卧室的床"），就把它们作为一次调用的多个航点。
"""


# --------------------------------------------------------------------------- #
# 模型客户端
# --------------------------------------------------------------------------- #

class ModelClient:
    """OpenAI 兼容的 chat/completions 客户端（只用标准库，方便直接丢到服务器上跑）。"""

    def __init__(self, base: str, model: str, key: str, timeout: float = 180.0):
        self.base = base.rstrip("/")
        self.model = model
        self.key = key
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
             temperature: float = 0.2) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base + "/chat/completions",
            data=data,
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError("模型网关返回 %s：%s" % (exc.code, detail[:500])) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("连不上模型网关 %s：%s" % (self.base, exc.reason)) from exc
        try:
            parsed = json.loads(body)
        except ValueError as exc:
            raise RuntimeError("模型网关返回的不是 JSON：%s" % body[:300]) from exc
        if "error" in parsed and not parsed.get("choices"):
            raise RuntimeError("模型网关报错：%s" % json.dumps(parsed["error"], ensure_ascii=False)[:400])
        return parsed


def read_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit.strip()
    path = os.environ.get("AGNES_LLM_KEY_FILE", DEFAULT_KEY_FILE)
    for candidate in (path, "/srv/agh-demo/agh-nav/agent/.llm-key"):
        try:
            with open(candidate, encoding="utf-8") as handle:
                value = handle.read().strip()
            if value:
                return value
        except OSError:
            continue
    raise SystemExit(
        "找不到模型密钥。请设 AGNES_LLM_KEY，或把密钥写到 %s" % DEFAULT_KEY_FILE
    )


# --------------------------------------------------------------------------- #
# 工作区 / 会话历史
# --------------------------------------------------------------------------- #

def _now() -> float:
    return time.time()


def _slug(text: str, limit: int = 28) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit] if text else "新对话"


class Workspaces:
    """一个工作区 = 一个目录；里面一个会话 = 一个 JSON 文件。

    刻意用**文件**而不是数据库：这种东西要能直接用眼睛看、用 diff 比、
    坏了能手工修；写入用"临时文件 + 原子替换"，避免半个 JSON。
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _dir(self, workspace: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", workspace) or "default"
        path = os.path.join(self.root, safe)
        os.makedirs(os.path.join(path, "threads"), exist_ok=True)
        return path

    def list(self) -> List[Dict[str, Any]]:
        out = []
        for name in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, name)
            if not os.path.isdir(path):
                continue
            meta_path = os.path.join(path, "workspace.json")
            meta = {}
            if os.path.isfile(meta_path):
                try:
                    meta = json.load(open(meta_path, encoding="utf-8"))
                except (OSError, ValueError):
                    meta = {}
            threads = self.threads(name)
            out.append({
                "workspace": name,
                "title": meta.get("title") or name,
                "created_at": meta.get("created_at"),
                "thread_count": len(threads),
                "last_active": max([t.get("updated_at") or 0 for t in threads] or [0]),
            })
        if not out:
            out.append(self.ensure("default", "导航工作区"))
        return out

    def ensure(self, name: str, title: Optional[str] = None) -> Dict[str, Any]:
        path = self._dir(name)
        meta_path = os.path.join(path, "workspace.json")
        if not os.path.isfile(meta_path):
            self._write(meta_path, {
                "workspace": name,
                "title": title or name,
                "created_at": _now(),
            })
        meta = json.load(open(meta_path, encoding="utf-8"))
        meta.setdefault("thread_count", len(self.threads(name)))
        return meta

    def threads(self, workspace: str) -> List[Dict[str, Any]]:
        directory = os.path.join(self._dir(workspace), "threads")
        out = []
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            try:
                data = json.load(open(os.path.join(directory, name), encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({
                "thread_id": data.get("thread_id"),
                "title": data.get("title"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "message_count": len([m for m in data.get("messages", []) if m.get("role") != "system"]),
                "scene_id": (data.get("context") or {}).get("scene_id"),
            })
        out.sort(key=lambda t: -(t.get("updated_at") or 0))
        return out

    def create_thread(self, workspace: str, title: Optional[str] = None) -> Dict[str, Any]:
        thread_id = "th-" + uuid.uuid4().hex[:12]
        data = {
            "thread_id": thread_id,
            "workspace": workspace,
            "title": title or "新对话",
            "created_at": _now(),
            "updated_at": _now(),
            "context": {},
            "messages": [],
            "trace": [],
        }
        self._save_thread(workspace, data)
        return data

    def get_thread(self, workspace: str, thread_id: str) -> Dict[str, Any]:
        path = os.path.join(self._dir(workspace), "threads", thread_id + ".json")
        if not os.path.isfile(path):
            raise KeyError("没有这个会话：%s" % thread_id)
        return json.load(open(path, encoding="utf-8"))

    def save_thread(self, workspace: str, data: Dict[str, Any]) -> None:
        data["updated_at"] = _now()
        self._save_thread(workspace, data)

    def delete_thread(self, workspace: str, thread_id: str) -> None:
        path = os.path.join(self._dir(workspace), "threads", thread_id + ".json")
        if os.path.isfile(path):
            os.remove(path)

    def _save_thread(self, workspace: str, data: Dict[str, Any]) -> None:
        self._write(
            os.path.join(self._dir(workspace), "threads", data["thread_id"] + ".json"),
            data,
        )

    @staticmethod
    def _write(path: str, payload: Any) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# 事件总线（前端轮询取增量）
# --------------------------------------------------------------------------- #

class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._events: List[Dict[str, Any]] = []

    def emit(self, kind: str, **payload: Any) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "ts": _now(), "kind": kind}
            event.update(payload)
            self._events.append(event)
            if len(self._events) > 800:
                del self._events[:-600]
            return event

    def since(self, seq: int) -> List[Dict[str, Any]]:
        with self._lock:
            return [e for e in self._events if e["seq"] > seq]


# --------------------------------------------------------------------------- #
# Agent 循环
# --------------------------------------------------------------------------- #

class Agent:
    """一次用户输入 → 若干轮"模型想 → 工具做 → 模型再说" → 返回最终文本。"""

    def __init__(self, model: ModelClient, tools: NavTools, bus: EventBus, store: Workspaces):
        self.model = model
        self.tools = tools
        self.bus = bus
        self.store = store
        self._lock = threading.Lock()  # 同一时刻只跑一轮，避免两条指令抢机器人

    def run(self, workspace: str, thread: Dict[str, Any], user_text: str,
            max_rounds: int = 8) -> Dict[str, Any]:
        with self._lock:
            return self._run(workspace, thread, user_text, max_rounds)

    def _run(self, workspace: str, thread: Dict[str, Any], user_text: str,
             max_rounds: int) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in thread.get("messages", []):
            if item.get("role") in ("user", "assistant") and item.get("content"):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": user_text})

        # —— 每个对话线有自己的仿真会话 ——
        # 之前所有对话线共用 NavTools 上的一个 session 指针，于是"新对话的指令跑到旧会话上"。
        # 现在把该对话线记住的 session 装进工具，跑完再读回来存下。
        context = thread.setdefault("context", {})
        saved_session = self.tools.session
        self.tools.session = context.get("session_id")
        try:
            return self._run_locked(workspace, thread, user_text, max_rounds, messages, context)
        finally:
            self.tools.session = saved_session

    def _run_locked(self, workspace: str, thread: Dict[str, Any], user_text: str,
                    max_rounds: int, messages: List[Dict[str, Any]],
                    context: Dict[str, Any]) -> Dict[str, Any]:
        thread.setdefault("messages", []).append({
            "role": "user", "content": user_text, "ts": _now(),
        })
        if thread.get("title") in (None, "", "新对话"):
            thread["title"] = _slug(user_text)

        self.bus.emit("user_message", thread_id=thread["thread_id"], text=user_text)
        tool_records: List[Dict[str, Any]] = []
        final_text = ""
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

        for round_index in range(1, max_rounds + 1):
            self.bus.emit("thinking", thread_id=thread["thread_id"], round=round_index)
            reply = self.model.chat(messages, tools=self.tools.catalog())
            usage = reply.get("usage") or {}
            usage_total["prompt_tokens"] += usage.get("prompt_tokens") or 0
            usage_total["completion_tokens"] += usage.get("completion_tokens") or 0
            choice = (reply.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            calls = message.get("tool_calls") or []

            if not calls:
                final_text = (message.get("content") or "").strip()
                self.bus.emit("assistant_message", thread_id=thread["thread_id"], text=final_text)
                break

            # 把模型的"我要调工具"这条也放进对话历史（协议要求）
            messages.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": calls,
            })
            for call in calls:
                name = ((call.get("function") or {}).get("name") or "").strip()
                raw_args = (call.get("function") or {}).get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except ValueError:
                    args = {}
                self.bus.emit(
                    "tool_start", thread_id=thread["thread_id"], tool=name,
                    args=args, round=round_index,
                )
                started = _now()
                try:
                    result = self.tools.call(name, args)
                    ok = True
                    error = None
                except Exception as exc:  # noqa: BLE001 —— 工具的错误要如实回给模型
                    result = {"ok": False, "error": str(exc)}
                    ok = False
                    error = str(exc)
                elapsed = round(_now() - started, 2)
                record = {
                    "tool": name, "args": args, "ok": ok, "error": error,
                    "result": result, "elapsed_s": elapsed, "ts": _now(),
                }
                tool_records.append(record)
                thread.setdefault("trace", []).append(record)
                self.bus.emit(
                    "tool_end", thread_id=thread["thread_id"], tool=name,
                    ok=ok, error=error, result=result, elapsed_s=elapsed,
                )
                # 工具结果回喂给模型（失败也要喂，让它知道并改口）
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or name,
                    "content": json.dumps(result, ensure_ascii=False)[:6000],
                })
        else:
            final_text = "（我连续调了很多次工具还没收敛，先停下。你再说一次想让它去哪？）"
            self.bus.emit("assistant_message", thread_id=thread["thread_id"], text=final_text)

        if final_text:
            thread["messages"].append({
                "role": "assistant", "content": final_text, "ts": _now(),
                "tool_calls_made": [r["tool"] for r in tool_records],
            })
        status = None
        try:
            status = self.tools.nav_observe()
            context["scene_id"] = status.get("scene_id")
            context["episode_id"] = status.get("episode_id")
            context["last_state"] = status
        except Exception:  # noqa: BLE001 —— 观察失败不该让整轮失败
            pass
        # 把这一轮用到的仿真会话记住：下次在这个对话线里说话，仍然作用在同一个会话上
        context["session_id"] = self.tools.session
        self.store.save_thread(workspace, thread)
        self.bus.emit("turn_done", thread_id=thread["thread_id"], usage=usage_total)
        return {
            "reply": final_text,
            "tool_calls": tool_records,
            "usage": usage_total,
            "state": status,
            "session_id": self.tools.session,
        }


# --------------------------------------------------------------------------- #
# HTTP 层
# --------------------------------------------------------------------------- #

class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "agnes-agent/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write("[agent] %s\n" % (fmt % args))

    # ---- 工具 --------------------------------------------------------------- #

    def _json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length).decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except ValueError:
            return {}

    # ---- 路由 --------------------------------------------------------------- #

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        path = parsed.path
        srv = self.server  # type: ignore[assignment]
        try:
            if path == "/health":
                self._json(200, {
                    "ok": True,
                    "model": srv.model.model,
                    "model_base": srv.model.base,
                    "navlive": srv.tools.base,
                    "workspaces": [w["workspace"] for w in srv.store.list()],
                    "uptime_s": round(_now() - srv.started_at, 1),
                })
                return
            if path == "/api/workspaces":
                self._json(200, {"workspaces": srv.store.list()})
                return
            if path == "/api/threads":
                workspace = (query.get("workspace") or ["default"])[0]
                self._json(200, {"workspace": workspace, "threads": srv.store.threads(workspace)})
                return
            if path == "/api/thread":
                workspace = (query.get("workspace") or ["default"])[0]
                thread_id = (query.get("thread_id") or [""])[0]
                self._json(200, srv.store.get_thread(workspace, thread_id))
                return
            if path == "/api/stream":
                since = int((query.get("since") or ["0"])[0])
                events = srv.bus.since(since)
                self._json(200, {"events": events, "last_seq": events[-1]["seq"] if events else since})
                return
            if path == "/api/tools":
                self._json(200, {"tools": [t["function"]["name"] for t in srv.tools.catalog()]})
                return
            if path == "/":
                page = CONSOLE_HTML.replace("__AGENT_BASE__", srv.agent_url).replace(
                    "__NAVLIVE_BASE__", srv.tools.base
                )
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self._json(404, {"error": "没有这个路径：%s" % path})
        except KeyError as exc:
            self._json(404, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": "%s: %s" % (type(exc).__name__, exc)})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        srv = self.server  # type: ignore[assignment]
        payload = self._read_json()
        try:
            if parsed.path == "/api/threads":
                workspace = payload.get("workspace") or "default"
                srv.store.ensure(workspace)
                thread = srv.store.create_thread(workspace, payload.get("title"))
                srv.bus.emit("thread_created", thread_id=thread["thread_id"], workspace=workspace)
                self._json(200, thread)
                return
            if parsed.path == "/api/chat":
                workspace = payload.get("workspace") or "default"
                thread_id = payload.get("thread_id")
                text = (payload.get("text") or "").strip()
                if not text:
                    self._json(400, {"error": "text 不能为空"})
                    return
                srv.store.ensure(workspace)
                if thread_id:
                    thread = srv.store.get_thread(workspace, thread_id)
                else:
                    thread = srv.store.create_thread(workspace)
                result = srv.agent.run(workspace, thread, text)
                self._json(200, {
                    "thread_id": thread["thread_id"],
                    "title": thread.get("title"),
                    **result,
                })
                return
            if parsed.path == "/api/thread/delete":
                srv.store.delete_thread(payload.get("workspace") or "default", payload.get("thread_id") or "")
                self._json(200, {"ok": True})
                return
            if parsed.path == "/api/reset":
                # 只重置工具侧的会话指针，不动历史
                srv.tools.session = None
                self._json(200, {"ok": True})
                return
            self._json(404, {"error": "没有这个路径：%s" % parsed.path})
        except KeyError as exc:
            self._json(404, {"error": str(exc)})
        except RuntimeError as exc:
            self._json(502, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": "%s: %s" % (type(exc).__name__, exc)})


#: 前端页面从这个文件加载（保持后端可读）
def _load_console() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("agent_console.html",):
        path = os.path.join(here, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                return handle.read()
    return "<h1>缺少 agent_console.html</h1>"


CONSOLE_HTML = _load_console()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=43123)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--navlive", default=DEFAULT_NAVLIVE)
    parser.add_argument("--workspaces", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspaces"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--key", default=os.environ.get("AGNES_LLM_KEY"))
    parser.add_argument("--agent-url", default=None, help="本服务对外的地址（前端用它回调自己）")
    args = parser.parse_args()

    key = read_key(args.key)
    model = ModelClient(args.base, args.model, key)
    tools = NavTools(args.navlive)
    bus = EventBus()
    store = Workspaces(args.workspaces)
    store.ensure("default", "导航工作区")
    agent = Agent(model, tools, bus, store)

    httpd = Server((args.host, args.port), Handler)
    httpd.model = model           # type: ignore[attr-defined]
    httpd.tools = tools           # type: ignore[attr-defined]
    httpd.bus = bus               # type: ignore[attr-defined]
    httpd.store = store           # type: ignore[attr-defined]
    httpd.agent = agent           # type: ignore[attr-defined]
    httpd.started_at = _now()     # type: ignore[attr-defined]
    httpd.agent_url = (args.agent_url or "http://127.0.0.1:%d" % args.port)  # type: ignore[attr-defined]

    sys.stderr.write(
        "[agent] Agnes 导航 Agent http://%s:%d/  模型=%s  导航=%s  工作区=%s\n"
        % (args.host, args.port, args.model, args.navlive, args.workspaces)
    )
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

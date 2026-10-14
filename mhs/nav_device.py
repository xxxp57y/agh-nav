#!/usr/bin/env python3
"""Expose the existing Habitat navlive service as a real MHS device.

The official agnes_mhs SDK owns registration, Nerve, resources, cancel and pause.
No synthetic battery, success, camera, pose or navigation data is generated here.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from agnes_mhs import CallError, Device
from PIL import Image

log = logging.getLogger("nav_mhs")


class NavClient:
    def __init__(self, base, session=None):
        self.base = base.rstrip("/")
        self.session = session
        self.token = uuid.uuid4().hex

    async def request(self, path, payload=None, binary=False, timeout=3):
        def fetch():
            body = None if payload is None else json.dumps(payload).encode()
            req = urllib.request.Request(self.base + path, data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as res:
                    data = res.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                raise RuntimeError(f"navlive HTTP {exc.code}: {detail}") from exc
            return data if binary else json.loads(data)
        return await asyncio.to_thread(fetch)

    def query(self):
        if not self.session:
            raise RuntimeError("No owned navlive session")
        return "?session=" + urllib.parse.quote(self.session, safe="")

    async def state(self):
        state = await self.request("/state" + self.query())
        if state.get("session_id") != self.session:
            raise RuntimeError("navlive returned another session")
        return state

    async def action(self, kind, **args):
        self.query()  # Never operate on the service's implicit current session.
        result = await self.request("/action", {"session": self.session, "kind": kind, **args})
        if not result.get("result", {}).get("ok"):
            raise RuntimeError(str(result.get("result", {}).get("error", "action failed")))
        return result.get("state", {})

    async def lease(self):
        await self.action("control_lease", token=self.token, ttl_s=4)

    async def stop(self):
        return await self.action("pause", reason="MHS stop/cancel/disconnect")


def map_pose(state, map_id):
    p = state["pose"]["position"]
    # navlive rotation is [w, x, y, z]; Habitat's camera looks along local -Z.
    w, qx, qy, qz = state["pose"]["rotation"]
    norm = math.sqrt(w*w + qx*qx + qy*qy + qz*qz)
    if not norm:
        raise ValueError("invalid orientation quaternion")
    w, qx, qy, qz = (q / norm for q in (w, qx, qy, qz))
    forward_x = -2 * (qx*qz + w*qy)
    forward_y = 1 - 2 * (qx*qx + qy*qy)  # MHS y = -Habitat z
    return {"map": map_id, "x": p[0], "y": -p[2],
            "yaw": math.degrees(math.atan2(forward_y, forward_x)) % 360, "ok": True}


def target_distance(state):
    p, t = state["pose"]["position"], state.get("target")
    return math.hypot(p[0] - t[0], p[2] - t[2]) if t else 0.0


def map_description(scene, state, landmarks):
    places = []
    for i, item in enumerate(landmarks):
        p = item.get("position")
        if not p or len(p) < 3:
            continue
        name = item.get("name") or item.get("name_zh") or item.get("label")
        if name:
            places.append({"id": f"place-{i}", "name": name,
                           "at": [p[0], -p[2]],
                           "description": item.get("region") or "Matterport3D semantic landmark"})
    occupancy = state.get("occupancy")
    if not occupancy:
        raise RuntimeError("navlive has no MOS occupancy grid; run the updated navlive_server.py")
    return {"id": "mp3d-" + scene.lower(), "name": "室内导览 · " + scene,
            "bounds": occupancy["bounds"], "places": places}


class NavDevice:
    def __init__(self, client, state, landmarks, frame, device_id="habitat-01"):
        self.client = client
        self.snapshot = state
        self.landmarks = landmarks
        self.map = map_description(state["scene_id"], state, landmarks)
        self.map_id = self.map["id"]
        self.initial = map_pose(state, self.map_id)
        self.odometry = {"x": 0, "y": 0, "yaw": 0, "v": 0, "w": 0}
        self.previous = None
        self.last_error = None
        self.moving = False
        self.run_timeout = 240
        with Image.open(io.BytesIO(frame)) as img:
            size = list(img.size)
        self.dev = Device(
            device_id, "robot", name="Habitat 室内导览机器人", model="Habitat + Matterport3D",
            mobile=True, localization="self", maps=[self.map],
            resources={"chassis": "reject"}, ui={"primary": "front"},
            state={
                "mode": {"type": "string"}, "scene": {"type": "string"},
                "target": {"type": "string"}, "steps": {"type": "integer", "min": 0},
                "collisions": {"type": "integer", "min": 0},
                "distance": {"type": "number", "unit": "m", "min": 0},
                "remaining": {"type": "integer", "min": 0},
            },
        )
        self.front = self.dev.source("front", "image", "Habitat 第一人称相机（真实仿真画面）",
                                     hz=2, mime="image/jpeg", encoding="rgb", size=size, default=True)
        self.floor = self.dev.source("map", "grid", "Habitat navmesh 可行走栅格（5 cm）",
                                     hz=0.2, model={"name": "habitat-navmesh", "version": "agh-navlive/1"})
        self.pose = self.dev.source("pose", "pose", "Habitat 仿真真值位姿",
                                    hz=4, model={"name": "habitat-ground-truth", "version": "agh-navlive/1"})
        self.odo = self.dev.source("odometry", "odometry", "相对接入起点的里程计", hz=4)
        self.telemetry = self.dev.source("navigation", "values", "当前导航目标的真实遥测",
            hz=2, fields={
                "steps": {"type": "integer"}, "collisions": {"type": "integer"},
                "distance": {"type": "number", "unit": "m"},
                "remaining": {"type": "integer"},
            })
        self.dev.on_stop = self.stop
        self.dev.after = lambda: {"pose": map_pose(self.snapshot, self.map_id), "odometry": dict(self.odometry)}
        self.dev.unsafe = lambda tool: "导航服务不可用，请先恢复连接" if self.last_error else None
        self._tools()
        self.update(state)

    def update(self, state):
        if state.get("scene_id") != self.snapshot.get("scene_id"):
            raise ValueError("session scene changed; reconnect with a matching map declaration")
        pose = map_pose(state, self.map_id)
        self.snapshot = state
        now = time.monotonic()
        # Odometry is in the initial BODY frame, rather than an untranslated map frame.
        dx, dy = pose["x"] - self.initial["x"], pose["y"] - self.initial["y"]
        a = math.radians(self.initial["yaw"])
        yaw = (pose["yaw"] - self.initial["yaw"] + 180) % 360 - 180
        v = w = 0.0
        if self.previous:
            prev, at = self.previous
            dt = max(now - at, 1e-3)
            v = math.hypot(pose["x"]-prev["x"], pose["y"]-prev["y"]) / dt
            w = ((pose["yaw"]-prev["yaw"]+180) % 360-180) / dt
        self.previous = (pose, now)
        self.odometry = {"x": dx*math.cos(a)+dy*math.sin(a),
                         "y": -dx*math.sin(a)+dy*math.cos(a), "yaw": yaw, "v": v, "w": w}
        m = state.get("metrics", {})
        mode = "paused" if state.get("paused") else ("navigating" if state.get("plan") else "idle")
        self.dev.update(mode=mode, scene=state["scene_id"], target=state.get("target_label") or "",
                        steps=int(state.get("step_count", 0)), collisions=int(m.get("collisions", 0)),
                        distance=round(target_distance(state), 3), remaining=len(state.get("plan", [])))
        return pose

    async def stop(self):
        self.update(await self.client.stop())

    def _tools(self):
        @self.dev.tool("列出地图中真实存在的目的地。先查名单再派活；禁止编造地点。", read_only=True,
                       timeout=5, ui={"label": "可去的地方", "group": "观察"})
        async def list_places(call):
            return {"detail": f"当前场景有 {len(self.map['places'])} 个地标",
                    "data": {"map": self.map_id, "places": self.map["places"]}}

        @self.dev.tool("读取导航状态与最近事件。距离是当前目标的平面直线距离，不是回合基准距离。",
                       read_only=True, timeout=5, ui={"label": "观察机器人", "group": "观察"})
        async def inspect_state(call):
            state = await self.client.state()
            self.update(state)
            return {"detail": "已读取真实仿真状态", "data": self.brief(state)}

        @self.dev.tool(
            "按地图里的精确中文地标名依次导航。任务有进度，可暂停/取消；只有真实到达才成功。"
            "失败时返回原因和状态，最多尝试一次合理的其他目的地，不重复盲目重试。",
            params={"places": {"type": "array", "items": {"type": "string"},
                               "minItems": 1, "maxItems": 8, "description": "list_places 返回的精确地标名，按顺序"}},
            required=["places"], uses=["chassis"], motion=True, pausable=True, timeout=250,
            ui={"label": "导航到地标", "group": "导航"})
        async def go_to_place(call, places):
            return await self.navigate(call, places)

        @self.dev.tool("回到当前场景的起点，清空路线。只在用户要求重新演示时使用。",
                       timeout=60, uses=["chassis"], motion=True,
                       ui={"label": "重置场景", "group": "导航", "confirm": "机器人会回到起点，当前路线将清空。"})
        async def reset_scene(call):
            try:
                await self.client.lease()
                # Reset may reload an environment. Use a bounded separate request.
                data = await self.client.request("/action", {"session": self.client.session, "kind": "restart"}, timeout=55)
                if not data.get("result", {}).get("ok"):
                    raise CallError("failed", "场景重置失败")
                self.update(data["state"])
                self.initial = map_pose(data["state"], self.map_id)
                self.previous = None
                self.update(data["state"])
                return {"detail": "已回到场景起点", "data": self.brief(data["state"])}
            finally:
                await self.stop()

    def brief(self, state):
        return {"scene": state["scene_id"], "target": state.get("target_label"),
                "pose": map_pose(state, self.map_id), "steps": state.get("step_count"),
                "collisions": state.get("metrics", {}).get("collisions"),
                "distance_m": round(target_distance(state), 3), "distance_kind": "planar straight-line",
                "remaining": [p["label"] for p in state.get("plan", [])],
                "paused": state.get("paused"), "events": state.get("events_tail", [])[-8:]}

    async def navigate(self, call, places):
        known = {p["name"] for p in self.map["places"]}
        missing = [p for p in places if p not in known]
        if missing:
            raise CallError("invalid", "地标名不存在或不精确：" + "、".join(missing),
                            data={"available": sorted(known)})
        state = await self.client.state()
        start_seq = state.get("seq", 0)
        start = time.monotonic()
        self.moving = True
        async def keep_lease():
            while True:
                await self.client.lease()
                await asyncio.sleep(0.5)
        heartbeat = None
        try:
            await self.client.lease()
            heartbeat = asyncio.create_task(keep_lease())
            await self.client.action("goto_landmark", names=places, note="AGH / MHS 自然语言导航")
            completed = 0
            while True:
                was_paused = call.paused
                await call.checkpoint(self.stop)
                if was_paused:
                    await self.client.lease()
                    await self.client.action("resume")
                await self.client.lease()
                if heartbeat.done():
                    await heartbeat  # Surface a failed heartbeat before requesting more motion.
                state = await self.client.state()
                self.update(state)
                recent = [e for e in state.get("events_tail", []) if e["seq"] > start_seq]
                for event in recent:
                    start_seq = max(start_seq, event["seq"])
                    if event["kind"] == "waypoint_reached":
                        completed += 1
                    if event["kind"] in ("stalled", "planner_error", "control_lease_expired"):
                        reason = {"stalled": "x_stalled", "planner_error": "x_unreachable",
                                  "control_lease_expired": "x_control_lost"}[event["kind"]]
                        raise CallError(reason, "导航停止：" + event["kind"], data=self.brief(state))
                call.progress(done=completed, total=len(places),
                              text=f"前往 {state.get('target_label')} · 直线距离 {target_distance(state):.1f} m",
                              data={"steps": state.get("step_count"), "distance_m": target_distance(state)})
                if not state.get("plan"):
                    if completed == len(places):
                        return {"detail": "已依次到达：" + " → ".join(places), "data": self.brief(state)}
                    raise CallError("x_incomplete", "路线结束，但没有收到全部航点的到达事件", data=self.brief(state))
                if time.monotonic() - start > self.run_timeout:
                    raise CallError("x_navigation_timeout", "导航超时，已停止；请检查位置后调整任务", data=self.brief(state))
                await asyncio.sleep(0.25)
        finally:
            self.moving = False
            if heartbeat:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
            # Lease is also enforced in the sim process if this process disappears or HTTP fails.
            await self.stop()

    async def stream(self):
        while True:
            try:
                state = await self.client.state()
                pose = self.update(state)
                self.last_error = None
                self.dev.problem = None
                if self.pose.wants():
                    self.pose.send(pose, t=state.get("frame_ts") or time.time())
                if self.odo.wants():
                    self.odo.send(self.odometry, t=state.get("frame_ts") or time.time())
                if self.telemetry.wants():
                    self.telemetry.send({k: self.dev.state[k] for k in ("steps", "collisions", "distance", "remaining")})
                if self.front.wants():
                    data = await self.client.request("/frame.jpg" + self.client.query(), binary=True)
                    with Image.open(io.BytesIO(data)) as img:
                        w, h = img.size
                    self.front.send({"w": w, "h": h}, data, t=state.get("frame_ts") or time.time())
                if self.floor.wants():
                    data = await self.client.request("/grid.png" + self.client.query(), binary=True)
                    grid = state["occupancy"]
                    self.floor.send({"id": self.map_id, "resolution": grid["resolution"], "origin": grid["origin"]}, data)
            except Exception as exc:
                self.last_error = str(exc)
                self.dev.problem = "导航服务不可用：" + str(exc)[:160]
                stale = map_pose(self.snapshot, self.map_id)
                stale["ok"] = False
                if self.pose.wants():
                    self.pose.send(stale)
                log.warning("telemetry: %s", exc)
            # Poll cached snapshots fast enough for the SDK's configured rates. A 250 ms sleep
            # plus HTTP work undersupplied 2 Hz sources; Source.wants() handles rate limiting.
            await asyncio.sleep(0.05)


async def main(args):
    client = NavClient(args.navlive, args.session)
    if not client.session:
        result = await client.request("/open", {"scene": args.scene}, timeout=90)
        client.session = result.get("session_id") or result.get("state", {}).get("session_id")
    await client.stop()
    state = await client.state()
    landmarks = await client.request("/landmarks" + client.query())
    frame = await client.request("/frame.jpg" + client.query(), binary=True)
    bridge = NavDevice(client, state, landmarks.get("landmarks", []), frame, args.device_id)
    if args.describe:
        print(json.dumps(bridge.dev.description(), ensure_ascii=False, indent=2))
        return
    streaming = asyncio.create_task(bridge.stream())
    try:
        log.info("MHS %s → %s, session=%s", args.device_id, args.hub, client.session)
        await bridge.dev.run(args.hub)
    finally:
        streaming.cancel()
        await asyncio.gather(streaming, return_exceptions=True)
        await client.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hub", default=os.environ.get("AGNES_HUB_URL", "ws://127.0.0.1:4180"))
    p.add_argument("--navlive", default=os.environ.get("NAVLIVE_URL", "http://127.0.0.1:43131"))
    p.add_argument("--scene", default="zsNo4HB9uLZ")
    p.add_argument("--session", help="Reuse this explicit dedicated session; never picks the global current session")
    p.add_argument("--device-id", default="habitat-01")
    p.add_argument("--describe", action="store_true")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(p.parse_args()))

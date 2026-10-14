#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""navlive_server.py —— 交互式导航服务（跑在导航服务器上，常驻）。

它把"已跑好的 MP3D 场景"变成一个**活的、可以被自然语言驱动的场景**：
打开一个会话 = 载入一个场景的一个 episode；之后可以随时改目标（"先送 C 站"），
机器人就真的在 Habitat 里一步步走过去；工作台通过 HTTP/SSE 实时看到
第一人称画面、楼层图与轨迹、位姿、逐决策遥测。

线程模型（这一条是踩过的坑，必须遵守）
--------------------------------------
**只有 sim 线程能碰 habitat / CUDA。** 第一版让 HTTP 处理函数直接调
`env.get_metrics()`，结果 Habitat 的渲染上下文跨线程炸掉（`GL::Context::current():
no current context`），整个服务退出。所以现在：

  * sim 线程按固定频率（默认 8 Hz）跑 tick：step 仿真、算指标、渲染画面与楼层图、
    生成一份**只有它写**的 JSON 快照。
  * HTTP 线程只读那份快照（读一个字符串，原子替换，无需加锁），
    所有需要动仿真的请求（open / goto / pause / resume）都**投递到命令队列**，
    由 sim 线程执行后把结果回给 HTTP 线程。
  * 浏览器侧的轨迹折线**不用服务端画**：给一个 world→map 像素的仿射变换，
    前端自己画（可缩放、可 hover，也省掉每次重画整张图）。

其他两个实测坑写在下面对应位置：场景切换代价、语义楼层图必须 colorize。

对外接口（工作台与插件只用这几个）
----------------------------------
    GET  /health                      服务与场景状态
    GET  /scenes                      可选场景与 episode 清单
    POST /open   {scene, episode}     载入场景+回合，创建会话
    GET  /state?session=…             当前快照（JSON，前端 200 ms 轮询）
    GET  /events?session=…            SSE 事件流（决策/状态变化，实时）
    GET  /frame.jpg?session=…         最近一帧第一人称画面
    GET  /map.png?session=…           最近一张楼层图（底图，不含轨迹）
    POST /action {session, kind, …}   指令：goto / pause / resume / note

用法（tmux 里常驻）::

    cd /srv/agh-demo/InternNav
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/srv/agh-demo/src \\
      /srv/agh-demo/habitat-env/bin/python -u \\
      /srv/agh-demo/agh-nav/navlive/navlive_server.py --port 43121
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = os.environ.get("ROUTE1_REPO", "/srv/agh-demo/InternNav")
CONFIG = os.environ.get(
    "NAVLIVE_CONFIG", os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py")
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")
HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (os.path.dirname(HERE), REPO):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from navlive import navlive_map  # noqa: E402
from navlive.landmarks import LandmarkBook  # noqa: E402

SCHEMA = "agh-navlive/1"
#: 「名字 → 坐标」的地名册（由 navlive/landmark_probe.py + build_landmarks.py 生成）
LANDMARKS = LandmarkBook(
    os.environ.get("NAVLIVE_LANDMARKS")
    or os.path.join(HERE, "data", "landmarks_semantic.json")
)
#: habitat 的 TopDownMap 每格像素对应的米数：map_resolution=1024 时是 0.01 m（见 habitat 源码）
METERS_PER_PIXEL = 0.01
#: 停滞判定：连续这么多步内累计位移不足 STALL_MIN_TRAVEL 米，才算真卡住。
#: （单步判据会把"边转向边小步挪"误判成卡住，实测单步约 0.01 m）
STALL_WINDOW = 30
STALL_MIN_TRAVEL = 0.20


def log(msg: str) -> None:
    sys.stderr.write("[navlive] %s\n" % msg)
    sys.stderr.flush()


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def scene_key_of(raw: Any) -> str:
    name = str(raw).replace("\\", "/").rstrip("/").split("/")[-1]
    for suffix in (".glb", ".navmesh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def instruction_text(episode) -> Optional[str]:
    value = getattr(episode, "instruction", None)
    if value is None:
        return None
    return getattr(value, "instruction_text", None) or str(value)


def goal_position(goal) -> Optional[List[float]]:
    if goal is None:
        return None
    for attr in ("position", "p"):
        value = getattr(goal, attr, None)
        if value is not None:
            return [float(v) for v in value]
    if isinstance(goal, dict) and "position" in goal:
        return [float(v) for v in goal["position"]]
    try:
        return [float(v) for v in goal]
    except TypeError:
        return None


def habitat_rotation(rotation: List[float]) -> List[float]:
    """Snapshots use w,x,y,z; Habitat's set_agent_state uses x,y,z,w.

    Passing the stored list directly turns a horizontal camera out of the scene on focus/reset.
    """
    w, x, y, z = rotation
    return [x, y, z, w]


def build_env():
    """建 habitat.Env（关雾、不画 habitat 自己的最短路径 —— 轨迹我们自己在 UI 上画）。"""
    import habitat
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat_baselines.config.default import get_config as get_habitat_config

    # oracle_success 这个测量项是 internnav 该模块 import 时注册的，必须先导入它
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401

    import importlib.util

    spec = importlib.util.spec_from_file_location("navlive_eval_config", CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["navlive_eval_config"] = module
    spec.loader.exec_module(module)
    cfg = module.eval_cfg

    config = get_habitat_config(cfg.env.env_settings["config_path"])
    with habitat.config.read_write(config):
        config.habitat.dataset.split = SPLIT
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=False,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(draw=False, visibility_dist=100.0, fov=180),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
        # —— 让 Habitat 的回合**永远不自己结束** ——
        # 这是"自由导览"能跑顺的关键。原来 habitat 会在两种情况下结束回合：
        #   ① 到达**它自己的** VLN 终点（成功判据）→ 与我们派的地方无关；
        #   ② 超过步数上限。
        # 一旦它结束了，`env.step()` 就一律返回 "Episode over, call reset before calling step"，
        # 机器人一步不动；而我之前用"结束后补救"（续轮）去兜，实测会出现
        # **一程几十次被拒 + 每次续轮重载场景**的恶性循环（一程要 2–3 分钟）。
        #
        # 正确做法是把这两个开关关掉：
        #   · 步数上限放大（自由导览本来就该允许长时间走动）
        #   · 成功判据的距离阈值设成 0（不可能满足）
        # 于是"到没到客人说的地方"完全由我们自己的距离判据决定（`Session.tick`），
        # habitat 只负责渲染与物理。回合制仍可用 `NAVLIVE_FREE_TOUR=0` 切回。
        if os.environ.get("NAVLIVE_FREE_TOUR", "1") != "0":
            try:
                config.habitat.environment.max_episode_steps = 1_000_000
            except Exception:  # noqa: BLE001 —— 字段名变了也不该让服务起不来
                pass
            for name in ("success", "oracle_success"):
                measurement = config.habitat.task.measurements.get(name)
                if measurement is None:
                    continue
                try:
                    measurement.success_distance = 0.0
                except Exception:  # noqa: BLE001
                    pass
    return habitat.Env(config)


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #


class Session(object):
    """一次"在场景里跑任务"。**所有对 env 的调用都必须发生在这个对象所属的 sim 线程里。**"""

    def __init__(self, env, episode, scene: str, session_id: str):
        import numpy as np
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

        self.env = env
        self.episode = episode
        self.scene = scene
        self.session_id = session_id

        env.current_episode = episode
        self.observation = env.reset()
        self.follower = ShortestPathFollower(env.sim, 0.2, return_one_hot=False)  # stop_distance 调小：0.5 会让它离目标老远就 STOP

        self.start_position = [float(v) for v in episode.start_position]
        goal = episode.goals[0] if getattr(episode, "goals", None) else None
        self.episode_goal = goal_position(goal)
        self.instruction = instruction_text(episode)

        self.target: List[float] = list(self.episode_goal or self.start_position)
        self.target_label = "episode 终点"
        self.plan: List[Dict[str, Any]] = []
        self.trail_world: List[List[float]] = [self.pose()["position"]]
        self.events: List[Dict[str, Any]] = []
        self.goal_reached = False
        self.paused = False
        self.pause_reason: Optional[str] = None
        self.step_count = 0
        self.episode_finished = False
        self._stall_count = 0
        self._stall_anchor: Optional[tuple] = None
        #: 最近一次"切到本会话"失败的原因（给前端显示，不静默）
        self.focus_error: Optional[str] = None
        self._last_distance: Optional[float] = None
        self.notes: Optional[str] = None
        #: 自由导览模式（默认开）：客人想去哪就去哪。
        #: 开着时，走完一程**不结束回合**（可以立刻接着去下一个地方）；
        #: 关掉则退回原来的"回合制"（到达回合终点即 episode_finished，要重开本轮）。
        self.free_tour = os.environ.get("NAVLIVE_FREE_TOUR", "1") != "0"

        self._frame: Optional[bytes] = None
        self._map_png: Optional[bytes] = None
        self._frame_ts = 0.0
        self._map_ts = 0.0
        self._map_origin = (0, 0)
        self._map_size = (0, 0)
        self._affine: Dict[str, Any] = {}
        self._pose: Dict[str, Any] = self.pose()
        self._metrics: Dict[str, Any] = {}
        self._calibration: Optional[Dict[str, Any]] = None
        self._cal_samples: List[Tuple[float, float, float, float]] = []
        self._seq = 0
        self._snapshot = "{}"
        # Optional MHS control lease. Existing clients without a lease keep their behavior.
        self._control_lease = None
        self._grid_png = None
        self._grid_meta = None

        # 首帧先渲染出来，工作台一打开就有画面。
        # 注意：这里**不**强行标定——标定靠机器人真实走动时积累的
        # 「世界位姿 ↔ agent_map_coord」样本自动解出（见 _observe_calibration）。
        self._tick_metrics()
        self.refresh_views(force=True)
        self.build_occupancy()
        self.publish()

        self.emit("session_opened", {
            "scene_id": scene,
            "episode_id": str(episode.episode_id),
            "instruction": self.instruction,
            "start_position": self.start_position,
            "goal": self.episode_goal,
        })

    # ---- 只允许在 sim 线程调用 ------------------------------------------- #

    def restart(self) -> Dict[str, Any]:
        """重置本回合（回到起点），好让演示能连续做下去。**只在 sim 线程调用。**

        为什么需要它：Habitat 的 VLN 回合在判定成功（距目标 < 3 m）后会结束，
        之后 `env.step()` 一律被拒（"Episode over, call reset before calling step"），
        于是再派一个更远的地标不会执行。工作台上表现为「派了活但机器人不动」。

        做法就是 `env.reset()` 后把会话状态清成新回合的样子——
        `reset()` 会把 agent 放回本回合的起点位姿，所以无需重新载入场景。
        """
        self.env.reset()
        self.plan = []
        self.trail_world = [self.pose()["position"]]
        self.goal_reached = False
        self.paused = False
        self.pause_reason = None
        self.step_count = 0
        self.episode_finished = False
        self._stall_count = 0
        self._last_distance = None
        self.notes = None
        self.target: Optional[List[float]] = None
        self.target_label: Optional[str] = None
        # 位姿回到起点了：清掉旧样本重新攒（旧样本的世界坐标已不再对应当前位姿序列）
        self._cal_samples = []
        self._tick_metrics()
        self.refresh_views(force=True)
        self.emit("episode_restarted", {
            "scene_id": self.scene,
            "episode_id": str(self.episode.episode_id),
            "position": self.pose()["position"],
        })
        self.publish()
        return {
            "ok": True,
            "action": "restart",
            "scene_id": self.scene,
            "episode_id": str(self.episode.episode_id),
            "position": self.pose()["position"],
            "session_id": self.session_id,
        }

    def pose(self) -> Dict[str, Any]:
        state = self.env.sim.get_agent_state()
        return {
            "position": [round(float(state.position[i]), 3) for i in range(3)],
            "rotation": [
                round(float(state.rotation.real), 4),
                round(float(state.rotation.imag[0]), 4),
                round(float(state.rotation.imag[1]), 4),
                round(float(state.rotation.imag[2]), 4),
            ],
        }

    def _map_info(self) -> Dict[str, Any]:
        return self.env.get_metrics()["top_down_map"]

    def _agent_rc(self) -> Tuple[float, float]:
        import numpy as np

        values = np.asarray(self._map_info()["agent_map_coord"]).reshape(-1)
        return (float(values[0]), float(values[1]))

    def _agent_angle(self) -> float:
        import numpy as np

        return float(np.asarray(self._map_info()["agent_angle"]).reshape(-1)[0])

    def _tick_metrics(self) -> None:
        current = self.env.get_metrics()
        collisions = current.get("collisions")
        distance = current.get("distance_to_goal")
        if distance is None and self.target:
            distance = self.env.sim.geodesic_distance(
                self.env.sim.get_agent_state().position, self.target
            )
        self._metrics = {
            "iter": self.step_count,
            "success": current.get("success"),
            "spl": current.get("spl"),
            "oracle_success": current.get("oracle_success"),
            "distance_to_goal": None if distance is None else round(float(distance), 3),
            "collisions": collisions.get("count") if isinstance(collisions, dict) else None,
            "plan_age": self._plan_age(),
            "episode_over": bool(self.env.episode_over),
        }
        self._pose = self.pose()

    def _plan_age(self) -> int:
        """距离上一次改变目标过了多少步（"计时器"信号，与 route1 的口径同名）。"""
        return self.step_count - getattr(self, "_last_plan_step", 0)

    def emit(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._seq += 1
        self.events.append({
            "seq": self._seq,
            "ts": round(time.time(), 3),
            "kind": kind,
            "payload": payload or {},
        })
        if len(self.events) > 600:
            del self.events[:300]

    # ---- 指令（sim 线程执行） -------------------------------------------- #

    def goto(self, waypoints: List[Dict[str, Any]], note: Optional[str] = None) -> Dict[str, Any]:
        clean: List[Dict[str, Any]] = []
        for item in waypoints or []:
            position = item.get("position") if isinstance(item, dict) else item
            if position is None or len(position) < 2:
                continue
            label = item.get("label") if isinstance(item, dict) else None
            clean.append({
                "label": label or "航点",
                "position": [
                    float(position[0]),
                    float(position[1]),
                    float(position[2]) if len(position) > 2 else 0.0,
                ],
            })
        if not clean:
            return {"ok": False, "error": "没有可用航点"}
        self.plan = clean
        self.target = list(clean[0]["position"])
        self.target_label = clean[0]["label"]
        self.goal_reached = False
        self.episode_finished = False
        self.paused = False
        self.pause_reason = None
        self.notes = note
        self._last_plan_step = self.step_count
        self._stall_count = 0
        # 这两个必须一起清：`_last_distance` 是**上一程**残留的距离，
        # 不清的话"停滞判定"（距离不再变小）会在新一程开头就误判，
        # 表现为"派了第二个地方，机器人动一下就报卡住"（踩过）。
        self._last_distance = None
        self._last_position = None
        self._stall_anchor = None
        self.emit("plan_changed", {
            "target": self.target,
            "label": self.target_label,
            "note": note,
            "remaining": len(self.plan),
        })
        return {"ok": True, "target": self.target, "label": self.target_label, "remaining": len(self.plan)}

    def pause(self, reason: str) -> Dict[str, Any]:
        self.paused = True
        self.pause_reason = reason
        self.emit("paused", {"reason": reason})
        return {"ok": True, "paused": True, "reason": reason}

    def resume(self) -> Dict[str, Any]:
        self.paused = False
        self.emit("resumed", {"reason": self.pause_reason})
        result = {"ok": True, "paused": False, "reason": self.pause_reason}
        self.pause_reason = None
        return result

    def set_note(self, note: str) -> Dict[str, Any]:
        self.notes = note
        self.emit("note", {"note": note})
        return {"ok": True, "note": note}

    # ---- 一步 ------------------------------------------------------------ #

    #: 一步动作的物理量，与 habitat 的 discrete action space 对齐
    STEP_FORWARD_M = 0.25
    TURN_RIGHT_DEG = 15.0

    def _apply_action(self, action: int) -> None:
        """执行一步动作。**只在 sim 线程调用**。

        这里的结论是踩出来的，写清楚免得再改坏：

        * **用 `env.step()`，不要自己改位姿。** 我试过绕开 habitat 直接
          `set_agent_state` + 四元数相乘来前进/转向，理由是"避开发动机自己的回合语义"。
          实测**失败**：连续转身时朝向会出现 +15°/+137°/+152°/0° 的怪异循环
          （`yaw_probe`/`t12` 两个最小复现都看到了），结果是机器人原地打转、规划器
          反复让它转、最后判停滞。自己重实现物理这条路不值得走。
        * 回合被 habitat 结束（到达**它自己的** VLN 终点、或超步数）之后，`env.step`
          会抛 AssertionError；那个由 `tick()` 里的**原地续轮**兜住，不在这一层处理。
        """
        self.observation = self.env.step(int(action))

    def _apply_action_unused_direct(self, action: int) -> None:
        """（已废弃）直接改位姿的版本，留作记录：这条路实测不可靠，见 `_apply_action`。

        保留这段是为了说明**为什么**不用它：
          · `quat_from_coeffs` 期望 [w,x,y,z]，而 `set_agent_state` 期望 quaternion 对象
            或 [x,y,z,w] —— 格式不一致，混用会让朝向整个乱掉；
          · 即使用四元数相乘修好了单次转向，连续转向的 yaw 仍会出现
            +15°/+137°/+152°/0° 的异常循环（两个最小复现都验过）；
          · 结果是机器人原地打转、规划器反复让它转、最后判停滞。
        """
        import habitat_sim  # noqa: F401  （仅供阅读这段历史实现时对照 API）
        return None

    def tick(self) -> None:
        self.check_control_lease()
        # 自由导览：机器人到地方后只是"这一程结束"，**不该结束回合**。
        # 但 habitat 会在三种情况下结束它自己的回合（到达它自己的终点 / 超过步数上限 /
        # reset 之后再走完一轮），一旦结束 `env.step()` 就被拒。
        # 所以统一在这里判断：只要还有没走完的航点，就原地续一轮接着走。
        #
        # 这里刻意**只看会话自己的状态**（`episode_finished` 与 `plan`），不去看
        # `env.episode_over` —— 那个字段的时机很微妙（reset 之后/habitat 内部判定），
        # 实测出现过 `env.episode_over=False` 但会话已被收口的情况，靠它判断会漏。
        if self.free_tour and self.episode_finished and self.plan:
            self.episode_finished = False
            self.goal_reached = False
            self._continue_after_episode_end()
            self._last_distance = None
        if self.episode_finished:
            self._tick_metrics()
            self._observe_calibration()
            self.refresh_views()
            return
        if not self.paused and self.plan:
            waypoint = self.plan[0]
            target = waypoint["position"]
            self.target = list(target)
            self.target_label = waypoint["label"]

            import numpy as np

            action = self.follower.get_next_action(np.asarray(target, dtype=np.float32))
            self._last_action = None if action is None else int(action)
            if action is None:
                self.emit("planner_error", {
                    "reason": "目标点不在导航网格上（unreachable）",
                    "target": target,
                })
                self.plan = []
            else:
                try:
                    # 动作一律交回 habitat 执行（它是唯一正确的物理/朝向实现）
                    self.observation = self.env.step(int(action))
                except AssertionError as exc:
                    # 回合被 habitat 结束了（它自己的终点 / 步数上限）：**原地续一轮**接着走。
                    # 注意不能在这里 `_finish_episode`，那会把计划清空、机器人永久停下。
                    log("step 被拒（%s），原地续一轮" % exc)
                    if self.plan:
                        self._continue_after_episode_end()
                    else:
                        self._finish_episode("episode_over")
                    self._tick_metrics()
                    self.refresh_views()
                    return
                # Physical steps are also counted in free-tour mode. A new waypoint does not
                # reset this counter; only an explicit scene restart does.
                self.step_count += 1
                position = self.pose()["position"]
                self.trail_world.append(position)
                if len(self.trail_world) > 6000:
                    del self.trail_world[:3000]
                distance = math.dist((position[0], position[2]), (target[0], target[2]))
                # 到达判据：0.25 m 以内（约一步）。
                # 注意：**不能要求它走到严格 0.25 m 以内**才认到达 —— ShortestPathFollower
                # 会在离目标一小段就返回 STOP，实测最近停在 0.27 m，于是"永远不到"，
                # 最后被 habitat 的回合上限结束，表现为"派了下一个地方但不动"（踩过）。
                # 所以再给一个宽松阈值：0.45 m 也算到了。
                if distance < 0.45:
                    self.emit("waypoint_reached", {
                        "label": waypoint["label"],
                        "target": target,
                        "step": self.step_count,
                        "distance": round(distance, 3),
                    })
                    self.plan.pop(0)
                    self._stall_count = 0
                    self._last_distance = None
                    self._last_position = None
                    self._stall_anchor = None
                    if not self.plan:
                        if self.free_tour:
                            # 自由导览：走完这一程只是"这一程结束"，**不结束回合**。
                            # 客人接着说下一个地方，直接接着走，不用重开本轮。
                            self.emit("tour_completed", {
                                "label": self.target_label,
                                "position": self.pose()["position"],
                                "steps": self.step_count,
                                "distance": round(distance, 3),
                            })
                        else:
                            self._finish_episode("arrived")
                else:
                    # 卡住判定：看**一段窗口内累计走了多远**，而不是单步位移。
                    # 踩过两次：
                    #   ① 用"距离不再变小"判定 → 被上一程残留的 _last_distance 干扰；
                    #   ② 用"单步位移 < 0.02 m"判定 → 机器人一边转向一边小步挪（实测
                    #      单步约 0.01 m）会被误判成卡住。
                    # 现在：窗口起点记住位置，连续 `STALL_WINDOW` 步累计位移不足
                    # `STALL_MIN_TRAVEL` 米才算真卡住。
                    if self._stall_anchor is None:
                        self._stall_anchor = (position[0], position[2])
                        self._stall_count = 0
                    else:
                        self._stall_count += 1
                        travelled = math.dist(
                            (position[0], position[2]),
                            (self._stall_anchor[0], self._stall_anchor[1]),
                        )
                        if travelled >= STALL_MIN_TRAVEL:
                            self._stall_anchor = (position[0], position[2])
                            self._stall_count = 0
                    if self._stall_count >= STALL_WINDOW:
                        self.emit("stalled", {
                            "label": waypoint["label"],
                            "distance": round(distance, 3),
                            "steps": self.step_count,
                            "window": STALL_WINDOW,
                            # 把"最近的动作"也带上：STOP(0)=规划器认为到了或到不了，
                            # 1/2/3=在走但在原地打转。没有这个字段只能猜（踩过）。
                            "last_action": getattr(self, "_last_action", None),
                            "mode": "free-tour" if self.free_tour else "episode",
                        })
                        # 卡住也**不清计划**：自由导览下客人还等着呢，
                        # 收口会把"它还在努力"变成"它彻底不干了"。
                        self.plan.pop(0)
                        self._stall_count = 0
                        self._stall_anchor = None
                self._last_position = [position[0], position[1], position[2]]
                self._last_distance = distance
        self._tick_metrics()
        self._observe_calibration()
        self.refresh_views()

    def renew_control_lease(self, token: str, ttl_s: float) -> Dict[str, Any]:
        if not token or not 1 <= ttl_s <= 10:
            raise ValueError("control lease requires a token and a TTL between 1 and 10 seconds")
        if self._control_lease and self._control_lease["token"] != token:
            raise ValueError("this session is controlled by another bridge")
        self._control_lease = {"token": token, "deadline": time.monotonic() + ttl_s, "expired": False}
        return {"ok": True, "ttl_s": ttl_s}

    def check_control_lease(self) -> None:
        lease = self._control_lease
        if lease and not lease["expired"] and time.monotonic() > lease["deadline"]:
            lease["expired"] = True
            self.pause("MHS control heartbeat lost")
            self.emit("control_lease_expired", {"reason": "MHS control heartbeat lost"})

    def build_occupancy(self) -> None:
        """Native MOS grid, in metres: MHS x=Habitat x, MHS y=-Habitat z.

        Use the navmesh directly, not the decorative screenshot or inferred pixel calibration.
        The top PNG row is the largest MHS y as required by MOS.
        """
        if self._grid_png is not None:
            return
        import numpy as np
        from PIL import Image

        resolution = 0.05
        low, high = self.env.sim.pathfinder.get_bounds()
        floor = self.env.sim.pathfinder.get_topdown_view(resolution, float(self._pose["position"][1]))
        pixels = np.where(floor, 255, 0).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="PNG")
        self._grid_png = buf.getvalue()
        self._grid_meta = {
            "resolution": resolution,
            "origin": [float(low[0]), -float(low[2]) - pixels.shape[0] * resolution],
            "bounds": [float(low[0]), -float(high[2]), float(high[0]), -float(low[2])],
            "size": [int(pixels.shape[1]), int(pixels.shape[0])],
        }

    def _continue_after_episode_end(self) -> None:
        """habitat 的回合结束后**原地续一轮**，好让自由导览接着走。**只在 sim 线程调用。**

        为什么需要它：habitat 会在三种情况下结束回合 —— 到达**它自己的**终点、
        超过步数上限、或调用了 `env.reset()` 之后又走完一轮。这些跟"客人想去哪"无关，
        但一旦结束，`env.step()` 就会返回 *"Episode over, call reset before calling step"*。
        现象就是：客人派了下一个地方，机器人一步不动。

        做法：`env.reset()`（把 agent 放回本回合起点）→ 立刻 `set_agent_state` 回到
        **它刚才所在的位置与朝向** → 继续走剩下航点。轨迹、步数、事件都保留，
        只是插入一条 `round_advanced` 说明发生过一次续轮 —— 不瞒着用户。
        代价是每次续轮要重载一次场景（数秒），比"让客人手动点重开本轮"好得多。
        """
        import numpy as np

        try:
            pose = self._pose
            self.env.reset()
            self.env.sim.set_agent_state(
                np.asarray(pose["position"], dtype=np.float32),
                np.asarray(habitat_rotation(pose["rotation"]), dtype=np.float32),
            )
            # 刷新画面：用 get_observations_at（按位姿渲染、不推进回合）。
            # 这里曾经写 `sim.step("move_forward")`，在这个环境下会抛
            # "No action move_forward in action space"，导致**整段续轮失败**，
            # 表现就是"机器人派了活但一步不动"（踩过）。
            self.observation = self.env.sim.get_observations_at(
                position=[float(v) for v in pose["position"]],
                rotation=habitat_rotation(pose["rotation"]),
                keep_agent_at_new_pose=True,
            )
            self._rounds = getattr(self, "_rounds", 0) + 1
            self.emit("round_advanced", {
                "rounds": self._rounds,
                "position": self.pose()["position"],
                "remaining_waypoints": len(self.plan),
                "note": "Habitat 的回合结束了，已原地续一轮继续走（自由导览）",
            })
        except Exception as exc:  # noqa: BLE001 —— 续轮失败就如实收口，不要静默
            log("续轮失败：%s" % exc)
            self._finish_episode("stalled")

    def _finish_episode(self, reason: str) -> None:
        self.goal_reached = True
        self.plan = []
        self.episode_finished = True
        self.emit("arrived" if reason == "arrived" else "episode_over", {
            "reason": reason,
            "label": self.target_label,
            "position": self.pose()["position"],
            "steps": self.step_count,
            "success": self.env.get_metrics().get("success"),
            "note": ("自由导览模式下不该走到这里；如果你看到这条，说明 free_tour 被关掉了"
                     if self.free_tour else None),
        })

    def refresh_views(self, force: bool = False) -> None:
        """渲染画面与楼层图，并重算 world→map 像素的仿射变换。只在 sim 线程调用。"""
        import numpy as np
        from PIL import Image

        now = time.time()
        if force or now - self._frame_ts > 0.2:
            rgb = np.asarray(self.observation["rgb"])
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="JPEG", quality=78)
            self._frame = buffer.getvalue()
            self._frame_ts = now
        if force or now - self._map_ts > 0.4:
            info = self._map_info()
            goal_rc = None
            values = np.asarray(info.get("goal_map_coord", [])).reshape(-1) if info.get("goal_map_coord") is not None else []
            if len(values) >= 2:
                goal_rc = (float(values[0]), float(values[1]))
            image, origin = navlive_map.render_with_origin(
                info, (), goal_rc=goal_rc, use_fog=False, landmarks=self.landmark_marks()
            )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            self._map_png = buffer.getvalue()
            self._map_ts = now
            self._map_origin = origin
            self._map_size = image.size
            arc_row, arc_col = self._agent_rc()
            # 终点像素**直接取 habitat 的 `goal_map_coord`**，不要再自己按
            # meters_per_pixel 推——两个轴的像素密度不同（见 _observe_calibration 的说明），
            # 粗算会把终点和地标一起推错。
            self._affine = {
                "map_size": [image.size[0], image.size[1]],
                "crop_origin": [int(origin[0]), int(origin[1])],
                "agent_px": [arc_col - origin[0], arc_row - origin[1]],
                "calibration": self._calibration,
                "meters_per_pixel_nominal": METERS_PER_PIXEL,
            }

    def _observe_calibration(self) -> None:
        """每 tick 用"当前世界位姿 ↔ 当前 agent_map_coord"喂一条样本。

        这是**真数据**：`agent_map_coord` 是 habitat 自己算的像素/栅格坐标，
        而世界位姿是同一时刻的位姿，两者天然对应，不需要瞬移也不需要猜边界。

        攒够两条**距离足够远**（>0.5 m）的样本就解出变换：

            col = scale_x * x + origin_col
            row = scale_z * z + origin_row
        """
        if not self._pose or self._calibration:
            return
        position = self._pose.get("position")
        if not position:
            return
        try:
            rc = self._agent_rc()  # (row, col)
        except Exception:  # noqa: BLE001
            return
        sample = (float(position[0]), float(position[2]), float(rc[0]), float(rc[1]))
        self._cal_samples.append(sample)
        if len(self._cal_samples) > 60:
            self._cal_samples.pop(0)
        if len(self._cal_samples) < 2:
            return
        # 取"离当前点最远"的那条样本，两点越远越准
        sx, sz, _, _ = sample
        best = max(self._cal_samples, key=lambda s: (s[0] - sx) ** 2 + (s[1] - sz) ** 2)
        bx, bz, brow, bcol = best
        dx, dz = sx - bx, sz - bz
        if abs(dx) < 0.5 and abs(dz) < 0.5:
            return  # 还没走开，先不解
        row_now, col_now = sample[2], sample[3]
        scale_x = (col_now - bcol) / dx if abs(dx) > 1e-6 else None
        scale_z = (row_now - brow) / dz if abs(dz) > 1e-6 else None
        if scale_x is None and scale_z is None:
            return
        if scale_x is None:
            scale_x = scale_z
        if scale_z is None:
            scale_z = scale_x
        self._calibration = {
            "kind": "affine-observed",
            "source": "agent_map_coord",
            "scale_x": float(scale_x),
            "scale_z": float(scale_z),
            "origin_col": float(col_now - scale_x * sx),
            "origin_row": float(row_now - scale_z * sz),
            "samples": [
                {"world": [bx, bz], "rc": [brow, bcol]},
                {"world": [sx, sz], "rc": [row_now, col_now]},
            ],
        }

    def _calibrate_map(self) -> Optional[Dict[str, Any]]:
        """兼容旧调用：返回当前已知的标定（没有就 None）。真正的样本收集在 `_observe_calibration`。"""
        return self._calibration

    def world_to_px(self, x: float, z: float) -> Optional[List[float]]:
        """世界 (x,z) → **裁剪后图像**像素 (col, row)。用标定出的变换。"""
        cal = self._calibration
        if not cal:
            return None
        ox, oy = self._map_origin
        return [
            cal["origin_col"] + cal["scale_x"] * x - ox,
            cal["origin_row"] + cal["scale_z"] * z - oy,
        ]

    def overlay_px(self) -> Dict[str, Any]:
        """给前端画覆盖层用的像素坐标（都在裁剪后图像坐标系里，和服务端 PNG 完全一致）。"""
        if not self._calibration:
            return {}
        marks = []
        for item in LANDMARKS.list_for(self.scene):
            position = item.get("position") or []
            if len(position) < 3:
                continue
            px = self.world_to_px(position[0], position[2])
            if px:
                marks.append({"name": item.get("name"), "name_en": item.get("name_en"), "px": px})
        goal_px = None
        if self.episode_goal:
            goal_px = self.world_to_px(self.episode_goal[0], self.episode_goal[2])
        target_px = None
        if self.target:
            target_px = self.world_to_px(self.target[0], self.target[2])
        trail_px = [p for p in (self.world_to_px(w[0], w[2]) for w in self.trail_world) if p]
        return {
            "calibration": self._calibration,
            "landmarks_px": marks,
            "goal_px": goal_px,
            "target_px": target_px,
            "trail_px": trail_px,
        }

    def landmark_marks(self) -> List[Dict[str, Any]]:
        """给楼层图渲染用：地标 → 裁剪后像素坐标 (row, col)。

        注意 `anchor_world` 只有 (x, z) 两维（地图是俯视的，y 不参与），
        所以取 `position[2]` 当"世界 z"，不要写成 position[1]（那是高度，实测踩过）。
        """
        if not self._affine:
            return []
        anchor_map = self._affine.get("anchor_map") or [0, 0]
        anchor_world = self._affine.get("anchor_world") or [0, 0]
        if len(anchor_world) < 2:
            return []
        marks: List[Dict[str, Any]] = []
        for item in LANDMARKS.list_for(self.scene):
            position = item.get("position") or []
            if len(position) < 3:
                continue
            col = anchor_map[0] + (position[0] - anchor_world[0]) / METERS_PER_PIXEL
            row = anchor_map[1] + (position[2] - anchor_world[1]) / METERS_PER_PIXEL
            marks.append({"name": item.get("name"), "rc": (row, col)})
        return marks

    def landmark_view(self) -> List[Dict[str, Any]]:
        """给快照用：地标的世界坐标 + 名字 + **所属房间**（前端和 Agent 都要用）。"""
        return [
            {
                "name": item.get("name"),
                "region": item.get("region") or item.get("region_zh"),
                "object": item.get("object"),
                "name_en": item.get("name_en"),
                "position": item.get("position"),
                "source": item.get("source"),
                "geodesic_from_start_m": item.get("geodesic_from_start_m"),
            }
            for item in LANDMARKS.list_for(self.scene)
        ]

    def publish(self) -> None:
        """生成给 HTTP 层读的 JSON 快照（只有一个字符串赋值，读方无需加锁）。"""
        payload = {
            "session_id": self.session_id,
            "scene_id": self.scene,
            "episode_id": str(self.episode.episode_id),
            "instruction": self.instruction,
            "start_position": self.start_position,
            "episode_goal": self.episode_goal,
            "target": self.target,
            "target_label": self.target_label,
            "plan": [{"label": w["label"], "position": w["position"]} for w in self.plan],
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "goal_reached": self.goal_reached,
            "episode_finished": self.episode_finished,
            "step_count": self.step_count,
            "pose": self._pose,
            "metrics": self._metrics,
            "trail": self.trail_world[-3000:],
            "notes": self.notes,
            "events_tail": self.events[-40:],
            "map": self._affine,
            "landmarks": self.landmark_view(),
            "overlay": self.overlay_px(),
            "frame_ts": self._frame_ts,
            "map_ts": self._map_ts,
            "seq": self._seq,
            "occupancy": self._grid_meta,
        }
        self._snapshot = json.dumps(payload, ensure_ascii=False)

    def snapshot_json(self) -> str:
        return self._snapshot

    def frame(self) -> Optional[bytes]:
        return self._frame

    def floor_map(self) -> Optional[bytes]:
        return self._map_png


# --------------------------------------------------------------------------- #
# 服务：命令队列 + sim 线程 + 快照
# --------------------------------------------------------------------------- #


class LiveService(object):
    def __init__(self) -> None:
        self.env = None
        #: 场景 → 环境实例。第一个环境也放进来（key 就是它的场景）。
        self.env_cache: Dict[str, Any] = {}
        self.sessions: Dict[str, Session] = {}
        self.current: Optional[Session] = None
        #: 切会话时把环境焦点换到目标会话。因为可能触发场景重载（20+ 秒），
        #: 这里加锁，避免两个请求同时切换把状态搅乱。
        self.focus_lock = threading.RLock()
        self.scene_index: Dict[str, List[Any]] = {}
        self.scene_meta: List[Dict[str, Any]] = []
        self.commands: "queue.Queue[Tuple[str, Callable[[], Any], threading.Event, Dict[str, Any]]]" = queue.Queue()
        self.started_at = time.time()
        self.last_error: Optional[str] = None
        self.tick_hz = float(os.environ.get("NAVLIVE_HZ", "8"))
        self.running = True
        self._counter = 0

    # ---- 命令投递（HTTP 线程 → sim 线程） ------------------------------- #

    def call(self, fn: Callable[[], Any], timeout: float = 120.0) -> Any:
        self._counter += 1
        command_id = "cmd-%d" % self._counter
        done = threading.Event()
        box: Dict[str, Any] = {}
        self.commands.put((command_id, fn, done, box))
        if not done.wait(timeout):
            raise TimeoutError("仿真线程在 %.0fs 内没有完成 %s" % (timeout, command_id))
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def drain(self, budget: int = 4) -> None:
        for _ in range(budget):
            try:
                _cid, fn, done, box = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                box["value"] = fn()
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc
                log("命令执行失败：%s" % traceback.format_exc(limit=6))
            finally:
                done.set()

    # ---- 场景（只在 sim 线程里建） --------------------------------------- #

    def _ensure_env(self) -> None:
        if self.env is not None:
            return
        os.chdir(REPO)
        t0 = time.time()
        self.env = build_env()
        # 注意：这里只读 episodes 元数据，不 step 仿真
        for episode in self.env.episodes:
            key = scene_key_of(episode.scene_id)
            self.scene_index.setdefault(key, []).append(episode)
        # 主环境登记进缓存，**只挂它自己那个场景**。别的场景第一次访问时再按需加载
        # （把 env 挂到所有场景上会让"切场景"永远拿不到新环境，等于没有缓存）。
        try:
            home_scene = scene_key_of(self.env.current_episode.scene_id)
        except Exception:  # noqa: BLE001
            home_scene = scene_key_of(self.env.episodes[0].scene_id)
        self.env_cache[home_scene] = self.env
        self.home_scene = home_scene
        log("habitat.Env 载入完成 %.1fs，episodes=%d，场景=%s"
            % (time.time() - t0, len(self.env.episodes), ", ".join(sorted(self.scene_index))))
        self.scene_meta = [
            {
                "scene_id": scene,
                "episode_count": len(items),
                "episodes": [
                    {
                        "episode_id": str(e.episode_id),
                        "instruction": instruction_text(e),
                        "start_position": [round(float(v), 2) for v in e.start_position],
                        "goal": goal_position(e.goals[0]) if getattr(e, "goals", None) else None,
                    }
                    for e in items
                ],
            }
            for scene, items in sorted(self.scene_index.items())
        ]

    def _env_for(self, scene: str) -> Any:
        """拿到某个场景的环境实例，**优先复用已经加载过的那个**。

        为什么要多环境：habitat 换场景要重建 stage 与 navmesh（实测 20–25 秒）。
        如果"切到另一个会话看一眼"每次都得等这么久，界面就没法用了。
        所以按场景缓存环境实例（上限 `NAVLIVE_ENV_CACHE`，默认 2）——
        两个场景之间来回看时，第二个环境建好之后切换就是**瞬时**的。

        建第二个环境有可能失败（显存/驱动限制）。失败不致命：退回单环境并如实写日志，
        那种情况下跨场景切换仍然慢，但会话隔离本身不受影响。
        """
        self._ensure_env()
        if scene in self.env_cache:
            return self.env_cache[scene]
        cap = int(os.environ.get("NAVLIVE_ENV_CACHE", "3"))
        if len(self.env_cache) >= cap:
            return self.env  # 超上限：退回主环境（调用方走"重载"路径）
        try:
            t0 = time.time()
            fresh = build_env()
            if scene not in {scene_key_of(ep.scene_id) for ep in fresh.episodes}:
                log("场景 %s 不在数据集里，不建独立环境" % scene)
                return self.env
            self.env_cache[scene] = fresh
            log("为场景 %s 单独载入环境（%.1fs，缓存 %d 个）" % (scene, time.time() - t0, len(self.env_cache)))
            return fresh
        except Exception as exc:  # noqa: BLE001
            log("建第二个环境失败，退回单环境（跨场景切换会变慢）：%s" % exc)
            return self.env

    def _open(self, scene: str, episode: Optional[str], instruction: Optional[str]) -> Session:
        self._ensure_env()
        if scene not in self.scene_index:
            raise KeyError("未知场景 %r；可选 %s" % (scene, ", ".join(sorted(self.scene_index))))
        episodes = self.scene_index[scene]
        chosen = episodes[0]
        if episode is not None:
            for candidate in episodes:
                if str(candidate.episode_id) == str(episode):
                    chosen = candidate
                    break
            else:
                raise KeyError("场景 %s 里没有 episode %r" % (scene, episode))
        # 注意：**不要**关掉旧会话。旧会话留在 self.sessions 里，它的 episode / 目标 /
        # 轨迹 / 事件都还在；只是不再抢仿真环境（见 focus）。下次 focus 到它会自动恢复。
        session_id = "nl-%d" % int(time.time() * 1000)
        t0 = time.time()
        session = Session(self._env_for(scene), chosen, scene, session_id)
        self.sessions[session_id] = session
        self.current = session
        # 只保留最近 12 个会话（不要淘汰焦点会话）
        while len(self.sessions) > 12:
            oldest = next(iter(self.sessions))
            if oldest in (session_id, self.current.session_id):
                break
            self.sessions.pop(oldest, None)
        log("会话 %s 打开（%s ep=%s）耗时 %.2fs" % (session_id, scene, chosen.episode_id, time.time() - t0))
        if instruction:
            session.goto(
                [{"label": "指令目标", "position": session.episode_goal or session.start_position}],
                note=instruction,
            )
        return session

    # ---- 会话隔离（关键） ------------------------------------------------ #
    #
    # 这台模拟器只有**一个 habitat.Env**，所以真正"两个会话同时跑"是做不到的。
    # 但"两个会话互不串台"必须做到，于是用一个显式的 **焦点会话** 机制：
    #
    #   · 每个 Session 自己记住 episode / 目标 / 轨迹 / 事件 / 标定（这些本来就是它自己的）；
    #   · 任何针对某会话的请求（读状态、下发动作、取图）都先调 `focus(session)`；
    #   · `focus` 若发现焦点不是它，就把环境切到该会话的 episode，并把 agent 瞬移回
    #     它的位姿 —— 于是**你看到的永远是你要看的那个会话**；
    #   · 非焦点会话**暂停推进**（不是丢失），切回去时从原处继续。
    #
    # 之前没有这一层，`_open` 直接把 `self.current` 换掉，于是新对话顶掉旧会话、
    # 前端也一直读"当前"会话 —— 就是"新对话的回答跑到旧对话里"和"换场景右侧不变"的根因。

    def focus(self, session: "Session") -> "Session":
        """把仿真环境切到该会话。**只在 sim 线程调用。** 已经是焦点就直接返回。"""
        with self.focus_lock:
            if self.current is session:
                return session
            self._ensure_env()
            previous = self.current
            t0 = time.time()
            reloaded = False
            try:
                # 优先拿该会话场景**自己的**环境（缓存命中就是瞬时的）
                target_env = self._env_for(session.scene)
                if target_env is not session.env:
                    reloaded = True
                    session.env = target_env
                self.env = target_env
                import numpy as np

                # 只有 episode **真的变了**才重建迭代器并 reset。
                # 这里踩过：无条件 reset 会让"同任务的两个会话来回切"每次都要 8–17 秒。
                same_episode = str(getattr(self.env.current_episode, "episode_id", None)) == str(
                    session.episode.episode_id
                )
                if not same_episode:
                    self.env.current_episode = session.episode
                    self.env.reset()
                    reloaded = True
                self.env.sim.set_agent_state(
                    np.asarray(session._pose["position"], dtype=np.float32),
                    np.asarray(habitat_rotation(session._pose["rotation"]), dtype=np.float32),
                )
                # 刷新第一人称画面：用 `get_observations_at` 在**当前位姿**取一次观测。
                # 绝不能换成 `sim.step(...)` —— 这个环境里三种写法都不通（实测）：
                #   sim.step("move_forward")            → AssertionError: No action … in action space
                #   sim.step("move_forward", 0.0)       → TypeError: takes 2 positional arguments
                #   sim.step({"action": "move_forward"}) → TypeError: list indices must be integers
                # 而它的异常会被下面的 except 吞掉，于是**整个焦点切换失效**：
                # 画面看着对（因为 reconcile 只到 set_agent_state 之前？不，是整段没跑），
                # 表现就是"机器人不动、没有轨迹"（踩过，很难查）。
                session.observation = self.env.sim.get_observations_at(
                    position=[float(v) for v in session._pose["position"]],
                    rotation=habitat_rotation(session._pose["rotation"]),
                    keep_agent_at_new_pose=True,
                )
                session._tick_metrics()
                session.refresh_views(force=True)
            except Exception as exc:  # noqa: BLE001 —— 切换失败不该让服务倒
                # **失败必须能被看见**：这里吞异常曾经导致最难查的一种故障 ——
                # 焦点其实没切过去，但 `self.current` 被设成了新会话，
                # 于是"界面看着对、机器人却不动、也没有轨迹"（用户看到就是这个）。
                # 现在：完整堆栈进日志，并且在会话状态里留一个 `focus_error` 给前端显示。
                import traceback as _tb
                self.last_error = "focus: %s" % exc
                session.focus_error = "%s: %s" % (type(exc).__name__, exc)
                log("切换会话焦点失败（%s → %s）：%s\n%s" % (
                    previous.session_id if previous else "-", session.session_id, exc,
                    _tb.format_exc(limit=8)))
            else:
                session.focus_error = None
            self.current = session
            log("焦点会话 %s → %s（%.2fs%s）" % (
                previous.session_id if previous else "-", session.session_id, time.time() - t0,
                "，含场景/任务重载" if reloaded else ""))
            return session

    # ---- sim 主循环 ------------------------------------------------------ #

    def loop(self) -> None:
        period = 1.0 / max(self.tick_hz, 0.5)
        while self.running:
            started = time.time()
            try:
                self.drain()
                session = self.current
                if session is not None:
                    session.tick()
                    session.publish()
                self.last_error = None
            except Exception as exc:  # noqa: BLE001
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                log("tick 出错：%s" % traceback.format_exc(limit=16))
                time.sleep(0.5)
            sleep_for = period - (time.time() - started)
            if sleep_for > 0:
                time.sleep(sleep_for)


SERVICE = LiveService()


def create_app():
    from fastapi import Body, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    app = FastAPI(title="AGH navlive", version=SCHEMA)
    # 工作台在另一个端口（AGH Web）上，浏览器直连需要 CORS；只读+本机，放开即可
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def session_of(session_id: Optional[str]) -> Session:
        if session_id and session_id in SERVICE.sessions:
            return SERVICE.sessions[session_id]
        if session_id:
            # 显式给了一个**不存在**的会话（例如服务重启后老对话还记着旧 id）：
            # 不能悄悄回退到当前会话 —— 那会让界面显示别人的场景（实测踩到）。
            # 明确报错，前端据此提示"这条对话的会话已失效，请重新开场景"。
            raise HTTPException(
                status_code=404,
                detail="会话 %s 已不存在（服务重启会清空会话；请在这条对话里重新载入场景）" % session_id,
            )
        if SERVICE.current is None:
            raise HTTPException(status_code=404, detail="还没有打开任何会话（先 POST /open）")
        return SERVICE.current

    @app.get("/health")
    def health():
        return {
            "schema": SCHEMA,
            "ok": True,
            "uptime_s": round(time.time() - SERVICE.started_at, 1),
            "env_loaded": SERVICE.env is not None,
            "scenes": len(SERVICE.scene_index),
            "episodes": sum(len(v) for v in SERVICE.scene_index.values()),
            "session_id": SERVICE.current.session_id if SERVICE.current else None,
            "tick_hz": SERVICE.tick_hz,
            "last_error": SERVICE.last_error,
            "queue_depth": SERVICE.commands.qsize(),
            "config": CONFIG,
            "split": SPLIT,
        }

    @app.get("/scenes")
    def scenes():
        def run() -> Dict[str, Any]:
            SERVICE._ensure_env()
            return {"schema": SCHEMA, "scenes": SERVICE.scene_meta}

        return SERVICE.call(run, timeout=180)

    @app.post("/open")
    def open_(payload: Dict[str, Any] = Body(...)):
        scene = str(payload.get("scene") or "")
        episode = payload.get("episode")
        instruction = payload.get("instruction")

        def run() -> Dict[str, Any]:
            session = SERVICE._open(scene, episode, instruction)
            return {"schema": SCHEMA, "state": json.loads(session.snapshot_json())}

        try:
            return SERVICE.call(run, timeout=300)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    def focus_call(fn: Callable[[Session], Any], session_id: Optional[str], timeout: float = 180.0,
                   block: bool = True) -> Any:
        """在**指定会话**上执行 `fn`：先把环境焦点切到它，再跑。只在 sim 线程里做。

        `block=False` 的用法很重要：**跨场景切换要 20+ 秒**（重建 stage/navmesh），
        如果让轮询接口一直等，界面就卡死了。所以前端那条路只在"已经是焦点"时立刻取数，
        否则先把该会话**缓存好的**快照返回，切换交给后台去完成 —— 界面立刻能切换显示，
        机器人状态随后自动刷新。
        """
        target = session_of(session_id)

        if not block and SERVICE.current is not target:
            # 后台补一次切换（丢掉结果即可；失败会在日志里）
            def kick() -> None:
                try:
                    SERVICE.call(lambda: SERVICE.focus(target), timeout=300)
                except Exception as exc:  # noqa: BLE001
                    log("后台切换会话失败：%s" % exc)

            threading.Thread(target=kick, daemon=True).start()
            return fn(target)

        def run() -> Any:
            SERVICE.focus(target)
            return fn(target)

        return SERVICE.call(run, timeout=timeout)

    @app.get("/state")
    def state(session: Optional[str] = Query(default=None)):
        return Response(
            content=focus_call(lambda s: s.snapshot_json(), session, timeout=60),
            media_type="application/json",
        )

    @app.get("/view")
    def view(session: Optional[str] = Query(default=None)):
        """一次拿全：状态快照 + 画面 + 楼层图的**版本号**。

        为什么要有这个接口：前端每次轮询要状态、要画面、要地图。
        分开请求就会让"切会话焦点"排队好几次（每次几百毫秒），
        合并成一次焦点切换 + 三个字节块，切换明显顺畅。
        图片本身仍走 `/frame.jpg` `/map.png`（可按版本号缓存）。
        """

        def build(s: Session) -> Dict[str, Any]:
            # 注意：这里**只读**缓存好的快照与图片时间戳，不重算画面/地图。
            # 原因：前端每次轮询都调它；若在这里渲染，切会话时一次操作要排队几十毫秒以上，
            # 而各会话自己的 `refresh_views` 已经把画面/地图维护好了（非焦点会话保持冻结）。
            return {
                "schema": SCHEMA,
                "session_id": s.session_id,
                # 这两个字段**以会话自身为准**（不是"当前焦点状态的快照"）。
                # 因为焦点切换可能还在后台进行，快照会短暂滞后一条；用它们显示就不会闪。
                "scene_id": s.scene,
                "episode_id": str(s.episode.episode_id),
                "instruction": s.instruction,
                "activating": SERVICE.current is not s,
                "state": json.loads(s.snapshot_json()),
                "frame_ts": s._frame_ts,
                "map_ts": s._map_ts,
                "has_frame": bool(s._frame),
                "has_map": bool(s._map_png),
            }

        # block=False：不是焦点时先给缓存快照，切换在后台做，界面不卡
        return focus_call(build, session, timeout=90, block=False)

    @app.get("/frame.jpg")
    def frame(session: Optional[str] = Query(default=None)):
        # 图片走**该会话缓存的字节**，不切焦点、不重渲染：
        # 切焦点可能触发 20 秒的场景重载，绝不能发生在取图这条路上。
        target = session_of(session)
        data = target.frame()
        if not data:
            raise HTTPException(status_code=503, detail="还没有渲染出画面")
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/map.png")
    def map_png(session: Optional[str] = Query(default=None)):
        target = session_of(session)
        data = target.floor_map()
        if not data:
            raise HTTPException(status_code=503, detail="还没有渲染出楼层图")
        return Response(content=data, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/grid.png")
    def grid_png(session: Optional[str] = Query(default=None)):
        target = session_of(session)
        if not target._grid_png:
            raise HTTPException(status_code=503, detail="occupancy grid not ready")
        return Response(content=target._grid_png, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/action")
    def action(payload: Dict[str, Any] = Body(...)):
        session_id = payload.get("session")
        current = session_of(session_id)
        kind = str(payload.get("kind") or "").strip()

        def run() -> Dict[str, Any]:
            SERVICE.focus(current)
            if kind == "goto":
                result = current.goto(payload.get("waypoints") or [], payload.get("note"))
            elif kind == "goto_landmark":
                # 自然语言那条路：给名字，服务端查地名册换成坐标
                names = payload.get("names") or payload.get("landmarks") or []
                if isinstance(names, str):
                    names = [names]
                resolved = LANDMARKS.resolve_many(current.scene, [str(n) for n in names])
                if not resolved.get("ok"):
                    raise KeyError(resolved.get("error") or "地标解析失败")
                waypoints = [
                    {"label": item["label"], "position": item["position"]}
                    for item in resolved["waypoints"]
                ]
                result = current.goto(waypoints, payload.get("note"))
                result["resolved"] = resolved["waypoints"]
            elif kind == "restart":
                # 回合已经结束时用它把机器人放回起点，接着派下一个活
                result = current.restart()
            elif kind == "pause":
                result = current.pause(str(payload.get("reason") or "人工暂停"))
            elif kind == "control_lease":
                result = current.renew_control_lease(str(payload.get("token") or ""), float(payload.get("ttl_s", 4)))
            elif kind == "resume":
                result = current.resume()
            elif kind == "note":
                result = current.set_note(str(payload.get("note") or ""))
            else:
                raise KeyError("未知指令 kind=%r" % kind)
            current.publish()
            return {"schema": SCHEMA, "result": result, "state": json.loads(current.snapshot_json())}

        try:
            return SERVICE.call(run, timeout=60)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/tour")
    def tour(session: Optional[str] = Query(default=None)):
        """自由导览诊断：`episode_over` / `plan` / `free_tour` / 续轮次数的真相。

        为什么要这个接口：自由导览最容易卡在"habitat 的回合结束了但代码没接住"，
        光看 `/state` 分不清是"没派活"、"回合结束"还是"续轮失败"。这里一次说清。
        """

        def run() -> Dict[str, Any]:
            target = session_of(session)
            import numpy as np

            return {
                "session_id": target.session_id,
                "free_tour": target.free_tour,
                "env_episode_over": bool(getattr(target.env, "episode_over", None)),
                "session_episode_finished": target.episode_finished,
                "focus_error": getattr(target, "focus_error", None),
                "plan_len": len(target.plan),
                "plan": [w.get("label") for w in target.plan],
                "target_label": target.target_label,
                "step_count": target.step_count,
                "rounds": getattr(target, "_rounds", 0),
                "paused": target.paused,
                "stall_count": target._stall_count,
                "last_distance": target._last_distance,
                "pose": target._pose["position"],
                "env_metrics_success": target.env.get_metrics().get("success"),
                "env_measure_distance": target.env.get_metrics().get("distance_to_goal"),
                "np_probe": float(np.asarray([0.0]).sum()),
            }

        return SERVICE.call(run, timeout=60)

    @app.get("/envs")
    def envs():
        """环境缓存诊断：每个场景是不是有**自己的**环境实例、上限是多少。

        为什么要这个接口：跨场景切换慢不慢，全看这里缓存命中没有。
        没有它只能靠猜（踩过）。
        """
        return {
            "cache_limit": int(os.environ.get("NAVLIVE_ENV_CACHE", "3")),
            "home_scene": getattr(SERVICE, "home_scene", None),
            "cached_scenes": sorted(SERVICE.env_cache),
            "session_env_map": {
                sid: (getattr(session, "scene", None), id(session.env))
                for sid, session in SERVICE.sessions.items()
            },
            "distinct_envs": len({id(env) for env in SERVICE.env_cache.values()}),
            "current_session": SERVICE.current.session_id if SERVICE.current else None,
        }

    @app.get("/mapdebug")
    def mapdebug(session: Optional[str] = Query(default=None)):
        """地图坐标诊断：把"原始数组尺寸 / agent 栅格坐标 / 裁剪原点 / 世界边界"一次打出来。

        为什么需要它：地图上的地标与轨迹一旦错位，光看 PNG 只能猜。
        这里给出可用于**精确**换算的全部量：`bounds` 是地图覆盖的世界范围，
        有了它就不必再用"当前位姿 ↔ 像素"这种会被裁剪/缩放带偏的锚点法。
        """

        def run() -> Dict[str, Any]:
            import numpy as np

            session_obj = session_of(session)
            info = session_obj._map_info()
            raw = np.asarray(info["map"])
            origin = session_obj._map_origin
            image = session_obj._map_size
            lower = info.get("map_lower_bound")
            upper = info.get("map_upper_bound")
            lower = np.asarray(lower).reshape(-1).tolist() if lower is not None else None
            upper = np.asarray(upper).reshape(-1).tolist() if upper is not None else None
            return {
                "schema": SCHEMA,
                "scene_id": session_obj.scene,
                "raw_shape": list(raw.shape),
                "raw_dtype": str(raw.dtype),
                "crop_origin": list(origin),
                "image_size": [int(image[0]), int(image[1])],
                "agent_map_coord": list(session_obj._agent_rc()),
                "meters_per_pixel": METERS_PER_PIXEL,
                "map_lower_bound": lower,
                "map_upper_bound": upper,
                "pose": session_obj.pose()["position"],
                "map_keys": sorted(info.keys()),
            }

        return SERVICE.call(run, timeout=20)

    @app.get("/landmarks")
    def landmarks(scene: Optional[str] = Query(default=None), session: Optional[str] = Query(default=None)):
        """地名册：列出某个场景可用的地标（不传场景就用当前会话的场景）。"""
        if LANDMARKS.error and not LANDMARKS.scenes:
            raise HTTPException(status_code=503, detail=LANDMARKS.error)
        target_scene = scene
        if not target_scene:
            try:
                target_scene = session_of(session).scene
            except KeyError:
                target_scene = None
        if target_scene:
            if not LANDMARKS.has_scene(target_scene):
                raise HTTPException(
                    status_code=404,
                    detail="场景 %s 没有地标表；可用：%s" % (target_scene, ", ".join(sorted(LANDMARKS.scenes))),
                )
            return {
                "schema": SCHEMA,
                "command": "landmarks",
                "scene_id": target_scene,
                "source_note": LANDMARKS.source_note,
                "landmarks": LANDMARKS.list_for(target_scene),
            }
        return {
            "schema": SCHEMA,
            "command": "landmarks",
            "source_note": LANDMARKS.source_note,
            "scenes": LANDMARKS.scene_summary(),
        }

    @app.post("/resolve")
    def resolve(payload: Dict[str, Any] = Body(...)):
        """把名字（可多个）解析成坐标。**不猜**：解析不出就明确失败并给候选。"""
        scene = str(payload.get("scene") or "")
        if not scene:
            try:
                scene = session_of(payload.get("session")).scene
            except KeyError:
                raise HTTPException(status_code=400, detail="必须给 scene，或者先打开一个会话")
        queries = payload.get("names") or payload.get("queries") or []
        if isinstance(queries, str):
            queries = [queries]
        result = LANDMARKS.resolve_many(scene, [str(q) for q in queries])
        status = 200 if result.get("ok") else 404
        return JSONResponse(status_code=status, content={"schema": SCHEMA, "scene_id": scene, **result})

    @app.get("/events")
    def events(session: Optional[str] = Query(default=None), since: int = Query(default=0)):
        current = session_of(session)

        def generator():
            cursor = int(since)
            idle = 0.0
            while True:
                tail = [e for e in current.events if e["seq"] > cursor]
                for event in tail:
                    cursor = event["seq"]
                    yield "data: %s\n\n" % json.dumps(event, ensure_ascii=False)
                if not tail:
                    idle += 0.4
                    if idle > 15:
                        yield ": ping\n\n"
                        idle = 0.0
                time.sleep(0.4)

        return StreamingResponse(generator(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    @app.exception_handler(Exception)
    async def unhandled(request, exc):  # noqa: ANN001
        log("未处理异常：%s" % traceback.format_exc(limit=8))
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=43121)
    parser.add_argument("--hz", type=float, default=float(os.environ.get("NAVLIVE_HZ", "8")))
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--open", default=None, help="启动后立刻打开这个场景，如 2azQ1b91cZZ")
    parser.add_argument("--episode", default=None)
    args = parser.parse_args()

    SERVICE.tick_hz = args.hz
    threading.Thread(target=SERVICE.loop, name="navlive-sim", daemon=True).start()

    if args.preload or args.open:
        SERVICE.call(lambda: SERVICE._ensure_env(), timeout=300)
        if args.open:
            SERVICE.call(lambda: SERVICE._open(args.open, args.episode, None), timeout=300)

    import uvicorn

    log("navlive 服务启动 http://%s:%d （tick %.1f Hz）" % (args.host, args.port, SERVICE.tick_hz))
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

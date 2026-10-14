"""
路线① 操作侧：Isaac Sim 6.0.1 最小「移动操作 pick」任务 + 子目标判定 + 失败注入。

设计要点
--------
* **standalone SimulationApp**（不走 Isaac Lab，6.0.1 无 isaacsim.core.api 之外的 Lab）。
* 场景全用**本地**基本几何体 + 真物理：一张桌子 + 1 个可抓盒子 + 1 个目标托盘
  + 1 个自建二指平行夹爪。**不下载任何远程资产**（本机 `assets/` 为空，YCB 等都要联网）。
* 夹爪的物理实现（这里踩过坑，记录一下）：
  - `isaacsim.core.api.World` 在 6.0.1 里是 `extsDeprecated` 包，可用；但
    `new_stage()` 会触发 StageEvent 把 `World._scene` 删掉 → 必须显式重建（见 build_scene）。
  - PhysX **不支持**铰接体里有 kinematic body：
    `Articulations with kinematic bodies are not supported`。
  - 用 PrismaticJoint 拼二指时，关节的初始位置/轴向在 PhysX 里的解释与 USD 直觉不一致
    （实测两指会沿同一世界方向漂移，见 report 的"排除过的方案"）。
  - 因此改用**最朴素也最可验证**的方案：两指是**独立动态刚体**，由**真实弹簧-阻尼力**
    （F = k·Δx − c·v，作用在质心、无力矩）驱动；指头与物体的接触是真 PhysX 接触，
    接触力由 `RigidContactView.get_contact_force_matrix` 实测。
    → 抓空时指头被弹簧推到目标宽度（≈0.0805 m，说明中间是空的）；
    → 抓稳时指头被物体挡住，停在物体宽度上（≈0.080 m），且接触力非零。
    这正是"抓空 vs 抓稳"能做物理区分的关键。
* 失败注入 X1/X2/X3 + success 控制组，统一输出 FailureEvent JSONL（字段严格照契约）。

CLI
---
    ./python.sh manip_task.py --mode success|X1|X2|X3 --trials 3 --out <jsonl> --headless
    可选：--steps 600  --record-trace  --seed N  --trial-id S  --gpu-sample
          --recover / --no-recover   --x2-variant friction|open_early
          --out-dir <dir>  --summary-json <path>

Python 3.9 兼容（本文件在 kit python 3.12 下运行，但语法不越 3.9）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 让 `import subgoal_judge` 在两种运行姿势下都成立
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from subgoal_judge import (  # noqa: E402
    FailureVerdict,
    StepObs,
    SubgoalJudge,
    SubgoalThresholds,
    detect_grasp_failure,
    detect_misplace_failure,
    detect_slip_failure,
)

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

DEFAULT_DT = 1.0 / 120.0

GRIPPER_ROOT = "/World/Gripper"
PALM_PATH = GRIPPER_ROOT + "/palm"
FINGER_LINK = (GRIPPER_ROOT + "/finger_1", GRIPPER_ROOT + "/finger_2")
OBJECT_PATH = "/World/object"
TABLE_PATH = "/World/table"
PLATE_PATH = "/World/goal_plate"
GROUND_PATH = "/World/ground"
MATERIALS_PATH = "/World/PhysicsMaterials"
PHYSICS_SCENE_PATH = "/physicsScene"


@dataclass
class TaskConfig:
    """场景 + 夹爪 + 控制参数。所有长度 m、质量 kg、力 N。"""

    # --- 桌子 ---
    table_pos: Tuple[float, float, float] = (0.05, 0.0, 0.725)
    table_size: Tuple[float, float, float] = (1.0, 0.7, 0.05)

    # --- 被操作物体 ---
    cube_size: float = 0.06
    cube_mass: float = 0.20
    object_init_xy: Tuple[float, float] = (0.0, 0.0)

    # --- 目标托盘 ---
    goal_xy: Tuple[float, float] = (0.35, 0.0)
    plate_size: Tuple[float, float, float] = (0.20, 0.20, 0.008)

    # --- 夹爪几何 ---
    #: 指头相对 base 的横向偏置（张开时指心位置 ±finger_x_max）
    finger_size: Tuple[float, float, float] = (0.02, 0.08, 0.07)
    finger_x_max: float = 0.0475
    #: 闭合指令位置：指心 ±0.0400 → 内表面正好贴到 ±0.030 的物体侧面（夹紧 0）
    finger_x_close: float = 0.0395
    palm_size: Tuple[float, float, float] = (0.16, 0.10, 0.03)
    palm_center_z: float = 0.075

    # --- 指头驱动（真实弹簧-阻尼力）---
    finger_mass: float = 0.05
    finger_stiffness: float = 400.0     # N/m
    finger_damping: float = 0.6         # N·s/m  → ζ≈0.95
    finger_force_limit: float = 200.0   # N
    #: 指头沿 x 速度限幅（防止穿透）：m/s
    finger_v_max: float = 0.25
    #: 指头 y/z 速度阻尼（把运动约束在 x 上，且允许接触把它顶回来）
    finger_lateral_damping: float = 8.0

    # --- 手掌驱动（根速度伺服）---
    palm_mass: float = 2.0
    palm_v_max: float = 1.2             # m/s
    palm_damping: float = 4.0

    # --- 世界 / 物理 ---
    dt: float = DEFAULT_DT
    gravity: float = -9.81
    ground_z: float = -0.05
    max_cmd_step_m: float = 0.012

    # --- 表面摩擦 ---
    obj_static_friction: float = 0.9
    obj_dynamic_friction: float = 0.8
    finger_static_friction: float = 0.9
    finger_dynamic_friction: float = 0.8
    table_static_friction: float = 0.7
    table_dynamic_friction: float = 0.6
    slip_static_friction: float = 0.03
    slip_dynamic_friction: float = 0.02

    def table_top_z(self) -> float:
        return float(self.table_pos[2] + self.table_size[2] / 2.0)

    def object_rest_z(self) -> float:
        return float(self.table_top_z() + self.cube_size / 2.0)

    def goal_object_z(self) -> float:
        return float(self.table_top_z() + self.plate_size[2] + self.cube_size / 2.0)

    def finger_gap(self, phase_grip: float) -> float:
        """把 "grip 参数"（0=全闭 1=全开）换成两指中心间距。"""
        half = (self.finger_x_close
                + (self.finger_x_max - self.finger_x_close) * float(phase_grip))
        return 2.0 * half

    def palm_offset_z(self) -> float:
        """手掌刚体中心相对"两指框住物体中心"这个参考点的 z 偏移。

        约定：**任务轨迹的 z 指的是两指框住所抓物体的那个高度**（手指中平面），
        手掌本体在这个参考点上方 `palm_center_z`。
        """
        return float(self.palm_center_z)


@dataclass
class Phase:
    name: str
    duration_s: float
    #: 夹爪开合：1.0=张开，0.0=闭合
    grip: float


def build_phases(cfg: TaskConfig) -> List[Phase]:
    """pick 序列：approach → descend → close → lift → transport → lower → release → retreat。"""
    return [
        Phase("approach", 0.30, 1.0),
        Phase("descend", 0.55, 1.0),
        Phase("close", 0.45, 0.0),
        Phase("settle_grasp", 0.25, 0.0),
        Phase("lift", 0.55, 0.0),
        Phase("transport", 0.90, 0.0),
        Phase("lower", 0.55, 0.0),
        Phase("release", 0.45, 1.0),
        Phase("retreat", 0.40, 1.0),
        Phase("settle", 0.50, 1.0),
    ]


def phase_waypoints(cfg: TaskConfig, place_offset_xy: Tuple[float, float] = (0.0, 0.0),
                    object_xy: Optional[Tuple[float, float]] = None
                    ) -> Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """每个相位的 (起点, 终点)（夹爪 base 的世界位置）。`object_xy` 同时承载 X1 的抓偏。"""
    if object_xy is None:
        object_xy = cfg.object_init_xy
    ox, oy = object_xy
    gx = cfg.goal_xy[0] + place_offset_xy[0]
    gy = cfg.goal_xy[1] + place_offset_xy[1]
    # 轨迹 z 用"手指中平面"高度，手掌本体再往上 palm_offset_z
    dz = cfg.palm_offset_z()
    z_rest = cfg.object_rest_z() + dz
    z_high = cfg.object_rest_z() + 0.16 + dz
    z_goal = cfg.goal_object_z() + dz
    return {
        "approach": ((ox, oy, z_high), (ox, oy, z_high)),
        "descend": ((ox, oy, z_high), (ox, oy, z_rest)),
        "close": ((ox, oy, z_rest), (ox, oy, z_rest)),
        "settle_grasp": ((ox, oy, z_rest), (ox, oy, z_rest)),
        "lift": ((ox, oy, z_rest), (ox, oy, z_high)),
        "transport": ((ox, oy, z_high), (gx, gy, z_high)),
        "lower": ((gx, gy, z_high), (gx, gy, z_goal)),
        "release": ((gx, gy, z_goal), (gx, gy, z_goal)),
        "retreat": ((gx, gy, z_goal), (gx, gy, z_high)),
        "settle": ((gx, gy, z_high), (gx, gy, z_high)),
    }


# --------------------------------------------------------------------------- #
# 失败注入
# --------------------------------------------------------------------------- #


@dataclass
class Injection:
    """一次尝试里的失败注入状态。"""

    mode: str = "success"
    lateral_offset_xy: Tuple[float, float] = (0.0, 0.0)
    object_friction_override: Optional[Tuple[float, float]] = None
    open_early_at_frac: Optional[float] = None
    place_offset_xy: Tuple[float, float] = (0.0, 0.0)
    applied_at_step: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "lateral_offset_xy": [round(float(v), 6) for v in self.lateral_offset_xy],
            "object_friction_override": (
                None if self.object_friction_override is None
                else [round(float(v), 6) for v in self.object_friction_override]),
            "open_early_at_frac": self.open_early_at_frac,
            "place_offset_xy": [round(float(v), 6) for v in self.place_offset_xy],
            "applied_at_step": self.applied_at_step,
        }


def make_injection(mode: str, cfg: TaskConfig, rng: random.Random,
                   x2_variant: str = "friction") -> Injection:
    """按模式构造注入参数。带 RNG 抖动，使同模式不同 seed 的行为不逐位相同。"""
    inj = Injection(mode=mode)
    if mode == "X1":
        # 抓偏：夹爪中心相对物体横向偏 ~0.045 m。
        # 物体半宽 0.03 + 指厚 0.02/2 = 0.04 → 偏 0.045 之后，一侧指头只会蹭到
        # 物体侧面之外，两指之间夹的是空气 → 指头能一路走到闭合目标、接触力≈0。
        magnitude = 0.045 + rng.uniform(-0.002, 0.008)
        angle = rng.choice((0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0))
        inj.lateral_offset_xy = (magnitude * math.cos(angle), magnitude * math.sin(angle))
    elif mode == "X2":
        if x2_variant == "open_early":
            inj.open_early_at_frac = 0.30
        else:
            inj.object_friction_override = (cfg.slip_static_friction,
                                           cfg.slip_dynamic_friction)
    elif mode == "X3":
        magnitude = 0.12
        angle = rng.choice((0.0, math.pi, math.pi / 2.0, 3.0 * math.pi / 2.0))
        inj.place_offset_xy = (magnitude * math.cos(angle), magnitude * math.sin(angle))
    return inj


def recovery_for(verdict_type: Optional[str]) -> Tuple[str, Injection, str]:
    """按失败类型返回 (恢复动作, "修好后"的注入, reason_code)。"""
    if verdict_type == "X1":
        return "retry_grasp", Injection(mode="success"), "regrasp_at_object_center"
    if verdict_type == "X2":
        return "change_grasp_point", Injection(mode="success"), "regrasp_with_clean_friction"
    if verdict_type == "X3":
        return "retry_grasp", Injection(mode="success"), "realign_place_target"
    return "none", Injection(mode="success"), "none"


# --------------------------------------------------------------------------- #
# USD 场景搭建
# --------------------------------------------------------------------------- #


class SceneBuilder:
    """纯 USD 搭场景（不用 isaacsim.core.api.objects，避免 6.0 的版本差异）。"""

    def __init__(self, cfg: TaskConfig) -> None:
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
        self.Gf = Gf
        self.Usd = Usd
        self.UsdGeom = UsdGeom
        self.UsdPhysics = UsdPhysics
        self.UsdShade = UsdShade
        self.PhysxSchema = PhysxSchema
        self.cfg = cfg
        self.stage = None
        self.materials: Dict[str, Any] = {}
        self.contact_report_prims: List[str] = []

    # -- 材质 ------------------------------------------------------------- #

    def make_material(self, name: str, static_friction: float, dynamic_friction: float,
                      restitution: float = 0.0) -> Any:
        self.UsdGeom.Scope.Define(self.stage, MATERIALS_PATH)
        path = "%s/%s" % (MATERIALS_PATH, name)
        material = self.UsdShade.Material.Define(self.stage, path)
        api = self.UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr().Set(float(static_friction))
        api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
        api.CreateRestitutionAttr().Set(float(restitution))
        self.materials[name] = material
        return material

    def bind_material(self, prim_path: str, material: Any) -> None:
        prim = self.stage.GetPrimAtPath(prim_path)
        self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=self.UsdShade.Tokens.strongerThanDescendants)

    # -- 几何 ------------------------------------------------------------- #

    def _define_box(self, path: str, size: Sequence[float],
                    position: Sequence[float]) -> Any:
        cube = self.UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        xform = self.UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(self.Gf.Vec3d(*[float(v) for v in position]))
        xform.AddScaleOp().Set(self.Gf.Vec3f(*[float(v) for v in size]))
        return cube

    def make_static_box(self, path: str, size: Sequence[float],
                        position: Sequence[float], material: Any = None) -> Any:
        cube = self._define_box(path, size, position)
        self.UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if material is not None:
            self.bind_material(path, material)
        return cube.GetPrim()

    def enable_contact_report(self, prim_path: str, pairs: Optional[List[str]] = None) -> None:
        """刚体上开接触报告（PhysX 要求 PhysxContactReportAPI 在**刚体** prim 上）。"""
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError("enable_contact_report: %s 不存在" % prim_path)
        api = self.PhysxSchema.PhysxContactReportAPI.Apply(prim)
        if pairs:
            api.CreateReportPairsRel().SetTargets(list(pairs))
        self.contact_report_prims.append(prim_path)

    def make_rigid_box(self, path: str, size: Sequence[float], position: Sequence[float],
                       mass: float, material: Any = None, disable_gravity: bool = False,
                       contact_report: bool = False) -> Any:
        cube = self._define_box(path, size, position)
        prim = cube.GetPrim()
        self.UsdPhysics.CollisionAPI.Apply(prim)
        self.UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr().Set(True)
        physx = self.PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if disable_gravity:
            physx.CreateDisableGravityAttr().Set(True)
        # 注意：**不要**设 sleepThreshold=0.0。实测那样做会让刚体立刻进入 sleeping
        # 状态（palm 的位姿再也不更新，停在 USD 局部坐标 0.075）。改为每步显式 wake_up。
        self.UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(mass))
        if contact_report:
            self.enable_contact_report(path)
        if material is not None:
            self.bind_material(path, material)
        return prim

    # -- 夹爪 ------------------------------------------------------------- #

    def build_gripper(self, cfg: TaskConfig) -> Tuple[Any, List[Any]]:
        """base(手掌) + 2 个指头，都是**独立动态刚体**，不用铰接体。

        重要（踩过的坑）：PhysX 只认**刚体 prim 自己**的 xform。如果刚体 API 打在
        一个父 Xform 上、几何放在子 prim 上，该刚体的 transform 在 PhysX 里会一直是
        父 Xform 的局部值（实测停在 USD 局部坐标 0.075，永不随仿真更新）。
        因此这里用 `make_rigid_box`：**Cube prim 本体**就是刚体 + 碰撞体 + 装配位置。
        """
        self.UsdGeom.Xform.Define(self.stage, GRIPPER_ROOT)
        palm = self.make_rigid_box(
            PALM_PATH, cfg.palm_size, (0.0, 0.0, cfg.palm_center_z),
            mass=cfg.palm_mass, material=self.materials["finger"], disable_gravity=True)
        fingers: List[Any] = []
        for index, link_path in enumerate(FINGER_LINK):
            sign = -1.0 if index == 0 else 1.0
            prim = self.make_rigid_box(
                link_path, cfg.finger_size,
                (sign * cfg.finger_x_max, 0.0, cfg.palm_center_z),
                mass=cfg.finger_mass, material=self.materials["finger"],
                disable_gravity=True, contact_report=True)
            fingers.append(prim)
        return palm, fingers


# --------------------------------------------------------------------------- #
# 夹爪驱动 / 测量
# --------------------------------------------------------------------------- #


class GripperRig:
    """用手算弹簧-阻尼力驱动三个刚体，并用 tensor API 测量接触力/位姿。

    * 手掌：速度伺服（v = clamp((target - x)/dt)），实现"移动基座"。
    * 指头：F = k·Δx − c·v（作用在质心，无力矩），另加横向（y/z）阻尼把运动
      约束在 x 上；接触时物体会把指头顶回来，因此指头位置是**真实物理结果**。
    """

    def __init__(self, cfg: TaskConfig, object_path: str, dt: float) -> None:
        import numpy as np
        self.np = np
        self.cfg = cfg
        self.dt = dt
        self.object_path = object_path
        self.body_view = None
        self.contact = None
        self.object_contact = None
        self.paths: List[str] = [object_path, object_path, object_path]
        self.sensor_paths: List[str] = []
        self.filter_paths: List[str] = []

    def build(self, sim_view: Any) -> None:
        np = self.np
        self.paths = [self.object_path, PALM_PATH] + list(FINGER_LINK)
        self.body_view = sim_view.create_rigid_body_view(self.paths)
        # sensor 用单个字符串 pattern；filter 用 list（见 report 的 API 说明）
        self.contact = sim_view.create_rigid_contact_view(
            GRIPPER_ROOT + "/finger_*", [self.object_path])
        if self.contact is None:
            raise RuntimeError("create_rigid_contact_view 返回 None")
        self.sensor_paths = [str(p) for p in self.contact.sensor_paths]
        self.filter_paths = [str(p) for p in self.contact.filter_paths]
        if len(self.sensor_paths) != len(FINGER_LINK):
            raise RuntimeError("contact view sensor_count=%d 期望 %d"
                               % (len(self.sensor_paths), len(FINGER_LINK)))
        try:
            self.object_contact = sim_view.create_rigid_contact_view(self.object_path)
        except Exception as exc:
            print("[rig] WARN object net-contact view failed: %s" % (exc,), flush=True)
        print("[rig] bodies=%s" % (self.paths,), flush=True)
        print("[rig] contact sensors=%s filters=%s" % (self.sensor_paths,
                                                       self.filter_paths), flush=True)

    # -- 控制 ------------------------------------------------------------- #

    def _transforms(self) -> Any:
        return self.np.asarray(self.body_view.get_transforms(), dtype=float)

    def _velocities(self) -> Any:
        return self.np.asarray(self.body_view.get_velocities(), dtype=float)

    def palm_position(self) -> Any:
        row = self._transforms()[1]
        return self.np.array([row[0], row[1], row[2]], dtype=float)

    def object_position(self) -> Any:
        row = self._transforms()[0]
        return self.np.array([row[0], row[1], row[2]], dtype=float)

    def object_velocity(self) -> Any:
        row = self._velocities()[0]
        return self.np.array([row[0], row[1], row[2]], dtype=float)

    def object_quaternion(self) -> Any:
        row = self._transforms()[0]
        return self.np.array([row[6], row[3], row[4], row[5]], dtype=float)

    def finger_positions(self) -> Any:
        data = self._transforms()
        return self.np.array([[data[2][0], data[2][1], data[2][2]],
                              [data[3][0], data[3][1], data[3][2]]], dtype=float)

    def finger_x(self) -> List[float]:
        """两指的实际 x 位置（真实物理结果，供判定器判断"是否被物体挡住"）。"""
        data = self._transforms()
        return [float(data[2][0]), float(data[3][0])]

    def gripper_width(self) -> float:
        """两指内侧间距 = 指心间距 − 指厚。"""
        pos = self.finger_positions()
        center_gap = float(abs(pos[1][0] - pos[0][0]))
        return center_gap - float(self.cfg.finger_size[0])

    def command_palm_step(self, target: Sequence[float], max_step_m: float) -> float:
        """手掌速度伺服：v = (target−cur)/dt，限速后写回（等效运动学基座）。

        实测结论（Isaac 6.0.1 + PhysX 110.1，tensor API）：
        * `RigidBodyView.set_velocities`：只在只写 1 个 body 时可靠；
          与别的 body 混写、或先调用 `wake_up()` 时，位置会卡住不动。
        * `RigidBodyView.set_transforms`：即使按 API 文档传满 (N,7) 数组，
          也会把**其他 body** 的变换写坏（实测指头 x 变成 4e13 量级）。
        因此这里只用 set_velocities，且**每步单独一次调用、不做 wake_up**。
        """
        np = self.np
        pos = self.palm_position()
        err = np.array([float(target[i]) - float(pos[i]) for i in range(3)], dtype=float)
        norm = float(np.linalg.norm(err))
        # 大误差 → 全速；小误差 → (err/dt)，避免在目标附近抖动
        if norm > self.cfg.palm_v_max * self.dt:
            vel = err * (self.cfg.palm_v_max / norm)
        else:
            vel = err / self.dt
        # 必须传满 (N,6)：只传 (1,6) 会报 "Incompatible size of velocity tensor"
        data = np.zeros((len(self.paths), 6), dtype=np.float32)
        data[1] = np.array([float(vel[0]), float(vel[1]), float(vel[2]),
                            0.0, 0.0, 0.0], dtype=np.float32)
        self.body_view.set_velocities(data, np.array([1], dtype=np.int32))
        return norm
    def command_fingers_step(self, target_gap: float) -> Any:
        """两指弹簧-阻尼伺服。target_gap = 指心间距（不含指厚）。

        返回 (施加的力数组, 位置误差数组)。
        """
        np = self.np
        cfg = self.cfg
        transforms = self._transforms()
        velocities = self._velocities()
        forces = np.zeros((len(self.paths), 3), dtype=np.float32)
        errors = []
        for i, index in enumerate((2, 3)):
            sign = -1.0 if i == 0 else 1.0
            target_x = sign * float(target_gap) / 2.0
            x = float(transforms[index][0])
            vx = float(velocities[index][0])
            error = target_x - x
            errors.append(error)
            force_x = cfg.finger_stiffness * error - cfg.finger_damping * vx
            force_x = max(-cfg.finger_force_limit, min(cfg.finger_force_limit, force_x))
            force_y = -cfg.finger_lateral_damping * float(velocities[index][1])
            force_z = -cfg.finger_lateral_damping * float(velocities[index][2])
            forces[index] = (force_x, force_y, force_z)
        # 注意：tensor API 要求传入的数组长度 == view 的 body 数（或与 indices 等长）
        self.body_view.apply_forces(forces, np.array([2, 3], dtype=np.int32), False)
        return forces, np.array(errors, dtype=float)

    # -- 测量 ------------------------------------------------------------- #

    def read_contact(self) -> Tuple[float, int, List[float], List[float]]:
        """返回 (指头↔物体合接触力之和 N, 有效接触对数, 每指合力, 每指净力)。"""
        np = self.np
        matrix = np.asarray(self.contact.get_contact_force_matrix(self.dt)).reshape(
            self.contact.sensor_count, self.contact.filter_count, 3)
        mags = np.linalg.norm(matrix, axis=2)
        per_sensor = mags.sum(axis=1)
        pairs = int(np.count_nonzero(mags > 1e-5))
        total = float(mags.sum())
        net = np.asarray(self.contact.get_net_contact_forces(self.dt)).reshape(
            self.contact.sensor_count, 3)
        net_mag = np.linalg.norm(net, axis=1)
        return total, pairs, [float(v) for v in per_sensor], [float(v) for v in net_mag]

    def read_object_net_contact(self) -> float:
        """物体受到的净接触力分量模长之和（无 filter 的兜底测量，用于交叉校验）。"""
        np = self.np
        if self.object_contact is None:
            return 0.0
        net = np.asarray(self.object_contact.get_net_contact_forces(self.dt)).reshape(
            self.object_contact.sensor_count, 3)
        if net.shape[0] == 0:
            return 0.0
        return float(np.abs(net[0]).sum())

    def wake(self) -> None:
        try:
            self.body_view.wake_up()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 统一失败事件（字段严格照 route1 契约）
# --------------------------------------------------------------------------- #


def build_failure_event(*, trial_id: str, step: int, unix_t: float, type_: Optional[str],
                        gripper_width_m: float, contact_force_n: float, object_dz_m: float,
                        dist_to_goal_m: Optional[float], collision_delta: int,
                        verdict: Optional[FailureVerdict], recovery_action: str,
                        reason_code: str, budget_used: int, budget_left: int,
                        recovered: Optional[bool], post_score: Optional[float],
                        steps_to_recover: Optional[int]) -> Dict[str, Any]:
    if verdict is not None:
        attribution = {
            "kind": verdict.kind,
            "summary_l1": verdict.summary_l1,
            "evidence": list(verdict.evidence),
        }
    else:
        attribution = {
            "kind": "nominal",
            "summary_l1": "本步四子目标无失败判定，任务按计划推进",
            "evidence": ["subgoal_states 见 trace；contact_force_n / object_dz_m / "
                         "dist_to_goal_m 均在标称范围"],
        }
    return {
        "id": "manip:%s:%d" % (trial_id, step),
        "t": float(unix_t),
        "source": "manip",
        "type": type_,
        "signals": {
            "feedback_score": None,
            "trajectory_drift": None,
            "plan_age": None,
            "collision_delta": int(collision_delta),
            "steps_since_last_s2": None,
            "semantic_instance_margin": None,
            "gripper_width_m": float(gripper_width_m),
            "contact_force_n": float(contact_force_n),
            "object_dz_m": float(object_dz_m),
            "dist_to_goal_m": None if dist_to_goal_m is None else float(dist_to_goal_m),
        },
        "attribution": attribution,
        "recovery": {
            "reason_code": reason_code,
            "action": recovery_action,
            "budget_used": int(budget_used),
            "budget_left": int(budget_left),
            "bypassed_cooldown": False,
        },
        "outcome": {
            "recovered": recovered,
            "post_score": post_score,
            "steps_to_recover": steps_to_recover,
        },
    }


class EventWriter:
    """append-only JSONL。"""

    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self.count = 0
        self._handle = None
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._handle = open(path, "a", encoding="utf-8")

    def write(self, event: Dict[str, Any]) -> None:
        self.count += 1
        if self._handle is not None:
            self._handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


# --------------------------------------------------------------------------- #
# 任务编排
# --------------------------------------------------------------------------- #


@dataclass
class AttemptResult:
    trial_id: str
    mode: str
    attempt_index: int
    steps_executed: int
    wall_seconds: float
    subgoal_summary: Dict[str, Any] = field(default_factory=dict)
    verdict: Dict[str, Any] = field(default_factory=dict)
    trace_path: Optional[str] = None
    final_success: bool = False
    trace: List[Dict[str, Any]] = field(default_factory=list)


class PickTask:
    """场景 + 控制循环 + 判定 + 事件输出的编排器。"""

    def __init__(self, cfg: TaskConfig, mode: str, seed: int, out: Optional[str],
                 max_steps: int, record_trace: bool, out_dir: str,
                 x2_variant: str, do_recover: bool) -> None:
        from isaacsim.core.api import World
        self.cfg = cfg
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.record_trace = record_trace
        self.out_dir = out_dir
        self.x2_variant = x2_variant
        self.do_recover = do_recover
        self.writer = EventWriter(out)
        os.makedirs(out_dir, exist_ok=True)
        self.world = World(stage_units_in_meters=1.0, physics_dt=cfg.dt,
                           rendering_dt=cfg.dt, physics_prim_path=PHYSICS_SCENE_PATH)
        self.injection = make_injection(mode, cfg, self.rng, x2_variant)
        self.builder: Optional[SceneBuilder] = None
        self.rig: Optional[GripperRig] = None
        self.object_rest_z = cfg.object_rest_z()

    # ---- 场景 ----------------------------------------------------------- #

    def build_scene(self) -> None:
        """搭舞台。

        注意（Isaac 6.0.1 的一个坑）：`omni.usd.get_context().new_stage()` 会派发
        StageEventType.OPENED 事件，而 `SimulationContext._stage_open_callback_fn`
        见到该事件时会 `del self._scene`（simulation_context.py:141-142）。而
        `World._world_initialized` 仍是 True → `World.__init__` 再也不会补建
        `_scene` → 之后 `world.reset()` 直接
        `AttributeError: 'World' object has no attribute '_scene'`。
        所以这里在 new_stage 之后**显式重建** World 的 scene。
        """
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.scenes.scene import Scene
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        cfg = self.cfg
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        builder = SceneBuilder(cfg)
        builder.stage = stage
        self.builder = builder
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        usd_scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
        usd_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        usd_scene.CreateGravityMagnitudeAttr().Set(abs(float(cfg.gravity)))
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(usd_scene.GetPrim())
        physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
        physx_scene.CreateTimeStepsPerSecondAttr().Set(int(round(1.0 / cfg.dt)))
        physx_scene.CreateEnableStabilizationAttr().Set(True)

        builder.make_material("object", cfg.obj_static_friction, cfg.obj_dynamic_friction)
        builder.make_material("finger", cfg.finger_static_friction, cfg.finger_dynamic_friction)
        builder.make_material("table", cfg.table_static_friction, cfg.table_dynamic_friction)
        builder.make_material("slip", cfg.slip_static_friction, cfg.slip_dynamic_friction)

        builder.make_static_box(GROUND_PATH, (4.0, 4.0, 0.02), (0.0, 0.0, cfg.ground_z),
                                builder.materials["table"])
        builder.make_static_box(TABLE_PATH, cfg.table_size, cfg.table_pos,
                                builder.materials["table"])
        plate_z = cfg.table_top_z() + cfg.plate_size[2] / 2.0
        builder.make_static_box(PLATE_PATH, cfg.plate_size,
                                (cfg.goal_xy[0], cfg.goal_xy[1], plate_z),
                                builder.materials["table"])
        self.object_rest_z = cfg.object_rest_z()
        builder.make_rigid_box(OBJECT_PATH, (cfg.cube_size,) * 3,
                               (cfg.object_init_xy[0], cfg.object_init_xy[1],
                                self.object_rest_z),
                               mass=cfg.cube_mass, material=builder.materials["object"],
                               contact_report=True)
        builder.build_gripper(cfg)
        # 登记接触报告对：指头 ↔ 物体
        builder.enable_contact_report(OBJECT_PATH, pairs=list(FINGER_LINK))
        for finger in FINGER_LINK:
            builder.enable_contact_report(finger, pairs=[OBJECT_PATH])
        print("[scene] contact report pairs: %s <-> %s" % (list(FINGER_LINK), OBJECT_PATH),
              flush=True)

        if not hasattr(self.world, "_scene"):
            self.world._scene = Scene()
            World._world_initialized = True
            print("[scene] rebuilt World._scene after new_stage() "
                  "(Isaac 6.0.1 stage-open callback deletes it)", flush=True)
        print("[scene] world attrs ok: _scene=%s" % (hasattr(self.world, "_scene"),),
              flush=True)

    # ---- 单次尝试 ------------------------------------------------------- #

    def run_attempt(self, attempt_index: int, injection: Injection,
                    trial_id: str) -> AttemptResult:
        cfg = self.cfg
        world = self.world
        world.reset()
        sim_view = world.physics_sim_view
        if sim_view is None:
            raise RuntimeError("world.physics_sim_view 为 None：物理未初始化")
        rig = GripperRig(cfg, OBJECT_PATH, cfg.dt)
        rig.build(sim_view)
        self.rig = rig

        obj_xy = (cfg.object_init_xy[0] + injection.lateral_offset_xy[0],
                  cfg.object_init_xy[1] + injection.lateral_offset_xy[1])
        wps = phase_waypoints(cfg, injection.place_offset_xy, obj_xy)
        home = list(wps["approach"][0])

        # --- 初始化：把指头/手掌摆到 home，物体落稳 ---
        for _ in range(80):
            rig.command_palm_step(home, 1e9)
            rig.command_fingers_step(cfg.finger_gap(1.0))
            world.step(render=False)
        rig.wake()
        obj_pos = rig.object_position()
        init_pos = (float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]))
        goal_xy = (cfg.goal_xy[0] + injection.place_offset_xy[0],
                   cfg.goal_xy[1] + injection.place_offset_xy[1])
        judge = SubgoalJudge(SubgoalThresholds(), goal_xy=goal_xy, object_init_pos=init_pos)
        print("[task] object settled at (%.4f, %.4f, %.4f); goal_xy=(%.4f, %.4f)"
              % (init_pos[0], init_pos[1], init_pos[2], goal_xy[0], goal_xy[1]), flush=True)

        phases = build_phases(cfg)
        timeline: List[Tuple[Phase, int, int]] = []
        cursor = 0
        for phase in phases:
            n = max(1, int(round(phase.duration_s / cfg.dt)))
            timeline.append((phase, cursor, cursor + n - 1))
            cursor += n
        total_planned = cursor

        friction_swapped = False
        opened_early = False
        verdict: FailureVerdict = FailureVerdict()
        grasp_seen = False
        collision_delta = 0
        last_phase = ""
        step_limit = min(self.max_steps, total_planned)
        debug_poses = bool(os.environ.get("MANIP_DEBUG_POSES"))
        t0 = time.time()
        steps = 0

        for step in range(step_limit):
            phase = timeline[-1][0]
            frac = 1.0
            for cand, start, end in timeline:
                if start <= step <= end:
                    phase = cand
                    span = max(1, end - start)
                    frac = float(step - start) / float(span)
                    break
            start_p, end_p = wps[phase.name]
            target = [start_p[i] + (end_p[i] - start_p[i]) * frac for i in range(3)]

            grip = phase.grip
            if phase.name == "lift":
                if (not friction_swapped) and injection.object_friction_override is not None:
                    self._set_object_friction()
                    friction_swapped = True
                    injection.applied_at_step = step
                    print("[inj] X2 friction drop applied at step %d" % step, flush=True)
                if (injection.open_early_at_frac is not None and not opened_early
                        and frac >= injection.open_early_at_frac):
                    grip = 1.0
                    opened_early = True
                    injection.applied_at_step = step
                    print("[inj] X2 early open applied at step %d" % step, flush=True)

            err = rig.command_palm_step(target, cfg.max_cmd_step_m)
            forces, finger_err = rig.command_fingers_step(cfg.finger_gap(grip))
            world.step(render=False)
            steps = step + 1

            width = rig.gripper_width()
            force, pairs, per_finger, net_per_finger = rig.read_contact()
            obj_net_force = rig.read_object_net_contact()
            obj_pos = rig.object_position()
            obj_vel = rig.object_velocity()
            palm_pos = rig.palm_position()
            finger_x = rig.finger_x()
            if obj_net_force > 1.0:
                collision_delta += 1

            judge.update(StepObs(
                step=step, sim_time=float(step) * cfg.dt, gripper_width_m=width,
                contact_force_n=force, contact_pairs=pairs,
                object_pos=(float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])),
                object_vel=(float(obj_vel[0]), float(obj_vel[1]), float(obj_vel[2])),
                gripper_pos=(float(palm_pos[0]), float(palm_pos[1]), float(palm_pos[2])),
                joint_positions=finger_x,
                joint_targets=[-(cfg.finger_gap(grip) / 2.0), cfg.finger_gap(grip) / 2.0],
                phase=phase.name))
            if judge.state.grasp:
                grasp_seen = True
            last_phase = phase.name
            entry = judge.trace[-1]
            dist_goal = entry["dist_to_goal_m"]
            dz = float(entry["object_dz_m"])

            self.writer.write(build_failure_event(
                trial_id=trial_id, step=step, unix_t=time.time(), type_=None,
                gripper_width_m=width, contact_force_n=force, object_dz_m=dz,
                dist_to_goal_m=dist_goal, collision_delta=collision_delta,
                verdict=None, recovery_action="none", reason_code="nominal",
                budget_used=0, budget_left=1, recovered=None, post_score=None,
                steps_to_recover=None))

            if debug_poses and (step % 20 == 0 or (force > 0.5 and step % 5 == 0)):
                print("[obs] step=%4d %-12s width=%.5f force=%8.4f pairs=%d objnet=%7.3f "
                      "f1x=%.5f f2x=%.5f objz=%.5f cz=%.5f dz=%+.5f"
                      % (step, phase.name, width, force, pairs, obj_net_force,
                         float(rig.finger_positions()[0][0]),
                         float(rig.finger_positions()[1][0]),
                         float(obj_pos[2]), float(palm_pos[2]), dz), flush=True)
            elif step % 60 == 0:
                print("[obs] step=%4d %-12s width=%.5f force=%8.4f pairs=%d objz=%.5f dz=%+.5f"
                      % (step, phase.name, width, force, pairs,
                         float(obj_pos[2]), dz), flush=True)

            # --- 在线失败判定（按相位优先级）---
            verdict = FailureVerdict()
            if phase.name in ("lift", "transport", "lower"):
                verdict = detect_slip_failure(judge, grasp_seen=grasp_seen)
            if verdict.type is None and phase.name in ("release", "retreat", "settle"):
                verdict = detect_misplace_failure(judge, dist_goal)
            if verdict.type is None and phase.name in ("settle_grasp", "lift"):
                verdict = detect_grasp_failure(judge)
            if verdict.type is None and phase.name in ("retreat", "settle"):
                verdict = detect_grasp_failure(judge)
            if verdict.type is not None:
                self.writer.write(build_failure_event(
                    trial_id=trial_id, step=step, unix_t=time.time(), type_=verdict.type,
                    gripper_width_m=width, contact_force_n=force, object_dz_m=dz,
                    dist_to_goal_m=dist_goal, collision_delta=collision_delta,
                    verdict=verdict, recovery_action=verdict.recovery_action,
                    reason_code=verdict.reason_code, budget_used=0, budget_left=1,
                    recovered=None, post_score=None, steps_to_recover=None))
                print("[judge] %s at step %d phase=%s width=%.5f force=%.4f dz=%.5f "
                      "dist=%s" % (verdict.type, step, phase.name, width, force, dz,
                                   dist_goal), flush=True)
                break

        wall = time.time() - t0
        final_pos = rig.object_position()
        final_dist = math.sqrt((float(final_pos[0]) - goal_xy[0]) ** 2
                               + (float(final_pos[1]) - goal_xy[1]) ** 2)
        final_success = bool(judge.state.in_place)

        trace_path = None
        if self.record_trace:
            safe = trial_id.replace(":", "_").replace("/", "_")
            trace_path = os.path.join(self.out_dir,
                                      "trace_%s_attempt%d.json" % (safe, attempt_index))
            judge.write_trace_jsonl(trace_path)

        summary = judge.summary()
        summary["final_dist_to_goal_m"] = round(final_dist, 6)
        summary["injection"] = injection.as_dict()
        summary["last_phase"] = last_phase
        summary["phases_planned"] = [
            {"name": pp.name, "steps": (end - start + 1), "grip": pp.grip}
            for pp, start, end in timeline]
        summary["planned_steps"] = total_planned
        summary["final_object_pos"] = [round(float(v), 6) for v in final_pos]
        return AttemptResult(
            trial_id=trial_id, mode=injection.mode, attempt_index=attempt_index,
            steps_executed=steps, wall_seconds=wall, subgoal_summary=summary,
            verdict=verdict.as_dict(), trace_path=trace_path,
            final_success=final_success, trace=judge.trace)

    def _set_object_friction(self) -> None:
        """X2：把物体换成低摩擦材质（同时改属性，确保 PhysX 立刻生效）。"""
        from pxr import UsdPhysics
        material = self.builder.materials["slip"]
        self.builder.bind_material(OBJECT_PATH, material)
        api = UsdPhysics.MaterialAPI(material.GetPrim())
        api.GetStaticFrictionAttr().Set(float(self.cfg.slip_static_friction))
        api.GetDynamicFrictionAttr().Set(float(self.cfg.slip_dynamic_friction))
        print("[inj] object material -> slip (static=%.2f dynamic=%.2f)"
              % (self.cfg.slip_static_friction, self.cfg.slip_dynamic_friction), flush=True)

    # ---- trial（含恢复）------------------------------------------------- #

    def run_trial(self, trial_id: str) -> Dict[str, Any]:
        attempt = self.run_attempt(0, self.injection, trial_id)
        record: Dict[str, Any] = {
            "trial_id": trial_id,
            "mode": self.mode,
            "seed": self.seed,
            "attempts": [self._attempt_record(attempt)],
            "final_success": attempt.final_success,
            "verdict_type": attempt.verdict.get("type"),
            "recovered": None,
            "steps_to_recover": None,
        }
        if attempt.verdict.get("type") and self.do_recover:
            action, fixed, reason = recovery_for(attempt.verdict.get("type"))
            retry = self.run_attempt(1, fixed, "%s-r1" % trial_id)
            record["attempts"].append(self._attempt_record(retry))
            record["recovered"] = bool(retry.final_success)
            record["steps_to_recover"] = int(retry.steps_executed)
            record["recovery_action"] = action
            last = retry.trace[-1] if retry.trace else None
            self.writer.write(build_failure_event(
                trial_id=trial_id,
                step=attempt.steps_executed + retry.steps_executed,
                unix_t=time.time(), type_=attempt.verdict.get("type"),
                gripper_width_m=float(last["gripper_width_m"]) if last else 0.0,
                contact_force_n=float(last["contact_force_n"]) if last else 0.0,
                object_dz_m=float(last["object_dz_m"]) if last else 0.0,
                dist_to_goal_m=retry.subgoal_summary.get("final_dist_to_goal_m"),
                collision_delta=0, verdict=None, recovery_action=action,
                reason_code="recovery_outcome:" + reason, budget_used=1, budget_left=0,
                recovered=bool(retry.final_success),
                post_score=(1.0 if retry.final_success else 0.0),
                steps_to_recover=int(retry.steps_executed)))
            print("[recover] action=%s recovered=%s steps=%d"
                  % (action, retry.final_success, retry.steps_executed), flush=True)
        return record

    @staticmethod
    def _attempt_record(attempt: AttemptResult) -> Dict[str, Any]:
        return {
            "attempt_index": attempt.attempt_index,
            "steps_executed": attempt.steps_executed,
            "wall_seconds": round(attempt.wall_seconds, 3),
            "final_success": attempt.final_success,
            "verdict": attempt.verdict,
            "subgoal_summary": attempt.subgoal_summary,
            "trace_path": attempt.trace_path,
        }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isaac Sim 6.0.1 最小 pick 任务 + 子目标判定 + 失败注入")
    parser.add_argument("--mode", default="success",
                        choices=["success", "X1", "X2", "X3", "smoke"])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--out", default=None, help="失败事件 JSONL 输出路径")
    parser.add_argument("--out-dir", default=None, help="trace / summary 输出目录")
    parser.add_argument("--summary-json", default=None, help="本次运行汇总 JSON 路径")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=600, help="单次尝试步数上限")
    parser.add_argument("--record-trace", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None, help="固定 seed（默认按 trial 递推）")
    parser.add_argument("--trial-id", default=None, help="覆盖 trial_id 前缀")
    parser.add_argument("--x2-variant", default="friction", choices=["friction", "open_early"])
    parser.add_argument("--recover", dest="do_recover", action="store_true", default=True)
    parser.add_argument("--no-recover", dest="do_recover", action="store_false")
    parser.add_argument("--gpu-sample", action="store_true", default=False)
    return parser.parse_args(argv)


class GpuSampler:
    """后台采样 GPU 计算进程显存峰值（nvidia-smi 轮询）。"""

    def __init__(self, interval_s: float = 0.5) -> None:
        import threading
        self.interval = interval_s
        self.peak_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
                for line in out.stdout.decode("utf-8", "replace").strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        try:
                            self.peak_mib = max(self.peak_mib, int(float(parts[1])))
                            self.samples += 1
                        except ValueError:
                            pass
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.time()

    from isaacsim import SimulationApp
    app = SimulationApp({
        "headless": bool(args.headless),
        "renderer": "RayTracedLighting",
        "width": 320, "height": 240,
        "anti_aliasing": 0,
        "fast_shutdown": True,
        "hide_ui": bool(args.headless),
    })

    exit_code = 0
    summary: Dict[str, Any] = {"mode": args.mode, "trials": [], "argv": list(sys.argv)}
    sampler = GpuSampler() if args.gpu_sample else None
    try:
        if sampler:
            sampler.start()
        cfg = TaskConfig()
        out = args.out
        out_dir = args.out_dir or os.path.join(
            os.path.dirname(os.path.abspath(out)) if out else os.getcwd(), "isaac_trace")
        os.makedirs(out_dir, exist_ok=True)
        for trial_index in range(args.trials):
            seed = (args.seed + trial_index) if args.seed is not None else (1000 + trial_index)
            trial_id = args.trial_id or ("%s-s%d-t%d" % (args.mode, seed, trial_index))
            print("[trial %d/%d] id=%s seed=%d" % (trial_index + 1, args.trials,
                                                   trial_id, seed), flush=True)
            task = PickTask(cfg, args.mode, seed, out, args.steps, args.record_trace,
                            out_dir, args.x2_variant, args.do_recover)
            task.build_scene()
            record = task.run_trial(trial_id)
            task.writer.close()
            summary["trials"].append(record)
            first = record["attempts"][0]
            print("[trial %s] verdict=%s final_success=%s steps=%d wall=%.2fs"
                  % (trial_id, first["verdict"].get("type"), first["final_success"],
                     first["steps_executed"], first["wall_seconds"]), flush=True)
        summary["all_final_success"] = all(t["final_success"] for t in summary["trials"])
        summary["any_final_success"] = any(t["final_success"] for t in summary["trials"])
        summary["recovered_count"] = sum(1 for t in summary["trials"] if t.get("recovered"))
        summary["verdicts"] = [t["verdict_type"] for t in summary["trials"]]
    except Exception:
        import traceback
        traceback.print_exc()
        summary["error"] = traceback.format_exc()
        exit_code = 3
    finally:
        if sampler:
            sampler.stop()
            summary["gpu_peak_mib"] = sampler.peak_mib
            summary["gpu_samples"] = sampler.samples
        summary["wall_seconds_total"] = round(time.time() - started, 3)
        if args.summary_json:
            directory = os.path.dirname(os.path.abspath(args.summary_json))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(args.summary_json, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=1)
        printable = {k: v for k, v in summary.items() if k != "trials"}
        print("[summary] %s" % json.dumps(printable, ensure_ascii=False), flush=True)
        try:
            app.close()
        except Exception:
            pass
    os._exit(exit_code)


def _install_signal_handlers() -> None:
    """timeout -s TERM 时干净退出，避免残留 kit 进程。"""

    def _handler(signum, frame):  # pragma: no cover
        print("[signal] got %s, exiting" % (signum,), flush=True)
        os._exit(130)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


_install_signal_handlers()

if __name__ == "__main__":
    sys.exit(main())

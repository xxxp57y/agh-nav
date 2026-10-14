#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""navlive_probe.py —— 交互式导航服务的最小可行性验证（在导航服务器上跑）。

目的只有一个：**证明"选一个场景 → 建环境 → 渲染出画面 → 拿到 2D 楼层图 → 拿到位姿"这条链
在服务器上真的能跑**，并且把每一步的耗时打出来。它不是最终服务，只是第一次跑通用的钉子。

用法（在 /srv/agh-demo/InternNav 下）::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/srv/agh-demo/src \\
      /srv/agh-demo/habitat-env/bin/python -u \\
      navlive_probe.py --scene 2azQ1b91cZZ --episode-index 0 --steps 40

产物落在 --out（默认 /tmp/navlive-probe）：rgb_NNN.jpg / map_NNN.png / trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

REPO = os.environ.get("ROUTE1_REPO", "/srv/agh-demo/InternNav")
CONFIG = os.environ.get(
    "NAVLIVE_CONFIG",
    os.path.join(REPO, "scripts/eval/configs/route1_multiscene_cfg.py"),
)
SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")

# 仓库根先进 sys.path：internnav / scripts.eval.configs.* 都从这里 import
# （PYTHONPATH 里通常只有 route1 所在的 src，不含 InternNav 仓库本身）
for _p in (REPO,):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def log(msg: str) -> None:
    sys.stderr.write("[probe] %s\n" % msg)
    sys.stderr.flush()


def load_eval_cfg(config_path: str):
    """照抄 internnav/scripts/eval/eval.py 的加载方式（同一份 config 工厂）。"""
    import importlib.util

    sys.path.append(".")
    spec = importlib.util.spec_from_file_location("navlive_eval_config", config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["navlive_eval_config"] = module
    spec.loader.exec_module(module)
    return module.eval_cfg


def build_config():
    """复刻 InternVLAN1Evaluator.__init__ 里配置 habitat 的那几行（不改上游）。

    关键坑（实测踩到）：`habitat-lab` 的 `oracle_success` 测量项**不是内置注册的**，
    它是 `internnav.habitat_extensions.vln.habitat_vln_evaluator` 这个模块**导入时**才注册的
    （所以 eval.py 能跑、裸 habitat.Env 会报 `invalid oracle_success type OracleSuccess`）。
    因此这里必须先 import 那个模块 —— 与 eval.py 走的是同一条注册路径。
    """
    import habitat
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat_baselines.config.default import get_config as get_habitat_config

    # 只为触发注册（VLN 任务的 measurements 由该模块 import 时注册）
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401

    cfg = load_eval_cfg(CONFIG)
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
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    return cfg, config


def instruction_text(episode) -> Any:
    """episode.instruction 在这个 habitat 版本里是 InstructionData（不是 str）。"""
    value = getattr(episode, "instruction", None)
    if value is None:
        return None
    text = getattr(value, "instruction_text", None)
    if text is not None:
        return text
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return None


def _round(value, digits: int = 3):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def scene_key_of(raw: Any) -> str:
    """把 episode.scene_id 归一成 MP3D 场景名。

    实测：habitat 的 episode.scene_id 形如
    `data/scene_data/mp3d_ce/mp3d/2azQ1b91cZZ.glb`（带扩展名），而日志里的 scene_id 是
    `2azQ1b91cZZ`。工作台要按干净的名字选场景，所以统一在这里剥掉路径与扩展名。
    """
    text = str(raw).replace("\\", "/").rstrip("/")
    name = text.split("/")[-1]
    for suffix in (".glb", ".navmesh", ".basis.glb"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def agent_pose(env) -> Dict[str, Any]:
    state = env.sim.get_agent_state()
    return {
        "position": [_round(state.position[0]), _round(state.position[1]), _round(state.position[2])],
        "rotation": [
            _round(state.rotation.real, 4),
            _round(state.rotation.imag[0], 4),
            _round(state.rotation.imag[1], 4),
            _round(state.rotation.imag[2], 4),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None, help="Matterport3D scene id，如 2azQ1b91cZZ")
    parser.add_argument("--episode-index", type=int, default=0, help="该场景内的第几个 episode（按清单顺序）")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--stride", type=int, default=4, help="每 N 步存一次帧")
    parser.add_argument("--out", default="/tmp/navlive-probe")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    os.chdir(REPO)
    cfg, config = build_config()
    log("config built in %.1fs; split=%s" % (time.time() - t0, config.habitat.dataset.split))

    import habitat

    t1 = time.time()
    env = habitat.Env(config)
    log("habitat.Env built in %.1fs; episodes=%d" % (time.time() - t1, len(env.episodes)))

    # 按场景分组（与 HabitatEnv.generate_episodes 同序：场景 id 字典序，组内清单序）
    by_scene: Dict[str, List[Any]] = {}
    for episode in env.episodes:
        scene_key = scene_key_of(episode.scene_id)
        by_scene.setdefault(scene_key, []).append(episode)

    inventory = {
        "config": CONFIG,
        "split": SPLIT,
        "total_episodes": len(env.episodes),
        "scenes": [
            {
                "scene_id": scene,
                "episodes": len(items),
                "episode_ids": [str(e.episode_id) for e in items],
                "instructions": [instruction_text(e) for e in items[:3]],
            }
            for scene, items in sorted(by_scene.items())
        ],
    }

    scene_id = args.scene or sorted(by_scene)[0]
    if scene_id not in by_scene:
        log("场景 %r 不在清单里；可用：%s" % (scene_id, ", ".join(sorted(by_scene))))
        return 2
    episode = by_scene[scene_id][args.episode_index]

    inventory["selected"] = {
        "scene_id": scene_id,
        "episode_id": str(episode.episode_id),
        "instruction": instruction_text(episode),
        "start_position": [float(v) for v in episode.start_position],
        "goals": [list(map(float, g.position)) for g in getattr(episode, "goals", [])],
    }
    print(json.dumps({"inventory": inventory}, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    if args.list_only:
        env.close()
        return 0

    os.makedirs(args.out, exist_ok=True)
    import numpy as np
    from PIL import Image

    t2 = time.time()
    # Env.reset() 用的是 current_episode，所以先指向目标 episode 再 reset（只 reset 一次）
    env.current_episode = episode
    obs = env.reset()
    log("reset to %s ep=%s in %.2fs" % (scene_id, episode.episode_id, time.time() - t2))

    metrics = env.get_metrics()
    tdm = metrics.get("top_down_map")
    log("metrics keys: %s" % sorted(metrics.keys()))
    log("rgb shape %s dtype %s" % (np.asarray(obs["rgb"]).shape, np.asarray(obs["rgb"]).dtype))
    log("pose at start: %s" % json.dumps(agent_pose(env)))
    if tdm is not None:
        log("top_down_map keys: %s" % sorted(tdm.keys()))
        log("map shape %s" % (np.asarray(tdm["map"]).shape,))
        Image.fromarray(np.asarray(tdm["map"])).save(os.path.join(args.out, "map_000.png"))
    else:
        log("top_down_map 是 None —— 需要再确认测量项是否真的生效")

    Image.fromarray(np.asarray(obs["rgb"])).save(os.path.join(args.out, "rgb_000.jpg"), quality=90)

    trace: List[Dict[str, Any]] = []
    t3 = time.time()
    for step in range(args.steps):
        action = 1  # ↑ forward，最朴素的直行；这一步只看"能不能动"
        result = env.step(action)
        # 这个 habitat 版本的 step 返回结构随分支不同：可能是 dict，也可能是 tuple/list。
        # 这里两种都接住，不做严格解包。
        if isinstance(result, dict):
            obs = result.get("observations", obs)
            done = bool(result.get("episode_over", result.get("done", False)))
            info = result.get("info") or env.get_metrics()
            reward = result.get("reward", 0.0)
        elif isinstance(result, (tuple, list)):
            obs, reward, done, info = result[0], result[1], result[2], result[3]
        else:
            obs = obs
            done = bool(env.episode_over)
            info = env.get_metrics()
            reward = 0.0
        if step == 0:
            log(
                "step() returned %s; info keys=%s"
                % (type(result).__name__, sorted(info.keys()) if isinstance(info, dict) else type(info))
            )
        current = env.get_metrics()
        collisions = current.get("collisions")
        trace.append(
            {
                "step": step,
                "action": action,
                "pose": agent_pose(env),
                "distance_to_goal": _round(current.get("distance_to_goal")),
                "collisions": collisions.get("count") if isinstance(collisions, dict) else None,
                "done": bool(done),
            }
        )
        if step % args.stride == 0:
            Image.fromarray(np.asarray(obs["rgb"])).save(
                os.path.join(args.out, "rgb_%03d.jpg" % (step + 1)), quality=85
            )
            m = current.get("top_down_map")
            if m is not None:
                Image.fromarray(np.asarray(m["map"])).save(
                    os.path.join(args.out, "map_%03d.png" % (step + 1))
                )
        if done:
            log("episode done at step %d" % step)
            break
    elapsed = time.time() - t3
    log("stepped %d in %.2fs → %.1f steps/s" % (len(trace), elapsed, len(trace) / max(elapsed, 1e-6)))

    with open(os.path.join(args.out, "trace.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "scene_id": scene_id,
                "episode_id": str(episode.episode_id),
                "instruction": instruction_text(episode),
                "trace": trace,
                "out": args.out,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    log("artifacts in %s" % args.out)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

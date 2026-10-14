#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agh_nav_probe.py —— AGH 导航插件的数据面（只跑在导航服务器上）。

为什么要有这一层
----------------
AGH 插件（`nav-agent-plugin` / `nav-board-plugin`）是 JS；导航闭环、日志、
契约实现都在服务器上（Python 3.9 + Habitat）。与其在 JS 里重新实现一遍
"读日志 → 算指标 → 判失败 → 选恢复动作"（离线一个样、在线另一个样），
不如**把已有的那份实现直接当数据面**：

- `route1.contracts`（FailureEvent / Signals / Attribution / Recovery / 恢复动作菜单）
- `route1.nav.channels`（碰撞密度 + 进度停滞两条通道，**与在线跑的是同一份代码**）

本脚本只做三件事：读日志 / 调上面的模块 / 把结果打成 JSON。
所有子命令的输出都是**一个 JSON 对象**（stdout），诊断信息走 stderr。

用法（在服务器上）::

    PYTHONPATH=/srv/agh-demo/src \\
      /srv/agh-demo/habitat-env/bin/python \\
      agh_nav_probe.py status
    ... agh_nav_probe.py runs
    ... agh_nav_probe.py run-summary --run r1ro4c
    ... agh_nav_probe.py series --run r1ro4c --scene 8194nk5LbLH --episode 220
    ... agh_nav_probe.py failures --run r1ro4c --limit 20
    ... agh_nav_probe.py triggers --run r1r2adapt_capture
    ... agh_nav_probe.py recovery --run r1ro4c
    ... agh_nav_probe.py board
    ... agh_nav_probe.py progress --run <label>
    ... agh_nav_probe.py launch --label r1new4 --budget-mode recovery_only \\
             --watchdog off --channels both
    ... agh_nav_probe.py cache-all

环境变量（全部有默认值，指向本项目在服务器上的位置）::

    ROUTE1_REPO  默认 /srv/agh-demo/InternNav
    ROUTE1_SRC   默认 /srv/agh-demo/src
    ROUTE1_LOG   默认 /srv/agh-demo/logs/route1
    ROUTE1_OUT   默认 /srv/agh-demo/out/route1
    ROUTE1_PY    默认 /srv/agh-demo/habitat-env/bin/python
    ROUTE1_SEED  默认 20260826
    ROUTE1_GPU   默认 1（GPU0 留给别人）
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

REPO = os.environ.get("ROUTE1_REPO", "/srv/agh-demo/InternNav")
SRC = os.environ.get("ROUTE1_SRC", "/srv/agh-demo/src")
LOG = os.environ.get("ROUTE1_LOG", "/srv/agh-demo/logs/route1")
OUT = os.environ.get("ROUTE1_OUT", "/srv/agh-demo/out/route1")
PY = os.environ.get("ROUTE1_PY", "/srv/agh-demo/habitat-env/bin/python")
SEED = os.environ.get("ROUTE1_SEED", "20260826")
GPU = os.environ.get("ROUTE1_GPU", "1")

RUNS_GLOB = os.path.join(REPO, "logs", "habitat", "route1_*_seed*")
DEFAULT_CONFIG = "scripts/eval/configs/route1_multiscene_v4_cfg.py"

SCHEMA = "agh-nav-probe/1"

#: 面板与工具展示用的五个对照配置（pool8_*.json 的文件名后缀）
CONFIGS: Tuple[Tuple[str, str], ...] = (
    ("R2A", "pool8_r1multi4.json"),
    ("original", "pool8_r1orig4.json"),
    ("recovery_only", "pool8_r1ro4c.json"),
    ("guarded", "pool8_r1grd4.json"),
    ("collision_only", "pool8_r1col4.json"),
)


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _log(msg: str) -> None:
    sys.stderr.write("[probe] %s\n" % msg)


def _read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _sh(cmd: Sequence[str], timeout: int = 20) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
        )
        return (
            proc.returncode,
            proc.stdout.decode("utf-8", "replace").strip(),
            proc.stderr.decode("utf-8", "replace").strip(),
        )
    except Exception as exc:  # noqa: BLE001 - 探针不能因为一个命令失败就整体挂掉
        return 127, "", "%s: %s" % (type(exc).__name__, exc)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def resolve_run(name: str) -> str:
    """把 'r1ro4c' / 'route1_r1ro4c_seed20260826' / 绝对路径 都解析成 run 目录。"""
    if not name:
        raise SystemExit("必须给 --run")
    candidates: List[str] = []
    if os.path.isabs(name):
        candidates.append(name)
    else:
        candidates.append(os.path.join(REPO, "logs", "habitat", name))
        candidates.append(os.path.join(REPO, "logs", "habitat", "route1_%s_seed%s" % (name, SEED)))
        candidates.extend(sorted(glob.glob(os.path.join(REPO, "logs", "habitat", "route1_%s_*" % name))))
    for path in candidates:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "progress.json")):
            return path
    raise SystemExit(
        "找不到 run %r；可用：%s"
        % (name, ", ".join(sorted(os.path.basename(p) for p in _run_dirs())) or "(无)")
    )


def _run_dirs() -> List[str]:
    return sorted(p for p in glob.glob(RUNS_GLOB) if os.path.isdir(p))


def _label_of(run_dir: str) -> str:
    base = os.path.basename(run_dir.rstrip("/"))
    match = re.match(r"^route1_(.+?)_seed\d+.*$", base)
    return match.group(1) if match else base


def _expected_episodes(label: str) -> int:
    """两个多场景清单都是 52；旧的单场景 capture run 是 50。"""
    return 50 if "capture" in label else 52


class RunData(object):
    """一个 run 的日志（progress.json + closed_loop_steps_rank*.jsonl）。"""

    def __init__(self, run_dir: str) -> None:
        self.dir = run_dir
        self.label = _label_of(run_dir)
        self.progress = _read_jsonl(os.path.join(run_dir, "progress.json"))
        self.events: List[Dict[str, Any]] = []
        for path in sorted(glob.glob(os.path.join(run_dir, "closed_loop_steps_rank*.jsonl"))):
            self.events.extend(_read_jsonl(path))
        self.by_episode: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
        for row in self.events:
            self.by_episode[int(row.get("episode_id", -1))].append(row)
        self.mode = None
        for row in self.progress:
            if row.get("closed_loop_mode"):
                self.mode = row["closed_loop_mode"]
                break
        self.instructions = {
            int(row.get("episode_id", -1)): row.get("episode_instruction") for row in self.progress
        }
        self.horizon = max([int(r.get("steps") or 0) for r in self.progress] or [0])

    # -- 聚合 -------------------------------------------------------------- #
    def decisions(self) -> List[Dict[str, Any]]:
        rows = [r for r in self.events if r.get("event") == "system1"]
        rows.sort(key=lambda r: (int(r.get("episode_id", 0)), int(r.get("step", 0))))
        return rows

    def per_scene(self) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for row in self.progress:
            grouped[str(row.get("scene_id"))].append(row)
        out = []
        for scene in sorted(grouped):
            rows = grouped[scene]
            terminations = collections.Counter(self.termination(r) for r in rows)
            out.append(
                {
                    "scene_id": scene,
                    "episodes": len(rows),
                    "success": _round(_mean(r.get("success") for r in rows)),
                    "spl": _round(_mean(r.get("spl") for r in rows)),
                    "ne": _round(_mean(r.get("ne") for r in rows), 3),
                    "collision": _round(_mean(r.get("collision_count") for r in rows), 2),
                    "steps": _round(_mean(r.get("steps") for r in rows), 1),
                    "replans": _round(_mean(r.get("replans") for r in rows), 2),
                    "system2_calls": _round(_mean(r.get("system2_calls") for r in rows), 2),
                    "watchdog_stops": sum(1 for r in rows if (r.get("collision_watchdog_stops") or 0) > 0),
                    "terminations": dict(terminations),
                }
            )
        return out

    def termination(self, row: Dict[str, Any]) -> str:
        """终止方式判定 —— **直接复用** scene_failure_diag 的实现，避免两套口径。

        口径局限（必须与结论一起写）：evaluator 不记录 Habitat `done` 的原因，
        所以"预算烧完没到目标"与"主动 STOP 在错位置"在单 run 日志里不可分，
        上游把两者合成 `budget_exhausted_no_goal`。
        """
        episode = int(row.get("episode_id", -1))
        events = list(self.by_episode.get(episode, []))
        try:
            from route1.experiments.scene_failure_diag import classify_termination  # noqa: WPS433

            klass, _evidence = classify_termination(row, events, self.horizon)
            return klass
        except Exception as exc:  # noqa: BLE001
            _log("classify_termination 不可用（%s），用本地兜底判定" % exc)
        if float(row.get("success") or 0.0) >= 0.5:
            return "success"
        if int(row.get("collision_watchdog_stops") or 0) > 0:
            return "watchdog_stop"
        return "budget_exhausted_no_goal"

    def trigger_stats(self) -> Dict[str, Any]:
        decisions = self.decisions()
        reasons = collections.Counter(
            str(r.get("replan_reason")) for r in decisions if r.get("replan_reason")
        )
        no_action = [r for r in decisions if r.get("has_executable_action") is False]
        official = [r for r in no_action if r.get("model_invoked") is True]
        return {
            "decisions": len(decisions),
            "replan_reasons": dict(reasons),
            "no_action_events": len(no_action),
            "official_label_events": len(official),
            "official_label_coverage": _round(
                (len(official) / len(no_action)) if no_action else None
            ),
        }

    def summary(self) -> Dict[str, Any]:
        rows = self.progress
        per_scene = self.per_scene()
        success_by_scene = [s["success"] for s in per_scene if s["success"] is not None]
        spread = None
        if success_by_scene:
            spread = {
                "n_scenes": len(success_by_scene),
                "min": _round(min(success_by_scene)),
                "max": _round(max(success_by_scene)),
                "std": _round(statistics.pstdev(success_by_scene)) if len(success_by_scene) > 1 else 0.0,
                "range": _round(max(success_by_scene) - min(success_by_scene)),
            }
        return {
            "label": self.label,
            "dir": self.dir,
            "mode": self.mode,
            "episodes": len(rows),
            "expected_episodes": _expected_episodes(self.label),
            "pooled_success": _round(_mean(r.get("success") for r in rows)),
            "pooled_spl": _round(_mean(r.get("spl") for r in rows)),
            "cost": {
                "collision_per_ep": _round(_mean(r.get("collision_count") for r in rows), 2),
                "steps_per_ep": _round(_mean(r.get("steps") for r in rows), 1),
                "system2_per_ep": _round(_mean(r.get("system2_calls") for r in rows), 2),
                "replans_per_ep": _round(_mean(r.get("replans") for r in rows), 2),
                "ne": _round(_mean(r.get("ne") for r in rows), 3),
            },
            "cross_scene_success_spread": spread,
            "per_scene": per_scene,
            "triggers": self.trigger_stats(),
            "mtime_iso": _iso(os.path.getmtime(self.dir)),
        }


def _iso(ts: Optional[float]) -> Optional[str]:
    if not ts:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #


def cmd_status(_args: argparse.Namespace) -> Dict[str, Any]:
    rc, gpu, _err = _sh(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    rc_t, tmux, _ = _sh(["tmux", "ls"])
    rc_p, procs, _ = _sh(
        ["bash", "-lc", "pgrep -af 'scripts/eval/eval.py' | grep -v pgrep || true"]
    )
    disk = shutil.disk_usage("/mnt/sda") if os.path.exists("/mnt/sda") else shutil.disk_usage("/")
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    rc_i, imp, err_i = _sh_env(
        [PY, "-c", "import route1, route1.nav.channels as c, route1.contracts as k; print(route1.__file__)"],
        env,
        timeout=120,
    )
    return {
        "schema": SCHEMA,
        "command": "status",
        "server": {
            "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "repo": REPO,
            "src": SRC,
            "logs": LOG,
            "python": PY,
            "python_version": sys.version.split()[0],
        },
        "gpu": gpu.splitlines() if rc == 0 else [],
        "tmux_sessions": [line.split(":")[0] for line in tmux.splitlines()] if rc_t == 0 else [],
        "eval_running": procs.splitlines() if rc_p == 0 else [],
        "disk": {
            "path": "/mnt/sda",
            "total_gb": round(disk.total / 1e9, 1),
            "free_gb": round(disk.free / 1e9, 1),
            "used_pct": round(100.0 * disk.used / disk.total, 1),
        },
        "route1_importable": rc_i == 0,
        "route1_import_note": (imp or err_i).splitlines()[-1] if (imp or err_i) else "",
        "runs_available": len(_run_dirs()),
        "checked_at_iso": _iso(time.time()),
    }


def _sh_env(cmd: Sequence[str], env: Dict[str, str], timeout: int = 60) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env
        )
        return (
            proc.returncode,
            proc.stdout.decode("utf-8", "replace").strip(),
            proc.stderr.decode("utf-8", "replace").strip(),
        )
    except Exception as exc:  # noqa: BLE001
        return 127, "", "%s: %s" % (type(exc).__name__, exc)


def cmd_runs(_args: argparse.Namespace) -> Dict[str, Any]:
    rows = []
    for run_dir in _run_dirs():
        try:
            data = RunData(run_dir)
        except Exception as exc:  # noqa: BLE001
            _log("跳过 %s：%s" % (run_dir, exc))
            continue
        summary = data.summary()
        scenes = sorted({str(r.get("scene_id")) for r in data.progress})
        rows.append(
            {
                "label": summary["label"],
                "dir": summary["dir"],
                "mode": summary["mode"],
                "episodes": summary["episodes"],
                "expected_episodes": summary["expected_episodes"],
                "scenes": scenes,
                "n_scenes": len(scenes),
                "pooled_success": summary["pooled_success"],
                "pooled_spl": summary["pooled_spl"],
                "worst_scene_success": summary["cross_scene_success_spread"]["min"]
                if summary["cross_scene_success_spread"]
                else None,
                "collision_per_ep": summary["cost"]["collision_per_ep"],
                "system2_per_ep": summary["cost"]["system2_per_ep"],
                "no_action_events": summary["triggers"]["no_action_events"],
                "mtime_iso": summary["mtime_iso"],
            }
        )
    rows.sort(key=lambda r: r["mtime_iso"] or "")
    return {"schema": SCHEMA, "command": "runs", "count": len(rows), "runs": rows}


def cmd_scenes(_args: argparse.Namespace) -> Dict[str, Any]:
    manifests = []
    for path in sorted(glob.glob(os.path.join(REPO, "data", "**", "val_unseen*"), recursive=True)):
        if os.path.isdir(path) or not path.endswith((".json", ".json.gz")):
            continue
        manifests.append(
            {"name": os.path.basename(path), "path": path, "size_bytes": os.path.getsize(path)}
        )
    scenes = collections.Counter()
    for run_dir in _run_dirs():
        for row in _read_jsonl(os.path.join(run_dir, "progress.json")):
            if row.get("scene_id"):
                scenes[str(row["scene_id"])] += 1
    return {
        "schema": SCHEMA,
        "command": "scenes",
        "manifests": manifests,
        "scenes_seen_in_runs": dict(sorted(scenes.items())),
        "configs": sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(REPO, "scripts", "eval", "configs", "route1_*"))
        ),
    }


def cmd_run_summary(args: argparse.Namespace) -> Dict[str, Any]:
    data = RunData(resolve_run(args.run))
    return {"schema": SCHEMA, "command": "run-summary", "summary": data.summary()}


def cmd_series(args: argparse.Namespace) -> Dict[str, Any]:
    data = RunData(resolve_run(args.run))
    episode = int(args.episode)
    rows = [r for r in data.by_episode.get(episode, []) if r.get("event") == "system1"]
    rows.sort(key=lambda r: int(r.get("step", 0)))
    if not rows:
        available = sorted(data.by_episode.keys())
        raise SystemExit("episode %s 没有 system1 决策；该 run 的 episode: %s" % (episode, available[:20]))
    series = []
    for row in rows:
        series.append(
            {
                "decision_index": row.get("system1_decision_index"),
                "step": row.get("step"),
                "action_step": row.get("action_step"),
                "distance_to_goal": _round(row.get("distance_to_goal"), 3),
                "collision_count": row.get("collision_count"),
                "plan_age": row.get("plan_age"),
                "trajectory_drift": _round(row.get("trajectory_drift"), 4),
                "feedback_score": _round(row.get("feedback_score"), 4),
                "availability": row.get("availability"),
                "should_replan": row.get("should_replan"),
                "replan_reason": row.get("replan_reason"),
                "trigger_kind": row.get("trigger_kind"),
                "has_executable_action": row.get("has_executable_action"),
                "model_invoked": row.get("model_invoked"),
                "rgb_path": row.get("rgb_path"),
            }
        )
    meta = None
    for row in data.progress:
        if int(row.get("episode_id", -1)) == episode:
            meta = {
                "scene_id": row.get("scene_id"),
                "success": row.get("success"),
                "spl": _round(row.get("spl")),
                "ne": _round(row.get("ne"), 3),
                "steps": row.get("steps"),
                "collision_count": row.get("collision_count"),
                "replans": row.get("replans"),
                "system2_calls": row.get("system2_calls"),
                "no_action_count": row.get("no_action_count"),
                "replan_reasons": row.get("replan_reasons"),
                "instruction": row.get("episode_instruction"),
                "termination": data.termination(row),
            }
            break
    return {
        "schema": SCHEMA,
        "command": "series",
        "run": data.label,
        "episode": episode,
        "episode_meta": meta,
        "decisions": series,
    }


def _failure_events(data: RunData, limit: int, scene: Optional[str], episode: Optional[int]) -> List[Any]:
    """把日志里的"真失败 / 通道触发"事件映射成统一 FailureEvent 契约。"""
    from route1.contracts import Attribution, FailureEvent, Recovery, Signals  # noqa: WPS433

    success_by_ep: Dict[int, float] = {}
    for row in data.progress:
        success_by_ep[int(row.get("episode_id", -1))] = float(row.get("success") or 0.0)

    decisions = data.decisions()
    next_score: Dict[int, Optional[float]] = {}
    for idx, row in enumerate(decisions):
        nxt = decisions[idx + 1] if idx + 1 < len(decisions) else None
        if nxt is not None and int(nxt.get("episode_id", -1)) == int(row.get("episode_id", -1)):
            next_score[id(row)] = nxt.get("feedback_score")
        else:
            next_score[id(row)] = None

    events = []
    for row in decisions:
        if scene and str(row.get("scene_id")) != scene:
            continue
        if episode is not None and int(row.get("episode_id", -1)) != int(episode):
            continue
        reason = row.get("replan_reason")
        no_action = row.get("has_executable_action") is False
        ftype: Optional[str] = None
        kind = "observation"
        summary = ""
        evidence: List[str] = []
        if no_action:
            ftype = "N1"
            kind = "no_executable_action"
            summary = "System-1 计划解不出可执行动作，只能走恢复路径"
            evidence.append("has_executable_action=False")
            if row.get("model_invoked") is False:
                evidence.append("model_invoked=False（官方标签不计入，属 padded local STOP）")
        elif reason == "collision_density_replan":
            ftype = "N5"
            kind = "collision_density"
            summary = "窗口内碰撞密度过高（v4 碰撞通道要求重规划）"
            evidence.append("replan_reason=collision_density_replan")
        elif reason == "progress_stall_replan":
            ftype = "N2"
            kind = "progress_stall"
            summary = "朝目标没有进展（v4 进度通道要求重规划）"
            evidence.append("replan_reason=progress_stall_replan")
        elif row.get("should_replan"):
            ftype = "N4"
            kind = "feedback_high"
            summary = "几何加权分越过迟滞高阈，触发重规划"
            evidence.append("should_replan=True score=%s" % _round(row.get("feedback_score"), 4))
        else:
            continue
        signals = Signals.from_dict(
            {
                "feedback_score": row.get("feedback_score"),
                "trajectory_drift": row.get("trajectory_drift"),
                "plan_age": row.get("plan_age"),
                "collision_delta": row.get("collision_count"),
                "steps_since_last_s2": row.get("steps_since_last_s2"),
                "dist_to_goal_m": row.get("distance_to_goal"),
            }
        )
        attribution = Attribution(
            kind=kind,
            summary_l1=summary,
            summary_l2="场景 %s / 回合 %s / 第 %s 步（决策 %s）"
            % (row.get("scene_id"), row.get("episode_id"), row.get("step"), row.get("system1_decision_index")),
            evidence=evidence,
            dominant_signal="collision_count" if kind == "collision_density" else (
                "distance_to_goal" if kind == "progress_stall" else "feedback_score"
            ),
            source="rule",
        )
        recovery = Recovery(
            reason_code=str(reason or kind),
            action="replan" if ftype in ("N1", "N2", "N4", "N5") else "none",
            budget_used=int(row.get("system2_call_count") or 0),
            budget_left=0,
        )
        from route1.contracts import Outcome  # noqa: WPS433

        post = next_score.get(id(row))
        outcome = Outcome(
            recovered=bool(success_by_ep.get(int(row.get("episode_id", -1)), 0.0) >= 1.0) if post is not None else None,
            post_score=post,
            steps_to_recover=None,
        )
        event = FailureEvent.create(
            source="nav",
            type=ftype,
            scene_id=row.get("scene_id"),
            episode_id=row.get("episode_id"),
            step=row.get("step"),
            action_step=row.get("action_step"),
            instruction=data.instructions.get(int(row.get("episode_id", -1))),
            signals=signals,
            attribution=attribution,
            recovery=recovery,
            outcome=outcome,
            trigger_kind=row.get("trigger_kind"),
            t=row.get("timestamp"),
        )
        events.append(event)
        if limit and len(events) >= limit:
            break
    return events


def cmd_failures(args: argparse.Namespace) -> Dict[str, Any]:
    data = RunData(resolve_run(args.run))
    events = _failure_events(data, int(args.limit), args.scene, args.episode)
    payloads = [e.to_dict() for e in events]
    types = collections.Counter(str(p.get("type")) for p in payloads)
    actions = collections.Counter(str((p.get("recovery") or {}).get("action")) for p in payloads)
    return {
        "schema": SCHEMA,
        "command": "failures",
        "run": data.label,
        "count": len(payloads),
        "type_histogram": dict(types),
        "recovery_histogram": dict(actions),
        "contract": "route1.contracts.FailureEvent（与 route1-bridge-plan.md §6 一一对应）",
        "events": payloads,
    }


def _p_at_k(decisions: List[Dict[str, Any]], k: int, official_only: bool) -> Dict[str, Any]:
    triggers = 0
    hits = 0
    for idx, row in enumerate(decisions):
        fired = bool(row.get("should_replan")) or (
            row.get("replan_reason") not in (None, "", "recovery_only_observe")
        )
        if not fired:
            continue
        triggers += 1
        window = decisions[idx + 1 : idx + 1 + k]
        for nxt in window:
            if nxt.get("has_executable_action") is False and (
                not official_only or nxt.get("model_invoked") is True
            ):
                hits += 1
                break
    return {"triggers": triggers, "hits": hits, "p_at_k": _round(hits / triggers) if triggers else None}


def _auroc(scores: List[float], labels: List[int]) -> Optional[float]:
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for pos in range(index, end + 1):
            if pairs[pos][1] == 1:
                rank_sum += average_rank
        index = end + 1
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def cmd_triggers(args: argparse.Namespace) -> Dict[str, Any]:
    data = RunData(resolve_run(args.run))
    decisions = data.decisions()
    out: Dict[str, Any] = {"schema": SCHEMA, "command": "triggers", "run": data.label, "decisions": len(decisions)}
    for k in (1, 3, 5):
        out["any_replan_p@%d" % k] = _p_at_k(decisions, k, official_only=False)
        out["official_label_p@%d" % k] = _p_at_k(decisions, k, official_only=True)
    scores: List[float] = []
    labels: List[int] = []
    for idx, row in enumerate(decisions):
        score = row.get("feedback_score")
        if score is None:
            continue
        window = decisions[idx + 1 : idx + 2]
        if not window:
            continue
        labels.append(1 if window[0].get("has_executable_action") is False else 0)
        scores.append(float(score))
    out["auroc_feedback_score_h1"] = _round(_auroc(scores, labels), 4)
    out["auroc_n"] = len(labels)
    out["base_rate_h1_any"] = _round((sum(labels) / len(labels)) if labels else None)
    official = [
        1 if (d.get("has_executable_action") is False and d.get("model_invoked") is True) else 0
        for d in decisions
    ]
    out["base_rate_official"] = _round(sum(official) / len(official) if official else None)
    out["triggers_detail"] = data.trigger_stats()
    out["note"] = (
        "any_replan = 任何真正重规划的决策；official_label = has_executable_action=False 且 "
        "model_invoked=True（与 notes/route1-findings.md 结论 3 的口径一致）"
    )
    return out


def cmd_recovery(args: argparse.Namespace) -> Dict[str, Any]:
    """用**在线那两条通道的同一份实现**离线复算：它们会在哪些决策点要求重规划。"""
    import route1.nav.channels as channels  # noqa: WPS433

    data = RunData(resolve_run(args.run))
    config = channels.ChannelConfig()
    if args.collision_min:
        config.collision_min_in_window = int(args.collision_min)
    if args.progress_window:
        config.progress_window_actions = int(args.progress_window)
    if args.max_per_episode:
        config.collision_max_per_episode = int(args.max_per_episode)
        config.progress_max_per_episode = int(args.max_per_episode)
    if args.channels == "none":
        config.collision_enabled = False
        config.progress_enabled = False
    elif args.channels == "collision":
        config.progress_enabled = False
    elif args.channels == "progress":
        config.collision_enabled = False

    episodes = collections.defaultdict(list)
    for row in data.events:
        episodes[int(row.get("episode_id", -1))].append(row)
    success_by_ep = {int(r.get("episode_id", -1)): float(r.get("success") or 0.0) for r in data.progress}
    scene_by_ep = {int(r.get("episode_id", -1)): r.get("scene_id") for r in data.progress}

    fires_by_scene: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    per_episode: List[Dict[str, Any]] = []
    for episode in sorted(episodes):
        if args.scene and str(scene_by_ep.get(episode)) != args.scene:
            continue
        if args.episode is not None and episode != int(args.episode):
            continue
        stream = sorted(episodes[episode], key=lambda r: (int(r.get("timestamp") or 0), int(r.get("step", 0))))
        tracker = channels.TriggerChannels(config)
        last_collision = 0
        fires: List[Dict[str, Any]] = []
        last_step = 0
        for row in stream:
            event = row.get("event")
            if event == "action":
                count = int(row.get("collision_count") or 0)
                if row.get("counts_as_motion"):
                    tracker.push_collision(count > last_collision)
                last_collision = count
                last_step = max(last_step, int(row.get("action_step") or 0))
            elif event == "system1":
                step = int(row.get("step") or 0)
                last_step = max(last_step, step)
                reason = tracker.evaluate(step, row.get("distance_to_goal"))
                if reason:
                    fires.append(
                        {
                            "step": step,
                            "reason": reason,
                            "distance_to_goal": _round(row.get("distance_to_goal"), 3),
                            "collisions_in_window": tracker.collision_count_in_window(),
                            "plan_age": row.get("plan_age"),
                            "feedback_score": _round(row.get("feedback_score"), 4),
                            "plan_age_at_fire": row.get("plan_age"),
                        }
                    )
                    fires_by_scene[str(scene_by_ep.get(episode))].append(fires[-1])
        per_episode.append(
            {
                "episode_id": episode,
                "scene_id": scene_by_ep.get(episode),
                "success": success_by_ep.get(episode),
                "steps": last_step,
                "fires": fires,
                "n_fires": len(fires),
            }
        )

    total_fires = sum(e["n_fires"] for e in per_episode)
    ep_with_fires = [e for e in per_episode if e["n_fires"] > 0]
    failed = [e for e in per_episode if (e["success"] or 0) < 1.0]
    covered = [e for e in ep_with_fires if (e["success"] or 0) < 1.0]
    reason_hist = collections.Counter(f["reason"] for e in per_episode for f in e["fires"])
    return {
        "schema": SCHEMA,
        "command": "recovery",
        "run": data.label,
        "channels": {
            "collision_enabled": config.collision_enabled,
            "progress_enabled": config.progress_enabled,
            "collision_window_actions": config.collision_window_actions,
            "collision_min_in_window": config.collision_min_in_window,
            "progress_window_actions": config.progress_window_actions,
            "progress_min_improvement_m": config.progress_min_improvement_m,
            "max_per_episode": config.collision_max_per_episode,
            "implementation": "route1.nav.channels.TriggerChannels（与在线 overlay_v4 同一份代码）",
        },
        "episodes": len(per_episode),
        "fires_total": total_fires,
        "fires_per_episode": _round(total_fires / len(per_episode)) if per_episode else None,
        "reason_histogram": dict(reason_hist),
        "episodes_with_fire": len(ep_with_fires),
        "failed_episodes": len(failed),
        "failed_episodes_covered": len(covered),
        "episode_level_precision": _round(len(covered) / len(ep_with_fires)) if ep_with_fires else None,
        "base_rate": _round(len(failed) / len(per_episode)) if per_episode else None,
        "detail": per_episode,
    }


def _load_pool(name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(LOG, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _config_table() -> Dict[str, Any]:
    table: Dict[str, Any] = {}
    scenes: List[str] = []
    for label, fname in CONFIGS:
        data = _load_pool(fname)
        if data is None:
            table[label] = None
            continue
        per_scene = {s["scene_id"]: s for s in data.get("per_scene", [])}
        for scene in per_scene:
            if scene not in scenes:
                scenes.append(scene)
        table[label] = {
            "source_file": fname,
            "pooled_success": _round(data.get("pooled_success")),
            "pooled_spl": _round(data.get("pooled_spl")),
            "episodes_total": data.get("episodes_total"),
            "scenes_total": data.get("scenes_total"),
            "spread": {
                k: _round(v)
                for k, v in (data.get("cross_scene_success_spread") or {}).items()
                if k in ("n_scenes", "min", "max", "std", "range")
            },
            "per_scene": {
                s: {
                    "success": _round(per_scene[s].get("success")),
                    "episodes": per_scene[s].get("episodes"),
                    "collision": _round(per_scene[s].get("collision"), 2),
                    "steps": _round(per_scene[s].get("steps"), 1),
                    "system2_calls": _round(per_scene[s].get("system2_calls"), 2),
                    "terminations": per_scene[s].get("terminations"),
                }
                for s in per_scene
            },
            "source_runs": data.get("sources"),
        }
    final_table = None
    final_path = os.path.join(LOG, "FINAL_TABLE.txt")
    if os.path.exists(final_path):
        with open(final_path, "r", encoding="utf-8", errors="replace") as handle:
            final_table = handle.read()[:20000]
    return {"scenes": sorted(scenes), "configs": table, "final_table_txt": final_table}


def _light_run(run_dir: str) -> Optional[Dict[str, Any]]:
    """只看 progress.json 的轻量 run 摘要（面板要快，不读逐决策日志）。"""
    rows = _read_jsonl(os.path.join(run_dir, "progress.json"))
    if not rows:
        return None
    per_scene: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        per_scene[str(row.get("scene_id"))].append(row)
    scene_success = [_mean(r.get("success") for r in group) for group in per_scene.values()]
    scene_success = [value for value in scene_success if value is not None]
    mode = None
    for row in rows:
        if row.get("closed_loop_mode"):
            mode = row["closed_loop_mode"]
            break
    return {
        "label": _label_of(run_dir),
        "dir": run_dir,
        "mode": mode,
        "episodes": len(rows),
        "expected_episodes": _expected_episodes(_label_of(run_dir)),
        "n_scenes": len(per_scene),
        "pooled_success": _round(_mean(r.get("success") for r in rows)),
        "pooled_spl": _round(_mean(r.get("spl") for r in rows)),
        "worst_scene_success": _round(min(scene_success)) if scene_success else None,
        "collision_per_ep": _round(_mean(r.get("collision_count") for r in rows), 2),
        "system2_per_ep": _round(_mean(r.get("system2_calls") for r in rows), 2),
        "mtime_iso": _iso(os.path.getmtime(run_dir)),
    }


def cmd_board(_args: argparse.Namespace) -> Dict[str, Any]:
    recent = [row for row in (_light_run(path) for path in _run_dirs()) if row]
    recent.sort(key=lambda row: row["mtime_iso"] or "")
    return {
        "schema": SCHEMA,
        "command": "board",
        "generated_at_iso": _iso(time.time()),
        "evaluation_protocol": {
            "manifest": "val_unseen_multiscene（主清单 4 场景×13 回合）+ val_unseen_multiscene_b（新场景 4×13）",
            "episodes_per_config": 104,
            "seeds": SEED,
            "metric": "Habitat VLN Success / SPL / NE / 碰撞 / 步数 / System-2 调用",
            "note": "4 个调参场景 + 4 个模型没见过的场景；每格 13 回合，SE≈14pt",
        },
        "recent_runs": recent[-8:][::-1],
        **_config_table(),
    }


def cmd_progress(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = resolve_run(args.run)
    rows = _read_jsonl(os.path.join(run_dir, "progress.json"))
    label = _label_of(run_dir)
    expected = _expected_episodes(label)
    per_scene = collections.Counter(str(r.get("scene_id")) for r in rows)
    rc, tmux, _ = _sh(["bash", "-lc", "tmux ls 2>/dev/null | cut -d: -f1"])
    sessions = [s for s in tmux.splitlines() if s]
    log_candidates = sorted(glob.glob(os.path.join(LOG, "*.log")), key=os.path.getmtime, reverse=True)
    tail = []
    for path in log_candidates[:3]:
        if label.split("4")[0] and os.path.basename(path).startswith(label[:4]):
            tail = _tail(path, 12)
            break
    return {
        "schema": SCHEMA,
        "command": "progress",
        "run": label,
        "dir": run_dir,
        "episodes": len(rows),
        "expected_episodes": expected,
        "complete": len(rows) >= expected,
        "success_so_far": _round(_mean(r.get("success") for r in rows)),
        "scenes": dict(per_scene),
        "tmux_sessions": sessions,
        "log_tail": tail,
        "checked_at_iso": _iso(time.time()),
    }


def _tail(path: str, lines: int) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in handle.readlines()[-lines:]]
    except OSError:
        return []


def cmd_launch(args: argparse.Namespace) -> Dict[str, Any]:
    """在服务器上起一次新的导航评估（tmux + 我们自己的 run_eval_v4 启动器）。"""
    label = args.label
    if not re.match(r"^[A-Za-z0-9_]{2,32}$", label):
        raise SystemExit("label 只能是字母数字下划线（2-32 位）")
    session = "r1_%s" % label[:16]
    run_dir = os.path.join(REPO, "logs", "habitat", "route1_%s_seed%s" % (label, SEED))
    if os.path.exists(os.path.join(run_dir, "progress.json")) and not args.force:
        raise SystemExit("产物已存在：%s（要覆盖请加 --force）" % run_dir)
    rc, sessions, _ = _sh(["bash", "-lc", "tmux ls 2>/dev/null | cut -d: -f1"])
    if session in sessions.split():
        raise SystemExit("tmux 会话 %s 已在运行" % session)
    if args.require_idle:
        rc, procs, _ = _sh(["bash", "-lc", "pgrep -af 'scripts/eval/eval.py' || true"])
        if procs.strip():
            raise SystemExit("已有 eval 在跑：\n%s" % procs)

    inner_path = os.path.join(LOG, "_agh_%s_inner.sh" % label)
    log_path = os.path.join(LOG, "agh_%s.log" % label)
    budget_mode = args.budget_mode
    channels = args.channels
    if budget_mode in ("original",) or channels == "none":
        launcher = "python -m route1.experiments.run_eval_v4"
    else:
        launcher = "python -m route1.experiments.run_eval_v4"
    env_lines = [
        "export CUDA_VISIBLE_DEVICES=%s" % args.gpu,
        "export PYTHONHASHSEED=%s" % SEED,
        "export PYTHONPATH=%s" % SRC,
        "export EPISODE_LIMIT=%d" % int(args.episodes),
        "export MULTISCENE_PER_SCENE=13",
        "export MULTISCENE_N_SCENES=4",
        "export CAPTURE_MAX_EPISODE_ID=400",
    ]
    run_env = [
        "BUDGET_MODE=%s" % budget_mode,
        "BUDGET_RUN_LABEL=%s" % label,
        "V4_WATCHDOG=%s" % args.watchdog,
        "V4_CHANNELS=%s" % channels,
        "V4_COLLISION_MIN=%d" % int(args.collision_min),
        "V4_PROGRESS_WINDOW=%d" % int(args.progress_window),
        "V4_COLLISION_MAX=%d" % int(args.collision_max),
        "V4_PROGRESS_MAX=%d" % int(args.collision_max),
        "PERIODIC_INTERVAL=%d" % int(args.periodic_interval),
    ]
    body = "\n".join(
        [
            "#!/usr/bin/env bash",
            "# 由 agh_nav_probe.py launch 生成（AGH 插件调用），不要手改：%s" % _iso(time.time()),
            "set -u",
            "cd %s" % REPO,
        ]
        + env_lines
        + [
            'echo "[agh] 等 GPU 空出来 $(date -Is)"',
            "while pgrep -f 'scripts/eval/eval.py' >/dev/null 2>&1; do sleep 30; done",
            "sleep 5",
            'if [ -s "%s/progress.json" ]; then echo "[agh] 产物已存在"; exit 0; fi' % run_dir,
            "rm -rf %s/route1/__pycache__ %s/route1/nav/__pycache__" % (SRC, SRC),
            'echo "[agh] START $(date -Is) mode=%s watchdog=%s channels=%s"' % (budget_mode, args.watchdog, channels),
            "%s \\" % " ".join(run_env),
            "  %s -u %s \\" % (PY, launcher),
            "    --config %s" % args.config,
            'echo "[agh] EXIT=$? $(date -Is)"',
        ]
    )
    with open(inner_path, "w", encoding="utf-8") as handle:
        handle.write(body + "\n")
    os.chmod(inner_path, 0o755)
    rc, out, err = _sh(
        ["tmux", "new-session", "-d", "-s", session, "bash %s 2>&1 | tee %s" % (inner_path, log_path)],
        timeout=30,
    )
    if rc != 0:
        raise SystemExit("tmux 启动失败：%s %s" % (out, err))
    return {
        "schema": SCHEMA,
        "command": "launch",
        "label": label,
        "tmux_session": session,
        "run_dir": run_dir,
        "log": log_path,
        "inner_script": inner_path,
        "config": args.config,
        "env": {
            "BUDGET_MODE": budget_mode,
            "V4_WATCHDOG": args.watchdog,
            "V4_CHANNELS": channels,
            "EPISODE_LIMIT": int(args.episodes),
            "CUDA_VISIBLE_DEVICES": args.gpu,
        },
        "expected_minutes": int(args.episodes) * 1.3,
        "expected_result": {
            "reference_pooled_success": {
                "R2A": 0.6058,
                "original": 0.6346,
                "recovery_only": 0.6635,
                "guarded": 0.7115,
                "collision_only": 0.6538,
            },
            "note": "参考值来自 8 场景×104 回合口径（logs/route1/FINAL_TABLE.txt）",
        },
        "next_steps": [
            "轮询 nav_run_progress 直到 complete=true（约 %d 分钟）" % int(int(args.episodes) * 1.3),
            "跑 nav_run_summary --run %s 看逐场景指标" % label,
            "跑 nav_recovery_options --run %s 看两条通道在该 run 上会怎么动" % label,
        ],
        "started_at_iso": _iso(time.time()),
    }


def cmd_quick(_args: argparse.Namespace) -> Dict[str, Any]:
    """没给子命令时的速览：一眼看全"环境 + 最近 run + 五套配置的头条数字"。

    为什么要有它：直接跑 `agh_nav_probe.py` 会得到 argparse 的 usage 报错（退出码 2），
    对第一次接触的人不友好。这里改成打印一份**结构化速览**（仍然是 JSON，JS 侧不受影响）。
    """
    status = cmd_status(_args)
    board = cmd_board(_args)
    runs = [row for row in (_light_run(path) for path in _run_dirs()) if row]
    runs.sort(key=lambda row: row["mtime_iso"] or "")
    configs = board.get("configs") or {}
    return {
        "schema": SCHEMA,
        "command": "quick",
        "hint": "没给子命令时打印速览；可用子命令：status/runs/scenes/board/run-summary/series/"
        "failures/triggers/recovery/progress/launch/cache-all（详见 --help）",
        "status": {
            key: status[key]
            for key in (
                "server",
                "gpu",
                "tmux_sessions",
                "eval_running",
                "disk",
                "route1_importable",
                "runs_available",
            )
        },
        "latest_runs": runs[-5:][::-1],
        "board_headline": {name: (cfg or {}).get("pooled_success") for name, cfg in configs.items()},
        "board_worst_scene": {
            name: ((cfg or {}).get("spread") or {}).get("min") for name, cfg in configs.items()
        },
        "board_note": "口径：8 场景 × 13 回合 = 104 回合/配置；完整表用 `board`",
        "generated_at_iso": _iso(time.time()),
    }


def cmd_cache_all(args: argparse.Namespace) -> Dict[str, Any]:
    """一次调用产出插件本地缓存所需的全部内容（断链时的回退数据）。"""
    runs = cmd_runs(args)["runs"]
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "command": "cache-all",
        "generated_at_iso": _iso(time.time()),
        "status": cmd_status(args),
        "runs": runs,
        "scenes": cmd_scenes(args),
        "board": cmd_board(args),
    }
    # 取一个"有通道触发"的 run 和一个"基线" run 做样本，供离线演示。
    # ⚠️ 样本必须是**子命令完整的 payload**（与实时返回同形），否则工具在缓存回退时
    # 会拿到形状不同的对象（这个坑在自检里被抓到过：`payload.data.summary` 变 undefined）。
    samples: Dict[str, Any] = {}
    preferred = args.sample_run or ("r1grd4" if any(r["label"] == "r1grd4" for r in runs) else runs[-1]["label"])
    try:
        samples["run-summary"] = cmd_run_summary(argparse.Namespace(run=preferred))
    except SystemExit as exc:
        _log("样本 run-summary 跳过：%s" % exc)
    try:
        recovery_payload = cmd_recovery(
            argparse.Namespace(
                run=preferred,
                scene=None,
                episode=None,
                channels="both",
                collision_min=None,
                progress_window=None,
                max_per_episode=None,
            )
        )
        recovery_payload["detail"] = recovery_payload["detail"][:6]
        samples["recovery"] = recovery_payload
    except SystemExit as exc:
        _log("样本 recovery 跳过：%s" % exc)
    try:
        samples["triggers"] = cmd_triggers(argparse.Namespace(run=preferred))
    except SystemExit as exc:
        _log("样本 triggers 跳过：%s" % exc)
    try:
        samples["failures"] = cmd_failures(
            argparse.Namespace(run=preferred, limit=8, scene=None, episode=None)
        )
    except SystemExit as exc:
        _log("样本 failures 跳过：%s" % exc)
    try:
        data = RunData(resolve_run(preferred))
        sample_ep = None
        for row in data.progress:
            if float(row.get("success") or 0) < 1.0:
                sample_ep = int(row.get("episode_id"))
                break
        if sample_ep is None and data.progress:
            sample_ep = int(data.progress[0].get("episode_id"))
        samples["series"] = cmd_series(argparse.Namespace(run=preferred, scene=None, episode=sample_ep))
    except SystemExit as exc:
        _log("样本 series 跳过：%s" % exc)
    payload["samples"] = samples
    payload["sample_run"] = preferred
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AGH 导航插件数据面（路线① 导航侧）")
    # required=False：不带子命令时打印速览（cmd_quick），不是报错
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("status", help="服务器/GPU/tmux/环境状态").set_defaults(func=cmd_status)
    sub.add_parser("runs", help="列出所有导航 run 与池化指标").set_defaults(func=cmd_runs)
    sub.add_parser("scenes", help="清单、场景、配置清单").set_defaults(func=cmd_scenes)
    sub.add_parser("board", help="五配置 × 8 场景对照表（面板用）").set_defaults(func=cmd_board)

    p = sub.add_parser("run-summary", help="单 run 逐场景指标 + 终止方式 + 触发统计")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_run_summary)

    p = sub.add_parser("series", help="单个回合的逐决策观测序列")
    p.add_argument("--run", required=True)
    p.add_argument("--scene", default=None)
    p.add_argument("--episode", required=True)
    p.set_defaults(func=cmd_series)

    p = sub.add_parser("failures", help="FailureEvent 契约事件流")
    p.add_argument("--run", required=True)
    p.add_argument("--limit", default=40)
    p.add_argument("--scene", default=None)
    p.add_argument("--episode", default=None)
    p.set_defaults(func=cmd_failures)

    p = sub.add_parser("triggers", help="触发质量 P@k / AUROC")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_triggers)

    p = sub.add_parser("recovery", help="两条通道在该 run 上会怎么动（离线复算）")
    p.add_argument("--run", required=True)
    p.add_argument("--scene", default=None)
    p.add_argument("--episode", default=None)
    p.add_argument("--channels", default="both", choices=["both", "collision", "progress", "none"])
    p.add_argument("--collision-min", type=int, default=None)
    p.add_argument("--progress-window", type=int, default=None)
    p.add_argument("--max-per-episode", type=int, default=None)
    p.set_defaults(func=cmd_recovery)

    p = sub.add_parser("progress", help="某个 run 跑到哪了")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_progress)

    p = sub.add_parser("launch", help="在 tmux 里起一次新的评估（会改服务器状态）")
    p.add_argument("--label", required=True)
    p.add_argument("--budget-mode", default="recovery_only")
    p.add_argument("--watchdog", default="off", choices=["on", "off"])
    p.add_argument("--channels", default="both", choices=["both", "collision", "progress", "none"])
    p.add_argument("--episodes", type=int, default=52)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--gpu", default=GPU)
    p.add_argument("--collision-min", type=int, default=5)
    p.add_argument("--progress-window", type=int, default=12)
    p.add_argument("--collision-max", type=int, default=8)
    p.add_argument("--periodic-interval", type=int, default=12)
    p.add_argument("--force", action="store_true")
    p.add_argument("--require-idle", action="store_true", default=True)
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("cache-all", help="一次性产出插件缓存")
    p.add_argument("--sample-run", default=None)
    p.set_defaults(func=cmd_cache_all)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) is None:
        print(json.dumps(cmd_quick(args), ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command in ("run-summary", "series", "failures", "triggers", "recovery", "progress") and not hasattr(
        args, "scene"
    ):
        args.scene = None
    if args.command in ("series", "failures", "recovery") and not hasattr(args, "episode"):
        args.episode = None
    try:
        payload = args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - 统一转成 JSON 错误，方便 JS 侧展示
        sys.stderr.write("[probe] %s: %s\n" % (type(exc).__name__, exc))
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

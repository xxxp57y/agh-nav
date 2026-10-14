#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路线① Agnes 视觉 Monitor —— 真实调用实测驱动脚本。

用途：在 omniServer001 上可复现地跑出「链路连通 / 3 组真实调用 / 稳定性 /
并发吞吐 / 是否真的看图 / 错误码」的全部数字，并落盘 JSON + Markdown 表格。

用法：
    cd /srv/agh-demo/src/route1/monitor/tests
    python3 agnes_realtest.py                 # 全部测试
    python3 agnes_realtest.py --only group1   # 只跑某一组（产出文件名带后缀，不覆盖全量结果）

产出：
    /srv/agh-demo/out/route1/agnes_realtest_results.json
    /srv/agh-demo/out/route1/agnes_realtest_table.md
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_MONITOR_DIR = os.path.dirname(_HERE)
_SRC_ROUTE1 = os.path.dirname(_MONITOR_DIR)
if _MONITOR_DIR not in sys.path:
    sys.path.insert(0, _MONITOR_DIR)

import agnes_monitor as am  # noqa: E402

ROOT = "/srv/agh-demo"
HAB = "/srv/agh-demo/InternNav/logs/habitat"
OUT_DIR = os.path.join(ROOT, "out", "route1")

# ---- 素材 ----------------------------------------------------------------

# 任务指定的那条 smoke：单帧当前观测
SMOKE_RUN = os.path.join(HAB, "budgeted_v20_50_long_smoke_20261009_seed20260826")
SMOKE_FRAME = os.path.join(SMOKE_RUN, "check_sim_0", "rgb_0.jpg")
INSTR_SMOKE = ("Walk into the living room and keep walking straight past the living room. "
               "Then walk into the entrance under the balcony. "
               "Wait in the entrance to the other room. ")

# 真实多帧轨迹：同一 episode（scene QUCTc6BB5sX / episode 19）的 22 张决策帧
TRAJ_DIR = os.path.join(HAB, "closed_loop_v03_full_20260825",
                        "early_replan_artifacts", "QUCTc6BB5sX_19")
INSTR_TRAJ = ("Walk through the living room around the backside of the couches. "
              "Walk into the kitchen area and walk along the barstools and countertop. "
              "Continue ahead towards the arched entryway that leads to a hall beside the stairs. ")

# 另一条完全不同场景的帧（内容差异极大，用于「是否真看图」）
OTHER_FRAME = os.path.join(HAB, "budgeted_v20_50_safe_drift_seed20260826",
                           "check_sim_0", "rgb_0.jpg")

RESULTS: List[Dict[str, Any]] = []
ROWS: List[Dict[str, Any]] = []


def log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def md5_12(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()[:12]


def pick_evenly(items: List[str], k: int) -> List[str]:
    """等距取 k 个（必含首尾）。"""
    if k >= len(items):
        return list(items)
    if k == 1:
        return [items[-1]]
    out = []
    for i in range(k):
        pos = int(round(i * (len(items) - 1) / float(k - 1)))
        out.append(items[pos])
    out[-1] = items[-1]
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen


def brief(res: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
    keys = keys or ["verdict", "confidence", "latency_ms", "http_status", "cache_hit", "n_images_sent"]
    return " ".join("%s=%s" % (k, res.get(k)) for k in keys if k in res)


def add_row(group: str, name: str, **kw) -> None:
    row = {"group": group, "name": name}
    row.update(kw)
    ROWS.append(row)


def record(tag: str, payload: Dict[str, Any]) -> None:
    RESULTS.append({"tag": tag, "ts": am.now_iso(), "payload": payload})


# --------------------------------------------------------------------------
# 素材装配
# --------------------------------------------------------------------------


def load_materials() -> Dict[str, Any]:
    traj = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.jpg")))
    mat: Dict[str, Any] = {
        "smoke_frame": SMOKE_FRAME,
        "smoke_frame_md5": md5_12(SMOKE_FRAME) if os.path.isfile(SMOKE_FRAME) else None,
        "traj_dir": TRAJ_DIR,
        "traj_n": len(traj),
        "traj": traj,
        "traj_first": traj[0] if traj else None,
        "traj_last": traj[-1] if traj else None,
        "f2": [traj[0], traj[-1]] if len(traj) >= 2 else traj,
        "f4": pick_evenly(traj, 4) if len(traj) >= 4 else traj,
        "f8": pick_evenly(traj, 8) if len(traj) >= 8 else traj,
        "other_frame": OTHER_FRAME,
        "other_frame_md5": md5_12(OTHER_FRAME) if os.path.isfile(OTHER_FRAME) else None,
        "instr_smoke": INSTR_SMOKE,
        "instr_traj": INSTR_TRAJ,
    }
    return mat


def print_materials(mat: Dict[str, Any]) -> None:
    log("== 素材 ==")
    log("  smoke 单帧      : %s (md5=%s)" % (mat["smoke_frame"], mat["smoke_frame_md5"]))
    log("  真实轨迹目录    : %s" % mat["traj_dir"])
    log("  轨迹帧数        : %d" % mat["traj_n"])
    log("  2 帧(首+尾)     : %s" % [os.path.basename(x) for x in mat["f2"]])
    log("  4 帧            : %s" % [os.path.basename(x) for x in mat["f4"]])
    log("  8 帧            : %s" % [os.path.basename(x) for x in mat["f8"]])
    log("  另一场景帧      : %s (md5=%s)" % (mat["other_frame"], mat["other_frame_md5"]))
    log("")


# --------------------------------------------------------------------------
# 各组测试
# --------------------------------------------------------------------------


def t_synthetic(mat: Dict[str, Any]) -> None:
    """0) PIL 合成示意房间图：验证链路 + JSON 可解析 + 语义敏感性。"""
    log("== [0] 合成图链路验证（PIL 画的示意房间图）==")
    syn_dir = os.path.join(OUT_DIR, "monitor_selftest")
    frames = am._make_synthetic_frames(syn_dir)
    by_name: Dict[str, str] = {}
    for f in frames:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem.startswith("synth_"):
            stem = stem[len("synth_"):]
        by_name[stem] = f
    log("  合成帧：%s" % {k: os.path.basename(v) for k, v in by_name.items()})

    # 当前观测 = 客厅 → 期望 PASS
    r_pass = am.monitor("Walk into the living room and stop near the sofa.",
                        [by_name["living_room"]], use_cache=False)
    log("  [客厅为当前观测] %s" % brief(r_pass))
    log("    evidence=%s" % json.dumps(r_pass["evidence"], ensure_ascii=False))
    record("synthetic_living_room", r_pass)
    add_row("0-合成", "客厅为当前观测", verdict=r_pass["verdict"],
            confidence=r_pass["confidence"], latency_ms=r_pass["latency_ms"],
            http_status=r_pass.get("http_status"), images=r_pass.get("n_images_sent"),
            note="expect PASS")

    # 当前观测 = 走廊 → 期望 LOST（同一指令）
    r_lost = am.monitor("Walk into the living room and stop near the sofa.",
                        [by_name["corridor"]], use_cache=False)
    log("  [走廊为当前观测] %s" % brief(r_lost))
    log("    evidence=%s" % json.dumps(r_lost["evidence"], ensure_ascii=False))
    record("synthetic_corridor", r_lost)
    add_row("0-合成", "走廊为当前观测", verdict=r_lost["verdict"],
            confidence=r_lost["confidence"], latency_ms=r_lost["latency_ms"],
            http_status=r_lost.get("http_status"), images=r_lost.get("n_images_sent"),
            note="expect LOST")
    log("")


def t_group1(mat: Dict[str, Any]) -> None:
    """① 单帧当前观测（真实帧，任务指定的 smoke）。"""
    log("== [1] 真实调用①  单帧当前观测 ==")
    log("  instruction(smoke, %d 字符): %s" % (len(INSTR_SMOKE), INSTR_SMOKE.strip()[:80] + "..."))
    res = am.monitor(INSTR_SMOKE, [mat["smoke_frame"]], use_cache=False)
    log("  %s" % brief(res, ["verdict", "confidence", "latency_ms", "api_latency_ms", "http_status", "n_images_sent"]))
    log("  usage=%s" % json.dumps(res.get("usage"), ensure_ascii=False))
    log("  reason_l1   = %s" % res["reason_l1"])
    log("  evidence    = %s" % json.dumps(res["evidence"], ensure_ascii=False))
    log("  l2_diagnosis= %s" % res["l2_diagnosis"])
    log("  subgoal     = %s" % res["subgoal_guess"])
    log("  response_format_used=%s attempts=%s" % (res.get("response_format_used"), res.get("attempts")))
    record("group1_single_frame", res)
    add_row("1-真实", "①单帧当前观测", frame="smoke rgb_0", verdict=res["verdict"],
            confidence=res["confidence"], latency_ms=res["latency_ms"],
            http_status=res.get("http_status"), images=res.get("n_images_sent"),
            tokens=json.dumps(res.get("usage"), ensure_ascii=False))
    log("")


def t_group2(mat: Dict[str, Any]) -> None:
    """② 2 帧历史 + 当前帧（同一 episode 的真实时间序列）。"""
    log("== [2] 真实调用②  2 帧（历史 + 当前）==")
    frames = mat["f2"]
    log("  frames = %s" % [os.path.basename(x) for x in frames])
    res = am.monitor(mat["instr_traj"], frames, use_cache=False)
    log("  %s" % brief(res, ["verdict", "confidence", "latency_ms", "api_latency_ms", "http_status", "n_images_sent"]))
    log("  usage=%s" % json.dumps(res.get("usage"), ensure_ascii=False))
    log("  reason_l1   = %s" % res["reason_l1"])
    log("  evidence    = %s" % json.dumps(res["evidence"], ensure_ascii=False))
    log("  l2_diagnosis= %s" % res["l2_diagnosis"])
    log("  subgoal     = %s" % res["subgoal_guess"])
    record("group2_two_frames", res)
    add_row("2-真实", "②2 帧(历史+当前)", frame="traj first+last", verdict=res["verdict"],
            confidence=res["confidence"], latency_ms=res["latency_ms"],
            http_status=res.get("http_status"), images=res.get("n_images_sent"),
            tokens=json.dumps(res.get("usage"), ensure_ascii=False))
    log("")


def t_group3(mat: Dict[str, Any]) -> None:
    """③ 8 帧上限（真实同 episode 轨迹）→ 必须下采样到 ≤4 张。"""
    log("== [3] 真实调用③  8 帧上限（验证 4 图硬上限的下采样）==")
    frames = mat["f8"]
    log("  输入帧数 = %d" % len(frames))
    log("  frames = %s" % [os.path.basename(x) for x in frames])
    res = am.monitor(mat["instr_traj"], frames, use_cache=False)
    log("  实际发送图片数 = %s（下标 %s）" % (res.get("n_images_sent"), res.get("sent_frame_indices")))
    log("  %s" % brief(res, ["verdict", "confidence", "latency_ms", "api_latency_ms", "http_status"]))
    log("  usage=%s" % json.dumps(res.get("usage"), ensure_ascii=False))
    log("  reason_l1   = %s" % res["reason_l1"])
    log("  evidence    = %s" % json.dumps(res["evidence"], ensure_ascii=False))
    log("  l2_diagnosis= %s" % res["l2_diagnosis"])
    log("  subgoal     = %s" % res["subgoal_guess"])
    record("group3_eight_frames", res)
    add_row("3-真实", "③8 帧上限", frame="traj 8 of 22", verdict=res["verdict"],
            confidence=res["confidence"], latency_ms=res["latency_ms"],
            http_status=res.get("http_status"), images=res.get("n_images_sent"),
            tokens=json.dumps(res.get("usage"), ensure_ascii=False))

    # 对照：正好 4 帧（等于上限，不下采样）
    log("  -- 对照：正好 4 帧（=上限，不下采样）--")
    res4 = am.monitor(mat["instr_traj"], mat["f4"], use_cache=False)
    log("  实际发送图片数 = %s" % res4.get("n_images_sent"))
    log("  %s" % brief(res4))
    log("  evidence = %s" % json.dumps(res4["evidence"], ensure_ascii=False))
    record("group3b_four_frames", res4)
    add_row("3-真实", "③b 正好 4 帧", frame="traj 4 of 22", verdict=res4["verdict"],
            confidence=res4["confidence"], latency_ms=res4["latency_ms"],
            http_status=res4.get("http_status"), images=res4.get("n_images_sent"),
            tokens=json.dumps(res4.get("usage"), ensure_ascii=False))
    log("")


def t_history(mat: Dict[str, Any]) -> None:
    """history 参数连通性：把前一次判决作为上下文再调一次。"""
    log("== [4] history 上下文连通性 ==")
    prev = am.monitor(mat["instr_traj"], [mat["f2"][0]], use_cache=False)
    log("  先跑第 1 帧 → verdict=%s conf=%.2f" % (prev["verdict"], prev["confidence"]))
    hist = [{"verdict": prev["verdict"], "confidence": prev["confidence"],
             "reason_l1": prev["reason_l1"], "subgoal_guess": prev["subgoal_guess"]}]
    cur = am.monitor(mat["instr_traj"], [mat["f2"][1]], history=hist, use_cache=False)
    log("  携带 history 再跑最后 1 帧 → verdict=%s conf=%.2f latency=%sms" % (
        cur["verdict"], cur["confidence"], cur["latency_ms"]))
    log("  usage=%s" % json.dumps(cur.get("usage"), ensure_ascii=False))
    log("  reason_l1 = %s" % cur["reason_l1"])
    record("history_context", {"prev": prev, "with_history": cur})
    add_row("4-history", "带历史上下文", frame="traj last", verdict=cur["verdict"],
            confidence=cur["confidence"], latency_ms=cur["latency_ms"],
            http_status=cur.get("http_status"), images=cur.get("n_images_sent"),
            tokens=json.dumps(cur.get("usage"), ensure_ascii=False))
    log("")


def t_stability(mat: Dict[str, Any]) -> None:
    """朴素一致性：同一批帧 ×3，temperature=0。"""
    log("== [5] 稳定性：同一批帧 ×3（temperature=0）==")
    frames = mat["f2"]
    runs = []
    for i in range(3):
        r = am.monitor(mat["instr_traj"], frames, temperature=0.0, use_cache=False)
        runs.append(r)
        log("  run%d: verdict=%-4s confidence=%.2f latency=%5dms reason=%s" % (
            i + 1, r["verdict"], r["confidence"], r["latency_ms"], r["reason_l1"][:40]))
        log("        evidence=%s" % json.dumps(r["evidence"], ensure_ascii=False))
        add_row("5-稳定性", "t=0 run%d" % (i + 1), frame="traj first+last",
                verdict=r["verdict"], confidence=r["confidence"],
                latency_ms=r["latency_ms"], http_status=r.get("http_status"),
                images=r.get("n_images_sent"), tokens=json.dumps(r.get("usage"), ensure_ascii=False))
    vs = [r["verdict"] for r in runs]
    cs = [r["confidence"] for r in runs]
    lat = [r["latency_ms"] for r in runs]
    log("  --> verdict 一致=%s (%s)；confidence 范围=[%.2f, %.2f] 极差=%.2f；延迟 %s ms" % (
        len(set(vs)) == 1, vs, min(cs), max(cs), max(cs) - min(cs), lat))
    record("stability_temp0", {"runs": runs, "verdicts": vs, "confidences": cs, "latencies_ms": lat})
    log("")


def t_image_sensitivity(mat: Dict[str, Any]) -> None:
    """关键验证：模型是否真的吃图（同指令 + 内容明显不同的图）。"""
    log("== [6] 关键验证：agnes-3.0-flash 是否真的吃图 ==")
    outs = []
    for name, frames in (("traj 首帧（立柱遮挡/岛台）", [mat["traj_first"]]),
                         ("traj 末帧（拱门/楼梯旁）", [mat["traj_last"]])):
        r = am.monitor(mat["instr_traj"], frames, use_cache=False)
        outs.append((name, r))
        log("  [%s] %s md5=%s" % (name, brief(r, ["verdict", "confidence", "latency_ms", "http_status"]),
                                  md5_12(frames[0])))
        log("      reason   = %s" % r["reason_l1"])
        log("      evidence = %s" % json.dumps(r["evidence"], ensure_ascii=False))
        log("      subgoal  = %s" % r["subgoal_guess"])
        add_row("6-看图", name, frame=os.path.basename(frames[0]), verdict=r["verdict"],
                confidence=r["confidence"], latency_ms=r["latency_ms"],
                http_status=r.get("http_status"), images=r.get("n_images_sent"),
                tokens=json.dumps(r.get("usage"), ensure_ascii=False))

    # 同一指令、两张内容差别极大的图 → 判决/证据必须跟着变
    log("  -- 同一指令 + 两张内容差别极大的图 --")
    a = am.monitor(INSTR_SMOKE, [mat["smoke_frame"]], use_cache=False)
    b = am.monitor(INSTR_SMOKE, [mat["other_frame"]], use_cache=False)
    for nm, r, f in (("A: smoke 帧", a, mat["smoke_frame"]),
                     ("B: safe_drift 帧", b, mat["other_frame"])):
        log("  %s md5=%s bytes=%d -> %s" % (
            nm, md5_12(f), os.path.getsize(f),
            brief(r, ["verdict", "confidence", "latency_ms", "http_status"])))
        log("      evidence = %s" % json.dumps(r["evidence"], ensure_ascii=False))
        log("      subgoal  = %s" % r["subgoal_guess"])
        add_row("6-看图", nm, frame=os.path.basename(f), verdict=r["verdict"],
                confidence=r["confidence"], latency_ms=r["latency_ms"],
                http_status=r.get("http_status"), images=r.get("n_images_sent"),
                tokens=json.dumps(r.get("usage"), ensure_ascii=False))
    same_verdict = a["verdict"] == b["verdict"]
    ev_overlap = len(set(a["evidence"]) & set(b["evidence"]))
    log("  --> 两图判决相同? %s (A=%s B=%s)；evidence 完全重合条数=%d" % (
        same_verdict, a["verdict"], b["verdict"], ev_overlap))
    record("image_sensitivity", {"traj_first": outs[0][1], "traj_last": outs[1][1],
                                 "smoke_A": a, "other_B": b,
                                 "same_verdict": same_verdict, "evidence_overlap": ev_overlap})
    log("")


def t_concurrency(mat: Dict[str, Any], n: int = 4) -> Dict[str, Any]:
    """并发吞吐：n 个并发请求，各自延迟 + 总墙钟。"""
    log("== [7] 并发 %d 吞吐（是否限流）==" % n)
    pool = mat["traj"][:n] if len(mat["traj"]) >= n else mat["traj"]
    while len(pool) < n:
        pool = pool + pool
    pool = pool[:n]

    def one(i: int) -> Dict[str, Any]:
        f = pool[i]
        t0 = time.time()
        try:
            r = am.monitor(mat["instr_traj"], [f], use_cache=False)
            return {"i": i, "frame": os.path.basename(f), "ok": True,
                    "latency_ms": r["latency_ms"], "wall_ms": int((time.time() - t0) * 1000),
                    "verdict": r["verdict"], "confidence": r["confidence"],
                    "http_status": r.get("http_status"), "usage": r.get("usage")}
        except Exception as exc:  # noqa: BLE001
            return {"i": i, "frame": os.path.basename(f), "ok": False,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                    "status": getattr(exc, "status", None),
                    "wall_ms": int((time.time() - t0) * 1000)}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        outs = list(ex.map(one, range(n)))
    wall_ms = int((time.time() - t0) * 1000)
    for o in outs:
        if o["ok"]:
            log("  #%d %-34s latency=%5dms wall=%5dms status=%s verdict=%s conf=%.2f" % (
                o["i"], o["frame"], o["latency_ms"], o["wall_ms"], o.get("http_status"),
                o["verdict"], o["confidence"]))
            add_row("7-并发", "并发%d #%d" % (n, o["i"]), frame=o["frame"], verdict=o["verdict"],
                    confidence=o["confidence"], latency_ms=o["latency_ms"],
                    http_status=o.get("http_status"), images=1,
                    tokens=json.dumps(o.get("usage"), ensure_ascii=False),
                    note="wall=%dms" % o["wall_ms"])
        else:
            log("  #%d %-34s FAILED wall=%dms error=%s" % (o["i"], o["frame"], o["wall_ms"], o["error"]))
            add_row("7-并发", "并发%d #%d" % (n, o["i"]), frame=o["frame"], verdict="ERROR",
                    note=o["error"][:120])
    lats = [o["latency_ms"] for o in outs if o["ok"]]
    statuses = [o.get("http_status") for o in outs if o["ok"]]
    summary = {
        "n": n, "wall_ms": wall_ms,
        "per_latency_ms": lats,
        "statuses": statuses,
        "n_429": sum(1 for s in statuses if s == 429),
        "n_ok": len(lats),
        "sum_latency_ms": sum(lats) if lats else 0,
    }
    log("  --> 并发 %d：总墙钟 %d ms；各请求延迟 %s ms；Σ延迟=%d ms；HTTP %s；429 数=%d" % (
        n, wall_ms, lats, summary["sum_latency_ms"], statuses, summary["n_429"]))
    record("concurrency_%d" % n, {"summary": summary, "calls": outs})
    log("")
    return summary


def t_probes(mat: Dict[str, Any]) -> None:
    """协议探针：response_format / temperature / 鉴权 / 超时。"""
    log("== [8] 协议与错误路径探针 ==")

    # 8.1 response_format 是否被接受（裸 HTTP，避免 monitor 内部降级掩盖真相）
    key = am.load_api_key()
    base = {
        "model": am.DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "只输出 JSON：{\"ok\": true}"}],
        "max_tokens": 32,
    }
    for label, payload in (
        ("带 response_format=json_object", dict(base, response_format={"type": "json_object"})),
        ("不带 response_format", dict(base)),
        ("带 temperature=0", dict(base, temperature=0)),
        ("带 temperature=0.7", dict(base, temperature=0.7)),
    ):
        try:
            resp, status, ms = am.post_chat(payload, key, timeout=60)
            txt = am.extract_content(resp)
            log("  [%s] HTTP %s (%d ms) content=%r" % (label, status, ms, txt[:60]))
            add_row("8-探针", label, verdict="HTTP %s" % status, latency_ms=ms,
                    http_status=status, note=repr(txt[:40]))
            record("probe_" + label.replace(" ", "_"), {"status": status, "ms": ms, "content": txt})
        except am.MonitorAPIError as exc:
            log("  [%s] 失败 status=%s: %s" % (label, exc.status, str(exc)[:200]))
            add_row("8-探针", label, verdict="ERROR", http_status=exc.status, note=str(exc)[:120])
            record("probe_" + label.replace(" ", "_"), {"status": exc.status, "error": str(exc)[:400]})

    # 8.2 错误 key → 期望 401（故意用明显假的值，且不带 sk- 前缀以免被密钥扫描误报）
    try:
        am.monitor("Walk to the kitchen.", [mat["smoke_frame"]], use_cache=False,
                   api_key="INVALID-PROBE-KEY-NOT-A-SECRET-0000")
        log("  [错误 key] 意外成功（不应发生）")
        add_row("8-探针", "错误 key", verdict="UNEXPECTED_OK")
        record("probe_bad_key", {"unexpected": "success"})
    except am.MonitorAPIError as exc:
        log("  [错误 key] 期望失败 → status=%s msg=%s" % (exc.status, str(exc)[:160]))
        add_row("8-探针", "错误 key", verdict="ERROR", http_status=exc.status, note="expect 401")
        record("probe_bad_key", {"status": exc.status, "error": str(exc)[:400]})
    except Exception as exc:  # noqa: BLE001
        log("  [错误 key] 其他异常：%r" % (exc,))
        record("probe_bad_key", {"other": repr(exc)})

    # 8.3 超时路径：不可路由主机
    orig = am.API_URL
    t0 = time.time()
    try:
        am.API_URL = "https://10.255.255.1/v1/chat/completions"
        am.monitor("Walk to the kitchen.", [mat["smoke_frame"]], use_cache=False, timeout=3.0)
        log("  [超时探针] 意外成功")
        record("probe_timeout_unroutable", {"unexpected": "success"})
    except am.MonitorAPIError as exc:
        outer_ms = int((time.time() - t0) * 1000)
        log("  [超时探针 不可路由主机 timeout=3s] 失败（符合预期）；monitor() 总耗时 %d ms（含内部重试）；%s" % (
            outer_ms, str(exc)[:200]))
        add_row("8-探针", "超时(不可路由,3s)", verdict="ERROR", latency_ms=outer_ms,
                note="monitor 总耗时含内部重试 %dms" % outer_ms)
        record("probe_timeout_unroutable", {"outer_ms": outer_ms, "error": str(exc)[:400]})
    except Exception as exc:  # noqa: BLE001
        log("  [超时探针] 其他异常：%r" % (exc,))
        record("probe_timeout_unroutable", {"other": repr(exc)})
    finally:
        am.API_URL = orig

    # 8.4 极短超时（同一真实端点）
    try:
        am.monitor("Walk to the kitchen.", [mat["smoke_frame"]], use_cache=False, timeout=0.001)
        log("  [极短超时 timeout=1ms] 意外成功")
        record("probe_timeout_1ms", {"unexpected": "success"})
    except am.MonitorAPIError as exc:
        log("  [极短超时 timeout=1ms] 失败（符合预期）：%s" % str(exc)[:160])
        add_row("8-探针", "超时(1ms)", verdict="ERROR", note=str(exc)[:100])
        record("probe_timeout_1ms", {"error": str(exc)[:400]})
    except Exception as exc:  # noqa: BLE001
        log("  [极短超时] 其他异常：%r" % (exc,))
        record("probe_timeout_1ms", {"other": repr(exc)})
    log("")


def t_cache(mat: Dict[str, Any]) -> None:
    """缓存 miss→hit 真实验证 + 备选模型可用性。"""
    log("== [9] 缓存 miss→hit + 备选模型 ==")
    # 用从未调用过的帧（traj[1], traj[2]），保证第 1 次是真 miss
    frames = mat["traj"][1:3] if len(mat["traj"]) >= 3 else mat["f2"]
    log("  帧 = %s（此前从未调用过）" % [os.path.basename(x) for x in frames])
    r1 = am.monitor(mat["instr_traj"], frames, use_cache=True)
    log("  第 1 次 cache_hit=%s latency=%sms verdict=%s conf=%.2f" % (
        r1["cache_hit"], r1["latency_ms"], r1["verdict"], r1["confidence"]))
    r2 = am.monitor(mat["instr_traj"], frames, use_cache=True)
    log("  第 2 次 cache_hit=%s latency=%sms verdict=%s conf=%.2f" % (
        r2["cache_hit"], r2["latency_ms"], r2["verdict"], r2["confidence"]))
    log("  判决一致=%s  cache_key=%s" % (
        r1["verdict"] == r2["verdict"], str(r1.get("cache_key"))[:24]))
    add_row("9-缓存", "第1次(真 miss)", verdict=r1["verdict"], confidence=r1["confidence"],
            latency_ms=r1["latency_ms"], http_status=r1.get("http_status"),
            note="cache_hit=%s" % r1["cache_hit"])
    add_row("9-缓存", "第2次(命中)", verdict=r2["verdict"], confidence=r2["confidence"],
            latency_ms=r2["latency_ms"], http_status=r2.get("http_status"),
            note="cache_hit=%s" % r2["cache_hit"])
    record("cache_miss_then_hit", {"first": r1, "second": r2})

    # 备选模型
    try:
        rf = am.monitor(mat["instr_traj"], [mat["traj_last"]], model=am.FALLBACK_MODEL,
                        use_cache=False)
        log("  [备选模型 %s] verdict=%s conf=%.2f latency=%sms status=%s" % (
            am.FALLBACK_MODEL, rf["verdict"], rf["confidence"], rf["latency_ms"], rf.get("http_status")))
        log("    usage=%s" % json.dumps(rf.get("usage"), ensure_ascii=False))
        log("    evidence=%s" % json.dumps(rf["evidence"], ensure_ascii=False))
        add_row("9-缓存", "备选模型 %s" % am.FALLBACK_MODEL, frame=os.path.basename(mat["traj_last"]),
                verdict=rf["verdict"], confidence=rf["confidence"], latency_ms=rf["latency_ms"],
                http_status=rf.get("http_status"), images=rf.get("n_images_sent"),
                tokens=json.dumps(rf.get("usage"), ensure_ascii=False))
        record("fallback_model", rf)
    except Exception as exc:  # noqa: BLE001
        log("  [备选模型 %s] 失败：%s: %s" % (am.FALLBACK_MODEL, type(exc).__name__, str(exc)[:300]))
        add_row("9-缓存", "备选模型 %s" % am.FALLBACK_MODEL, verdict="ERROR", note=str(exc)[:120])
        record("fallback_model", {"error": "%s: %s" % (type(exc).__name__, str(exc)[:400])})
    log("")


def t_stability_borderline(mat: Dict[str, Any]) -> None:
    """边界帧稳定性：语义含糊的单帧最容易翻判，各跑 3 次量化翻转率。"""
    log("== [11] 边界帧稳定性（语义含糊 vs 明确，各 ×3，temperature=0）==")
    cases = [
        ("含糊: smoke 帧(台球桌/大窗)", mat["smoke_frame"], INSTR_SMOKE),
        ("含糊: traj 首帧(立柱/门廊)", mat["traj_first"], INSTR_TRAJ),
        ("明确: traj 末帧(拱门/楼梯)", mat["traj_last"], INSTR_TRAJ),
    ]
    summary = []
    for name, frame, instr in cases:
        vs, cs, lats, evs = [], [], [], []
        log("  -- %s  md5=%s" % (name, md5_12(frame)))
        for i in range(3):
            r = am.monitor(instr, [frame], temperature=0.0, use_cache=False)
            vs.append(r["verdict"])
            cs.append(r["confidence"])
            lats.append(r["latency_ms"])
            evs.append(r["evidence"])
            log("     run%d verdict=%-4s conf=%.2f latency=%5dms reason=%s" % (
                i + 1, r["verdict"], r["confidence"], r["latency_ms"], r["reason_l1"][:36]))
            add_row("11-边界稳定性", "%s run%d" % (name.split(":")[0], i + 1),
                    frame=os.path.basename(frame), verdict=r["verdict"],
                    confidence=r["confidence"], latency_ms=r["latency_ms"],
                    http_status=r.get("http_status"), images=r.get("n_images_sent"),
                    tokens=json.dumps(r.get("usage"), ensure_ascii=False))
        stable = len(set(vs)) == 1
        log("     --> verdict=%s 稳定=%s；confidence=[%.2f, %.2f]；延迟 %s" % (
            vs, stable, min(cs), max(cs), lats))
        # 证据文本重合度：看每次给的视觉证据是否指向同一物体
        all_ev = [e for sub in evs for e in sub]
        summary.append({"case": name, "frame": os.path.basename(frame),
                        "verdicts": vs, "confidences": cs, "latencies_ms": lats,
                        "stable": stable, "evidence_flat": all_ev})
    n_unstable = sum(1 for s in summary if not s["stable"])
    log("  ==> %d/%d 组出现判决翻转" % (n_unstable, len(summary)))
    record("stability_borderline", summary)
    log("")


def t_flip(mat: Dict[str, Any], n: int = 5) -> None:
    """翻转率量化：3 类帧各 n 次、交错发车，逐次核对同一输入是否同判。"""
    log("== [12] 翻转率量化（每类帧 ×%d，交错发车，temperature=0）==" % n)
    cases = [
        ("A 含糊: smoke 帧(台球桌)", mat["smoke_frame"], INSTR_SMOKE),
        ("B 含糊: traj 首帧(立柱)", mat["traj_first"], INSTR_TRAJ),
        ("C 明确: traj 末帧(拱门)", mat["traj_last"], INSTR_TRAJ),
    ]
    acc: Dict[str, Dict[str, Any]] = {}
    for name, frame, instr in cases:
        acc[name] = {"frame": os.path.basename(frame), "md5": md5_12(frame),
                     "instr_sha16": hashlib.sha256(instr.encode("utf-8")).hexdigest()[:16],
                     "results": []}
    # 交错：外层轮次，内层三类帧 → 每类帧的 n 次调用分散在整个时间窗内
    for i in range(n):
        for name, frame, instr in cases:
            try:
                r = am.monitor(instr, [frame], temperature=0.0, use_cache=False)
                acc[name]["results"].append({"i": i, "verdict": r["verdict"],
                                             "confidence": r["confidence"],
                                             "latency_ms": r["latency_ms"],
                                             "ts": am.now_iso()})
            except Exception as exc:  # noqa: BLE001
                acc[name]["results"].append({"i": i, "verdict": "ERROR",
                                             "error": "%s: %s" % (type(exc).__name__, str(exc)[:120]),
                                             "ts": am.now_iso()})
    log("")
    for name, frame, instr in cases:
        a = acc[name]
        vs = [x["verdict"] for x in a["results"]]
        cs = [x["confidence"] for x in a["results"] if isinstance(x.get("confidence"), float)]
        lats = [x["latency_ms"] for x in a["results"] if x.get("latency_ms")]
        n_lost = vs.count("LOST")
        n_pass = vs.count("PASS")
        flip = min(n_lost, n_pass) / float(len(vs)) if vs else 0.0
        log("  %s  md5=%s" % (name, a["md5"]))
        log("    逐次: %s" % " ".join("%s@%.2f" % (x["verdict"], x.get("confidence", -1))
                                     for x in a["results"]))
        log("    --> PASS=%d LOST=%d 少数派占比=%.0f%%；confidence=[%s]；延迟 %s" % (
            n_pass, n_lost, flip * 100,
            ("%.2f, %.2f" % (min(cs), max(cs))) if cs else "n/a", lats))
        for j, x in enumerate(a["results"]):
            add_row("12-翻转率", "%s #%d" % (name.split()[0], j + 1), frame=a["frame"],
                    verdict=x["verdict"], confidence=x.get("confidence"),
                    latency_ms=x.get("latency_ms"))
        a["n_pass"] = n_pass
        a["n_lost"] = n_lost
        a["minority_ratio"] = flip
    record("flip_rate", acc)
    tot_min = sum(min(a["n_pass"], a["n_lost"]) for a in acc.values())
    tot = sum(len(a["results"]) for a in acc.values())
    log("  ==> 合计少数派 %d/%d 次（%.0f%%）；含糊帧 A/B 各 %s" % (
        tot_min, tot, 100.0 * tot_min / tot,
        ["PASS=%d LOST=%d" % (acc[n]["n_pass"], acc[n]["n_lost"]) for n in list(acc)[:2]]))
    log("")


def t_burst(mat: Dict[str, Any], n: int = 8) -> None:
    """额外：8 并发瞬时脉冲，看有没有 429。"""
    log("== [10] 额外：并发 %d 瞬时脉冲（限流观察）==" % n)
    frames = [mat["traj"][i % len(mat["traj"])] for i in range(n)]

    def one(i: int) -> Dict[str, Any]:
        t0 = time.time()
        try:
            r = am.monitor(mat["instr_traj"], [frames[i]], use_cache=False)
            return {"i": i, "ok": True, "latency_ms": r["latency_ms"],
                    "status": r.get("http_status"), "verdict": r["verdict"],
                    "wall_ms": int((time.time() - t0) * 1000)}
        except Exception as exc:  # noqa: BLE001
            return {"i": i, "ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:160]),
                    "status": getattr(exc, "status", None),
                    "wall_ms": int((time.time() - t0) * 1000)}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        outs = list(ex.map(one, range(n)))
    wall_ms = int((time.time() - t0) * 1000)
    for o in outs:
        log("  #%d ok=%s status=%s latency=%s wall=%sms %s" % (
            o["i"], o["ok"], o.get("status"), o.get("latency_ms"), o["wall_ms"],
            o.get("error", "")))
    lats = [o["latency_ms"] for o in outs if o["ok"]]
    statuses = [o.get("status") for o in outs]
    log("  --> 并发 %d：总墙钟 %d ms；延迟 %s；HTTP %s；429=%d；失败=%d" % (
        n, wall_ms, lats, statuses,
        sum(1 for s in statuses if s == 429), sum(1 for o in outs if not o["ok"])))
    record("burst_%d" % n, {"wall_ms": wall_ms, "calls": outs})
    add_row("10-脉冲", "并发%d" % n, verdict="-%d ok" % len(lats),
            latency_ms=(sum(lats) // len(lats)) if lats else None,
            note="wall=%dms statuses=%s" % (wall_ms, statuses))
    log("")


# --------------------------------------------------------------------------
# 收尾
# --------------------------------------------------------------------------


def write_outputs(mat: Dict[str, Any], extras: Dict[str, Any], suffix: str = "") -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "ts": am.now_iso(),
        "python": sys.version,
        "api_url": am.API_URL,
        "default_model": am.DEFAULT_MODEL,
        "prompt_version": am.PROMPT_VERSION,
        "max_frames": am.MAX_FRAMES,
        "max_images_per_request": am.MAX_IMAGES_PER_REQUEST,
        "materials": {k: v for k, v in mat.items() if k != "traj"},
        "traj_files": [os.path.basename(x) for x in mat.get("traj", [])],
        "results": RESULTS,
        "extras": extras,
    }
    jp = os.path.join(OUT_DIR, "agnes_realtest_results%s.json" % suffix)
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    log("写出 JSON: %s (%d bytes)" % (jp, os.path.getsize(jp)))

    cols = ["group", "name", "frame", "verdict", "confidence", "latency_ms", "http_status",
            "images", "note", "tokens"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in ROWS:
        cells = []
        for c in cols:
            v = r.get(c)
            v = "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")
            cells.append(v)
        lines.append("| " + " | ".join(cells) + " |")
    mp = os.path.join(OUT_DIR, "agnes_realtest_table%s.md" % suffix)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log("写出 Markdown 表: %s" % mp)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="只跑某一组：synthetic/group1/group2/group3/history/stability/image/probes/cache")
    ap.add_argument("--skip-slow", action="store_true")
    args = ap.parse_args(argv)

    mat = load_materials()
    print_materials(mat)

    groups = {
        "synthetic": lambda: t_synthetic(mat),
        "group1": lambda: t_group1(mat),
        "group2": lambda: t_group2(mat),
        "group3": lambda: t_group3(mat),
        "history": lambda: t_history(mat),
        "stability": lambda: t_stability(mat),
        "image": lambda: t_image_sensitivity(mat),
        "probes": lambda: t_probes(mat),
        "cache": lambda: t_cache(mat),
        "stability2": lambda: t_stability_borderline(mat),
        "flip": lambda: t_flip(mat, n=5),
    }
    order = ["synthetic", "group1", "group2", "group3", "history", "stability",
             "image", "probes", "cache", "stability2", "flip"]
    if args.only:
        order = [g for g in order if g == args.only]
        if not order:
            log("未知组：%s" % args.only)
            return 2

    extras: Dict[str, Any] = {}
    for g in order:
        try:
            groups[g]()
        except Exception as exc:  # noqa: BLE001
            log("!! 组 %s 抛异常：%r" % (g, exc))
            log(traceback.format_exc())
            record("EXCEPTION_" + g, {"error": repr(exc), "tb": traceback.format_exc()})

    if not args.only:
        try:
            extras["concurrency_4"] = t_concurrency(mat, 4)
        except Exception as exc:  # noqa: BLE001
            log("!! 并发 4 抛异常：%r" % (exc,))
            record("EXCEPTION_conc4", {"error": repr(exc)})
        if not args.skip_slow:
            try:
                t_burst(mat, 8)
            except Exception as exc:  # noqa: BLE001
                log("!! 脉冲 8 抛异常：%r" % (exc,))
                record("EXCEPTION_burst8", {"error": repr(exc)})

    write_outputs(mat, extras, suffix=("_" + args.only) if args.only else "")
    log("\n== 完成 ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""实验① 的正式跑法：几何判据 vs 语义判据，在**同一批决策点**上比 AUROC 与 P@k。

为什么用 AUROC 做主指标
--------------------
这批数据的正样本极稀（`no_action_within_1` 的基率约 1.5–3%），
按触发率对齐的 P@k 会被单点噪声支配（我们已经在几何判据上见到 P@1 恒为 0）。
AUROC 对类别不平衡稳健得多，所以**主指标用 AUROC，P@k 作为辅助**。

两阶段，刻意分开
--------------
1. `collect`：**唯一花钱的一步**。从带帧 run 里抽决策点，调 Agnes Monitor（多票 + 一致度），
   把结果写进缓存 JSONL。缓存按 `(run, scene, episode, action_step)` 去重，
   **重跑不再花钱**。
2. `analyze`：纯离线。读缓存，算
   - 语义风险 vs 标签的 AUROC
   - **几何分在完全相同的子集上**的 AUROC（公平对照，必须同子集）
   - 几何触发（N2）与语义触发（N7）各自的 P@k
   - `independent` / `mixed` 两种接法下决策层会选什么动作、消耗多少 System-2

抽样说明（必须写进报告）
----------------------
正样本太稀，若纯随机抽样，抽不到几个正例 → AUROC 无法估计。
所以默认用**分层抽样**：把"未来 k 步内出现无可动作"的决策点**全部**取上，
再补随机负样本。这会让样本的正例率高于真实分布，
因此：
  - 报告里同时给 `raw`（抽样内）与 `reweighted`（按真实基率重加权）两版 AUROC；
  - **同一子集内**的几何 vs 语义比较是**无偏**的（两者面对同一批点）。

Python 3.9 兼容。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..nav.monitor_eval import (
    DEFAULT_REPO_ROOT,
    build_frame_history,
    make_agnes_monitor,
    vote_monitor,
)
from ..nav.replay import label_future_failures, load_run

DEFAULT_CACHE = "/srv/agh-demo/logs/route1/monitor_cache_dataset.jsonl"


# --------------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------------- #


def _episode_index(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[Tuple[str, int], List[Dict[str, Any]]], Dict[int, int]]:
    """把决策按回合分组，**存行本身**（`build_frame_history` 要的是行，不是下标），
    同时给出"全局下标 → 回合内位置"的映射。"""
    per_episode: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    local: Dict[int, int] = {}
    for index, row in enumerate(rows):
        key = (str(row.get("scene_id")), int(row.get("episode_id") or -1))
        local[index] = len(per_episode.setdefault(key, []))
        per_episode[key].append(row)
    return per_episode, local


def select_points(
    run_dir: str,
    horizon: int = 3,
    n_negatives: int = 40,
    seed: int = 20261009,
    require_frames: bool = True,
) -> Dict[str, Any]:
    """分层抽样：所有可见正例 + 随机负例。"""
    run = load_run(run_dir)
    rows = run["system1"]
    labels = label_future_failures(rows)
    key_name = "no_action_within_%d" % horizon

    positives: List[int] = []
    negatives: List[int] = []
    censored = 0
    for index, (row, label) in enumerate(zip(rows, labels)):
        if require_frames and not row.get("rgb_path"):
            continue
        value = label.get(key_name)
        if value is None:
            censored += 1
            continue
        (positives if value else negatives).append(index)

    rng = np.random.RandomState(seed)
    rng.shuffle(negatives)
    chosen = sorted(positives + negatives[:n_negatives])
    return {
        "run_dir": run_dir,
        "horizon": horizon,
        "decisions_total": len(rows),
        "decisions_with_label": len(positives) + len(negatives),
        "positives_available": len(positives),
        "censored": censored,
        "selected": chosen,
        "positive_indices": positives,
    }


# --------------------------------------------------------------------------- #
# 阶段 1：收集（唯一花钱的一步）
# --------------------------------------------------------------------------- #


def load_cache(path: str) -> Dict[Tuple[str, str, int, Any], Dict[str, Any]]:
    cache: Dict[Tuple[str, str, int, Any], Dict[str, Any]] = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            cache[(record["run"], record["scene_id"], int(record["episode_id"]),
                   record["action_step"])] = record
    return cache


def collect(
    run_dir: str,
    cache_path: str = DEFAULT_CACHE,
    horizon: int = 3,
    n_negatives: int = 40,
    votes: int = 3,
    max_frames: int = 8,
    repo_root: str = DEFAULT_REPO_ROOT,
    model: Optional[str] = None,
    max_new: Optional[int] = None,
) -> Dict[str, Any]:
    plan = select_points(run_dir, horizon=horizon, n_negatives=n_negatives)
    run = load_run(run_dir)
    rows = run["system1"]
    instructions = run["instruction_by_episode"]
    per_episode, local_index = _episode_index(rows)

    cache = load_cache(cache_path)
    cache_key_run = run_dir.rstrip("/")
    monitor = make_agnes_monitor(model=model, use_cache=False, n_votes=votes)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    added = 0
    skipped = 0
    errors = 0
    with open(cache_path, "a", encoding="utf-8") as sink:
        for index in plan["selected"]:
            row = rows[index]
            key = (str(row.get("scene_id")), int(row.get("episode_id") or -1))
            identity = (cache_key_run, key[0], key[1], row.get("action_step"))
            if identity in cache:
                skipped += 1
                continue
            if max_new is not None and added >= max_new:
                break
            frames = build_frame_history(per_episode.get(key, []), local_index[index],
                                        max_frames, repo_root=repo_root)
            instruction = instructions.get(key) or ""
            try:
                raw = monitor(instruction, frames) or {}
            except Exception as exc:
                errors += 1
                raw = {"error": "%s: %s" % (type(exc).__name__, exc)}
            record = {
                "run": cache_key_run,
                "scene_id": key[0],
                "episode_id": key[1],
                "action_step": row.get("action_step"),
                "n_frames": len(frames),
                "geometric_score": row.get("feedback_score"),
                "geometric_drift": row.get("trajectory_drift"),
                "geometric_plan_age": row.get("plan_age"),
                "geometric_reason": row.get("replan_reason"),
                "verdict": raw.get("verdict"),
                "confidence": raw.get("confidence"),
                "agreement": raw.get("agreement"),
                "votes": raw.get("votes"),
                "vote_counts": raw.get("vote_counts"),
                "model": raw.get("model"),
                "latency_ms": raw.get("latency_ms"),
                "reason_l1": raw.get("reason_l1"),
                "error": raw.get("error"),
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            added += 1
            print("  [+%d] ep=%s step=%s score=%s -> %s conf=%s agree=%s lat=%sms"
                  % (added, record["episode_id"], record["action_step"],
                     record["geometric_score"], record["verdict"], record["confidence"],
                     record["agreement"], record["latency_ms"]))
    return {"plan": plan, "added": added, "skipped": skipped, "errors": errors,
            "cache_path": cache_path}


# --------------------------------------------------------------------------- #
# 阶段 2：分析（纯离线）
# --------------------------------------------------------------------------- #


def auroc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    mask = np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    for index in range(1, sorted_scores.size + 1):
        if index == sorted_scores.size or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = (start + 1 + index) / 2.0
            start = index
    rank_sum = ranks[labels > 0.5].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def semantic_risk_of(record: Dict[str, Any], min_agreement: float = 0.6) -> Optional[float]:
    from ..nav.feedback5 import semantic_risk
    return semantic_risk(record.get("verdict"), record.get("confidence"),
                         record.get("agreement"), min_agreement)


def analyze(
    run_dir: str,
    cache_path: str = DEFAULT_CACHE,
    horizons: Sequence[int] = (1, 3, 5),
    min_agreement: float = 0.6,
) -> Dict[str, Any]:
    run = load_run(run_dir)
    rows = run["system1"]
    labels = label_future_failures(rows)
    index_by_identity: Dict[Tuple[str, int, Any], Dict[str, Any]] = {}
    for row, label in zip(rows, labels):
        index_by_identity[(str(row.get("scene_id")), int(row.get("episode_id") or -1),
                           row.get("action_step"))] = label

    cache = load_cache(cache_path)
    key_run = run_dir.rstrip("/")
    records = [value for key, value in cache.items() if key[0] == key_run]

    # 真实基率（全体可见决策），用于重加权
    base_rates: Dict[int, Optional[float]] = {}
    for horizon in horizons:
        name = "no_action_within_%d" % horizon
        values = [label.get(name) for label in labels if label.get(name) is not None]
        base_rates[horizon] = float(np.mean([bool(v) for v in values])) if values else None

    report: Dict[str, Any] = {
        "run_dir": run_dir,
        "cached_records": len(records),
        "min_agreement": min_agreement,
        "true_base_rates": base_rates,
        "n_horizons": {},
    }

    for horizon in horizons:
        name = "no_action_within_%d" % horizon
        geom: List[float] = []
        sem: List[float] = []
        lab: List[float] = []
        trig: List[float] = []
        for record in records:
            label = index_by_identity.get((record["scene_id"], int(record["episode_id"]),
                                           record["action_step"]))
            if label is None or label.get(name) is None:
                continue
            risk = semantic_risk_of(record, min_agreement)
            geom.append(float(record["geometric_score"])
                        if record.get("geometric_score") is not None else np.nan)
            sem.append(risk if risk is not None else np.nan)
            lab.append(1.0 if label[name] else 0.0)
            trig.append(1.0 if (risk is not None and risk >= 0.70) else 0.0)
        geom_arr = np.asarray(geom, dtype=np.float64)
        sem_arr = np.asarray(sem, dtype=np.float64)
        lab_arr = np.asarray(lab, dtype=np.float64)
        trig_arr = np.asarray(trig, dtype=np.float64)
        n = int(lab_arr.size)
        positives = int(lab_arr.sum())
        entry: Dict[str, Any] = {
            "n": n,
            "positives": positives,
            "sample_positive_rate": (positives / n) if n else None,
            "auroc_geometric": auroc(lab_arr, geom_arr) if n else None,
            "auroc_semantic": auroc(lab_arr, -sem_arr) if n else None,  # 风险越大越像失败 → 取负
            "semantic_available_rate": float(np.mean(np.isfinite(sem_arr))) if n else None,
            "semantic_trigger_count": int(trig_arr.sum()) if n else 0,
            "semantic_trigger_rate": float(trig_arr.mean()) if n else None,
            "semantic_trigger_precision": (
                float(lab_arr[trig_arr > 0.5].mean()) if n and trig_arr.sum() > 0 else None
            ),
        }
        report["n_horizons"][str(horizon)] = entry

    # 判决与一致度分布
    verdicts = [str(r.get("verdict")).upper() for r in records if r.get("verdict")]
    agreements = [float(r["agreement"]) for r in records if r.get("agreement") is not None]
    latencies = [float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None]
    report["verdict_distribution"] = {
        "PASS": verdicts.count("PASS"), "LOST": verdicts.count("LOST"),
        "UNSURE": verdicts.count("UNSURE"), "missing": len(records) - len(verdicts),
    }
    if agreements:
        report["agreement"] = {
            "mean": float(np.mean(agreements)),
            "unanimous_rate": float(np.mean([1.0 if a >= 1.0 else 0.0 for a in agreements])),
            "below_gate_rate": float(np.mean([1.0 if a < min_agreement else 0.0 for a in agreements])),
        }
    if latencies:
        report["latency_ms"] = {
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "max": float(np.max(latencies)),
        }
    return report


def _cmd_collect(args: argparse.Namespace) -> int:
    result = collect(args.run, cache_path=args.cache, horizon=args.horizon,
                     n_negatives=args.negatives, votes=args.votes,
                     max_frames=args.max_frames, model=args.model, max_new=args.max_new)
    print(json.dumps({k: v for k, v in result.items() if k != "plan"}, ensure_ascii=False, indent=2))
    print(json.dumps(result["plan"], ensure_ascii=False, indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    report = analyze(args.run, cache_path=args.cache, min_agreement=args.min_agreement)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="几何判据 vs 语义判据（同子集 AUROC）")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect", help="抽样并调用 Monitor（唯一花钱的一步，有缓存）")
    p.add_argument("--run", required=True)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--negatives", type=int, default=40)
    p.add_argument("--votes", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--model", default=None)
    p.add_argument("--max-new", type=int, default=None, help="本次最多新增多少个点（控成本）")
    p.set_defaults(func=_cmd_collect)
    p = sub.add_parser("analyze", help="纯离线分析")
    p.add_argument("--run", required=True)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--min-agreement", type=float, default=0.6)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_analyze)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

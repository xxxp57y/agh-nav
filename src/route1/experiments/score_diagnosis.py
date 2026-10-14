"""诊断：几何加权分到底有没有信息量？阈值是不是只是"设错了"？

为什么必须做这个诊断
------------------
我们已经知道两件事：
  1. 早漂移触发的 P@1 = 0.0000；
  2. 几何分在参考 run 里 **从未越过 0.70**（1018 个决策里 0 个）。
这两件事有两种完全不同的解释，而它们指向**完全不同的下一步**：

  解释 A（阈值设错）：分数其实有信息，只是 0.70 这条线相对当前信号量级太高，
                      换个阈值/重新标定就能救。
  解释 B（分数无信息）：分数与"真失败"几乎独立，怎么调阈值都没用，
                      必须引入新信息（语义信号）。

本模块用**ROC/阈值扫描**把两者分开，并给出每个信号的可用率与贡献占比，
从而定位"分数为什么这么小"。

只依赖 numpy 与既有日志，Python 3.9 兼容。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..nav.replay import PRECURSOR_HORIZONS, label_future_failures, load_run

#: 与 `habitat_vln_evaluator.py:395-407` 一致的默认权重（日志里也有，优先用日志）
DEFAULT_WEIGHTS = {
    "uncertainty": 0.35,
    "trajectory_drift": 0.35,
    "plan_age": 0.15,
    "collision_risk": 0.15,
}


def auroc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """秩和法 AUROC（含并列处理）。"""
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


def _quantiles(values: np.ndarray) -> Dict[str, Optional[float]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: None for name in ("min", "p25", "p50", "p75", "p90", "p99", "max", "mean")}
    return {
        "min": float(finite.min()),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def diagnose_run(run_dir: str, horizon: int = 1) -> Dict[str, Any]:
    run = load_run(run_dir)
    rows = run["system1"]
    labels = label_future_failures(rows)
    config = run["closed_loop_config"] or {}
    weights = dict(DEFAULT_WEIGHTS)
    for name, value in (config.get("feedback") or {}).items():
        if name in weights and isinstance(value, (int, float)):
            weights[name] = float(value)

    scores = np.asarray(
        [r.get("feedback_score") if r.get("feedback_score") is not None else np.nan for r in rows],
        dtype=np.float64,
    )
    key = "no_action_within_%d" % horizon
    labels_arr = np.asarray(
        [1.0 if (label.get(key) is True) else 0.0 for label in labels], dtype=np.float64
    )
    observed = np.asarray([label.get(key) is not None for label in labels], dtype=bool)

    # ---- 逐信号可用率与归一化分量分布 ---- #
    signal_stats: Dict[str, Any] = {}
    contrib_sums = {name: 0.0 for name in weights}
    avail_counts = {name: 0 for name in weights}
    for row in rows:
        availability = row.get("availability") or {}
        normalized = row.get("normalized_components") or {}
        active_weights = row.get("active_weights") or {}
        denominator = sum(active_weights.values()) or 1.0
        for name in weights:
            if availability.get(name):
                avail_counts[name] += 1
                value = normalized.get(name)
                if value is not None:
                    contrib_sums[name] += float(weights[name]) * float(value) / denominator
    total = len(rows) or 1
    for name in weights:
        column = np.asarray(
            [
                (r.get("normalized_components") or {}).get(name)
                if (r.get("availability") or {}).get(name) else np.nan
                for r in rows
            ],
            dtype=np.float64,
        )
        signal_stats[name] = {
            "weight": weights[name],
            "available_rate": avail_counts[name] / float(total),
            "normalized": _quantiles(column),
            "mean_contribution_to_score": contrib_sums[name] / float(total),
        }

    # ---- 分母（活跃权重）分布：真正决定"分数能有多大"的是它 ---- #
    # 注意：score 本身是 Σ(w·n)/Σ(w|可用)，所以"理论上限"恒为 1.0；
    # 有意义的是**分母**——可用信号占总权重的比例，以及**哪些信号根本不可用**。
    active_weights_seen = np.asarray(
        [
            sum(
                (row.get("active_weights") or {}).values()
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    unavailable: Dict[str, int] = {}
    for row in rows:
        availability = row.get("availability") or {}
        for name in weights:
            if not availability.get(name):
                unavailable[name] = unavailable.get(name, 0) + 1

    # ---- 阈值扫描 ---- #
    sweep: List[Dict[str, Any]] = []
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        triggered = np.isfinite(scores) & (scores >= threshold)
        count = int(triggered.sum())
        entry: Dict[str, Any] = {"threshold": threshold, "triggers": count,
                                 "trigger_rate": count / float(total)}
        if count:
            labelled = labels_arr[triggered][observed[triggered]]
            entry["precision_at_h%d" % horizon] = float(labelled.mean()) if labelled.size else None
            entry["observed"] = int(labelled.size)
        sweep.append(entry)

    # ---- 用"达到同样触发次数"的方式扫分位阈值 ---- #
    percentile_sweep: List[Dict[str, Any]] = []
    finite_scores = scores[np.isfinite(scores)]
    for rate in (0.01, 0.02, 0.05, 0.10, 0.20):
        if finite_scores.size == 0:
            continue
        cut = float(np.quantile(finite_scores, 1.0 - rate))
        triggered = np.isfinite(scores) & (scores >= cut)
        labelled = labels_arr[triggered][observed[triggered]]
        percentile_sweep.append(
            {
                "target_rate": rate,
                "cut": cut,
                "triggers": int(triggered.sum()),
                "precision_at_h%d" % horizon: float(labelled.mean()) if labelled.size else None,
                "observed": int(labelled.size),
            }
        )

    return {
        "run_dir": run_dir,
        "mode": config.get("mode"),
        "decisions": len(rows),
        "positives_h%d" % horizon: int(labels_arr[observed].sum()),
        "base_rate_h%d" % horizon: float(labels_arr[observed].mean()) if observed.any() else None,
        "score": _quantiles(scores),
        "auroc_score_vs_failure": auroc(labels_arr[observed], scores[observed]),
        "active_weight_total": _quantiles(active_weights_seen),
        "unavailable_counts": unavailable,
        "unavailable_rates": {name: count / float(total) for name, count in unavailable.items()},
        "unnormalized_weight_total": float(sum(weights.values())),
        "signal_stats": signal_stats,
        "threshold_sweep": sweep,
        "rate_matched_sweep": percentile_sweep,
        "max_score_seen": float(np.nanmax(scores)) if np.isfinite(scores).any() else None,
    }


def _cmd_run(args: argparse.Namespace) -> int:
    report = diagnose_run(args.run, horizon=args.horizon)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("signal_stats",)}, ensure_ascii=False, indent=2))
    print("-- 逐信号 --")
    print("  %-18s %6s %10s %10s %10s %14s" % ("signal", "weight", "avail", "p50(norm)", "p90(norm)", "mean贡献"))
    for name, stat in report["signal_stats"].items():
        norm = stat["normalized"]
        print("  %-18s %6.2f %9.1f%% %10s %10s %14.5f" % (
            name, stat["weight"], 100 * stat["available_rate"],
            "-" if norm["p50"] is None else "%.4f" % norm["p50"],
            "-" if norm["p90"] is None else "%.4f" % norm["p90"],
            stat["mean_contribution_to_score"]))
    print("-- 不可用信号占比（这就是分数被压小的原因） --")
    for name, rate in sorted(report["unavailable_rates"].items()):
        print("  %-18s 不可用 %.1f%%" % (name, 100 * rate))
    print("-- 活跃权重（分母）分布 --")
    print("  ", json.dumps(report["active_weight_total"], ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


def _cmd_table(args: argparse.Namespace) -> int:
    rows = [diagnose_run(run_dir, horizon=args.horizon) for run_dir in args.run]

    def f(value: Any) -> str:
        return "-" if value is None else "%.4f" % value

    print("%-54s %6s %5s %8s %8s %8s %8s %8s"
          % ("run", "decis", "pos", "AUROC", "max分", "p99分", ">=0.70", "活跃权重p50"))
    for row in rows:
        key = "positives_h%d" % args.horizon
        ge70 = [item for item in row["threshold_sweep"] if item["threshold"] == 0.70]
        print("%-54s %6d %5d %8s %8s %8s %8d %8s"
              % (os.path.basename(row["run_dir"])[:54],
                 row["decisions"], row.get(key, 0),
                 f(row["auroc_score_vs_failure"]),
                 f(row["max_score_seen"]),
                 f((row["score"] or {}).get("p99")),
                 ge70[0]["triggers"] if ge70 else -1,
                 f((row.get("active_weight_total") or {}).get("p50"))))
    print()
    print("-- 同触发率下的精度（这时阈值是按分位选的，不是 0.70） --")
    print("%-54s %8s %8s %8s %8s %8s"
          % ("run", "@1%", "@2%", "@5%", "@10%", "@20%"))
    for row in rows:
        cells = {}
        for item in row["rate_matched_sweep"]:
            cells[item["target_rate"]] = item.get("precision_at_h%d" % args.horizon)
        print("%-54s %8s %8s %8s %8s %8s"
              % (os.path.basename(row["run_dir"])[:54],
                 f(cells.get(0.01)), f(cells.get(0.02)), f(cells.get(0.05)),
                 f(cells.get(0.10)), f(cells.get(0.20))))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="几何分信息量诊断")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="单个 run 的完整诊断")
    p.add_argument("--run", required=True)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_run)
    p = sub.add_parser("table", help="多个 run 的对照表")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_table)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

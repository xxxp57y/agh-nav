"""实验二：前兆预测器 —— 把「失败前 N 步」变成可预测的信号。

动机（route1-bridge-plan.md §4-2 / failure-taxonomy.md §8-2）
-----------------------------------------------------------
你们已经有**前兆标注**这套机制（`counterfactual.py:430-436` 把 no-action 发生前
`precursor_window` 个决策回溯标成 `pre_no_executable_action_<offset>`），但那批标注
只活在"分析产物"里，没有被当成**训练目标**用起来。

同时，你们自己的报表给出了代价：**208 次主动重规划里 70 次有害 → P(Harmful)=0.337**，
而触发那一刻的指向性 **P@1 = 0.0000**。也就是说现在的决策层是"事后救火 + 盲选"。

本模块做的是最小可用的替代品：**用决策时刻可见的特征，预测"未来 k 个决策内会出现
无可执行动作"**，并与现有几何判据在同一批决策点上比 P@k。

诚实性声明（必须写进报告）
------------------------
1. 训练/评测用的是**完整决策分布**（50 回合 run 的全部 `system1` 决策），不是上游
   反事实数据集的**分层采样分布**——因此这里的数字与 `analysis_results/*` 不可直接横比，
   但**可以和我自己复算的几何判据 P@k 横比**（同一分布、同一标签、同一批点）。
2. 特征只用**当前及历史**决策可见的量（无标签泄漏）；`has_executable_action` 本身
   **不作为特征**（它就是标签的构成部分）。
3. 交叉验证按 group 分组。**实测发现这批 50 回合的 episode 全部来自同一个场景**
   `2azQ1b91cZZ`，因此场景级留一交叉验证不可能成立，`auto` 会自动退到留一 episode，
   并在报告里写明这一点（这是 manifest 的局限，不是模型的）。

只依赖 numpy，Python 3.9 兼容。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..nav.replay import PRECURSOR_HORIZONS, label_future_failures, load_run

#: 直接用日志里就有的当前时刻特征（不含任何未来信息）
CURRENT_FEATURES: Tuple[str, ...] = (
    "feedback_score",
    "trajectory_drift",
    "plan_age",
    "collision_risk",
    "uncertainty",
    "steps_since_last_s2",
    "distance_to_goal",
    "collision_count",
    "system2_call_count",
    "action_step",
)

#: 在历史窗口上做时序聚合得到的特征（对应他们 22 维设计里的"趋势"部分）
HISTORY_WINDOW = 4


def _num(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def build_features(
    rows: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """构造特征矩阵。时序聚合都在**回合内**做，跨回合不串。"""
    feature_names: List[str] = list(CURRENT_FEATURES)
    for base in ("trajectory_drift", "feedback_score", "plan_age"):
        for stat in ("mean", "max", "slope"):
            feature_names.append("%s_%s_%d" % (base, stat, HISTORY_WINDOW))
    feature_names += ["drift_delta_1", "drift_delta_3", "score_delta_1", "age_ratio"]

    matrix: List[List[float]] = []
    scenes: List[str] = []
    episode_history: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    for row, _label in zip(rows, labels):
        key = (str(row.get("scene_id")), int(row.get("episode_id") or -1))
        history = episode_history.setdefault(key, [])
        window = history[-HISTORY_WINDOW:]
        values: List[float] = [_num(row.get(name)) for name in CURRENT_FEATURES]
        for base in ("trajectory_drift", "feedback_score", "plan_age"):
            series = np.array([_num(item.get(base)) for item in window], dtype=np.float64)
            finite = series[np.isfinite(series)]
            values.append(float(np.mean(finite)) if finite.size else np.nan)
            values.append(float(np.max(finite)) if finite.size else np.nan)
            if finite.size >= 2:
                x = np.arange(finite.size, dtype=np.float64)
                values.append(float(np.polyfit(x, finite, 1)[0]))
            else:
                values.append(np.nan)
        drift_now = _num(row.get("trajectory_drift"))
        drift_prev = _num(history[-1].get("trajectory_drift")) if history else np.nan
        drift_prev3 = _num(history[-3].get("trajectory_drift")) if len(history) >= 3 else np.nan
        score_now = _num(row.get("feedback_score"))
        score_prev = _num(history[-1].get("feedback_score")) if history else np.nan
        age = _num(row.get("plan_age"))
        values.append(drift_now - drift_prev)
        values.append(drift_now - drift_prev3)
        values.append(score_now - score_prev)
        values.append(age / 20.0)
        matrix.append(values)
        scenes.append(key[0])
        history.append(row)

    return np.asarray(matrix, dtype=np.float64), feature_names, np.asarray(scenes)


def _impute_and_standardize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """用训练折的中位数填补缺失，再标准化（**只用训练折统计量**，避免泄漏）。"""
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train = np.where(np.isfinite(train), train, median)
    test = np.where(np.isfinite(test), test, median)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (train - mean) / std, (test - mean) / std


def _add_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    lr: float = 0.2,
    iters: int = 2000,
    class_weight: bool = True,
) -> np.ndarray:
    """带 L2 的 logistic 回归（numpy 手写）。

    服务器实测：`sklearn` 与 `pandas` **都不可用**（只有 numpy/scipy/torch），
    所以这里不能依赖它们，也不能依赖 `scipy.optimize` 的便利封装。
    """
    x = _add_bias(x)
    n = x.shape[0]
    if class_weight:
        positives = max(1.0, float(y.sum()))
        negatives = max(1.0, float(n - y.sum()))
        w = np.where(y > 0.5, n / (2.0 * positives), n / (2.0 * negatives))
    else:
        w = np.ones(n, dtype=np.float64)
    theta = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(iters):
        z = x @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = (x.T @ (w * (p - y))) / n
        grad[:-1] += l2 * theta[:-1] / n
        theta -= lr * grad
    return theta


def predict_proba(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    z = _add_bias(x) @ theta
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auroc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    pos = score[y > 0.5]
    neg = score[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    sorted_score = score[order]
    start = 0
    for index in range(1, sorted_score.size + 1):
        if index == sorted_score.size or sorted_score[index] != sorted_score[start]:
            if index - start > 1:
                ranks[order[start:index]] = (start + 1 + index) / 2.0
            start = index
    rank_sum = ranks[y > 0.5].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def auprc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    order = np.argsort(-score)
    y = y[order]
    if y.sum() == 0:
        return None
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    recall = tp / y.sum()
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(precision, recall))
    return float(np.trapz(precision, recall))


def precision_at_rate(labels: np.ndarray, score: np.ndarray, rate: float) -> Optional[float]:
    """在**触发率 = rate** 的工作点上算精度（保证与几何判据的触发次数可比）。"""
    if labels.size == 0 or rate <= 0:
        return None
    k = max(1, int(round(rate * labels.size)))
    k = min(k, labels.size)
    order = np.argsort(-score)[:k]
    return float(labels[order].mean())


def group_folds(groups: np.ndarray, n_folds: int = 5, seed: int = 20261009) -> List[np.ndarray]:
    """按 group 分组的 K 折（同一 group 不会同时出现在训练和测试里）。"""
    unique = sorted(set(groups.tolist()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    buckets: List[List[str]] = [[] for _ in range(n_folds)]
    for index, group in enumerate(unique):
        buckets[index % n_folds].append(group)
    return [np.isin(groups, np.asarray(bucket)) for bucket in buckets]


def episode_groups(
    scenes: np.ndarray,
    episodes: np.ndarray,
    group_by: str = "auto",
    n_folds: int = 5,
) -> Tuple[np.ndarray, str, str]:
    """决定交叉验证的**分组键**，并解释为什么。

    实测发现（重要）：`budgeted_v20_50_*_seed20260826` 这批 run 的 50 个 episode
    **全部来自同一个场景** `2azQ1b91cZZ`（`progress.json` 与 `system1` 事件都一致）。
    因此"留一场景交叉验证"在这份 manifest 上**根本无法成立**——若强行按 scene 分组，
    会退化成"训练集无正样本 / 测试集无正样本"。`auto` 在这种情况下自动退到**留一 episode**，
    并在报告里写明：**场景级泛化无法用这份 manifest 度量**。

    要真正度量场景泛化，需要一个跨场景的 episode 清单——这是本轮暴露出来的一条待办。
    """
    distinct_scenes = sorted(set(scenes.tolist()))
    if group_by == "scene" or (group_by == "auto" and len(distinct_scenes) >= n_folds):
        return scenes, "scene", "按场景分组（%d 个场景）" % len(distinct_scenes)
    return episodes, "episode", (
        "按 episode 分组（**只有 %d 个场景，无法做场景级交叉验证**；这批 50 回合全部来自 %s）"
        % (len(distinct_scenes), distinct_scenes[:3])
    )


def resolve_scene_ids(run: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """补齐 `system1` 事件里缺失的 `scene_id`，并报告 episode_id 是否唯一。"""
    rows = run["system1"]
    missing = sum(1 for row in rows if not row.get("scene_id"))
    mapping: Dict[int, str] = {}
    collisions = 0
    for row in run["progress"]:
        episode = int(row.get("episode_id") or -1)
        scene = str(row.get("scene_id"))
        if episode in mapping and mapping[episode] != scene:
            collisions += 1
        mapping[episode] = scene
    if missing:
        patched = []
        for row in rows:
            if row.get("scene_id"):
                patched.append(row)
                continue
            row = dict(row)
            row["scene_id"] = mapping.get(int(row.get("episode_id") or -1), "unknown")
            patched.append(row)
        rows = patched
    return rows, {
        "rows_missing_scene_id": missing,
        "episode_id_collisions": collisions,
        "episode_id_unique": collisions == 0,
        "scenes_in_manifest": len(set(mapping.values())) if mapping else None,
    }


def _empty_result(run_dirs: Sequence[str], horizon: int, feature_names: List[str],
                  join_info: Dict[str, Any], note: str) -> Dict[str, Any]:
    return {
        "run_dirs": list(run_dirs),
        "horizon": horizon,
        "decisions": 0,
        "positives": 0,
        "base_rate": None,
        "scenes": 0,
        "oof_auroc": None,
        "oof_auprc": None,
        "folds": [],
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "scene_id_join": join_info,
        "cv_group": "n/a",
        "cv_note": note,
        "geometric_early_drift": {"triggers": 0, "trigger_rate": 0.0,
                                  "precision_at_trigger": None},
        "any_replan": {"triggers": 0, "trigger_rate": 0.0, "precision_at_trigger": None},
    }


def train_and_evaluate(
    run_dirs: Sequence[str],
    horizon: int = 1,
    n_folds: int = 5,
    l2: float = 1.0,
    group_by: str = "auto",
) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    all_labels: List[Dict[str, Any]] = []
    join_info: Dict[str, Any] = {}
    for run_dir in run_dirs:
        run = load_run(run_dir)
        rows, info = resolve_scene_ids(run)
        join_info[os.path.basename(run_dir.rstrip("/"))] = info
        all_labels.extend(label_future_failures(rows))
        all_rows.extend(rows)

    x, feature_names, scenes = build_features(all_rows, all_labels)
    key = "no_action_within_%d" % horizon
    mask = np.asarray([label.get(key) is not None for label in all_labels], dtype=bool)
    if x.shape[0] == 0 or not mask.any():
        return _empty_result(run_dirs, horizon, feature_names, join_info, "无可用决策点")

    x = x[mask]
    scenes = scenes[mask]
    episodes = np.asarray(
        ["%s:%s" % (row.get("scene_id"), row.get("episode_id"))
         for row, keep in zip(all_rows, mask) if keep]
    )
    y = np.asarray([1.0 if label[key] else 0.0 for label, keep in zip(all_labels, mask) if keep])
    if y.sum() < 2:
        return _empty_result(run_dirs, horizon, feature_names, join_info,
                             "正样本少于 2 个，无法训练")

    groups, cv_group, cv_note = episode_groups(scenes, episodes, group_by, n_folds)
    folds = group_folds(groups, n_folds)

    scores = np.full(y.size, np.nan, dtype=np.float64)
    per_fold: List[Dict[str, Any]] = []
    for index, test_mask in enumerate(folds):
        train_mask = ~test_mask
        if y[train_mask].sum() < 2 or y[test_mask].sum() < 1:
            per_fold.append(
                {
                    "fold": index,
                    "skipped": True,
                    "train_pos": float(y[train_mask].sum()),
                    "test_pos": float(y[test_mask].sum()),
                    "groups": sorted(set(groups[test_mask].tolist()))[:8],
                }
            )
            continue
        x_train, x_test = _impute_and_standardize(x[train_mask], x[test_mask])
        theta = fit_logistic(x_train, y[train_mask], l2=l2)
        scores[test_mask] = predict_proba(theta, x_test)
        per_fold.append(
            {
                "fold": index,
                "train": int(train_mask.sum()),
                "test": int(test_mask.sum()),
                "train_pos": int(y[train_mask].sum()),
                "test_pos": int(y[test_mask].sum()),
                "auroc": auroc(y[test_mask], scores[test_mask]),
                "auprc": auprc(y[test_mask], scores[test_mask]),
                "groups": sorted(set(groups[test_mask].tolist()))[:8],
            }
        )

    valid = np.isfinite(scores)
    geometric_trigger = np.asarray(
        [label.get("reason") == "early_drift" for label, keep in zip(all_labels, mask) if keep]
    )
    any_replan = np.asarray(
        [bool(label.get("should_replan")) for label, keep in zip(all_labels, mask) if keep]
    )

    result: Dict[str, Any] = {
        "run_dirs": list(run_dirs),
        "horizon": horizon,
        "decisions": int(y.size),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "scenes": int(len(set(scenes.tolist()))),
        "folds": per_fold,
        "oof_auroc": auroc(y[valid], scores[valid]) if valid.any() else None,
        "oof_auprc": auprc(y[valid], scores[valid]) if valid.any() else None,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "scene_id_join": join_info,
        "cv_group": cv_group,
        "cv_note": cv_note,
        "cv_folds_requested": n_folds,
    }

    if not valid.any():
        result["note"] = "所有折都被跳过（正样本不足）"
        for name, trigger in (("geometric_early_drift", geometric_trigger),
                              ("any_replan", any_replan)):
            count = int(trigger.sum())
            result[name] = {
                "triggers": count,
                "trigger_rate": count / float(y.size) if y.size else 0.0,
                "precision_at_trigger": float(y[trigger].mean()) if count else None,
            }
        return result

    for name, trigger in (("geometric_early_drift", geometric_trigger),
                          ("any_replan", any_replan)):
        trigger_count = int(trigger.sum())
        rate = trigger_count / float(y.size) if y.size else 0.0
        entry: Dict[str, Any] = {
            "triggers": trigger_count,
            "trigger_rate": rate,
            "precision_at_trigger": float(y[trigger].mean()) if trigger_count else None,
        }
        if trigger_count:
            entry["geometric_hits"] = int(y[trigger].sum())
            entry["predictor_precision_at_matched_rate"] = precision_at_rate(
                y[valid], scores[valid], rate * valid.mean()
            )
            k = max(1, int(round(rate * valid.sum())))
            top = np.argsort(-scores[valid])[:k]
            entry["predictor_hits"] = int(y[valid][top].sum())
        result[name] = entry

    for rate in (0.01, 0.02, 0.05, 0.10):
        result["predictor_precision_at_rate_%.2f" % rate] = precision_at_rate(
            y[valid], scores[valid], rate
        )
    return result


def _cmd_single(args: argparse.Namespace) -> int:
    result = train_and_evaluate(
        args.run, horizon=args.horizon, n_folds=args.folds, l2=args.l2, group_by=args.group_by
    )
    print(json.dumps({k: v for k, v in result.items() if k != "folds"},
                     ensure_ascii=False, indent=2))
    print("-- 逐折 --")
    for fold in result["folds"]:
        print("  ", json.dumps(fold, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


def _cmd_table(args: argparse.Namespace) -> int:
    """在所有 50 回合 run 上跑一遍，出一张对照表。"""
    rows = []
    for run_dir in args.run:
        for horizon in args.horizons:
            result = train_and_evaluate(
                [run_dir], horizon=horizon, n_folds=args.folds, l2=args.l2,
                group_by=args.group_by,
            )
            rows.append(
                {
                    "run": os.path.basename(run_dir.rstrip("/")),
                    "horizon": horizon,
                    "decisions": result["decisions"],
                    "positives": result["positives"],
                    "base_rate": result["base_rate"],
                    "auroc": result["oof_auroc"],
                    "auprc": result["oof_auprc"],
                    "geometric_triggers": result["geometric_early_drift"]["triggers"],
                    "geometric_precision": result["geometric_early_drift"]["precision_at_trigger"],
                    "predictor_precision_matched": result["geometric_early_drift"].get(
                        "predictor_precision_at_matched_rate"
                    ),
                    "any_replan_triggers": result["any_replan"]["triggers"],
                    "any_replan_precision": result["any_replan"]["precision_at_trigger"],
                    "predictor_precision_matched_any": result["any_replan"].get(
                        "predictor_precision_at_matched_rate"
                    ),
                }
            )

    def f(value: Any) -> str:
        if value is None:
            return "-"
        return "%.4f" % value if isinstance(value, float) else str(value)

    print("%-56s %2s %6s %5s %7s %7s %7s %6s %7s %7s %6s %7s %7s"
          % ("run", "h", "decis", "pos", "base", "AUROC", "AUPRC",
             "geoN", "geoP", "predP", "replN", "replP", "predP2"))
    for row in rows:
        print("%-56s %2d %6d %5d %7s %7s %7s %6d %7s %7s %6d %7s %7s"
              % (row["run"][:56], row["horizon"], row["decisions"], row["positives"],
                 f(row["base_rate"]), f(row["auroc"]), f(row["auprc"]),
                 row["geometric_triggers"], f(row["geometric_precision"]),
                 f(row["predictor_precision_matched"]),
                 row["any_replan_triggers"], f(row["any_replan_precision"]),
                 f(row["predictor_precision_matched_any"])))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="前兆预测器：预测未来 k 个决策内是否出现无可执行动作"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("single", _cmd_single), ("table", _cmd_table)):
        p = sub.add_parser(name)
        p.add_argument("--run", action="append", required=True)
        p.add_argument("--folds", dest="folds", type=int, default=5)
        p.add_argument("--l2", type=float, default=1.0)
        p.add_argument("--group-by", dest="group_by", default="auto",
                       choices=["auto", "scene", "episode"])
        p.add_argument("--horizon", type=int, default=1)
        p.add_argument("--horizons", type=int, nargs="*", default=list(PRECURSOR_HORIZONS))
        p.add_argument("--out", default=None)
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

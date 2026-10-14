"""实验一：几何 Monitor vs 语义 Monitor，在同一批决策点上比 P@1/P@3/P@5。

为什么这个实验值得做（route1-bridge-plan.md §4-1）
------------------------------------------------
AVERT-VLN 的核心立论是「几何判据不足以判断偏航」，但**它没有给几何判据的对照**。
我们手上恰好有几何判据的完整实现（4 信号 + 早漂移触发）与真实失败标签，
把两者放在同一批决策点上比，**无论结果如何都是一条独立证据**：
  - 若语义 Monitor 显著更高 → AVERT 的立论被独立证实，且我们给出了它缺的对照；
  - 若不显著 → 说明"语义 > 几何"这条断言不能照搬，我们的实现给出反例。

口径
----
- 候选决策点 = 日志里所有 `system1` 决策（可配置为只在几何触发点上比，成本更低）；
- 标签 = 未来 k 个决策内是否出现 `has_executable_action=False`（与上游 `analyze_*` 同口径）；
- 语义 Monitor 的"触发" = `verdict == LOST 且 confidence ≥ 阈值`；
- 几何 Monitor 的"触发" = `replan_reason == 'early_drift'`（原系统实际用的判据）；
- P@k 只在**非删失**样本上求均值（删失样本从分母剔除）。

Python 3.9 兼容。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .replay import (
    PRECURSOR_HORIZONS,
    label_future_failures,
    load_run,
)

#: Monitor 函数的签名：`(instruction, frame_paths) -> {"verdict","confidence",...}`
MonitorFn = Callable[[str, Sequence[str]], Dict[str, Any]]

#: 日志里的 `rgb_path` 是**相对 InternNav 仓库根**的（如 `./logs/habitat/...`），
#: 直接拿它去开文件会 FileNotFoundError。所有帧路径都要先按这个根解析。
DEFAULT_REPO_ROOT = os.environ.get("INTERNNAV_REPO_ROOT", "/srv/agh-demo/InternNav")


def resolve_frame_path(path: str, repo_root: str = DEFAULT_REPO_ROOT) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(repo_root, path))


def _episode_frames(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """按回合收集**有帧**的决策，按文件顺序（≈时间顺序）排列。"""
    per_episode: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        if not row.get("rgb_path"):
            continue
        key = (str(row.get("scene_id")), int(row.get("episode_id") or -1))
        per_episode.setdefault(key, []).append(row)
    return per_episode


def build_frame_history(
    episode_rows: Sequence[Dict[str, Any]],
    index: int,
    max_frames: int = 8,
    repo_root: str = DEFAULT_REPO_ROOT,
) -> List[str]:
    """取当前帧 + 至多 `max_frames-1` 张历史帧（等距下采样），返回**绝对**文件路径。"""
    end = index + 1
    start = max(0, end - max_frames)
    window = list(episode_rows[start:end])
    if len(window) > max_frames:
        step = len(window) / float(max_frames)
        window = [window[int(i * step)] for i in range(max_frames)]
    return [
        resolve_frame_path(str(row["rgb_path"]), repo_root)
        for row in window if row.get("rgb_path")
    ]


def select_candidates(
    labels: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    predicate: str,
    max_candidates: Optional[int] = None,
    seed: int = 20261009,
) -> List[Dict[str, Any]]:
    """挑选要送 Monitor 的决策点。

    `predicate`:
      - `all`         ：所有带帧的决策（最贵，但对照最完整）
      - `geometric`   ：几何触发点（`reason == 'early_drift'`）——与 P@1=0 那批完全同一批点
      - `band`        ：分数落在迟滞中间带 [low, high) 的决策（"该不该动"最不确定的那些点）
      - `replan`      ：任何真正重规划的决策
      - `stratified`  ：正样本全取 + 负样本随机等量（成本可控且类平衡）
    """
    candidates: List[Dict[str, Any]] = []
    for entry, row in zip(labels, rows):
        if not row.get("rgb_path"):
            continue
        reason = entry.get("reason")
        score = entry.get("score")
        keep = False
        if predicate == "all":
            keep = True
        elif predicate == "geometric":
            keep = reason == "early_drift"
        elif predicate == "replan":
            keep = bool(entry.get("should_replan"))
        elif predicate == "band":
            keep = score is not None and 0.40 <= float(score) < 0.70
        elif predicate == "stratified":
            keep = True
        else:
            raise ValueError("unknown candidate predicate %r" % predicate)
        if keep:
            candidates.append({"label": entry, "row": row})

    if predicate == "stratified" and max_candidates:
        positives = [c for c in candidates if c["label"].get("no_action_within_5")]
        negatives = [c for c in candidates if not c["label"].get("no_action_within_5")]
        rng = np.random.RandomState(seed)
        rng.shuffle(positives)
        rng.shuffle(negatives)
        half = max(1, max_candidates // 2)
        candidates = positives[:half] + negatives[:half]
        candidates.sort(key=lambda c: c["label"]["index"])
    elif max_candidates and len(candidates) > max_candidates:
        rng = np.random.RandomState(seed)
        chosen = rng.choice(len(candidates), size=max_candidates, replace=False)
        candidates = [candidates[i] for i in sorted(chosen)]
    return candidates


def vote_monitor(monitor_fn: MonitorFn, instruction: str, frames: Sequence[str],
                 n_votes: int = 3) -> Dict[str, Any]:
    """多次独立调用取多数派，并给出**一致度**。

    为什么必须这样（实测结论）
    ------------------------
    Agnes `agnes-3.0-flash` 在含糊帧上判决不稳定：同帧同指令、temperature=0，
    少数派比例可达 20–40%；而 confidence **挡不住**这个错误。
    另外：**投票时必须绕开磁盘缓存**，否则重复调用命中同一份缓存，
    会把"不稳定"伪装成"5/5 稳定"（我第一版就被这个坑过一次）。
    """
    if n_votes <= 1:
        raw = monitor_fn(instruction, frames) or {}
        out = dict(raw)
        out["agreement"] = 1.0
        out["votes"] = 1
        out["vote_verdicts"] = [raw.get("verdict")]
        return out

    verdicts: List[str] = []
    confidences: List[float] = []
    rials: List[Dict[str, Any]] = []
    last: Dict[str, Any] = {}
    for _ in range(n_votes):
        try:
            raw = monitor_fn(instruction, list(frames)) or {}
        except Exception as exc:
            rials.append({"error": "%s: %s" % (type(exc).__name__, exc)})
            continue
        last = raw
        rials.append(raw)
        verdict = raw.get("verdict")
        if verdict is not None:
            verdicts.append(str(verdict).strip().upper())
        if raw.get("confidence") is not None:
            confidences.append(float(raw["confidence"]))

    if not verdicts:
        out = dict(last)
        out["agreement"] = None
        out["votes"] = n_votes
        out["vote_verdicts"] = []
        out["vote_trials"] = rials
        return out

    counts: Dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    majority = max(counts, key=lambda key: counts[key])
    agreement = counts[majority] / float(len(verdicts))
    if agreement < 1.0 and len(counts) > 1:
        majority = "UNSURE"
    out = dict(last)
    out["verdict"] = majority
    out["confidence"] = float(np.mean(confidences)) if confidences else None
    out["agreement"] = float(agreement)
    out["votes"] = len(verdicts)
    out["vote_verdicts"] = verdicts
    out["vote_counts"] = counts
    return out


def make_agnes_monitor(model: Optional[str] = None, use_cache: bool = False,
                       n_votes: int = 3) -> MonitorFn:
    """把 `route1.monitor.agnes_monitor.monitor` 包成 `(instruction, frames) -> dict`。

    `use_cache=False` 是刻意的：投票需要真正的独立采样。
    """
    from ..monitor.agnes_monitor import monitor as agnes_monitor  # type: ignore

    def _call(instruction: str, frames: Sequence[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"use_cache": use_cache, "temperature": 0.0}
        if model:
            kwargs["model"] = model
        return agnes_monitor(instruction, list(frames), **kwargs)

    def _voted(instruction: str, frames: Sequence[str]) -> Dict[str, Any]:
        return vote_monitor(_call, instruction, frames, n_votes=n_votes)

    return _voted


def evaluate_semantic_monitor(
    run_dir: str,
    monitor: MonitorFn,
    predicate: str = "geometric",
    max_candidates: Optional[int] = None,
    max_frames: int = 8,
    confidence_threshold: float = 0.70,
    limit_episodes: Optional[int] = None,
    repo_root: str = DEFAULT_REPO_ROOT,
) -> Dict[str, Any]:
    """在候选决策点上跑语义 Monitor，产出与几何 Monitor 同口径的 P@k 表。"""
    run = load_run(run_dir)
    rows = run["system1"]
    labels = label_future_failures(rows)
    instructions = run["instruction_by_episode"]
    per_episode = _episode_frames(rows)
    index_in_episode: Dict[int, int] = {}
    for key, episode_rows in per_episode.items():
        for local_index, row in enumerate(episode_rows):
            index_in_episode[id(row)] = local_index

    candidates = select_candidates(labels, rows, predicate, max_candidates)

    records: List[Dict[str, Any]] = []
    errors = 0
    for item in candidates:
        entry, row = item["label"], item["row"]
        key = (str(row.get("scene_id")), int(row.get("episode_id") or -1))
        instruction = instructions.get(key) or ""
        local_index = index_in_episode.get(id(row), 0)
        frames = build_frame_history(per_episode.get(key, []), local_index, max_frames,
                                     repo_root=repo_root)
        verdict: Optional[str] = None
        confidence: Optional[float] = None
        agreement: Optional[float] = None
        raw: Dict[str, Any] = {}
        try:
            raw = monitor(instruction, frames) or {}
            verdict = raw.get("verdict")
            confidence = raw.get("confidence")
            agreement = raw.get("agreement")
        except Exception as exc:  # 单点失败不拖垮整轮
            errors += 1
            raw = {"error": "%s: %s" % (type(exc).__name__, exc)}
        triggered = bool(
            verdict is not None and str(verdict).upper() == "LOST"
            and confidence is not None and float(confidence) >= confidence_threshold
        )
        abstained = bool(verdict is not None and str(verdict).upper() not in ("PASS", "LOST"))
        records.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "action_step": row.get("action_step"),
                "instruction": instruction,
                "frames": frames,
                "geometric_score": entry.get("score"),
                "geometric_drift": entry.get("drift"),
                "geometric_reason": entry.get("reason"),
                "geometric_triggered": entry.get("reason") == "early_drift",
                "semantic_verdict": verdict,
                "semantic_confidence": confidence,
                "semantic_agreement": agreement,
                "semantic_triggered": triggered,
                "semantic_abstained": abstained,
                "monitor_raw": raw,
                "no_action_within_1": entry.get("no_action_within_1"),
                "no_action_within_3": entry.get("no_action_within_3"),
                "no_action_within_5": entry.get("no_action_within_5"),
                "first_no_action_offset": entry.get("first_no_action_offset"),
            }
        )

    report: Dict[str, Any] = {
        "run_dir": run_dir,
        "predicate": predicate,
        "repo_root": repo_root,
        "candidates_requested": len(candidates),
        "monitor_errors": errors,
        "confidence_threshold": confidence_threshold,
        "max_frames": max_frames,
        "monitors": {},
        "records": records,
    }
    for name, key_name in (("geometric", "geometric_triggered"), ("semantic", "semantic_triggered")):
        subset = [r for r in records if r[key_name]]
        entry: Dict[str, Any] = {
            "triggers": len(subset),
            "trigger_rate": (len(subset) / len(records)) if records else None,
        }
        for horizon in PRECURSOR_HORIZONS:
            name_k = "no_action_within_%d" % horizon
            observed = [bool(r[name_k]) for r in subset if r.get(name_k) is not None]
            entry["precision_at_%d" % horizon] = float(np.mean(observed)) if observed else None
            entry["observed_at_%d" % horizon] = len(observed)
        report["monitors"][name] = entry

    all_observed = [r for r in records if r.get("no_action_within_1") is not None]
    report["base_rate_at_1"] = (
        float(np.mean([bool(r["no_action_within_1"]) for r in all_observed]))
        if all_observed else None
    )
    if records:
        verdicts = [str(r["semantic_verdict"]).upper() for r in records if r["semantic_verdict"]]
        report["semantic_verdict_distribution"] = {
            "PASS": verdicts.count("PASS"), "LOST": verdicts.count("LOST"),
            "UNSURE": verdicts.count("UNSURE"),
            "missing": len(records) - len(verdicts),
        }
        agreements = [float(r["semantic_agreement"]) for r in records
                      if r.get("semantic_agreement") is not None]
        if agreements:
            report["semantic_agreement_mean"] = float(np.mean(agreements))
            report["semantic_agreement_unanimous_rate"] = float(
                np.mean([1.0 if a >= 1.0 else 0.0 for a in agreements])
            )
        report["semantic_abstain_rate"] = float(
            np.mean([1.0 if r["semantic_abstained"] else 0.0 for r in records])
        )
    return report


def format_comparison(report: Dict[str, Any]) -> str:
    """把对照表渲染成 Markdown（直接进作品报告）。"""
    lines = [
        "## 几何 Monitor vs 语义 Monitor",
        "",
        "同一批决策点、同一套失败标签（未来 k 个决策内出现 `has_executable_action=False`）。",
        "",
        "| Monitor | 触发数 | 触发率 | P@1 | P@3 | P@5 |",
        "|---|---|---|---|---|---|",
    ]

    def f(value: Any) -> str:
        return "-" if value is None else "%.4f" % value

    for name, label in (("geometric", "几何（早漂移判据，现有系统）"),
                        ("semantic", "语义（Agnes Monitor，第 5 信号）")):
        item = report["monitors"].get(name, {})
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (label, item.get("triggers"), f(item.get("trigger_rate")),
               f(item.get("precision_at_1")), f(item.get("precision_at_3")),
               f(item.get("precision_at_5")))
        )
    lines += [
        "",
        "- 候选决策点：%d（predicate=%s）" % (report["candidates_requested"], report["predicate"]),
        "- Monitor 调用失败：%d" % report["monitor_errors"],
        "- 全体基率 P(未来 1 步内出现无可执行动作) = %s" % f(report.get("base_rate_at_1")),
        "- 语义判决分布：%s" % json.dumps(report.get("semantic_verdict_distribution", {}),
                                     ensure_ascii=False),
        "- 投票一致度均值：%s ｜ 全票一致比例：%s ｜ 弃权率：%s"
        % (f(report.get("semantic_agreement_mean")),
           f(report.get("semantic_agreement_unanimous_rate")),
           f(report.get("semantic_abstain_rate"))),
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="几何 vs 语义 Monitor 对照实验")
    parser.add_argument("--run", required=True, help="带 rgb_path 的 run 目录")
    parser.add_argument("--predicate", default="geometric",
                        choices=["all", "geometric", "band", "replan", "stratified"])
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--mock", action="store_true",
                        help="不调用网络，用几何分数伪装成 Monitor（用于自检管线）")
    parser.add_argument("--votes", type=int, default=3,
                        help="每个决策点的独立投票次数（>1 时取多数派并给出一致度）")
    parser.add_argument("--model", default=None, help="覆盖 Monitor 模型名")
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT,
                        help="解析日志里相对 rgb_path 的仓库根")
    parser.add_argument("--use-cache", action="store_true",
                        help="允许命中磁盘缓存（**投票实验不要开**）")
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args(argv)

    if args.mock:
        def _mock_call(instruction: str, frames: Sequence[str]) -> Dict[str, Any]:
            # 自检用：判决只跟帧数有关，确保管线（取帧→投票→聚合）是通的
            return {"verdict": "LOST" if len(frames) >= 3 else "PASS",
                    "confidence": 0.8, "reason_l1": "mock", "mock": True}

        def monitor(instruction: str, frames: Sequence[str]) -> Dict[str, Any]:
            return vote_monitor(_mock_call, instruction, frames, n_votes=args.votes)
    else:
        monitor = make_agnes_monitor(model=args.model, use_cache=bool(args.use_cache),
                                     n_votes=args.votes)

    report = evaluate_semantic_monitor(
        args.run, monitor, predicate=args.predicate,
        max_candidates=args.max_candidates, max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold,
        repo_root=args.repo_root,
    )
    markdown = format_comparison(report)
    print(markdown)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        payload = {k: v for k, v in report.items() if k != "records"}
        payload["records_path"] = args.out.replace(".json", "_records.json")
        with open(payload["records_path"], "w", encoding="utf-8") as handle:
            json.dump(report["records"], handle, ensure_ascii=False, indent=2)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print("wrote %s" % args.out)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        print("wrote %s" % args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

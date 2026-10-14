"""路线① 端到端 demo 编排：把导航侧、语义 Monitor、操作侧合到一条失败事件总线上。

一条命令做完的事
--------------
1. **导航侧**：`nav/adapter.py` 把既有 50 回合日志映射成 `FailureEvent` 流（零改造）。
2. **语义富化**（可选）：只在几何分落进迟滞中间带的决策点上调 Agnes Monitor
   （p90 ≈ 10 s，很贵），把 `semantic_*` 信号写回同一批事件；票数不一致时记 `UNSURE`。
3. **操作侧**：读 Isaac 侧产出的 `FailureEvent` JSONL（X1/X2/X3）。
4. **桥**：按时间合流到 append-only 总线，写出 `bridge_report.json` 与一份 Markdown。

这就是"路线① 的仿真"的可重复入口——两个仿真器各自跑，桥只交换事件。

Python 3.9 兼容。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..bridge import Bridge, load_events_jsonl
from ..contracts import (
    RECOVERY_MENU,
    Attribution,
    FailureEvent,
    Recovery,
    Signals,
)
from ..nav.adapter import events_from_run
from ..nav.replay import load_run
from ..nav.monitor_eval import (
    DEFAULT_REPO_ROOT,
    build_frame_history,
    make_agnes_monitor,
    vote_monitor,
)


def band_candidates(run_dir: str, low: float = 0.40, high: float = 0.70) -> List[Dict[str, Any]]:
    """列出几何分落进迟滞中间带的决策点（= 值得付一次 Monitor 调用的点）。"""
    run = load_run(run_dir)
    rows = run["system1"]
    out: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        score = row.get("feedback_score")
        if score is None or not row.get("rgb_path"):
            continue
        if low <= float(score) < high:
            out.append({"row": row, "index": index, "score": float(score)})
    return out


def score_histogram(run_dir: str, bins: Sequence[float] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01)) -> Dict[str, Any]:
    """几何分分布 + 各区间里"未来 1 步内出现无可执行动作"的基率。

    用途：判断"只在中间带调 Monitor"会不会根本调不到几次（如果分数几乎都 < 0.4，
    这条路就是死路），以及中间带是不是真的有信息量。
    """
    from ..nav.replay import label_future_failures

    run = load_run(run_dir)
    rows = run["system1"]
    labels = label_future_failures(rows)
    scores = np.asarray([r.get("feedback_score") if r.get("feedback_score") is not None else np.nan
                         for r in rows], dtype=np.float64)
    with_frames = np.asarray([bool(r.get("rgb_path")) for r in rows], dtype=bool)
    outcome: Dict[str, Any] = {
        "run_dir": run_dir,
        "decisions": len(rows),
        "decisions_with_frames": int(with_frames.sum()),
        "score_available": int(np.isfinite(scores).sum()),
        "bins": [],
    }
    for low, high in zip(bins[:-1], bins[1:]):
        mask = np.isfinite(scores) & (scores >= low) & (scores < high)
        count = int(mask.sum())
        labelled = [
            label.get("no_action_within_1") for label, keep in zip(labels, mask) if keep
        ]
        labelled = [value for value in labelled if value is not None]
        outcome["bins"].append(
            {
                "range": "[%.2f, %.2f)" % (low, high),
                "decisions": count,
                "with_frames": int((mask & with_frames).sum()),
                "base_rate_at_1": float(np.mean([bool(v) for v in labelled])) if labelled else None,
                "observed": len(labelled),
            }
        )
    total_frames = int((np.isfinite(scores) & with_frames).sum())
    band = [b for b in outcome["bins"] if b["range"].startswith("[0.40")]
    outcome["band_decisions_with_frames"] = band[0]["with_frames"] if band else 0
    outcome["frame_coverage"] = (with_frames.sum() / len(rows)) if rows else None
    outcome["_total_frames_scored"] = total_frames
    return outcome


def enrich_events_with_monitor(
    run_dir: str,
    events: Sequence[FailureEvent],
    monitor,
    max_candidates: Optional[int] = None,
    repo_root: str = DEFAULT_REPO_ROOT,
    max_frames: int = 8,
    confidence_threshold: float = 0.70,
    band_low: float = 0.40,
    band_high: float = 0.70,
) -> Tuple[List[FailureEvent], List[Dict[str, Any]]]:
    """给中间带里带帧的决策点补上语义信号，返回 (新事件列表, 逐点记录)。

    事件的身份用 (scene_id, episode_id, action_step) 匹配 —— 与日志一致。
    """
    run = load_run(run_dir)
    rows = run["system1"]
    instructions = run["instruction_by_episode"]

    per_episode: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    local_index: Dict[int, int] = {}
    for row in rows:
        if not row.get("rgb_path"):
            continue
        key = (str(row.get("scene_id")), int(row.get("episode_id") or -1))
        local_index[id(row)] = len(per_episode.setdefault(key, []))
        per_episode[key].append(row)

    candidates = band_candidates(run_dir, band_low, band_high)
    if max_candidates is not None and len(candidates) > max_candidates:
        rng = np.random.RandomState(20261009)
        pick = rng.choice(len(candidates), size=max_candidates, replace=False)
        candidates = [candidates[i] for i in sorted(pick)]

    records: List[Dict[str, Any]] = []
    enriched: Dict[Tuple[str, int, Any], Dict[str, Any]] = {}
    for item in candidates:
        row = item["row"]
        key = (str(row.get("scene_id")), int(row.get("episode_id") or -1))
        frames = build_frame_history(per_episode.get(key, []), local_index.get(id(row), 0),
                                     max_frames, repo_root=repo_root)
        instruction = instructions.get(key) or ""
        raw: Dict[str, Any] = {}
        try:
            raw = monitor(instruction, frames) or {}
        except Exception as exc:
            raw = {"error": "%s: %s" % (type(exc).__name__, exc)}
        enriched[(key[0], key[1], row.get("action_step"))] = raw
        records.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "action_step": row.get("action_step"),
                "geometric_score": item["score"],
                "n_frames": len(frames),
                "verdict": raw.get("verdict"),
                "confidence": raw.get("confidence"),
                "agreement": raw.get("agreement"),
                "latency_ms": raw.get("latency_ms"),
                "abstained": raw.get("verdict") not in (None, "PASS", "LOST"),
                "raw": raw,
            }
        )

    out: List[FailureEvent] = []
    for event in events:
        raw = enriched.get((event.scene_id, event.episode_id, event.action_step))
        if raw is None:
            out.append(event)
            continue
        verdict = raw.get("verdict")
        confidence = raw.get("confidence")
        agreement = raw.get("agreement")
        from ..nav.feedback5 import semantic_risk

        risk = semantic_risk(verdict, confidence, agreement, min_agreement=0.6)
        signals = Signals.from_dict(event.signals.to_dict())
        signals.semantic_verdict = None if verdict is None else str(verdict).upper()
        signals.semantic_confidence = None if confidence is None else float(confidence)
        signals.semantic_agreement = None if agreement is None else float(agreement)
        signals.semantic_votes = int(raw.get("votes") or 0) or None
        signals.semantic_source = str(raw.get("model") or "agnes")
        signals.semantic_instance_margin = risk

        attribution = event.attribution
        recovery = event.recovery
        event_type = event.type
        if verdict is not None and str(verdict).upper() not in ("PASS", "LOST"):
            event_type = "N7"
            attribution = Attribution(
                kind="semantic_low_agreement_abstain",
                summary_l1="语义 Monitor 的多次投票结果不一致，无法判定是否偏航",
                summary_l2="按设计弃权：请求上层澄清，而不是用抖动信号去动计划。",
                evidence=["votes=%s" % raw.get("votes"), "agreement=%s" % agreement],
                dominant_signal="semantic", source="llm",
            )
            recovery = Recovery(reason_code="semantic_low_agreement_abstain",
                                action="ask_clarification",
                                budget_used=event.recovery.budget_used,
                                budget_left=event.recovery.budget_left)
        elif risk is not None and risk >= 0.70:
            event_type = "N7"
            attribution = Attribution(
                kind="semantic_instruction_mismatch",
                summary_l1="Agnes Monitor 判定执行与指令不一致（语义偏航）",
                summary_l2=str(raw.get("l2_diagnosis") or ""),
                evidence=[str(x) for x in (raw.get("evidence") or [])][:3],
                dominant_signal="semantic", source="llm",
            )
            recovery = Recovery(
                reason_code="semantic_lost_high_confidence",
                action="switch_instance" if risk >= 0.90 else "reparse_instruction",
                budget_used=event.recovery.budget_used,
                budget_left=event.recovery.budget_left,
            )

        rebuilt = FailureEvent(
            id=event.id, t=event.t, source=event.source, type=event_type,
            signals=signals, attribution=attribution, recovery=recovery,
            outcome=event.outcome, scene_id=event.scene_id, episode_id=event.episode_id,
            step=event.step, action_step=event.action_step, instruction=event.instruction,
            trigger_kind=event.trigger_kind,
        )
        out.append(rebuilt)
    return out, records


def _markdown(report: Dict[str, Any], nav_events: int, manip_events: int,
              enrich_records: Sequence[Dict[str, Any]]) -> str:
    summary = report["summary"]
    lines = [
        "# 路线① 端到端 demo 报告",
        "",
        "> 导航侧（Habitat/MP3D）与操作侧（Isaac Sim）各自持有自己的仿真，",
        "> 桥只交换 `FailureEvent` 与任务状态；因此任一侧崩掉都能从事件流重放。",
        "",
        "## 总线汇总",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        "| 事件总数 | %s |" % summary["count"],
        "| 判定为失败 | %s |" % summary["failure_events"],
        "| 导航侧 / 操作侧 | %s / %s |" % (summary["by_source"].get("nav", 0),
                                        summary["by_source"].get("manip", 0)),
        "| 失败类型分布 | %s |" % json.dumps(summary["by_type"], ensure_ascii=False),
        "| 恢复动作分布 | %s |" % json.dumps(summary["by_action"], ensure_ascii=False),
        "| 归因种类分布 | %s |" % json.dumps(summary["by_attribution_kind"], ensure_ascii=False),
        "| 已恢复 / 未恢复 / 未知 | %s / %s / %s |" % (
            summary["recovered_true"], summary["recovered_false"], summary["recovered_unknown"]),
        "",
        "## 本回合纳入的事件",
        "",
        "- 导航侧导入：%d" % nav_events,
        "- 操作侧导入：%d" % manip_events,
        "",
    ]
    if enrich_records:
        lines += [
            "## 语义 Monitor 富化（只在迟滞中间带调用）",
            "",
            "| 回合 | 步 | 几何分 | 帧数 | verdict | conf | 一致度 | 延迟(ms) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for record in enrich_records:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    record["episode_id"], record["action_step"],
                    ("%.3f" % record["geometric_score"]) if record["geometric_score"] is not None else "-",
                    record["n_frames"], record["verdict"],
                    ("%.2f" % record["confidence"]) if record["confidence"] is not None else "-",
                    ("%.2f" % record["agreement"]) if record["agreement"] is not None else "-",
                    record["latency_ms"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def _cmd_bandstats(args: argparse.Namespace) -> int:
    report = score_histogram(args.run)
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("_")},
                     ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    nav_events = events_from_run(args.nav_run, include_non_acting=args.include_non_acting)
    enrich_records: List[Dict[str, Any]] = []
    if args.enrich:
        if args.mock:
            def _mock(_instruction: str, frames: Sequence[str]) -> Dict[str, Any]:
                return {"verdict": "PASS", "confidence": 0.9, "agreement": 1.0,
                        "votes": 1, "model": "mock"}
            monitor = _mock
        else:
            monitor = make_agnes_monitor(model=args.model, use_cache=False, n_votes=args.votes)
        nav_events, enrich_records = enrich_events_with_monitor(
            args.nav_run, nav_events, monitor,
            max_candidates=args.max_candidates, max_frames=args.max_frames,
        )
    manip_events = load_events_jsonl(args.manip) if args.manip else []

    bridge = Bridge(args.root)
    bridge.bus.write_meta({
        "kind": "route1_demo",
        "nav_run": os.path.abspath(args.nav_run),
        "manip_jsonl": os.path.abspath(args.manip) if args.manip else None,
        "enriched_with_monitor": bool(args.enrich),
    })
    bridge.merge(nav_events, manip_events, label=os.path.basename(args.root.rstrip("/")))
    report_path = bridge.write_report(os.path.join(args.root, "bridge_report.json"))
    report = json.load(open(report_path, encoding="utf-8"))
    markdown = _markdown(report, len(nav_events), len(manip_events), enrich_records)
    md_path = os.path.join(args.root, "demo_report.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(markdown)
    print("\nwrote", report_path)
    print("wrote", md_path)
    if enrich_records:
        with open(os.path.join(args.root, "monitor_enrichment.json"), "w", encoding="utf-8") as handle:
            json.dump(enrich_records, handle, ensure_ascii=False, indent=2)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="路线① 端到端 demo 编排")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("band-stats", help="几何分分布 + 中间带候选量（决定 Monitor 值不值得调）")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_bandstats)

    p = sub.add_parser("demo", help="合成一条失败事件总线并出报告")
    p.add_argument("--nav-run", required=True)
    p.add_argument("--manip", default=None, help="Isaac 侧产出的 FailureEvent JSONL")
    p.add_argument("--root", required=True, help="事件总线目录")
    p.add_argument("--include-non-acting", action="store_true")
    p.add_argument("--enrich", action="store_true", help="用 Agnes Monitor 富化中间带决策点")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--votes", type=int, default=1)
    p.add_argument("--model", default=None)
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=8)
    p.set_defaults(func=_cmd_demo)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

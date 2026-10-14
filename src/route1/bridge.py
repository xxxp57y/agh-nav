"""桥：把导航侧与操作侧的 FailureEvent 合流到一条事件总线，并可重放。

设计（route1-bridge-plan.md §1 / §5-R5）
--------------------------------------
- 桥**只传事件与任务状态**，不传渲染帧、不共享仿真状态；
- 事件流是 append-only JSONL，因此任一侧崩掉都能从事件流重放；
- 合流时按 `t` 排序，但**保留来源标记**，便于分侧统计。

Python 3.9 兼容。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .bus import EventBus, read_jsonl
from .contracts import FailureEvent, validate_stream


def load_events_jsonl(path: str) -> List[FailureEvent]:
    """读一个已落盘的 FailureEvent JSONL（容错跳过坏行并计数）。"""
    events: List[FailureEvent] = []
    bad = 0
    if not os.path.exists(path):
        return events
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(FailureEvent.from_json(line))
            except Exception:
                bad += 1
    if bad:
        print("[bridge] 跳过 %d 行无法解析的事件" % bad)
    return events


class Bridge(object):
    """跨仿真器事件总线。"""

    def __init__(self, root: str) -> None:
        self.bus = EventBus(root)

    # ---- 合流 ---------------------------------------------------------- #
    def merge(self,
              nav_events: Sequence[FailureEvent] = (),
              manip_events: Sequence[FailureEvent] = (),
              label: Optional[str] = None) -> Dict[str, Any]:
        merged = list(nav_events) + list(manip_events)
        merged.sort(key=lambda event: float(event.t))
        count = self.bus.emit_many(merged)
        self.bus.write_meta(
            {
                "label": label,
                "last_merge": {
                    "nav": len(nav_events),
                    "manip": len(manip_events),
                    "written": count,
                },
            }
        )
        return self.summary()

    def merge_files(self, paths: Iterable[str], label: Optional[str] = None) -> Dict[str, Any]:
        nav: List[FailureEvent] = []
        manip: List[FailureEvent] = []
        for path in paths:
            for event in load_events_jsonl(path):
                (nav if event.source == "nav" else manip).append(event)
        return self.merge(nav, manip, label=label)

    # ---- 统计 ---------------------------------------------------------- #
    def summary(self) -> Dict[str, Any]:
        events = self.bus.load_events()
        report = validate_stream(events)
        report["root"] = self.bus.root
        report["failure_events"] = sum(1 for event in events if event.type is not None)
        report["recovered_true"] = sum(1 for event in events if event.outcome.recovered is True)
        report["recovered_false"] = sum(1 for event in events if event.outcome.recovered is False)
        report["recovered_unknown"] = sum(1 for event in events if event.outcome.recovered is None)
        report["by_attribution_kind"] = _count(events, lambda event: event.attribution.kind)
        report["by_recovery_action"] = _count(events, lambda event: event.recovery.action)
        report["task_state"] = self.bus.latest_state()
        return report

    def episode_report(self) -> List[Dict[str, Any]]:
        """按 (source, scene, episode) 聚合，便于写进实验报告。"""
        buckets: Dict[Any, Dict[str, Any]] = {}
        for event in self.bus.load_events():
            key = (event.source, event.scene_id, event.episode_id)
            bucket = buckets.setdefault(
                key,
                {
                    "source": event.source,
                    "scene_id": event.scene_id,
                    "episode_id": event.episode_id,
                    "events": 0,
                    "failures": 0,
                    "types": {},
                    "actions": {},
                    "recovered": 0,
                    "not_recovered": 0,
                    "instruction": event.instruction,
                },
            )
            bucket["events"] += 1
            if event.type is not None:
                bucket["failures"] += 1
                bucket["types"][event.type] = bucket["types"].get(event.type, 0) + 1
            bucket["actions"][event.recovery.action] = (
                bucket["actions"].get(event.recovery.action, 0) + 1
            )
            if event.outcome.recovered is True:
                bucket["recovered"] += 1
            elif event.outcome.recovered is False:
                bucket["not_recovered"] += 1
        return sorted(buckets.values(), key=lambda item: (item["source"], str(item["scene_id"]),
                                                          item["episode_id"] or -1))

    def replay(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self.bus.replay(**kwargs)]

    def write_report(self, path: str) -> str:
        payload = {
            "summary": self.summary(),
            "episodes": self.episode_report(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path


def _count(events: Iterable[FailureEvent], key) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        name = key(event)
        counts[name] = counts.get(name, 0) + 1
    return counts


def demo_bridge(root: str) -> Dict[str, Any]:
    """不依赖任何仿真器的桥自检：造一条 nav 事件 + 一条 manip 事件并合流。"""
    from .contracts import Attribution, Outcome, Recovery, Signals

    nav = FailureEvent.create(
        source="nav", type="N2", scene_id="DEMO", episode_id=0, step=10, action_step=10,
        instruction="走进客厅后直走，然后进入阳台下方的入口。",
        signals=Signals(feedback_score=0.71, trajectory_drift=0.23, plan_age=9,
                        collision_delta=0, steps_since_last_s2=9),
        attribution=Attribution(kind="feedback_high:trajectory_drift",
                                summary_l1="机器人在客厅门口开始偏离计划轨迹",
                                evidence=["drift 0.23m > high 0.20m"]),
        recovery=Recovery(reason_code="early_drift_high_plan_fresh", action="local_escape",
                          budget_used=3, budget_left=5),
        outcome=Outcome(recovered=True, post_score=0.32, steps_to_recover=4),
    )
    manip = FailureEvent.create(
        source="manip", type="X1", scene_id="DEMO", episode_id=0, step=42, action_step=42,
        instruction="pick the cracker box on the table",
        signals=Signals(gripper_width_m=0.079, contact_force_n=0.0, object_dz_m=0.001,
                        dist_to_goal_m=0.31),
        attribution=Attribution(kind="grasp_empty",
                                summary_l1="夹爪闭合时未建立有效接触（抓空）",
                                evidence=["contact_force_n=0.00 N", "gripper_width 未达预期"]),
        recovery=Recovery(reason_code="grasp_empty", action="retry_grasp", budget_used=1,
                          budget_left=1),
        outcome=Outcome(recovered=True, post_score=0.0, steps_to_recover=1),
    )
    bridge = Bridge(root)
    return bridge.merge([nav], [manip], label="demo")

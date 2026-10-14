"""事件总线：append-only JSONL + 重放 + 跨仿真器任务状态快照。

为什么是 append-only
--------------------
路线① 的桥只传事件（不传帧），因此**任一侧崩掉都能从事件流重放**。总线不做
原地更新、不做事务、不依赖任何数据库——一个文件 + 顺序读就够，且便于赛后复现。

Python 3.9 兼容，只依赖标准库。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .contracts import FailureEvent, validate_stream


class EventBus:
    """把一个目录当作事件总线的根。

    布局::

        <root>/failure_events.jsonl   # FailureEvent 流（nav + manip 合流）
        <root>/task_state.jsonl       # 任务状态快照（跨仿真器共享的低带宽状态）
        <root>/bus_meta.json          # 总线元信息
    """

    EVENT_FILE = "failure_events.jsonl"
    STATE_FILE = "task_state.jsonl"
    META_FILE = "bus_meta.json"

    def __init__(self, root: str, create: bool = True) -> None:
        self.root = os.path.abspath(root)
        if create:
            os.makedirs(self.root, exist_ok=True)
        self.event_path = os.path.join(self.root, self.EVENT_FILE)
        self.state_path = os.path.join(self.root, self.STATE_FILE)
        self.meta_path = os.path.join(self.root, self.META_FILE)

    # ---- 事件 ---------------------------------------------------------- #
    def emit(self, event: FailureEvent, fsync: bool = False) -> FailureEvent:
        event.validate()
        line = event.to_json()
        with open(self.event_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        return event

    def emit_many(self, events: Iterable[FailureEvent]) -> int:
        count = 0
        with open(self.event_path, "a", encoding="utf-8") as handle:
            for event in events:
                event.validate()
                handle.write(event.to_json() + "\n")
                count += 1
            handle.flush()
        return count

    def replay(self, source: Optional[str] = None,
               failure_only: bool = False) -> Iterator[FailureEvent]:
        """顺序重放事件流，可按 source / 是否判定为失败过滤。"""
        if not os.path.exists(self.event_path):
            return
        with open(self.event_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = FailureEvent.from_json(line)
                if source is not None and event.source != source:
                    continue
                if failure_only and event.type is None:
                    continue
                yield event

    # ---- 任务状态 ------------------------------------------------------ #
    def put_state(self, key: str, value: Any, source: str = "nav") -> None:
        record = {"t": time.time(), "key": key, "value": value, "source": source}
        with open(self.state_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    def latest_state(self) -> Dict[str, Any]:
        """回放状态流，得到每个 key 的当前值。"""
        state: Dict[str, Any] = {}
        if not os.path.exists(self.state_path):
            return state
        with open(self.state_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                state[record["key"]] = record["value"]
        return state

    # ---- 汇总 ---------------------------------------------------------- #
    def summary(self) -> Dict[str, Any]:
        events = list(self.replay())
        report = validate_stream(events)
        report["root"] = self.root
        report["event_path"] = self.event_path
        report["recovery_actions"] = report.pop("by_action")
        return report

    def write_meta(self, extra: Optional[Dict[str, Any]] = None) -> str:
        meta = {
            "created": time.time(),
            "root": self.root,
            "event_file": self.EVENT_FILE,
            "state_file": self.STATE_FILE,
        }
        meta.update(extra or {})
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2, sort_keys=True)
        return self.meta_path

    # ---- 便捷读取 ------------------------------------------------------ #
    def load_events(self, **kwargs: Any) -> List[FailureEvent]:
        return list(self.replay(**kwargs))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读一个普通 JSONL（日志侧用，容错跳过坏行）。"""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows

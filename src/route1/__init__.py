"""路线①：Habitat 导航 + Isaac 操作，用统一 FailureEvent 桥接。

    route1/
      contracts.py   跨仿真器契约（FailureEvent / 失败类型 / 恢复动作空间）
      bus.py         append-only 事件总线 + 任务状态
      bridge.py      nav ⇄ manip 合流与重放
      nav/           导航侧：第 5 信号、决策层 v3、离线重放、日志适配器
      monitor/       语义 Monitor（Agnes 视觉大模型）
      isaac/         操作侧：最小 pick 任务 + 子目标判定 + 失败注入
      cli.py         统一命令行入口
"""

from __future__ import annotations

__all__ = ["contracts", "bus", "bridge"]
__version__ = "0.1.0"

"""路线① 导航侧带帧捕获的运行配置。

**本文件是新增文件**，不修改 InternNav 任何既有文件（`budgeted_closed_loop_50_cfg.py`
保持原样）。它继承同目录的 50 回合配置，只做两件事：

1. 把输出目录改到 `logs/habitat/route1_<label>_seed20260826`，与上游产物隔离；
2. 打开 `closed_loop.diagnostics.capture_episode_ids` —— 这是**语义 Monitor 的前提**：
   现有 50 回合日志里没有 `rgb_path`，不打开帧捕获就无法在决策点上跑 Agnes Monitor。

用法（在 `/srv/agh-demo/InternNav` 下）::

    BUDGET_MODE=budgeted_drift BUDGET_RUN_LABEL=r1pilot EPISODE_LIMIT=20 \
      /srv/agh-demo/habitat-env/bin/python -u scripts/eval/eval.py \
      --config scripts/eval/configs/route1_capture_cfg.py

环境变量：
    CAPTURE_EPISODE_IDS   逗号分隔的 episode id 白名单（优先级最高）
    CAPTURE_MAX_EPISODE_ID  未给白名单时，捕获 [0, MAX] 区间（默认 400）
"""

import copy
import os

from scripts.eval.configs.budgeted_closed_loop_50_cfg import eval_cfg as base_eval_cfg

eval_cfg = copy.deepcopy(base_eval_cfg)

run_label = os.environ.get("BUDGET_RUN_LABEL", "capture")
eval_cfg.eval_settings["output_path"] = f"./logs/habitat/route1_{run_label}_seed20260826"

whitelist = os.environ.get("CAPTURE_EPISODE_IDS", "").strip()
if whitelist:
    capture_ids = [int(value) for value in whitelist.split(",") if value.strip()]
else:
    capture_ids = list(range(0, int(os.environ.get("CAPTURE_MAX_EPISODE_ID", "400")) + 1))

closed_loop = eval_cfg.eval_settings["closed_loop"]
closed_loop.setdefault("diagnostics", {})
closed_loop["diagnostics"]["capture_episode_ids"] = capture_ids

# 帧捕获本身要写盘，给一个显式开关便于关掉
if os.environ.get("CAPTURE_DISABLED", "0") == "1":
    closed_loop["diagnostics"]["capture_episode_ids"] = []

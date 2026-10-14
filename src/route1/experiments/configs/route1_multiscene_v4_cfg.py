"""路线① 导航侧 v4：**跨场景稳定性**取向的触发/恢复策略配置（新增文件，不动上游）。

为什么要有这个文件
-----------------
跨场景那一轮（`route1_r1multi4`，4 场景 × 13 回合）的实测结果：
`2azQ1b91cZZ` 0.6923 / `TbHJrupSAjP` 0.6923 / `zsNo4HB9uLZ` 0.8462，
而 `QUCTc6BB5sX` **0.1538** → 池化 0.5962，**场景间 success 的 std = 0.3037、极差 0.6923**。
即：现在的"最好配置"（Recovery-2-Adaptive = `drift_recovery_periodic`）**不稳定**，
而它所有的调参都是在**单个场景**（`2azQ1b91cZZ`）上做的。

同一次跨场景 run 的失败模式诊断（`route1.experiments.scene_failure_diag`）给出两条硬事实：

1. **最差场景 11 个失败回合里有 5 个是"碰撞看门狗把 episode 提前 STOP 掉"**
   （显式 `collision_watchdog_stop` 事件，`trigger=collision_density`，窗口 8 动作内 ≥6 次碰撞）。
   这 5 个回合分别在 41/45/58/66/88 步就结束，而该 run 的最长 episode 是 202 步
   —— **60%–80% 的步数预算被安全层直接扔掉**，且 QUCTc6BB5sX 的碰撞数（3.00/回合）
   是另外三个场景（0.00/0.00/0.62）的 5–∞ 倍。
2. **在 4 个场景里有 3 个，success 与 replans 负相关**（`2azQ1b91cZZ` −0.79、
   `TbHJrupSAjP` −0.66；`system2_calls` 同向 −0.59/−0.65），只有最差场景是 +0.29。
   与之前"几何加权分 AUROC≈0.44（低于随机）"的结论一致：**现在的触发通道整体上是净有害的**。

因此 v4 的两条设计原则（都是"减少场景依赖"，而不是"再调一个阈值"）：

- **原则一：安全层不再单方面判死。** 碰撞是**唯一**能把最差场景与其它场景分开的观测量，
  它应该变成"要求重规划/脱离"的**触发信号**，而不是"结束 episode"的**终止信号**。
  在无法改上游代码的前提下（`InternNav/` 只读），本配置用 `V4_WATCHDOG=off` 先做**单变量消融**，
  验证"提前 STOP"就是那个害人机制；真正的替代（碰撞通道 → 重规划）走 out-of-tree 覆盖
  （见 `notes/route1-nav-next.md` 的 Stage 3）。
- **原则二：把"纯计时器"从决策里拿掉。** `plan_age` 是设计上的保险丝
  （`failure-taxonomy.md` §2 原话），但实测它对几何分的平均贡献最大（0.0438 > drift 0.0315），
  于是"分数 ≈ 距上次重规划多久"。`V4_W_AGE` 让它可调（默认沿用上游 0.15），
  `V4_W_AGE=0` 是"计时器只做保险丝、不参与打分"的候选。

可切换的环境变量（都只影响新增配置，不改上游默认）
------------------------------------------------
- `V4_WATCHDOG=off|on`    off（默认）：`safety` 三个阈值全 0 → 看门狗不终止 episode
- `V4_W_AGE=<float>`      默认 0.15（= 上游）；0 表示计时器不参与打分
- `V4_W_DRIFT=<float>`    默认 0.35
- `V4_LABEL_TAG=<str>`    产物目录后缀，便于与对照组区分

用法（在 `/srv/agh-demo/InternNav` 下）::

    BUDGET_MODE=drift_recovery_periodic BUDGET_RUN_LABEL=r1v4a4 \
    EPISODE_LIMIT=52 V4_WATCHDOG=off V4_W_AGE=0.0 \
      /srv/agh-demo/habitat-env/bin/python -u scripts/eval/eval.py \
      --config scripts/eval/configs/route1_multiscene_v4_cfg.py
"""

import copy
import os

from scripts.eval.configs.route1_multiscene_cfg import eval_cfg as base_eval_cfg

eval_cfg = copy.deepcopy(base_eval_cfg)
settings = eval_cfg.eval_settings
closed_loop = settings["closed_loop"]

# ---- 原则一：安全层不再"STOP 掉 episode" ----
if os.environ.get("V4_WATCHDOG", "off").strip().lower() == "off":
    closed_loop["safety"] = {
        "max_consecutive_collision_actions": 0,
        "collision_window_actions": 0,
        "max_collisions_in_window": 0,
    }

# ---- 原则二：计时器（plan_age）降权 / 取消 ----
feedback = closed_loop.setdefault("feedback", {})
feedback["w_age"] = float(os.environ.get("V4_W_AGE", feedback.get("w_age", 0.15)))
feedback["w_drift"] = float(os.environ.get("V4_W_DRIFT", feedback.get("w_drift", 0.35)))

# ---- 产物隔离（用 BUDGET_RUN_LABEL 决定，这里只保证一定带 v4 前缀便于事后筛选） ----
_tag = os.environ.get("V4_LABEL_TAG", "").strip()
if _tag:
    settings["output_path"] = "%s_%s" % (settings["output_path"], _tag)

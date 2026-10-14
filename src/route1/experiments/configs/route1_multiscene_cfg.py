"""路线① 跨场景 50 回合配置（新增文件，不动上游）。

为什么需要它 —— 一个上游行为的量化后果
------------------------------------
`habitat_vln_evaluator.py:160-165` 用 `self.env.episodes[:episode_limit]` 截断，
而 `self.env.episodes` 是**按场景分组**的（见 :144 的注释 "The stock wrapper groups
episodes by scene"）。于是 `EPISODE_LIMIT=50` 实际拿到的是**第一个场景的全部前 50 个
episode**——按场景 id 字典序，第一个是 `2azQ1b91cZZ`。

实测证实：`budgeted_v20_50_*_seed20260826` 这 13 个 run 的 50 个 episode
**全部来自 `2azQ1b91cZZ`**，episode id 10..215。也就是说：
- 所有 "50 回合 val_unseen" 的数字（含最好的 0.66 / 0.5734）都是**单场景**结果；
- 场景级泛化从未被度量；
- "留一场景"交叉验证在这份清单上无法成立。

修法（不需要改上游一行）：`eval_settings.dataset_split` 换成
`val_unseen_multiscene` —— habitat 的 yaml 里
`data_path: data/vln_ce/raw_data/r2r/{split}/{split}.json.gz` 会自动解析到
我们新写的 `data/vln_ce/raw_data/r2r/val_unseen_multiscene/val_unseen_multiscene.json.gz`。
该清单取 **4 个场景 × 13 个 episode = 52 个**，`episode_limit=52`，
分组后每个场景的 13 个都落在截断区间内 → **4 个场景全部出现**（已数值验证）。

场景选的是 `2azQ1b91cZZ`（与现有单场景基准同场景，可直接对比）、
`QUCTc6BB5sX`、`TbHJrupSAjP`、`zsNo4HB9uLZ`。

用法（在 `/srv/agh-demo/InternNav` 下）::

    BUDGET_MODE=drift_recovery_periodic BUDGET_RUN_LABEL=r1multi EPISODE_LIMIT=52 \
      /srv/agh-demo/habitat-env/bin/python -u scripts/eval/eval.py \
      --config scripts/eval/configs/route1_multiscene_cfg.py
"""

import copy
import os

from scripts.eval.configs.budgeted_closed_loop_50_cfg import eval_cfg as base_eval_cfg

MULTISCENE_SPLIT = os.environ.get("MULTISCENE_SPLIT", "val_unseen_multiscene")
EPISODES_PER_SCENE = int(os.environ.get("MULTISCENE_PER_SCENE", "13"))
N_SCENES = int(os.environ.get("MULTISCENE_N_SCENES", "4"))

#: 清单里的 4 个场景（每场景 13 个 episode，共 52）。刻意包含 `2azQ1b91cZZ`——
#: 单体基准那 50 回合就是它，这样"单场景 vs 跨场景"的对照里有一格是同场景的，
#: 差异只来自 episode 子集抽样，而不是场景本身。
MULTISCENE_SCENES = ("2azQ1b91cZZ", "QUCTc6BB5sX", "TbHJrupSAjP", "zsNo4HB9uLZ")

eval_cfg = copy.deepcopy(base_eval_cfg)
settings = eval_cfg.eval_settings

run_label = os.environ.get("BUDGET_RUN_LABEL", "multiscene")
settings["output_path"] = f"./logs/habitat/route1_{run_label}_seed20260826"
settings["dataset_split"] = MULTISCENE_SPLIT
settings["episode_limit"] = int(os.environ.get("EPISODE_LIMIT", str(EPISODES_PER_SCENE * N_SCENES)))

closed_loop = settings["closed_loop"]
closed_loop.setdefault("diagnostics", {})
if os.environ.get("CAPTURE_DISABLED", "0") == "1":
    closed_loop["diagnostics"]["capture_episode_ids"] = []
else:
    closed_loop["diagnostics"]["capture_episode_ids"] = list(
        range(0, int(os.environ.get("CAPTURE_MAX_EPISODE_ID", "400")) + 1)
    )

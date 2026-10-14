"""Stage 3 的启动器：装上 out-of-tree 覆盖，然后**原样**跑上游 `scripts/eval/eval.py`。

用法（在任意目录，只要 `PYTHONPATH` 能 import 到 `route1`）::

    cd /srv/agh-demo/InternNav
    PYTHONPATH=/srv/agh-demo/src \
    BUDGET_MODE=drift_recovery_periodic BUDGET_RUN_LABEL=r1ovl4 EPISODE_LIMIT=52 \
    V4_WATCHDOG=off V4_CHANNELS=both \
      /srv/agh-demo/habitat-env/bin/python -u \
      -m route1.experiments.run_eval_v4 --config scripts/eval/configs/route1_multiscene_v4_cfg.py

关键点：`install()` 必须在 `Evaluator.init()` 之前执行——所以这里**不** import
`internnav.evaluator`，而是先装覆盖，再用 `runpy` 跑上游入口脚本。
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys

from ..nav.overlay_v4 import install


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="装上 v4 覆盖后跑上游 eval 入口")
    parser.add_argument("--config", required=True, help="eval 配置路径（相对 InternNav 根）")
    args = parser.parse_args(argv)

    repo = os.environ.get("INTERNNAV_REPO_ROOT", "/srv/agh-demo/InternNav")
    os.chdir(repo)
    sys.path.insert(0, repo)
    sys.path.append("./third_party/diffusion-policy")

    channels = os.environ.get("V4_CHANNELS", "both").strip().lower()
    install(enable_collision=channels in ("both", "collision"),
            enable_progress=channels in ("both", "progress"))

    sys.argv = ["scripts/eval/eval.py", "--config", args.config]
    try:
        runpy.run_path(os.path.join(repo, "scripts/eval/eval.py"), run_name="__main__")
    finally:
        # 覆盖在进程内生效，统计只在这里拿得到；日志尾部会留下这两行
        from ..nav.overlay_v4 import stats
        print("[overlay_v4] final stats: %s" % stats(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

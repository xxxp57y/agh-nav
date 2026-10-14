# @agh-nav/nav-agent-plugin

路线① 导航侧的 **AGH 后端能力包**：11 个工具 + 1 个 Skill。

- 工具与 Skill 的完整清单、环境变量、验证命令：见上级 [`../README.md`](../README.md)。
- 自检：`node selftest.mjs --run <run> --episode <ep>`（在线，实测 55/55）；
  加 `--offline` 验证断链回退（实测 54/54，没缓存的工具**显式失败**而不是给空结果）。
- AGH 原生校验（用 AGH 自己的 `checkToolDef` / `validateAgainst`）：
  `corepack pnpm exec tsx ../tests/agh-integration.mjs`（实测 22/22，含看板 SSR 渲染与清理）。
- 数据面在服务器：`../bin/agh_nav_probe.py`（`local` 直接跑 / `ssh` 送过去）。
- 缓存快照：`data/cache/nav-cache.json`，用 `../scripts/refresh-cache.mjs` 刷新。

依赖：无 npm 依赖（纯 ESM + Node 内置模块），Node ≥ 24.10。工具全部只读，
唯一写操作 `nav_run_eval` 声明 `requiresApproval: 'always'` 与代价提示。

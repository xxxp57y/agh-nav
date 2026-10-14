# @agh-nav/nav-board-plugin

路线① 导航侧的 **AGH 看板包**：后端行注册只读查询服务 `nav.board`，
浏览器侧把对照表挂到 `ui:sidebar` 槽（priority −1，替换内置左栏）。

- 数据链路：面板 → `ctx.agnes.services.call('nav.board')` → 同源 BFF → daemon/Host 授权 → handler → 服务器数据面。
- 降级：服务不可用时显示包内静态快照 `extensions/main/client/baseline.js`
  （由 `../scripts/refresh-cache.mjs` 生成），并在角标写明"静态快照"。
- 自检：`node selftest.mjs`（包形状 + 服务注册 + 真取数）、`node selftest.mjs --offline`（回退）。

⚠️ 本包需要 `policy.capabilityCeiling` 放开 **`services`** 与 **`ui`** 能力
（AGH 默认不含 services，见 `docs/develop/fullstack.zh-CN.md`）。
因此工具包与看板包**故意拆成两个包**：策略卡住看板时，工具仍然能装能用。

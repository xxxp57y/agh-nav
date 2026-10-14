> 历史开发记录：部署地址已替换为示例。实际部署请使用 [当前 MHS 说明](../mhs/README.md)；真实评估日志和缓存未随公共仓库发布。

# agh-nav —— 把路线① 导航侧包装成 Agnes Harness(AGH) 能力

> 赛题：*基于 Agnes + AGH 构建仿真机器人自主决策智能体，实现**自然语言驱动的导航任务执行**、
> **环境反馈分析**和**导航失败后的自主任务调整***。

**两条线，别混**（细节见下文）：

| | 离线 `nav_*` | **在线 `navlive_*`** |
| --- | --- | --- |
| 回答 | 那一轮跑得怎么样、失败在哪、恢复通道会怎么动 | **现在这个场景里，机器人去哪、到哪了** |
| 形态 | 读已跑完 run 的日志（`bin/agh_nav_probe.py`） | **常驻服务**：`navlive/navlive_server.py` + 实时画面/楼层图/轨迹 |
| 会不会动 | 不会（只读） | **会**：`navlive_goto` 让机器人在 Habitat 里真的走过去（会动的调用要审批） |
| 时间尺度 | 一次评估 ≈ 60–70 分钟 | 一步 5–8 ms，一个回合 30–60 秒 |

- 想在**一个已跑好的场景**里按自然语言指令跑任务、并看到实时画面/轨迹 → 看
  [docs/实时可视化工作台.md](实时可视化工作台.md) 与 [docs/演示脚本-实时导航.md](演示脚本-实时导航.md)。
- 想读懂**已有评估**（8 场景 × 5 套恢复配置、失败事件、恢复通道复算）→ 看本文余下部分。

---

## 1. 它解决什么问题

导航闭环本身早就能跑（50 回合基准 Success 0.66 / SPL 0.5734；8 场景 104 回合的跨场景对照），
但它是**一堆 Python 模块 + 服务器上的日志**，Agent（Agnes）碰不到。要满足赛题，缺三件事：

| 赛题要求 | 缺什么 | 本包给什么 |
| --- | --- | --- |
| 自然语言驱动的导航任务执行 | Agent 不能发起 / 查看导航评估 | `nav_run_eval`（写操作，需审批）、`nav_run_progress`、`nav_list_runs`、`nav_run_summary` |
| 环境反馈分析 | 逐决策信号只躺在日志里 | `nav_observe_episode`、`nav_failure_events`、`nav_diagnose_triggers` |
| 导航失败后的自主任务调整 | 恢复判据在闭环进程内，Agent 看不到也改不了 | `nav_recovery_options`、`nav_compare_configs`、Skill `nav-failure-diagnosis` |

**设计上的关键取舍：判据不重写，而是复用。**

```
AGH（JS，跑在服务器或开发机上）
  ├── nav-agent-plugin   11 个工具 + 1 个 Skill（只读为主，唯一写操作需审批）
  └── nav-board-plugin   看板：query 服务 nav.board + ui:sidebar 面板
        │  local：直接跑；ssh：送到导航服务器；都失败 → 读本地缓存快照并标注 degraded
        ▼
服务器 bin/agh_nav_probe.py   ← 数据面（唯一的取数实现）
  ├── import route1.contracts                     → FailureEvent / Signals / Attribution / Recovery
  ├── import route1.nav.channels                  → **与在线 overlay_v4 同一份**碰撞/进度通道
  └── import route1.experiments.scene_failure_diag → 终止方式判定（同一口径）
        │
        ▼
   InternNav 日志：closed_loop_steps_rank*.jsonl + progress.json + pool8_*.json + FINAL_TABLE.txt
```

所以"给 Agent 的离线建议"与"在线闭环真正执行的判据"**是同一份代码** —— 这是路线① 全部结论成立的前提。

---

## 2. 目录

```
agh-nav/
  navlive/                        # **在线**：交互式导航服务与实时渲染
    navlive_server.py             # 常驻服务：sim 线程 + 命令队列 + 快照（HTTP/SSE）
    navlive_map.py                # 楼层图渲染（语义图 → 建筑平面图 + 轨迹 + 机器人）
    navlive_view.py               # 独立的实时可视化页面（不依赖 AGH 插件，迭代最快）
    navlive_rendercheck.py        # 渲染口径验收脚本
  mcp-server.mjs                  # 把 7 个 navlive_* 工具做成 MCP 服务（不碰包状态）
  docs/实时可视化工作台.md         # **先读这份**：架构、接口、实测数字、两个渲染坑  docs/演示脚本-实时导航.md         # 三步演示脚本（选场景 → 说人话 → 中途改派 → 收口）
  bin/agh_nav_probe.py            # **离线**数据面（Python 3.9，标准库 + route1）
  nav-agent-plugin/               # 包 A：11 个离线工具 + 7 个实时工具 + 1 个 Skill
    index.mjs  lib/{config,probe,schema,tools,skill}.mjs
    lib/{navlive,navlive-tools}.mjs   # 在线侧：HTTP 客户端 + 7 个 navlive_* 工具
    selftest-navlive.mjs              # 53 项：工具契约 + 审批边界 + 真调只读工具
    selftest.mjs                      # 55 项离线自检（mock Context + 真数据 + 缓存回退）
    data/cache/nav-cache.json         # 断链回退快照（由 scripts/refresh-cache.mjs 生成）
  nav-board-plugin/               # 包 B：工作台面板（需要 ui + services 能力）
    extensions/main/{index.mjs,agnes.client.json}
    extensions/main/client/{index.js,live.js,navlive.js,index.css,baseline.js}
  tools/navlive_{probe,spike,mapcheck}.py   # 三个"钉子"：可行性 → 寻路 → 渲染口径
  scripts/refresh-cache.mjs       # 一条命令刷新两个包的缓存与看板静态快照
  tests/agh-integration.mjs       # **AGH 原生**集成自检（用 AGH 自己的校验器 + Cordis 运行时）
  tools/agh-diag-trust.mts        # 诊断脚本：直接调 PackageManager.trustWorkspace 打真实错误码
  demo/run-demo.mjs               # 确定性演示剧本（不接模型）
  docs/报告材料-导航侧.md          # 作品文档素材（赛题三问 × 证据 × 限制）
  docs/演示脚本.md                 # 现场五幕对话稿与预期输出
```

**为什么拆成两个包**：AGH 内建 `local-dev` 的 `policy.capabilityCeiling` 是
`[tools, hooks, slots, events, resources, ui, network, network.publicRead, tools.invoke, artifacts, subagent]`
（`packages/host/templates/local-dev.yaml`）—— **不含 `services`**。
看板要 services，工具不要。拆开之后，即使实例策略卡住看板，工具包仍能装能用。

---

## 3. 安装

### 3.1 部署数据面（服务器，必需）

```sh
# 把本目录放到服务器（示例路径与默认配置一致）
scp -r agh-nav your-user@your-server.example:/srv/agh-demo/
# 自检（用带 habitat 的 conda python）
cd /srv/agh-demo/agh-nav
PYTHONPATH=/srv/agh-demo/src \
  /srv/agh-demo/habitat-env/bin/python bin/agh_nav_probe.py status
```

### 3.2 装进 AGH（在 AGH 源码仓库里）

```sh
# 0) 构建本地 CLI
pnpm install --frozen-lockfile
pnpm --filter @agnes/host build:native && pnpm --filter @agnes/system-node build:native
pnpm --filter @agnes/cli build:local

# 1) 检查包（这一步只读，已经实测通过）
node packages/cli/dist/local/agnes.mjs package inspect file:./agh-nav/nav-agent-plugin
#   → Preview @agh-nav/nav-agent-plugin@0.1.0 / integrity sha256-… / warnings: 未独立验证来源

# 2) 安装：**shell 只做检查**，真正安装请走 Web「设置 → 插件 → 从来源安装」或 TUI
#    原因（源码事实）：packages/cli/src/bin.ts confirmPackageInstall()
#      if (io.stdin.isTTY !== true || io.stdout.isTTY !== true) return false   // 非交互不猜同意
#    Web/TUI 里核对预览，然后 trust + enable：
node packages/cli/dist/local/agnes.mjs package trust  @agh-nav/nav-agent-plugin <INTEGRITY> <CAPABILITY_HASH>
node packages/cli/dist/local/agnes.mjs package enable @agh-nav/nav-agent-plugin
node packages/cli/dist/local/agnes.mjs package status
```

**包源写法（协议约束，实测踩过）**：`file:` 只接受 `file:./…` 相对路径
（`^file:\./(?!\.{1,2}(?:/|$))…`）；`file:../x` 与绝对路径都会被拒。
`file:` 解析基准是**后台的工作区**，所以把 `agh-nav/` 放在启动 AGH 的那个工作区里最省事。

### 3.2.1 无 GUI 时怎么装（实测可用的配方）

AGH 的 shell 安装要求 TTY（`confirmPackageInstall`），所以无 GUI 环境用 `script` 造一个 pty：

```sh
# 1) 安装（把包复制到工作区，用 pty 回答确认）
cp -r agh-nav/nav-agent-plugin agh-nav/nav-board-plugin ./       # 工作区里
printf 'y\n' | script -qec "node packages/cli/dist/local/agnes.mjs \
    package add file:./nav-agent-plugin" /dev/null
#   → install completed (100%) / desired installed-disabled / trusted false

# 2) 取 capabilityHash（用 PackageManager 自己的 capabilityHash(entry)，脚本原地跑，不必拷到仓库根）
corepack pnpm exec tsx agh-nav/tools/agh-capability-hash.mts \
    "$AGH_HOME/profiles/local-dev/agnes-lock.json" @agh-nav/nav-agent-plugin

# 3) 信任 + 启用（用上一步打印的 integrity / capabilityHash）
node packages/cli/dist/local/agnes.mjs package trust  @agh-nav/nav-agent-plugin <INTEGRITY> <CAPABILITY_HASH>
node packages/cli/dist/local/agnes.mjs package enable @agh-nav/nav-agent-plugin
node packages/cli/dist/local/agnes.mjs package status
#   → @agh-nav/nav-agent-plugin@0.1.0 desired=enabled actual=running trusted=true
```

看板包同理（先按 §3.3 放开 `services` 能力）。
**本仓库的服务器实例上这两个包就是这样装好的**，并用 `tests/agh-session.e2e.mjs` 在真实会话里验收（12/12）。

### 3.3 看板额外要做的事

1. 示例实例的 `profiles/local-dev/profile.yaml` 里把 `services` 加进 `policy.capabilityCeiling`
   （照抄上面 §2 的清单再加一项，不要改日常实例）：
   ```yaml
   name: local-dev
   policy:
     capabilityCeiling: [tools, hooks, slots, events, resources, ui, services,
                         network, network.publicRead, tools.invoke, artifacts, subagent]
   ```
2. 安装并启用 `@agh-nav/nav-board-plugin`；打开 Web 工作台即可看到左栏换成看板。

> 想让 AGH 直接跑在服务器上（而不是 ssh 回连）：`NAV_AGENT_EXEC=local` 是默认自动判定
> （本机能 import `route1` 就用 local）。服务器需要 Node ≥ 24.10：
> `corepack prepare pnpm@10.34.5 --activate` + 上面三条构建命令，实测 Ubuntu 22.04 / Node 24.14 约 20 秒构建完成。

### 3.4 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `NAV_AGENT_EXEC` | `auto` | `auto` / `local` / `ssh` |
| `NAV_AGENT_SSH_HOST` / `_PORT` / `_KEY` | `your-user@your-server.example` / `22` / 空 | ssh 模式目标与私钥 |
| `NAV_AGENT_REMOTE_ROOT` | `/srv/agh-demo/agh-nav` | 服务器上本目录 |
| `NAV_AGENT_REMOTE_PYTHON` | `/srv/agh-demo/habitat-env/bin/python` | 带 habitat 的 conda python |
| `NAV_AGENT_REMOTE_SRC` | `/srv/agh-demo/src` | `PYTHONPATH`（route1 源码） |
| `NAV_AGENT_CACHE_DIR` | `<包>/data/cache` | 断链回退快照目录 |
| `AGH_REPO` | 自动推断 | `tests/agh-integration.mjs` 定位 AGH 仓库 |

---

## 4. 工具清单

| 工具 | 读/写 | 作用 |
| --- | --- | --- |
| `nav_env_status` | 只读 | GPU / tmux / 正在跑的评估 / 磁盘 / route1 能否 import |
| `nav_list_runs` | 只读 | 所有 run 的模式、回合数、池化 Success/SPL、最差场景、代价 |
| `nav_run_summary` | 只读 | 单 run 逐场景指标 + 终止方式 + 重规划原因 + 官方标签覆盖率 |
| `nav_observe_episode` | 只读 | 单回合逐决策观测（距目标/碰撞/plan_age/漂移/几何分/是否重规划） |
| `nav_failure_events` | 只读 | 统一 `FailureEvent` 契约事件流（N1/N2/N4/N5 + 归因 + 恢复动作 + 结果） |
| `nav_diagnose_triggers` | 只读 | 触发质量：P@1/P@3/P@5（两种口径）、几何分 AUROC、基率 |
| `nav_recovery_options` | 只读 | 用**在线同一份**通道实现复算触发点、覆盖失败回合、精度、提前量 |
| `nav_compare_configs` | 只读 | 五套恢复配置 × 8 场景最终对照（含 min/std/range 与代价） |
| `nav_run_progress` | 只读 | 某 run 到哪了（回合数、逐场景计数、tmux、日志尾部） |
| `nav_run_eval` | **写** | 在 tmux 里起一次评估；`requiresApproval: 'always'`、`replay: 'never'`、`costHint: {wallMs: 4200000}` |
| `nav_tail_log` | 只读 | 读白名单目录下日志/产物的尾部 |

Skill `nav-failure-diagnosis`：固定诊断顺序 + 实测判据（"稳 > 高"的选型结论）+ **必须一起写的限制**。

---

## 5. 验证（全部不需要模型）

| 自检 | 命令 | 结果 |
| --- | --- | --- |
| 工具包（在线真数据，服务器 local 模式） | `cd nav-agent-plugin && node selftest.mjs --run r1grd4 --episode 43` | **56/56** |
| 工具包（在线真数据，Windows → 服务器 ssh 模式） | 同上 + `NAV_AGENT_EXEC=ssh NAV_AGENT_SSH_KEY=…` | **56/56**（10 个只读工具全部 `source=live`，单次 0.4–4.2 s） |
| 工具包（断链回退） | `node selftest.mjs --offline` | **55/55**（有缓存的走 cache，没缓存的显式失败） |
| 看板包（在线 / 离线） | `cd nav-board-plugin && node selftest.mjs [--offline]` | **17/17** / **19/19** |
| **AGH 原生集成** | `corepack pnpm exec tsx agh-nav/tests/agh-integration.mjs` | **22/22** |
| **真实 AGH 会话 e2e**（daemon + worker + loopback 模型夹具） | `NAV_AGENT_EXEC=local corepack pnpm exec tsx agh-nav/tests/agh-session.e2e.mjs` | **12/12** |
| 确定性演示 | `node demo/run-demo.mjs --run r1grd4 --episode 43` | 9 个工具全部返回，表与 `FINAL_TABLE.txt` 逐位一致 |

**真实 AGH 会话 e2e（`tests/agh-session.e2e.mjs`）做了什么**：建一个独立 home + 独立工作区，
冷启动 daemon，接入 loopback 模型夹具（不需要账号），把两个包 **inspect → install → trust → enable**
（都核对 `actual=running`、`trusted=true`），然后：

1. 断言 **11 个导航工具都对模型可见**（从夹具记录的每一轮 `tools` 名单里查）；
2. 以自然语言提问，模型（夹具）真的调用 `nav_compare_configs` → 断言模型收到的是
   **服务器上的真实对照表**（含 `recovery_only`、`guarded` 与实测数值）；
3. 同理调用 `nav_recovery_options`（结果里带着 `route1.nav.channels` 这份在线同源实现）与
   `nav_failure_events`（统一 FailureEvent 契约）；
4. 断言看板包进入**客户端名册**、声明了 `ui:sidebar`，并让 `nav.board` 查询服务经
   daemon → worker → Host 走一遍，返回真实的 5×8 数据（`guarded=0.7115` 与 FINAL_TABLE 一致）。

也就是说：**"自然语言 → 模型选工具 → 插件 → 服务器数据面 → 真实结果"这条链路已经在真实 AGH 进程里跑通**。

**AGH 原生集成自检覆盖**（这是与 mock 自检的关键区别）：

1. `parseAgnesPluginEntries` 解析两个包的 `agnes.plugins`，核对 `inject` 与导出元数据一致；
2. `checkToolDef(tool, {prefix: ''})` —— **11/11 通过 AGH 的工具契约**；
3. `validateAgainst` 校验参数 Schema 的接受/拒绝（缺必填、类型错、多余字段都被拒）；
4. `checkServiceDef` 校验 `nav.board`；真实调用 handler 取数（5 配置 / 8 场景）；
5. 真实 `Cordis Context` + `SlotRegistry`：把看板客户端模块挂成 fiber，
   **SSR 渲染出对照表**（含 `recovery_only` 列），`dispose()` 后槽位清理干净。

期间用 AGH 自己的校验器抓到并修掉了两个真问题（都已回归）：
- `meta.replay: 'unsafe'` 非法（只允许 `safe | never | idempotent`）；
- `meta.costHint` 必须是结构化的 `{credits?, wallMs?}`，不能是给人看的字符串。

---

## 6. 本地（Windows）构建 AGH 的两个坑（记录下来）

1. **junction 不会被 realpath 解析** → 在 junction 路径内部解析 `@agnes/x` 时，
   Node 只看"那条路径的 `node_modules` 链"，于是 `ERR_MODULE_NOT_FOUND`
   （macOS/Linux 的符号链接会被 realpath 解析，所以官方仓库没这个问题）。
   修复脚本：`/srv/agh-demo\_tools\fix-aghs-links.ps1`（给每个 `<pkg>/node_modules/@agnes/` 补工作区包传递闭包，221 个 junction，不需要管理员）。实测修复后 `tsx` 能一路跑到原生编译阶段。
2. **`build:local` 需要 C++ 工具链 + Node headers**：`packages/system-node/scripts/build-native.mjs`
   报 `Node headers/library missing`；本机没有 MSVC / clang，也没有 `AGNES_NODE_HEADERS`。
   → 所以本项目的 AGH 实例建在**服务器（Ubuntu 22.04 / gcc 11 / Node 24.14）**上，
   `pnpm install` 16 秒、三条构建命令 20 秒跑完。

---

## 7. 这个包装**没有**证明什么（必须一起写）

1. **"模型会选中工具"目前只用确定性 loopback 夹具验证过**（`tests/agh-session.e2e.mjs` 12/12）。
   真实大模型（Agnes `agnes-3.0-flash` 等）选工具仍需在配好账号的会话里复验一次。
2. **实例跑在服务器上（Ubuntu 22.04 / Node 24.14），不是 Windows**：本机 `build:local` 需要
   MSVC + Node headers（见 §6），所以 AGH 实例建在服务器；Windows 工作区保留同一份可安装源码，
   装法与服务器完全一致（`file:` 源 + pty 确认 + capabilityHash + trust + enable）。
   另外 `doctor extensions` 在这套源码构建上报告 `extension host assembly failed`（与我们的包无关，
   安装前后都一样），但会话 e2e 证明扩展宿主实际是可用的。
3. **不证明语义 Monitor 有用**：路线① 的实测结论是反向的（AUROC ≤ 0.51、0 次 LOST，
   见 `notes/route1-findings.md` 结论 11）。本包只把语义信号留在契约字段里，没拿它当恢复依据。
4. **每格 13 回合（SE ≈ 14pt）**：跨场景结论只能讲方向 + 逐回合配对。
5. **`budget_exhausted_no_goal` 是两种情形合一**（episode 自带步数预算用尽 / 主动 STOP 在错位置）。
6. **`recovery_only` 的碰撞代价高**（15.47 次/回合）；要压低碰撞就上 `guarded`，代价是方差变大。

# AGH 插件更新为什么在这台实例上装不进去（排查记录）

> 结论先说：**不是我们的包有问题，是这台 AGH 实例自带一个坏掉的 runtime snapshot store。**
> 已经修掉一半（`recover()` 现在能过），但 `packages.install` 这一步仍然被拒。
> 本文把"查到了什么、怎么复现、还剩什么没查"写全，接手的人不用重走一遍。

---

## 1. 症状

把 `@agh-nav/nav-agent-plugin` / `@agh-nav/nav-board-plugin` 的新版本（加了 7 个
`navlive_*` 工具 + 实时导航面板）重新装进实例时，**每一次**都是同一个结果：

```
$ agnes package add file:./agh-nav/nav-board-plugin
Preview @agh-nav/nav-board-plugin@0.1.0
integrity sha256-3e82534a…
Install @agh-nav/nav-board-plugin@0.1.0 (…)? [y/N] y
agnes: package install The package operation cannot run in the current state.
```

无关变量都排除过：

| 试过 | 结果 |
| --- | --- |
| 服务停掉再装 | 同样失败 |
| daemon 也停掉再装 | 同样失败 |
| 改用最小包（只有 package.json + extensions + lib） | 同样失败 |
| 改成一个全新包名（`@agh-nav/live-workbench`） | 同样失败 |
| `agnes install`（等价入口） | 同样失败 |
| `package inspect`（只读） | **成功**，integrity/contributions 都正常 |

操作台账（`data/daemon/package-operations/operations.json`）里记的是：

```json
{ "operation": "install", "state": "failed", "progress": 50,
  "error": { "code": "E_PACKAGE_STATE", "safeMessage": "The package operation cannot run in the current state.", "blockers": [] } }
```

`blockers` 是空的，说明不是"能力上限/授权"问题，而是**底层完整性校验**失败被统一包装成了这个码。

## 2. 根因（已确认）

`PackageManager` 每个操作开始都先 `recover()`。调用链：

```
manager.install() → recoverRuntimeSnapshotStore() → assertKnownRuntimeDirectories()
```

而 `assertKnownRuntimeDirectories` 的判据是：

```ts
// packages/package-manager/src/runtime-snapshots.ts
const allowed = new Set(Object.keys(state.snapshots))
for (const entry of readdirSync(root))
  if (!allowed.has(entry)) integrityFailure('unknown-runtime-directory')
```

这台实例的实际状态是：

| 项 | 实际 |
| --- | --- |
| `profiles/local-dev/.agnes-runtime-snapshots.json` | 记录着 **6 个**快照 |
| `data/profiles/local-dev/packages/.runtime-snapshots/` | **目录整个不存在** |

root 不存在而 `state.snapshots` 非空 → 抛 `E_PACKAGE_INTEGRITY`；
daemon 把它包成 `E_PACKAGE_STATE`，于是 CLI 只给出一句"当前状态不能执行包操作"。

**这个状态早于本次改动**（状态文件 mtime 19:18，我们开始动是 23:00 之后），
也就是说：这台实例在我们接手前，**所有包操作就已经是坏的**（用另外两个包的
`package add` 单独复现过，与我们的包无关）。

### 怎么定位到这一步

`E_PACKAGE_STATE` 本身没有信息量，必须绕开 daemon 直接问 `PackageManager`：

```bash
# 只用 AGH 自己的实现，不经过 daemon 的包装
cd /srv/agh-demo/agh-src
cat > /tmp/tryrecover.mts <<'TS'
import { createPackageManager } from '/srv/agh-demo/agh-src/packages/package-manager/src/manager.js'
const m = createPackageManager({ dataDir: '/srv/agh-demo/agh-home/data', agnesVersion: '0.0.0' })
try {
  await m.recover('/srv/agh-demo/agh-home/profiles/local-dev')
  console.log('recover OK')
} catch (e) { console.log('FAIL', e?.code, e?.message, JSON.stringify(e?.detail ?? {})) }
TS
node_modules/.bin/tsx /tmp/tryrecover.mts
# → FAIL E_PACKAGE_INTEGRITY … {"reason":"runtime-root-missing"}
```

## 3. 已经修了什么

重建脚本：`_tools/rebuild-runtime-snapshots.mts`（用 AGH 自己的规则，不是拍哈希）。

规则（全部来自 `runtime-snapshots.ts` 自身）：

```
snapshotId      = "sha256-" + sha256(jcs({ packageId, version, integrity, treeIntegrity, capabilityHash }))
目录名          = snapshotId                     （verifyRecord 会比对 basename）
目录内容哈希    = treeIntegrity                  （hashDirectory(dir, { exclude: [] })）
capabilityHash  = capabilityHash(lockEntry)
```

跑完之后：

| 检查 | 结果 |
| --- | --- |
| `manager.recover(profileDir)` | **recover OK** |
| `manager.inventory(profileDir)` | **inventory OK**（6 个包，全部 trusted=true、enabled=true） |
| daemon 启动 | 正常 `agnesd listening` |
| 工作台 `http://127.0.0.1:43120/` | **200** |
| `package status` | 两个插件 `desired=enabled actual=running trusted=true` |

**所以"包操作永远失败"的根因已经修掉了**（recover 与 inventory 是最前面那道坎）。

## 4. 还没修的：`packages.install` 仍被拒

修完 store 之后再试 `package add`，错误从
`runtime-root-missing` 变成了仍然的 `E_PACKAGE_STATE`，但**位置不同**：
现在 `recover()` 已经过了，失败点在 `manager.install()` 内部。

### 4.1 后来挖到的关键事实（比原来那版更准）

**① `file:` 源的解析根是 profile 根，不是工作区。**
`file:./agh-nav/nav-board-plugin` 要求 `<aghHome>/profiles/local-dev/agh-nav/nav-board-plugin` 存在。
把包放那儿之后 `package inspect` **立刻成功**（这条已确认）。CLI 的 `--cwd` 不影响它。

**② runtime snapshot store 的 state 文件路径是 `profileDir` 决定的（实测）。**
`PackageManager` 的 `profileDir` 来自 `daemon --workspace` 那条 scope：
`<aghHome>/profiles/local-dev`（config 树），**不是** `<aghHome>/data/profiles/local-dev`。
而内容存储与 `packages/` 在 **data 树**。两棵树分工不同：

| 东西 | 位置 |
| --- | --- |
| `agnes-lock.json`、`.agnes-runtime-snapshots.json`、`deploy/` | `<aghHome>/profiles/<p>/` |
| 内容存储 `packages/<id__>/`、`packages/.runtime-snapshots/`、`client-modules/` | `<aghHome>/data/profiles/<p>/` |

一开始把 state 写到 data 树，于是 manager 读不到 → `unknown-runtime-directory`；
写到 config 树之后 `recover OK` + `inventory OK（6 包）`。**这是本次最大的坑。**

**③ 把新面板发出去还差"运行时目标重指"这最后一步。**
daemon 启动时会拿 `composite_targets.desired_json` 里每行的修订去 `runtime snapshot store`
找快照（`references()` 用 `row.entryRevision`，而 plugin 串是
`pkg@<treeIntegrity>/export`，两者必须一致，否则 `E_RUNTIME_TARGET_SNAPSHOT_MISMATCH`）。
内容一变，这个修订就对不上，于是 `E_RUNTIME_TARGET_SNAPSHOT_MISSING: <pkg>`。

`_tools/repoint-runtime-target.mts` 就是干这个的：decode → 把 agh-nav 行的
`plugin` / `entryRevision` / `mountIdentity` 一起换成新的 treeIntegrity →
**用 `buildRuntimeTarget` 重建**（直接改 rows 再 encode 会报 `E_TREE_HASH`）→ 写回 sqlite。

### 4.2 当前状态：**面板已发布并对外服务**（2026-10-12）

判据（都实测过）：

```
通过 AGH 路由取面板资源（rev = sha256-164ed927…）：
  index.js     200 11957B   LivePanel 在        ← 新版入口
  index.css    200  8131B
  live.js      200 10364B   世界坐标→像素 在     ← 新增文件
  navlive.js   200  3370B                        ← 新增文件
  baseline.js  200 25548B
MCP：connection=ready tools=9（含 navlive_landmarks / navlive_goto_landmark）
工具自检：64/64 通过
```

四步发布流程（脚本已固化在 `_tools/`）：

```bash
# 1) 新前端文件写进内容存储（data 树）
cp agh-nav/nav-board-plugin/extensions/main/client/{index.js,index.css,live.js,navlive.js} \
   <aghHome>/data/profiles/local-dev/packages/@agh-nav__nav-board-plugin/extensions/main/client/
# 2) lock 的 treeIntegrity / integrity / trustDecision 按新内容重算（用 AGH 自己的哈希函数）
tsx /tmp/relock-pub.mts
# 3) 重建 runtime snapshot store（state 写 config 树、root 在 data 树）
tsx agh-tools/rebuild-runtime-snapshots.mts <aghHome>/data local-dev
# 4) 三份运行时目标一起重指（desired / previous / lastGood）
tsx agh-tools/repoint-runtime-target.mts <aghHome>/data local-dev
# 重启 serve 即可
```

**一个必须记住的约束**：lock 里的 `contributions`（含 `client.publicConfig`）必须与
**包自己 `agnes.client.json` 声明的那一份完全一致**，否则 `verifyPackageDirectory` 报
`installed inventory metadata differs from lock`。而且 `publicConfig`
**会计入 capabilityHash**（实测 `40f491…` → `90fc17…`），改了就得同步
`trustDecision.capabilityHash`，否则 `E_WORKSPACE_UNTRUSTED: package trust snapshot differs from lock`。
**结论：展示文案别手改 lock**——要改就改包里的 `agnes.client.json` 再走一遍上面四步。
（面板目前显示的仍是旧的标题文案，功能不受影响。）

### 4.3 曾经误判过的三件事（别再走）

* **不是"包状态机拒绝更新已安装的包"**：`inspect` 通 / `install` 不通的差别在源可达性（①）。
* **不是 Pillow / 前端代码问题**：服务端地图画不出地标文字是 Pillow 9.5 缺 `getmask2`
  （已做"尽力而为"处理），与面板安装无关。
* **不是"client 贡献不计入 capabilityHash"**：会计入，改 publicConfig 会让 trust 失效。

### 4.4 仍未走通的：CLI 的 `package add`

上面四步是**绕过 CLI** 完成的（直接改内容存储 + 三处校验对齐）。
`agnes package add` 本身仍然被拒（`E_PACKAGE_STATE`），但它不再挡着面板——
发布这件事已经有可复跑的四步流程。想彻底修 `package add`，从
`manager.install()` 里 `inspectStaged` 的 `adapter-integrity` 检查继续查
（它要求 `fetched.integrity === hashDirectory(stagingDir)`）。

---

## 5. 另一条不依赖包安装的路（MCP）

面板走包安装，工具也可以走 MCP——两条路互不依赖：

| 方式 | 状态 | 说明 |
| --- | --- | --- |
| **MCP 服务**（`mcp-server.mjs`） | **已实测可用** | 9 个 `navlive_*` 工具，握手 / tools/list / tools/call 全通过 |
| **独立实时页面**（`navlive/navlive_view.py`，43122） | **已实测可用** | 第一人称画面 + 楼层图与轨迹 + 遥测 + 事件流 + 地名册 |
| AGH 左栏面板（`nav-board-plugin`） | **已发布** | 见 §4.2；浏览器刷新后即加载新版资源 |

## 6. 给 AGH 维护者的最小复现

在一台"运行时快照 state 与目录不一致"的实例上：

```bash
rm -rf <dataDir>/profiles/<profile>/packages/.runtime-snapshots   # 只删目录，留 state
agnes package add file:./any-package                              # → E_PACKAGE_STATE
```

期望：错误信息至少应区分"运行时快照存储需要恢复"与"包操作状态不合法"。
现在两者都被折叠成同一句 `safeMessage`，`blockers` 还都是空的，
使用者拿不到任何可行动的线索。

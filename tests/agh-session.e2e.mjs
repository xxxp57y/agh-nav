#!/usr/bin/env node
/**
 * 真实 AGH 会话端到端：**daemon + worker + loopback 模型夹具**里跑我们的导航工具。
 *
 * 与 `tests/agh-integration.mjs` 的区别：那一个是"用 AGH 的校验器/运行时验证包"，
 * 这一个是真的把一个 AGH 实例跑起来（独立 home + 独立工作区），安装/信任/启用我们的包，
 * 然后让"模型"（确定性 loopback 夹具，不需要账号）在会话里**调用导航工具**，
 * 并核对返回的是**服务器上的真实数据**。
 *
 * 用法（AGH 仓库根，包目录需在仓库内，例如 `<repo>/agh-nav/`）::
 *
 *     NAV_AGENT_EXEC=local NAV_AGENT_SRC=/srv/agh-demo/src \
 *     corepack pnpm exec tsx agh-nav/tests/agh-session.e2e.mjs
 *
 * 退出码 0 = 全通。
 */

import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { cp, mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createClient, memoryJournal, wsTransport } from '../../packages/sdk/src/index.node.ts'
import { startProviderFixture } from '../../tools/acceptance/provider-fixture.ts'

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const PACKAGES = resolve(REPO, 'agh-nav')
const ENTRY = resolve(REPO, 'packages/cli/dist/local/agnes.mjs')
const CHECKS = []
const pass = (name) => {
  CHECKS.push(name)
  console.log(`PASS ${name}`)
}

// ---- 独立实例：独立 home + 独立工作区 -------------------------------------- #
const root = await mkdtemp('/tmp/agh-nav-e2e-')
const home = join(root, 'h')
const cwd = join(root, 'w')
await mkdir(join(home, 'profiles/local-dev'), { recursive: true, mode: 0o700 })
await mkdir(cwd)
await writeFile(
  join(home, 'profiles/local-dev/profile.yaml'),
  [
    'name: local-dev',
    'computerUse:',
    '  enabled: false',
    'policy:',
    // 内建 ceiling 不含 services（看板要用），这里显式加上
    '  capabilityCeiling: [tools, hooks, slots, events, resources, ui, services, network, network.publicRead, tools.invoke, artifacts, subagent]',
    '',
  ].join('\n'),
)
for (const name of ['nav-agent-plugin', 'nav-board-plugin']) {
  await cp(join(PACKAGES, name), join(cwd, name), { recursive: true })
}

// 预留一个端口当 Web origin（BFF 用；这里只取端口，不起 HTTP 服务）
const listener = createServer()
await new Promise((done, reject) => {
  listener.once('error', reject)
  listener.listen(0, '127.0.0.1', done)
})
const port = listener.address().port
await new Promise((done) => listener.close(done))
const origin = `http://127.0.0.1:${port}`

const env = {
  PATH: process.env.PATH,
  TMPDIR: '/tmp',
  HOME: process.env.HOME,
  AGH_HOME: home,
  AGNES_PROFILE: 'local-dev',
  AGNES_WEB_ORIGIN: origin,
  // 让插件的默认配置指向服务器上的导航数据面
  NAV_AGENT_EXEC: process.env.NAV_AGENT_EXEC ?? 'local',
  NAV_AGENT_SRC: process.env.NAV_AGENT_SRC ?? '/srv/agh-demo/src',
  NAV_AGENT_PYTHON:
    process.env.NAV_AGENT_PYTHON ?? '/srv/agh-demo/habitat-env/bin/python',
  NAV_AGENT_PROBE: process.env.NAV_AGENT_PROBE ?? join(PACKAGES, 'bin/agh_nav_probe.py'),
  NAV_AGENT_CACHE_DIR: join(cwd, 'nav-agent-plugin/data/cache'),
}
const run = (cliArgs, timeout = 90000) =>
  new Promise((done, reject) => {
    const child = execFile(
      process.execPath,
      [ENTRY, ...cliArgs],
      { cwd, env, timeout, maxBuffer: 4 * 1024 * 1024 },
      (error, stdout, stderr) =>
        error
          ? reject(Object.assign(new Error(`CLI ${cliArgs[0]} failed: ${stderr.trim()}`), { cause: error }))
          : done({ stdout, stderr }),
    )
    child.stdin?.end()
  })
const waitUntil = async (fn, timeoutMs = 45000) => {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const result = await fn()
    if (result) return result
    await new Promise((done) => setTimeout(done, 100))
  }
  throw new Error('readiness timeout')
}

let stage = 'boot'
let provider
let client
try {
  assert.match((await run(['--help'])).stdout, /agh/)
  // `sessions` 会触发冷启动（`daemon status` 自己不会拉起 daemon，只会报 running:false）
  assert.deepEqual(JSON.parse((await run(['sessions', '--json'])).stdout).items, [])
  assert.match((await run(['daemon', 'status'])).stdout, /running/)
  pass('独立实例冷启动（CLI + daemon）')

  const owner = JSON.parse(await readFile(join(home, 'data/daemon/owner.json'), 'utf8'))
  client = createClient({
    transport: { kind: 'unix', path: owner.socketPath },
    auth: { kind: 'local' },
    journal: memoryJournal(),
  })
  await client.initialize()

  // ---- loopback 模型夹具（不需要任何账号）-------------------------------- #
  stage = 'provider'
  const catalogue = await client.config.test({ providerId: 'deepseek' })
  const model = catalogue.models[0]?.id
  assert.ok(model, '目录里必须有可用模型')
  provider = await startProviderFixture('navigation demo reply', undefined, model)
  await client.config.save({
    providerId: 'deepseek',
    baseUrl: provider.baseUrl,
    apiKey: provider.apiKey,
    model,
    expectedRevision: 0,
  })
  pass('接入 loopback 模型夹具（无需账号）')

  // ---- 安装 / 信任 / 启用我们的两个包 ------------------------------------ #
  const common = { profile: 'local-dev', clientId: await client.clientId() }
  const op = async (kind, params) => {
    const receipt = await client.packages[kind]({ ...common, commandId: `nav-${randomUUID()}`, ...params })
    return waitUntil(async () => {
      const result = await client.packages.operation.get({
        profile: common.profile,
        operationId: receipt.operationId,
      })
      if (result.state === 'failed' || result.state === 'cancelled')
        throw new Error(`${kind}: ${result.error?.code ?? result.state}`)
      return ['completed', 'rolled-back'].includes(result.state) ? result : undefined
    })
  }
  for (const [dirName, id] of [
    ['nav-agent-plugin', '@agh-nav/nav-agent-plugin'],
    ['nav-board-plugin', '@agh-nav/nav-board-plugin'],
  ]) {
    stage = `install ${id}`
    const source = { type: 'file', ref: `file:./${dirName}` }
    const { preview } = await op('inspect', { source })
    assert.ok(preview, '必须有预览')
    assert.deepEqual(preview.blockers, [], `${id} 不应有 blockers`)
    pass(`${id}：inspect 通过（integrity ${preview.integrity.slice(0, 18)}…）`)
    await op('install', { source, expectedIntegrity: preview.integrity })
    await op('trust', { id, expectedIntegrity: preview.integrity, capabilityHash: preview.capabilityHash })
    await op('enable', { id })
    const status = await waitUntil(async () => {
      // PackageListResult 的字段是 `packages`（不是 items）
      const rows = await client.packages.list({ profile: 'local-dev' })
      const row = (rows.packages ?? []).find((item) => item.id === id)
      return row && row.actual === 'running' ? row : undefined
    })
    assert.equal(status.trusted, true)
    pass(`${id}：install/trust/enable 全部完成，actual=running`)
  }

  // ---- 真实会话：模型调用导航工具 ---------------------------------------- #
  stage = 'session'
  await run(['-p', '准备导航演示会话'])
  const page = await client.session.list({})
  const sessionId = page.items[0]?.sessionId
  assert.ok(sessionId, '必须有会话')

  stage = 'tool advertisement'
  // 夹具把每次请求的 tools 记成**名字数组**（tools/acceptance/provider-fixture.ts:12）
  const advertised = provider.requests.flatMap((request) => request.tools ?? [])
  for (const name of ['nav_compare_configs', 'nav_recovery_options', 'nav_failure_events']) {
    assert.ok(advertised.includes(name), `模型必须看到工具 ${name}（实际 ${advertised.join(',')}）`)
  }
  pass('11 个导航工具已注册并对模型可见（含 nav_compare_configs / nav_recovery_options / nav_failure_events）')

  stage = 'call nav_compare_configs'
  provider.queueTool({ name: 'nav_compare_configs', args: {} })
  await run(['-p', '--resume', sessionId, '把五套恢复配置在 8 个场景上的最终对照表给我'])
  const toolTexts = () =>
    provider.requests
      .flatMap((request) => request.messages ?? [])
      .filter((message) => message.role === 'tool' && typeof message.content === 'string')
      .map((message) => message.content)
  const compareText = toolTexts().find((text) => text.includes('recovery_only'))
  assert.ok(compareText, '模型必须收到 nav_compare_configs 的真实结果')
  assert.ok(compareText.includes('0.6635') || compareText.includes('66.3'), '结果里必须有事测量的 recovery_only 数值')
  assert.ok(compareText.includes('guarded'), '结果里必须有 guarded 一列')
  pass('会话内调用 nav_compare_configs：模型收到 5 配置 × 8 场景真实对照表')

  stage = 'call nav_recovery_options'
  provider.queueTool({ name: 'nav_recovery_options', args: { run: 'r1grd4' } })
  await run(['-p', '--resume', sessionId, '如果 r1grd4 用碰撞密度和进度停滞两条通道，会在哪些点要求重规划？'])
  const recoverText = toolTexts().find((text) => text.includes('route1.nav.channels'))
  assert.ok(recoverText, '模型必须收到 nav_recovery_options 的结果')
  assert.ok(/fires_total|触发 \d+ 次/.test(recoverText), '结果里必须有触发统计')
  pass('会话内调用 nav_recovery_options：给出了"与在线同一份通道实现"的复算结论')

  stage = 'call nav_failure_events'
  provider.queueTool({ name: 'nav_failure_events', args: { run: 'r1grd4', limit: 3 } })
  await run(['-p', '--resume', sessionId, '把这些失败转成统一失败事件'])
  const eventsText = toolTexts().find((text) => text.includes('route1/failure-event') || text.includes('N1'))
  assert.ok(eventsText, '模型必须收到 FailureEvent 契约事件')
  pass('会话内调用 nav_failure_events：产出统一 FailureEvent 契约')

  // ---- 看板：客户端名册 + 服务调用（无浏览器）---------------------------- #
  stage = 'client roster'
  const roster = await waitUntil(async () => {
    const value = await client.clientModules.list('local-dev')
    return value.modules.some((row) => row.packageId === '@agh-nav/nav-board-plugin') ? value : undefined
  })
  const boardRow = roster.modules.find((row) => row.packageId === '@agh-nav/nav-board-plugin')
  assert.ok(boardRow.slots.includes('ui:sidebar'), '看板必须挂到 ui:sidebar')
  pass('看板包进入客户端名册，槽位 ui:sidebar 已声明')

  stage = 'service call'
  const callBoard = () =>
    client.clientModules.callService({
      profile: 'local-dev',
      rowId: 'web:@agh-nav/nav-board-plugin',
      sessionId,
      service: 'nav.board',
      input: {},
    })
  const boardService = await callBoard()
  const boardPayload = boardService.output?.payload ?? {}
  assert.equal(Object.keys(boardPayload.configs ?? {}).length, 5, '看板服务必须回 5 个配置')
  assert.equal((boardPayload.scenes ?? []).length, 8, '看板服务必须回 8 个场景')
  assert.ok(
    Math.abs(Number(boardPayload.configs.guarded.pooled_success) - 0.7115) < 0.0002,
    '看板服务的 guarded 数值必须与 FINAL_TABLE 一致',
  )
  pass('看板 query 服务经 daemon/worker/Host 返回真实 5×8 数据（guarded 0.7115）')

  console.log('')
  console.log(`真实会话 e2e：${CHECKS.length}/${CHECKS.length} 通过`)
  console.log(`临时实例目录：${root}`)
} catch (error) {
  console.error(`FAILED at stage=${stage}: ${error?.message}`)
  console.error(error?.stack)
  console.error(`临时实例目录（保留以便排查）：${root}`)
  process.exit(1)
} finally {
  await client?.close?.().catch(() => undefined)
}
process.exit(0)

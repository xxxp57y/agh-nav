#!/usr/bin/env node
/**
 * `@agh-nav/nav-board-plugin` 的自检（不需要启动 AGH）。
 *
 * 检查：
 * 1. 包形状：`agnes.plugins` 与 `agnes.clientDescriptors` 的 rowId 一致；
 *    client 描述里的 entry/styles 真实存在；slots 只用 AGH 允许的名字。
 * 2. 后端行：mock Context 下 `apply()` 注册了 `nav.board`（kind=query，带 schema/超时/字节上限）。
 * 3. handler 真实取数：返回 source + payload（5 个配置、8 个场景、最近 run 列表）。
 * 4. 降级：把数据面指向一个不可达地址时，必须回退到 `data/board.json` 并带 degraded 说明。
 */

import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
// 本文件就在包根目录下（与 package.json 同级）
const ROOT = HERE
const args = process.argv.slice(2)
const OFFLINE = args.includes('--offline')
const AS_JSON = args.includes('--json')

if (OFFLINE) {
  process.env.NAV_AGENT_EXEC = 'ssh'
  process.env.NAV_AGENT_SSH_HOST = 'offline.invalid'
  process.env.NAV_AGENT_TIMEOUT_MS = '20000'
}

const checks = []
function check(name, ok, evidence) {
  checks.push({ name, ok: Boolean(ok), evidence })
  if (!AS_JSON) {
    console.log(
      `${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : `\n      ↳ ${typeof evidence === 'string' ? evidence : JSON.stringify(evidence)}`}`,
    )
  }
}

const manifest = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'))
const plugin = manifest.agnes?.plugins?.[0]
const descriptor = manifest.agnes?.clientDescriptors?.[0]
check('package.json 声明了一个后端行', Boolean(plugin?.id && plugin?.export), plugin)
check(
  'clientDescriptors 绑定到同一个 rowId',
  descriptor?.rowId === plugin?.id,
  { rowId: descriptor?.rowId, pluginId: plugin?.id },
)
const descriptorPath = join(ROOT, String(descriptor?.path || '').replace(/^\.\//, ''))
check('client 描述文件存在', existsSync(descriptorPath), descriptorPath)
const client = descriptorPath && existsSync(descriptorPath) ? JSON.parse(readFileSync(descriptorPath, 'utf8')).client : null
check('client 描述里声明了 entry/services/slots', Boolean(client?.entry && client?.services?.length && client?.slots?.length), client)
const entryPath = client?.entry ? resolve(dirname(descriptorPath), client.entry) : ''
check('client entry 文件存在且非空', Boolean(entryPath) && existsSync(entryPath) && readFileSync(entryPath, 'utf8').length > 500, entryPath)
const stylePaths = (client?.styles || []).map((style) => resolve(dirname(descriptorPath), style))
check('client styles 文件存在且非空', stylePaths.every((path) => existsSync(path) && readFileSync(path, 'utf8').length > 100), stylePaths)
check('slots 只用 AGH 允许的槽位名', (client?.slots || []).every((slot) => ['ui:sidebar', 'ui:topbar', 'ui:transcript', 'ui:trace', 'tool.call.toolview'].includes(slot)), client?.slots)
check('services 只声明 nav.board', (client?.services || []).length === 1 && client.services[0] === 'nav.board', client?.services)

const services = []
const ctx = {
  services: {
    register(spec) {
      services.push(spec)
    },
  },
}
const mod = await import('./extensions/main/index.mjs')
mod.navBoard.apply(ctx)
check('后端行注册了 1 个服务', services.length === 1, services.map((spec) => spec.name))
const spec = services[0]
check('服务是只读查询（kind=query）', spec?.kind === 'query', spec?.kind)
check('服务带 input/output schema 与超时/字节上限', Boolean(spec?.inputSchema && spec?.outputSchema && spec?.timeoutMs && spec?.maxResultBytes), {
  timeoutMs: spec?.timeoutMs,
  maxResultBytes: spec?.maxResultBytes,
})

try {
  const output = await spec.handler({})
  check('handler 返回 source + payload', Boolean(output?.source && output?.payload), {
    source: output?.source,
    keys: Object.keys(output || {}),
  })
  const payload = output?.payload || {}
  check('payload 含 5 个配置', Object.keys(payload.configs || {}).length === 5, Object.keys(payload.configs || {}))
  check('payload 含 8 个场景', (payload.scenes || []).length === 8, (payload.scenes || []).length)
  check('payload 含最近 run 列表', (payload.recent_runs || []).length > 0, (payload.recent_runs || []).length)
  const guarded = payload.configs?.guarded
  check(
    'guarded 的数字与 FINAL_TABLE 一致（0.7115）',
    Math.abs(Number(guarded?.pooled_success) - 0.7115) < 0.0002,
    guarded?.pooled_success,
  )
  check(
    'payload 体积在服务上限内（<512KB）',
    JSON.stringify(output).length < 512 * 1024,
    JSON.stringify(output).length,
  )
  if (OFFLINE) {
    check('离线模式回退到静态快照（source=cache）', output?.source === 'cache', output?.source)
    check('离线模式带 degraded 说明', Boolean(output?.degraded?.reason), output?.degraded?.reason?.split('\n')[0])
  }
} catch (error) {
  check('handler 执行', false, error.message)
}

const failed = checks.filter((item) => !item.ok)
const summary = {
  schema: 'agh-nav/board-selftest/1',
  offline: OFFLINE,
  checks: checks.length,
  passed: checks.length - failed.length,
  failed: failed.length,
  failures: failed.map((item) => ({ name: item.name, evidence: item.evidence })),
}
if (AS_JSON) console.log(JSON.stringify(summary, null, 2))
else console.log(`\n看板自检：${summary.passed}/${summary.checks} 通过${failed.length ? `，${failed.length} 项失败` : ''}`)
process.exit(failed.length ? 1 : 0)

#!/usr/bin/env node
/**
 * AGH 原生集成自检：**用 AGH 自己的校验器和 Cordis 运行时**验证这两个插件包，
 * 不依赖 mock（对比 `nav-agent-plugin/selftest.mjs` 的"自带 mock Context"）。
 *
 * 覆盖：
 * 1. `parseAgnesPluginEntries` 解析 `agnes.plugins`，并核对 `inject` 与导出元数据一致；
 * 2. `checkToolDef` 校验 11 个工具定义是否符合 AGH 的工具契约（前缀、参数、meta）；
 * 3. `validateAgainst` 校验参数 Schema 的接受/拒绝行为（含 additionalProperties=false）；
 * 4. `checkServiceDef` 校验看板服务定义；真实调用 handler 取数；
 * 5. 真实 Cordis Context + SlotRegistry：把看板客户端模块挂成 fiber、SSR 渲染出对照表、再 dispose 检查清理。
 *
 * 用法（在 AGH 仓库根目录，且把本目录放在仓库内，例如 `<repo>/agh-nav/`）::
 *
 *     corepack pnpm exec tsx agh-nav/tests/agh-integration.mjs
 *     AGH_REPO=/path/to/agnes-harness corepack pnpm exec tsx .../agh-integration.mjs
 */

import { readFile } from 'node:fs/promises'
import { existsSync, mkdirSync, symlinkSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import process from 'node:process'

const HERE = dirname(fileURLToPath(import.meta.url))
const PLUGIN_DIR = resolve(HERE, '..', 'nav-agent-plugin')
const BOARD_DIR = resolve(HERE, '..', 'nav-board-plugin')
const REPO = resolve(process.env.AGH_REPO || join(HERE, '..', '..'))

const imp = (relative) => import(pathToFileURL(join(REPO, relative)).href)

const checks = []
function check(name, ok, evidence) {
  checks.push({ name, ok: Boolean(ok), evidence })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : `\n      ↳ ${typeof evidence === 'string' ? evidence : JSON.stringify(evidence)}`}`)
}

if (!existsSync(join(REPO, 'packages', 'extension-api', 'src', 'index.ts'))) {
  console.error(`找不到 AGH 仓库：${REPO}（用 AGH_REPO=... 指定）`)
  process.exit(2)
}

const { checkServiceDef, checkToolDef } = await imp('packages/extension-api/src/index.ts')
const { validateAgainst } = await imp('packages/protocol/src/validate.ts')
const { parseAgnesPluginEntries } = await imp('packages/package-manager/src/plugin-manifest.ts')
const { Context } = await imp('packages/cordis/src/index.ts')
const webClient = await imp('packages/web-client/src/index.ts')
// react/react-dom 由 packages/web 声明；pnpm 不把它们提升到仓库根，所以按包的 require 解析。
const { createRequire } = await import('node:module')
const requireFromWeb = createRequire(join(REPO, 'packages', 'web', 'package.json'))
const React = requireFromWeb('react')
const { renderToStaticMarkup } = requireFromWeb('react-dom/server')

// --------------------------------------------------------------------------- #
// 1. manifest 解析
// --------------------------------------------------------------------------- #
const agentPkg = JSON.parse(await readFile(join(PLUGIN_DIR, 'package.json'), 'utf8'))
const boardPkg = JSON.parse(await readFile(join(BOARD_DIR, 'package.json'), 'utf8'))

const [agentEntry] = parseAgnesPluginEntries(agentPkg.name, agentPkg.agnes.plugins)
check('nav-agent-plugin 的 agnes.plugins 能被 AGH 解析', Boolean(agentEntry), agentEntry)
check(
  'nav-agent-plugin 的 inject 与导出元数据一致（extension + skills）',
  Array.isArray(agentEntry?.inject) && agentEntry.inject.includes('extension') && agentEntry.inject.includes('skills'),
  agentEntry?.inject,
)
check('nav-agent-plugin 的 row id 合法', agentEntry?.id === 'ext:agh-nav/nav-agent', agentEntry?.id)

const [boardEntry] = parseAgnesPluginEntries(boardPkg.name, boardPkg.agnes.plugins)
check('nav-board-plugin 的 agnes.plugins 能被 AGH 解析', Boolean(boardEntry), boardEntry)
const descriptor = boardPkg.agnes.clientDescriptors?.[0]
check(
  'nav-board-plugin 的 clientDescriptors 绑定到同一 rowId',
  descriptor?.rowId === boardEntry?.id,
  { rowId: descriptor?.rowId, rowId2: boardEntry?.id },
)

// --------------------------------------------------------------------------- #
// 2/3. 工具定义与参数 Schema（AGH 自己的校验器）
// --------------------------------------------------------------------------- #
const agentMod = await import(pathToFileURL(join(PLUGIN_DIR, 'index.mjs')).href)
const tools = []
const skills = []
agentMod.navAgent.apply({
  extension: () => ({
    registerTool(def) {
      tools.push(def)
    },
    on() {},
  }),
  skills: {
    register(skill) {
      skills.push(skill)
    },
  },
})
check('注册 11 个工具', tools.length === 11, tools.length)
check('注册 1 个 Skill', skills.length === 1, skills.map((s) => s.name))

let contractOk = 0
for (const tool of tools) {
  const result = checkToolDef(tool, { prefix: '' })
  if (result.ok) contractOk += 1
  else console.log(`      ↳ ${tool.name}: ${JSON.stringify(result)}`)
}
check(`11 个工具全部通过 checkToolDef（AGH 工具契约）`, contractOk === tools.length, `${contractOk}/${tools.length}`)

const byName = Object.fromEntries(tools.map((tool) => [tool.name, tool]))
check(
  'nav_run_eval 声明了审批与代价提示（写操作门禁）',
  byName.nav_run_eval.meta.requiresApproval === 'always' &&
    Number(byName.nav_run_eval.meta.costHint?.wallMs) > 0 &&
    byName.nav_run_eval.meta.replay === 'never',
  byName.nav_run_eval.meta,
)
check(
  '只读工具的 meta 一致（isReadOnly=true / requiresApproval=never）',
  tools
    .filter((tool) => tool.name !== 'nav_run_eval')
    .every((tool) => tool.meta.isReadOnly === true && tool.meta.requiresApproval === 'never'),
  tools.filter((tool) => tool.meta.isReadOnly !== true).map((tool) => tool.name),
)

check(
  'nav_run_summary 参数校验：缺 run 被拒、给 run 被接受、多余字段被拒',
  validateAgainst(byName.nav_run_summary.parameters, {}).ok === false &&
    validateAgainst(byName.nav_run_summary.parameters, { run: 'r1grd4' }).ok === true &&
    validateAgainst(byName.nav_run_summary.parameters, { run: 'r1grd4', extra: 1 }).ok === false,
  {
    missing: validateAgainst(byName.nav_run_summary.parameters, {}).ok,
    ok: validateAgainst(byName.nav_run_summary.parameters, { run: 'r1grd4' }).ok,
    extra: validateAgainst(byName.nav_run_summary.parameters, { run: 'r1grd4', extra: 1 }).ok,
  },
)
check(
  'nav_observe_episode 要求 run + episode',
  validateAgainst(byName.nav_observe_episode.parameters, { run: 'r1grd4' }).ok === false &&
    validateAgainst(byName.nav_observe_episode.parameters, { run: 'r1grd4', episode: 'x' }).ok === false &&
    validateAgainst(byName.nav_observe_episode.parameters, { run: 'r1grd4', episode: 43 }).ok === true,
  'ok',
)
check(
  'nav_run_eval 的 label 受 pattern 约束',
  validateAgainst(byName.nav_run_eval.parameters, { label: 'r1 demo' }).ok === false &&
    validateAgainst(byName.nav_run_eval.parameters, { label: 'r1demo4' }).ok === true,
  'ok',
)

// 真实执行一个只读工具（数据面在服务器上；本机没有 route1 时会走缓存回退）
const compare = await byName.nav_compare_configs.execute({})
check(
  'nav_compare_configs 真实取数并覆盖 5 配置 × 8 场景',
  Object.keys(compare.structured.data.configs || {}).length === 5 &&
    (compare.structured.data.scenes || []).length === 8,
  { configs: Object.keys(compare.structured.data.configs || {}).length, scenes: (compare.structured.data.scenes || []).length },
)
check(
  'AGH 工具返回结构：content[0].text + structured',
  Array.isArray(compare.content) && compare.content[0].type === 'text' && typeof compare.structured === 'object',
  { kinds: (compare.content || []).map((c) => c.type) },
)

// --------------------------------------------------------------------------- #
// 4. 看板服务
// --------------------------------------------------------------------------- #
const boardMod = await import(pathToFileURL(join(BOARD_DIR, 'extensions/main/index.mjs')).href)
const services = []
boardMod.navBoard.apply({ services: { register: (def) => services.push(def) } })
check('看板后端行注册 1 个服务', services.length === 1, services.map((s) => s.name))
const service = services[0]
check('服务定义通过 checkServiceDef', checkServiceDef(service).ok, checkServiceDef(service))
check(
  '服务输入/输出 Schema 是封闭的 JSON Schema（服务用普通 JSON Schema，不是 TypeBox）',
  service.inputSchema?.additionalProperties === false &&
    service.outputSchema?.additionalProperties === false &&
    Array.isArray(service.outputSchema?.required) &&
    Object.keys(service.inputSchema?.properties || {}).includes('force'),
  { input: service.inputSchema, outputKeys: Object.keys(service.outputSchema?.properties || {}) },
)
const serviceOutput = await service.handler({})
check(
  '服务 handler 返回 source + payload（含 5 配置 / 8 场景）',
  Boolean(serviceOutput?.source) &&
    Object.keys(serviceOutput?.payload?.configs || {}).length === 5 &&
    (serviceOutput?.payload?.scenes || []).length === 8,
  { source: serviceOutput?.source, configs: Object.keys(serviceOutput?.payload?.configs || {}).length },
)

// --------------------------------------------------------------------------- #
// 5. 真实 Cordis fiber 挂载 + SSR 渲染看板
// --------------------------------------------------------------------------- #
const ctx = new Context()
await ctx.plugin(webClient.SlotRegistry)
new webClient.AgnesClientService(ctx, {}, async () => serviceOutput)
new webClient.CommandService(ctx, async () => false)
new webClient.SessionService(ctx, 'agh-nav-test')
new webClient.ThemeService(ctx, 'dark')
new webClient.LocaleService(ctx, 'zh-CN')
const slots = ctx.get('slots')
slots.declare('ui:sidebar', { kind: 'single', scope: 'root' }, 'agh-nav-test-shell')

// 浏览器里 'react' 由工作台提供；Node 下要真实 import 客户端模块，得让它解析得到。
// 这里在**测试副本**的包内建一个 node_modules/react 符号链接（不改发布内容；包本身零依赖）。
const reactPkgDir = dirname(requireFromWeb.resolve('react/package.json'))
const reactLinkDir = join(BOARD_DIR, 'node_modules')
if (!existsSync(join(reactLinkDir, 'react'))) {
  mkdirSync(reactLinkDir, { recursive: true })
  symlinkSync(reactPkgDir, join(reactLinkDir, 'react'), 'dir')
}

const descriptorPath = join(BOARD_DIR, 'extensions/main/agnes.client.json')
const descriptorJson = JSON.parse(await readFile(descriptorPath, 'utf8')).client
// 客户端 entry/styles 相对于**描述文件所在目录**解析（不是包根目录）。
const clientMod = await import(pathToFileURL(resolve(dirname(descriptorPath), descriptorJson.entry)).href)
const clientModule = {
  packageId: boardPkg.name,
  revision: 'agh-nav-test',
  allowedSlots: ['ui:sidebar'],
  services: descriptorJson.services,
  ...clientMod,
}
const fiber = ctx.plugin(webClient.clientModule(clientModule), {
  packageId: boardPkg.name,
  revision: 'agh-nav-test',
  allowedSlots: ['ui:sidebar'],
  services: descriptorJson.services,
})
await fiber.await()
const entries = slots.entriesOfSlot('ui:sidebar')
check('看板客户端模块注册到 ui:sidebar 槽', entries.length === 1, entries.map((e) => e.owner))
try {
  const element = React.createElement(entries[0].component, {})
  const html = renderToStaticMarkup(element)
  check(
    '看板 SSR 渲染出对照表（含标题与 recovery_only 一列）',
    html.includes('路线①') && html.includes('recovery_only') && html.includes('R2A'),
    { length: html.length, head: html.slice(0, 120) },
  )
} catch (error) {
  check('看板 SSR 渲染', false, error.message)
}
await fiber.dispose()
check('dispose 后槽位清理干净', slots.entriesOfSlot('ui:sidebar').length === 0, slots.entriesOfSlot('ui:sidebar').length)
await ctx.fiber.dispose()

const failed = checks.filter((item) => !item.ok)
console.log('')
console.log(`AGH 原生集成自检：${checks.length - failed.length}/${checks.length} 通过`)
if (failed.length) {
  console.log('失败项：')
  for (const item of failed) console.log(` - ${item.name}: ${JSON.stringify(item.evidence)}`)
}
process.exit(failed.length ? 1 : 0)

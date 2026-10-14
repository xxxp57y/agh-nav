#!/usr/bin/env node
/**
 * 用真实账号配置 AGH 的模型 Provider（读 key 文件、写进实例配置，**不打印 key**）。
 *
 * 为什么需要它：AGH 官方流程是 Web「设置 → 模型」向导；只有 SSH 时，
 * 用 SDK 的 `config.test` + `config.save` 走同一条配置路径。
 *
 * 用法（AGH 源码仓库根目录）::
 *
 *     AGH_HOME=/srv/agh-demo/agh-home \
 *       corepack pnpm exec tsx agh-nav/tools/configure-provider.mts \
 *         --key-file /srv/agh-demo/.secrets/agnes-key.txt \
 *         --provider agnes-ai --model agnes-3.0-flash
 *
 * 关键协议事实（踩过，别再踩）：
 * - `config.test` 的返回是 `{ models: [...], verified: boolean }` —— **没有 baseUrl**。
 *   所以绝不能把 `catalogue.baseUrl` 直接塞给 `config.save`：`baseUrl` 一旦出现就必须 `minLength>=1`，
 *   空串会被协议层拒绝（`_agnes/v1/config.save params: ...`）。
 * - `baseUrl` 是**可选**的：不传就用实例里该 provider 的默认网关。只有要覆盖时才传 `--base-url`。
 * - `config.test` 需要带着 key 才能列目录（等价于 Web 上的"测试连接"）。
 * - `config.save` 的 `expectedRevision` 用 `config.get().revision`（并发保护）。
 *
 * 退出码：0 成功；1 配置失败；2 用法/前置条件不对。
 */

import { readFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import process from 'node:process'
import { createClient, memoryJournal } from '../../packages/sdk/src/index.node.ts'

function arg(name, fallback) {
  const index = process.argv.indexOf(`--${name}`)
  if (index === -1) return fallback
  const next = process.argv[index + 1]
  return next && !next.startsWith('--') ? next : true
}

const step = (n, text) => console.log(`[${n}] ${text}`)
/** AGH 把配置失败映射成 SEMANTIC_REJECTED + data.reason=CONFIG_xxx；这个表用来把原因说清楚。 */
const REASON_HINTS = {
  CONFIG_TEST_FAILED: '测试连接没通过：key 无效/过期，或网关拒绝（Web 上「测试连接」也会失败）',
  CONFIG_MODEL_UNAVAILABLE: '模型不在该 provider 的目录里（脚本会把目录里的候选打出来，用 --model 选一个）',
  CONFIG_UNKNOWN_PROVIDER: 'provider id 不对（Agnes AI 是 agnes-ai；可用列表见 agh doctor provider）',
  CONFIG_CREDENTIAL_REQUIRED: '没读到可用凭据：key 文件为空/格式不对，或该 provider 走的是订阅 OAuth',
  CONFIG_CREDENTIAL_STORE: '凭据写不进实例的凭据库（目录权限或磁盘问题）',
  CONFIG_ENDPOINT_OVERRIDE_UNSUPPORTED: '这个 provider 不允许自定义 baseUrl：去掉 --base-url',
  CONFIG_REVISION_CONFLICT: '配置被别处同时改了：脚本重跑一次即可（它每次都读最新 revision）',
  CONFIG_INVALID_INPUT: '入参不合法（例如 key 里有换行/空白，或模型名超长）',
  CONFIG_PERSIST_FAILED: '配置落盘失败（磁盘/权限）',
}
const explain = (error) => {
  const data = error?.data ?? {}
  const reason = typeof data.reason === 'string' ? data.reason : undefined
  const code = typeof data.code === 'string' ? data.code : undefined
  return {
    message: error?.message ?? String(error),
    code,
    reason,
    hint: reason && REASON_HINTS[reason] ? REASON_HINTS[reason] : undefined,
  }
}
const fail = (message, code = 1, extra) => {
  console.error(`\n✗ ${message}`)
  if (extra?.reason) console.error(`  reason: ${extra.reason}${extra.hint ? ` → ${extra.hint}` : ''}`)
  if (extra?.code && extra.code !== extra.reason) console.error(`  code: ${extra.code}`)
  process.exit(code)
}

const home = process.env.AGH_HOME
if (!home) {
  fail('必须设置 AGH_HOME（要配置哪个实例），例如：AGH_HOME=/srv/agh-demo/agh-home', 2)
}
const profile = process.env.AGNES_PROFILE ?? 'local-dev'
const keyFile = arg('key-file')
const providerId = String(arg('provider', 'agnes-ai'))
const wantedModel = arg('model') ? String(arg('model')) : undefined
const baseUrlOverride = arg('base-url') ? String(arg('base-url')) : undefined
const keepOnTestFailure = Boolean(arg('continue-on-test-failure', false))
if (!keyFile || keyFile === true) {
  fail('用法：--key-file <文件> [--provider agnes-ai] [--model agnes-3.0-flash] [--base-url …]', 2)
}
if (!existsSync(String(keyFile))) fail(`key 文件不存在：${keyFile}`, 2)

const ownerPath = join(home, 'data/daemon/owner.json')
if (!existsSync(ownerPath)) {
  fail(
    `找不到 daemon 描述文件 ${ownerPath}\n` +
      '  → 这个实例还没起过。先跑一次 `node <SRC>/packages/cli/dist/local/agnes.mjs sessions list`（会冷启动 daemon）再回来。',
    2,
  )
}

const apiKey = (await readFile(String(keyFile), 'utf8')).trim()
if (!apiKey) fail(`key 文件是空的：${keyFile}`, 2)
step(1, `读取 key（${apiKey.length} 字节，不打印明文）→ 连接 daemon ${ownerPath}`)

const owner = JSON.parse(await readFile(ownerPath, 'utf8'))
const client = createClient({
  transport: { kind: 'unix', path: owner.socketPath },
  auth: { kind: 'local' },
  journal: memoryJournal(),
})
await client.initialize()

const before = await client.config.get()
step(2, `当前配置：configured=${before.configured} revision=${before.revision} provider=${before.provider?.id ?? '—'}`)

// ---- 测试连接（等价于 Web 上的「测试连接」）-------------------------------- #
let models = []
let verified = false
let testError = null
try {
  const catalogue = await client.config.test({
    providerId,
    apiKey,
    ...(baseUrlOverride ? { baseUrl: baseUrlOverride } : {}),
  })
  models = (catalogue.models ?? []).map((entry) => (typeof entry === 'string' ? entry : entry.id)).filter(Boolean)
  verified = Boolean(catalogue.verified)
  step(3, `测试连接：verified=${verified}，目录里 ${models.length} 个模型${models.length ? `（前几个：${models.slice(0, 5).join(', ')}）` : ''}`)
} catch (error) {
  testError = error?.message ?? String(error)
  const info = explain(error)
  step(3, `测试连接失败：${info.message.split('\n')[0]}${info.reason ? `（reason=${info.reason}）` : ''}`)
  if (!keepOnTestFailure && !wantedModel) {
    fail(
      '测试连接失败，而且没有用 --model 指定模型，无法继续。\n' +
        '  可能原因：key 不对/过期、baseUrl 不对、网关不可达。\n' +
        '  想强行写入（例如网关临时抖动）加 --continue-on-test-failure --model agnes-3.0-flash。',
      1,
      info,
    )
  }
}
if (!wantedModel && models.length === 0) {
  fail('目录里没有可用模型，且未指定 --model；请显式给一个模型名（如 --model agnes-3.0-flash）')
}
const model = wantedModel ?? models[0]
step(4, `准备保存：provider=${providerId} model=${model} baseUrl=${baseUrlOverride ?? '(用实例默认)'}`)

// ---- 保存（baseUrl 只在显式覆盖时才带，避免空串违反 minLength:1）---------- #
let after
try {
  after = await client.config.save({
    providerId,
    model: String(model),
    apiKey,
    expectedRevision: before.revision ?? 0,
    ...(baseUrlOverride ? { baseUrl: baseUrlOverride } : {}),
  })
} catch (error) {
  const info = explain(error)
  fail(`保存失败：${info.message}`, 1, info)
}

await client.close?.().catch(() => undefined)
console.log(
  JSON.stringify(
    {
      ok: true,
      providerId: after.provider?.id ?? providerId,
      baseUrl: after.provider?.baseUrl ?? baseUrlOverride ?? '(实例默认)',
      model: after.provider?.model ?? String(model),
      configured: after.configured,
      credentialConfigured: after.provider?.credentialConfigured ?? null,
      effect: after.effect,
      revision: after.revision,
      test: { verified, candidates: models.slice(0, 8), error: testError ? testError.split('\n')[0] : null },
      keyBytes: apiKey.length,
      note: 'key 只写进实例配置；本输出不含 key 明文。新配置对“新会话”生效。',
    },
    null,
    2,
  ),
)
process.exit(0)

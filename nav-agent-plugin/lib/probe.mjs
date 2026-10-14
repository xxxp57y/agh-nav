/**
 * 数据面调用：把 `agh_nav_probe.py` 当唯一数据源。
 *
 * 设计要点（为什么值得这么绕）：
 * 1. **在线与离线共用同一份实现**——`route1.contracts` 与 `route1.nav.channels`
 *    就是导航闭环自己用的那份代码，插件不重新实现一遍判据。
 * 2. **实时 + 缓存回退**——服务器断链/GPU 排队/conda 环境异常时，工具仍能回答，
 *    但会显式标注 `source: "cache"` 与缓存时间，绝不把旧数据当实时数据讲。
 * 3. 所有工具都返回**可核对的结构化 JSON**，面板与报告材料复用同一份。
 */

import { spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
export const PACKAGE_ROOT = resolve(HERE, '..')
export const DEFAULT_CACHE_DIR = join(PACKAGE_ROOT, 'data', 'cache')
const CACHE_FILE = 'nav-cache.json'

export class ProbeError extends Error {
  constructor(message, detail) {
    super(message)
    this.name = 'ProbeError'
    this.detail = detail || {}
  }
}

function quote(value) {
  const text = String(value)
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(text)) return text
  return `'${text.replace(/'/g, `'\\''`)}'`
}

/** 组装远端命令（ssh 模式）—— 单引号规则与 bash 一致。
 *
 * ⚠️ 必须是 `cd <root> && PYTHONPATH=… python …`：
 * 少一个 `&&` 时 bash 会把 `cd /x PYTHONPATH=/y python` 当成 `cd` 的三个参数
 * （`cd: too many arguments`），于是所有工具都会静默回退到缓存。
 */
export function buildRemoteCommand(config, argv) {
  return [
    `cd ${quote(config.remote.root)}`,
    '&&',
    `PYTHONPATH=${quote(config.remote.src)}`,
    quote(config.remote.python),
    quote(config.remote.probe),
    ...argv.map(quote),
  ].join(' ')
}

function spawnCollect(bin, args, { timeoutMs, env }) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(bin, args, { env: env || process.env, windowsHide: true })
    let stdout = ''
    let stderr = ''
    let settled = false
    const timer = setTimeout(() => {
      if (settled) return
      settled = true
      try {
        child.kill()
      } catch {
        /* 忽略 */
      }
      rejectPromise(new ProbeError(`数据面调用超时（${timeoutMs} ms）`, { bin, args }))
    }, timeoutMs)
    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString('utf8')
    })
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString('utf8')
    })
    child.on('error', (error) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      rejectPromise(new ProbeError(`无法启动 ${bin}：${error.message}`, { bin }))
    })
    child.on('close', (code) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolvePromise({ code, stdout, stderr })
    })
  })
}

/** 直接跑一次探针，返回解析后的 JSON。失败一律抛 ProbeError（带 stderr 尾部）。 */
export async function runProbe(config, argv, options = {}) {
  const startedAt = Date.now()
  let bin
  let args
  let env = process.env
  if (config.exec === 'local') {
    bin = config.local.python
    args = [config.local.probe, ...argv]
    env = { ...process.env, PYTHONPATH: [config.local.src, process.env.PYTHONPATH].filter(Boolean).join(':') }
  } else {
    bin = config.ssh.sshBin
    args = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15']
    if (config.ssh.key) args.push('-i', config.ssh.key)
    if (config.ssh.port) args.push('-p', String(config.ssh.port))
    args.push(config.ssh.host, buildRemoteCommand(config, argv))
  }
  const timeoutMs = options.timeoutMs || config.timeoutMs
  const { code, stdout, stderr } = await spawnCollect(bin, args, { timeoutMs, env })
  const trimmed = stdout.trim()
  if (code !== 0) {
    const tail = (stderr || trimmed).split('\n').slice(-6).join('\n')
    throw new ProbeError(
      `数据面命令失败（exit=${code}）：${argv.join(' ')}${tail ? `\n${tail}` : ''}`,
      { code, argv, stderr: tail },
    )
  }
  try {
    return { data: JSON.parse(trimmed), ms: Date.now() - startedAt, transport: config.exec }
  } catch (error) {
    throw new ProbeError(`数据面返回的不是 JSON：${error.message}`, {
      stdout: trimmed.slice(0, 400),
      argv,
    })
  }
}

export function cachePath(config) {
  return join(config.cacheDir || DEFAULT_CACHE_DIR, CACHE_FILE)
}

/** 读本地缓存（`data/cache/nav-cache.json`，由 `scripts/refresh-cache.mjs` 生成）。 */
export function readCache(config) {
  const path = cachePath(config)
  try {
    const payload = JSON.parse(readFileSync(path, 'utf8'))
    return { path, payload }
  } catch {
    return { path, payload: null }
  }
}

/** 实时优先、缓存回退的统一入口。 */
export async function probe(config, argv, options = {}) {
  try {
    const live = await runProbe(config, argv, options)
    return { ok: true, source: 'live', transport: live.transport, ms: live.ms, data: live.data }
  } catch (error) {
    const cached = readCache(config)
    const sample = options.cacheSample && cached.payload ? options.cacheSample(cached.payload) : null
    if (!sample) throw error
    return {
      ok: true,
      source: 'cache',
      transport: 'cache',
      ms: 0,
      cachePath: cached.path,
      cacheGeneratedAt: cached.payload.generated_at_iso || null,
      degraded: {
        reason: error.message,
        note: '实时数据面不可用，返回的是服务器上次生成的缓存快照；实时结论请恢复连接后重跑。',
      },
      data: sample,
    }
  }
}

export function envelope(result, extra = {}) {
  return {
    schema: 'agh-nav/tool-result/1',
    source: result.source,
    transport: result.transport,
    generated_at: new Date().toISOString(),
    latency_ms: result.ms,
    cache_path: result.cachePath || null,
    cache_generated_at: result.cacheGeneratedAt || null,
    degraded: result.degraded || null,
    ...extra,
    data: result.data,
  }
}

/**
 * 跑一条（白名单内的）远程只读命令，用于读日志/产物文本。
 * 与 `runProbe` 共用 ssh 参数，但不要求输出是 JSON。
 */
export async function runShell(config, command, options = {}) {
  const startedAt = Date.now()
  let bin
  let args
  if (config.exec === 'local') {
    bin = 'bash'
    args = ['-lc', command]
  } else {
    bin = config.ssh.sshBin
    args = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15']
    if (config.ssh.key) args.push('-i', config.ssh.key)
    if (config.ssh.port) args.push('-p', String(config.ssh.port))
    args.push(config.ssh.host, command)
  }
  const { code, stdout, stderr } = await spawnCollect(bin, args, {
    timeoutMs: options.timeoutMs || config.timeoutMs,
    env: process.env,
  })
  if (code !== 0) {
    throw new ProbeError(`远程命令失败（exit=${code}）：${(stderr || '').split('\n').slice(-4).join('\n')}`, {
      code,
    })
  }
  return { stdout, ms: Date.now() - startedAt }
}

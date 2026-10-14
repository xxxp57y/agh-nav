/**
 * 看板的数据来源（与 nav-agent-plugin 同一套"实时 + 缓存回退"约定）。
 *
 * 这里刻意**不 import** nav-agent-plugin 的模块：两个包是独立安装单元，
 * 一个包被禁用/卸载时另一个必须还能工作。重复的只有这 60 行 IO 代码。
 */

import { spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
export const PACKAGE_ROOT = resolve(HERE, '..')

export const DEFAULTS = {
  exec: 'auto',
  sshHost: 'your-user@your-server.example',
  sshPort: 22,
  sshKey: '',
  remoteRoot: '/srv/agh-demo/agh-nav',
  remoteProbe: 'bin/agh_nav_probe.py',
  remotePython: '/srv/agh-demo/habitat-env/bin/python',
  remoteSrc: '/srv/agh-demo/src',
  localSrc: '',
  localPython: 'python3',
  localProbe: '',
  timeoutMs: 120000,
}

function quote(value) {
  const text = String(value)
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(text)) return text
  return `'${text.replace(/'/g, `'\\''`)}'`
}

function spawnCollect(bin, args, timeoutMs) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(bin, args, { windowsHide: true })
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
      rejectPromise(new Error(`看板数据面超时（${timeoutMs} ms）`))
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
      rejectPromise(new Error(`无法启动 ${bin}：${error.message}`))
    })
    child.on('close', (code) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      code === 0
        ? resolvePromise(stdout)
        : rejectPromise(new Error(`看板数据面失败（exit=${code}）：${(stderr || '').trim().split('\n').slice(-3).join('\n')}`))
    })
  })
}

export function loadConfig(env = process.env) {
  const localSrc = env.NAV_AGENT_SRC || DEFAULTS.localSrc || DEFAULTS.remoteSrc
  const requested = String(env.NAV_AGENT_EXEC || DEFAULTS.exec).toLowerCase()
  let isLocal = false
  try {
    isLocal = Boolean(localSrc) && readFileSync(`${String(localSrc).replace(/[\\/]+$/, '')}/route1/__init__.py`)
  } catch {
    isLocal = false
  }
  const exec = requested === 'auto' ? (isLocal ? 'local' : 'ssh') : requested
  return {
    exec: exec === 'local' ? 'local' : 'ssh',
    timeoutMs: Number(env.NAV_AGENT_TIMEOUT_MS || DEFAULTS.timeoutMs),
    cacheFile: env.NAV_BOARD_CACHE || join(PACKAGE_ROOT, 'data', 'board.json'),
    ssh: {
      host: env.NAV_AGENT_SSH_HOST || DEFAULTS.sshHost,
      port: Number(env.NAV_AGENT_SSH_PORT || DEFAULTS.sshPort),
      key: env.NAV_AGENT_SSH_KEY || DEFAULTS.sshKey,
      bin: env.NAV_AGENT_SSH_BIN || 'ssh',
    },
    remote: {
      root: env.NAV_AGENT_REMOTE_ROOT || DEFAULTS.remoteRoot,
      probe: env.NAV_AGENT_REMOTE_PROBE || DEFAULTS.remoteProbe,
      python: env.NAV_AGENT_REMOTE_PYTHON || DEFAULTS.remotePython,
      src: env.NAV_AGENT_REMOTE_SRC || DEFAULTS.remoteSrc,
    },
    local: {
      probe: env.NAV_AGENT_PROBE || DEFAULTS.localProbe || `${localSrc}/../agh-nav/bin/agh_nav_probe.py`,
      python: env.NAV_AGENT_PYTHON || DEFAULTS.localPython,
      src: localSrc,
    },
  }
}

async function runProbe(config, argv) {
  let bin
  let args
  if (config.exec === 'local') {
    bin = config.local.python
    args = [config.local.probe, ...argv]
  } else {
    bin = config.ssh.bin
    args = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15']
    if (config.ssh.key) args.push('-i', config.ssh.key)
    if (config.ssh.port) args.push('-p', String(config.ssh.port))
    const command = [
      `cd ${quote(config.remote.root)}`,
      '&&',
      `PYTHONPATH=${quote(config.remote.src)}`,
      quote(config.remote.python),
      quote(config.remote.probe),
      ...argv.map(quote),
    ].join(' ')
    args.push(config.ssh.host, command)
  }
  const stdout = await spawnCollect(bin, args, config.timeoutMs)
  return JSON.parse(stdout.trim())
}

export function readCache(config) {
  try {
    return JSON.parse(readFileSync(config.cacheFile, 'utf8'))
  } catch {
    return null
  }
}

/** 看板只需要一个 board payload：实时优先，失败回退到 `data/board.json`。 */
export async function loadBoard({ force = false } = {}) {
  const config = loadConfig()
  try {
    const data = await runProbe(config, ['board'])
    return {
      ok: true,
      source: force ? 'live' : 'live',
      transport: config.exec,
      generated_at: data.generated_at_iso || new Date().toISOString(),
      payload: data,
      degraded: null,
    }
  } catch (error) {
    const cached = readCache(config)
    if (!cached) throw error
    return {
      ok: true,
      source: 'cache',
      transport: 'cache',
      generated_at: cached.generated_at_iso || cached.generated_at || null,
      payload: cached,
      degraded: {
        reason: error.message,
        note: '实时数据面不可用，显示服务器上次生成的看板快照。',
      },
    }
  }
}

/**
 * 配置解析：把"导航闭环在哪儿跑"这件事收敛到一个地方。
 *
 * 三种执行模式（`NAV_AGENT_EXEC`）：
 * - `local` —— AGH 直接跑在导航服务器上（route1 代码、日志、conda 环境都在本机）。
 * - `ssh`   —— AGH 跑在开发机（如 Windows），通过 SSH 把命令送到导航服务器。
 * - `auto`（默认）—— 本机能 import route1 就用 local，否则用 ssh。
 *
 * 所有路径都可用环境变量覆盖；默认值就是本项目在服务器上的真实位置
 * （见 notes/route1-prep-brief.md §2）。
 */

import { existsSync } from 'node:fs'

export const DEFAULTS = {
  exec: 'auto',
  ssh: {
    host: 'your-user@your-server.example',
    port: 22,
    key: '',
    sshBin: 'ssh',
  },
  remote: {
    root: '/srv/agh-demo/agh-nav',
    probe: 'bin/agh_nav_probe.py',
    python: '/srv/agh-demo/habitat-env/bin/python',
    src: '/srv/agh-demo/src',
    repo: '/srv/agh-demo/InternNav',
    gpu: '1',
  },
  local: {
    probe: '',
    python: 'python3',
    src: '',
    repo: '',
  },
  timeoutMs: 240000,
}

function num(value, fallback) {
  const parsed = Number.parseInt(String(value ?? ''), 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

/** 本机是否就是导航服务器（route1 源码在本地存在）。 */
export function looksLikeNavServer(srcPath) {
  const src = String(srcPath || '')
  if (!src) return false
  try {
    return existsSync(`${src.replace(/[\\/]+$/, '')}/route1/__init__.py`)
  } catch {
    return false
  }
}

export function loadConfig(env = process.env) {
  const src = env.NAV_AGENT_SRC || DEFAULTS.local.src || DEFAULTS.remote.src
  const requested = String(env.NAV_AGENT_EXEC || DEFAULTS.exec).toLowerCase()
  const isLocalHost = looksLikeNavServer(src)
  const exec = requested === 'auto' ? (isLocalHost ? 'local' : 'ssh') : requested
  if (exec !== 'local' && exec !== 'ssh') {
    throw new Error(`NAV_AGENT_EXEC 只能是 auto/local/ssh，收到 ${requested}`)
  }
  return {
    exec,
    requested,
    isLocalHost,
    timeoutMs: num(env.NAV_AGENT_TIMEOUT_MS, DEFAULTS.timeoutMs),
    cacheDir: env.NAV_AGENT_CACHE_DIR || '',
    ssh: {
      host: env.NAV_AGENT_SSH_HOST || DEFAULTS.ssh.host,
      port: num(env.NAV_AGENT_SSH_PORT, DEFAULTS.ssh.port),
      key: env.NAV_AGENT_SSH_KEY || DEFAULTS.ssh.key,
      sshBin: env.NAV_AGENT_SSH_BIN || DEFAULTS.ssh.sshBin,
    },
    remote: {
      root: env.NAV_AGENT_REMOTE_ROOT || DEFAULTS.remote.root,
      probe: env.NAV_AGENT_REMOTE_PROBE || DEFAULTS.remote.probe,
      python: env.NAV_AGENT_REMOTE_PYTHON || DEFAULTS.remote.python,
      src: env.NAV_AGENT_REMOTE_SRC || DEFAULTS.remote.src,
      repo: env.NAV_AGENT_REMOTE_REPO || DEFAULTS.remote.repo,
      gpu: env.NAV_AGENT_GPU || DEFAULTS.remote.gpu,
    },
    local: {
      probe: env.NAV_AGENT_PROBE || DEFAULTS.local.probe || `${src}/../agh-nav/bin/agh_nav_probe.py`,
      python: env.NAV_AGENT_PYTHON || DEFAULTS.local.python,
      src,
      repo: env.NAV_AGENT_REPO || DEFAULTS.local.repo || DEFAULTS.remote.repo,
    },
  }
}

/** 给模型/面板看的一句话：这次调用是"真跑服务器"还是"读本地缓存"。 */
export function describeConfig(config) {
  if (config.exec === 'local') {
    return `local:${config.local.python} (${config.local.src})`
  }
  return `ssh:${config.ssh.host}:${config.ssh.port}`
}

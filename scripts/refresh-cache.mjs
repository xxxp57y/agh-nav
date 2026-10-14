#!/usr/bin/env node
/**
 * 一条命令刷新两个插件的本地缓存：
 *
 *   1. `nav-agent-plugin/data/cache/nav-cache.json` —— 工具在服务器不可达时的回退快照
 *   2. `nav-board-plugin/data/board.json`          —— 看板服务的回退快照
 *   3. `nav-board-plugin/extensions/main/client/baseline.js` —— 看板浏览器模块的静态快照
 *
 * 用法（在能访问导航服务器的机器上跑；AGH 跑在服务器上时用 `--exec local`）::
 *
 *     NAV_AGENT_EXEC=local NAV_AGENT_SRC=/srv/agh-demo/src \
 *       node scripts/refresh-cache.mjs
 *
 *     # Windows/开发机（走 ssh）
 *     node scripts/refresh-cache.mjs --exec ssh --key /srv/agh-demo\_tools\ssh\id_ed25519
 */

import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runProbe } from '../nav-agent-plugin/lib/probe.mjs'
import { loadConfig } from '../nav-agent-plugin/lib/config.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..')
const args = process.argv.slice(2)
const arg = (name, fallback) => {
  const index = args.indexOf(`--${name}`)
  return index === -1 ? fallback : args[index + 1]
}

if (arg('exec')) process.env.NAV_AGENT_EXEC = String(arg('exec'))
if (arg('key')) process.env.NAV_AGENT_SSH_KEY = String(arg('key'))
if (arg('src')) process.env.NAV_AGENT_SRC = String(arg('src'))
if (arg('python')) process.env.NAV_AGENT_PYTHON = String(arg('python'))

const config = loadConfig()
console.log(`[refresh-cache] 数据面模式：${config.exec}`)

const cacheAll = await runProbe(config, ['cache-all', '--sample-run', String(arg('sample-run', 'r1grd4'))], {
  timeoutMs: config.timeoutMs,
})
const cacheDir = join(ROOT, 'nav-agent-plugin', 'data', 'cache')
mkdirSync(cacheDir, { recursive: true })
const cachePath = join(cacheDir, 'nav-cache.json')
writeFileSync(cachePath, JSON.stringify(cacheAll.data, null, 2), 'utf8')
console.log(`[refresh-cache] 写入 ${cachePath}（${JSON.stringify(cacheAll.data).length} 字节）`)

const board = cacheAll.data.board
const boardDir = join(ROOT, 'nav-board-plugin', 'data')
mkdirSync(boardDir, { recursive: true })
const boardPath = join(boardDir, 'board.json')
writeFileSync(boardPath, JSON.stringify(board, null, 2), 'utf8')
console.log(`[refresh-cache] 写入 ${boardPath}（${JSON.stringify(board).length} 字节）`)

const baselinePath = join(ROOT, 'nav-board-plugin', 'extensions', 'main', 'client', 'baseline.js')
const baseline = `/**
 * 静态快照（自动生成，请勿手改）。
 *
 * 生成命令：\`node scripts/refresh-cache.mjs\`
 * 数据来源：导航服务器 \`agh_nav_probe.py board\`
 * 生成时间：${new Date().toISOString()}
 */

export const BASELINE = ${JSON.stringify({ generated_at_iso: board.generated_at_iso, payload: board }, null, 2)}
`
writeFileSync(baselinePath, baseline, 'utf8')
console.log(`[refresh-cache] 写入 ${baselinePath}（${baseline.length} 字节）`)
console.log('[refresh-cache] 完成')

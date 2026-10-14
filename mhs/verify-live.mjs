// Verify the real MHS connection via AgnesHub's supported northbound API.
import { createRequire } from 'node:module'
import { writeFileSync } from 'node:fs'
const root = process.env.AGH_MHS_RUNTIME
if (!root) throw new Error('Set AGH_MHS_RUNTIME to your AGH checkout')
const require = createRequire(root + '/packages/mhs/package.json')
const WebSocket = require('ws')
const url = process.env.AGNES_HUB_URL ?? 'ws://127.0.0.1:43132/ws/hub'
const ws = new WebSocket(url)
const wait = new Map(), seen = new Set(), results = new Map()
let id = 0, binaryFrames = 0
const deadline = setTimeout(() => { console.error('MHS verification timed out'); ws.terminate(); process.exitCode = 1 }, 25000)
ws.on('message', (raw, binary) => {
  if (binary) { binaryFrames++; return }
  const m = JSON.parse(raw.toString())
  if (m.id && wait.has(m.id)) {
    const { resolve, reject } = wait.get(m.id)
    wait.delete(m.id)
    if (m.error) reject(new Error(m.error.message)); else resolve(m.result)
  }
  if (m.method === 'hub/data') seen.add(m.params.item.source)
  if (m.method === 'hub/result') results.set(m.params.job, m.params)
})
const req = (method, params = {}) => new Promise((resolve, reject) => {
  const key = String(++id); wait.set(key, { resolve, reject })
  ws.send(JSON.stringify({ jsonrpc: '2.0', id: key, method, params }))
})
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
try {
  await new Promise((resolve, reject) => { ws.once('open', resolve); ws.once('error', reject) })
  await req('hub/hello', { role: 'ui', client: { name: 'nav-mhs-acceptance' } })
  const devices = (await req('hub/devices')).devices
  const d = devices.find((d) => d.id === 'habitat-01')
  if (!d?.available) throw new Error('habitat-01 is not available')
  const worlds = (await req('hub/world')).worlds
  const world = worlds.find((w) => w.map.startsWith('mp3d-'))
  if (!world?.places?.length) throw new Error('No semantic landmarks in World')
  await req('hub/subscribe', { device: d.id, sources: ['front','map','pose','odometry','navigation'], binary: true, hz: 2 })
  for (let i = 0; i < 80 && seen.size < 5; i++) await sleep(100)
  if (seen.size !== 5 || binaryFrames < 2) throw new Error(`Missing streams: ${[...seen]}, binary=${binaryFrames}`)
  const started = await req('hub/call', { device: d.id, tool: 'inspect_state', arguments: {} })
  for (let i = 0; i < 50 && !results.has(started.job); i++) await sleep(100)
  const result = results.get(started.job)
  if (result?.status !== 'done') throw new Error('inspect_state did not finish successfully: ' + JSON.stringify(result))
  const stop = await req('hub/stop', { device: d.id })
  await sleep(200)
  const final = await req('hub/read', { device: d.id })
  if (final.state.values.mode !== 'paused') throw new Error('Stop did not pause the simulator')
  const report = { device: d.id, available: d.available, sourceIds: [...seen], binaryFrames,
    map: world.map, places: world.places.length, position: final.position,
    inspectResult: result.status, stopResult: stop, finalMode: final.state.values.mode,
    verifiedAt: new Date().toISOString() }
  if (process.env.MHS_REPORT) writeFileSync(process.env.MHS_REPORT, JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 2))
} finally {
  clearTimeout(deadline)
  ws.close()
}

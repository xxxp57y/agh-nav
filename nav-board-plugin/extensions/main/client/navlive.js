/**
 * 工作台（浏览器侧）访问 navlive 服务的小客户端。
 *
 * 浏览器直连 navlive 的 HTTP 端口（默认 http://127.0.0.1:43121），
 * 与 AGH 的通道是两条独立的路：AGH 那条走同源 BFF（服务调用），
 * 这条走普通 HTTP（画面/地图/状态轮询）。所以：
 *   - 服务器上 navlive 没起 → 这一栏显示"服务未连接"，不影响 AGH 与 `nav_*` 工具；
 *   - 端口转发没做 → 同上，且错误文案里直接给出该做什么。
 */

export const DEFAULT_URL = 'http://127.0.0.1:43121'

export function resolveBase(explicit) {
  const fromWindow =
    typeof window !== 'undefined' && window.__AGH_NAVLIVE_URL__ ? String(window.__AGH_NAVLIVE_URL__) : ''
  return String(explicit || fromWindow || DEFAULT_URL).replace(/\/+$/, '')
}

async function getJson(base, path, timeoutMs = 8000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${base}${path}`, { signal: controller.signal, cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return await response.json()
  } finally {
    clearTimeout(timer)
  }
}

export function fetchHealth(base) {
  return getJson(base, '/health', 5000)
}

export function fetchScenes(base) {
  return getJson(base, '/scenes', 180000)
}

export function fetchState(base, session) {
  return getJson(base, `/state${session ? `?session=${encodeURIComponent(session)}` : ''}`, 8000)
}

async function postJson(base, path, body, timeoutMs = 60000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${base}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    const text = await response.text()
    if (!response.ok) throw new Error(text.slice(0, 240) || `HTTP ${response.status}`)
    return JSON.parse(text)
  } finally {
    clearTimeout(timer)
  }
}

export function openScene(base, { scene, episode, instruction }) {
  return postJson(base, '/open', { scene, episode: episode ?? null, instruction: instruction ?? null }, 300000)
}

export function sendAction(base, payload) {
  return postJson(base, '/action', payload)
}

/** 给图片 URL 加时间戳，绕开浏览器缓存（画面必须每帧都是新的）。 */
export function imageUrl(base, path, session, stamp) {
  const params = new URLSearchParams()
  if (session) params.set('session', session)
  if (stamp) params.set('t', String(stamp))
  const query = params.toString()
  return `${base}${path}${query ? `?${query}` : ''}`
}

/**
 * world(x,z) → 地图像素(col,row)。
 * navlive 给的仿射是"当前位姿 ↔ 当前 agent_map_coord"的锚点 + 0.01 m/px，
 * 所以这里只是一次平移换算，前端拿到就能自己画轨迹折线（可缩放、可 hover）。
 */
export function worldToMap(affine, x, z) {
  if (!affine?.anchor_map || !affine?.anchor_world) return null
  const scale = Number(affine.meters_per_pixel || 0.01)
  return [
    Number(affine.anchor_map[0]) + (Number(x) - Number(affine.anchor_world[0])) / scale,
    Number(affine.anchor_map[1]) + (Number(z) - Number(affine.anchor_world[1])) / scale,
  ]
}

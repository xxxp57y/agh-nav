/**
 * `navlive` 服务的 HTTP 客户端（插件后端行用）。
 *
 * 与 `probe.mjs` 的区别很重要：`probe.mjs` 是**离线数据面**（读日志、算指标、给建议），
 * 本文件是**在线数据面**（真的在 Habitat 里让机器人走）。两者回答的是不同问题：
 *
 *   离线：这一轮跑得怎么样、失败在哪、恢复通道会怎么动 → `nav_*` 工具
 *   在线：现在这个场景里，机器人去 A → C，走到哪了 → `navlive_*` 工具
 *
 * 地址来自 `NAVLIVE_URL`（默认本机 43121）。AGH 与导航服务同机时就是直接 localhost，
 * 跨机时由操作者做端口转发（见 agh-nav/docs/实时可视化工作台.md）。
 */

export const DEFAULT_URL = 'http://127.0.0.1:43121'

export class NavLiveError extends Error {
  constructor(message, detail = {}) {
    super(message)
    this.name = 'NavLiveError'
    this.detail = detail
  }
}

export function serviceUrl(env = process.env) {
  const raw = env.NAVLIVE_URL || env.NAV_LIVE_URL || DEFAULT_URL
  return String(raw).replace(/\/+$/, '')
}

async function request(path, { method = 'GET', body, timeoutMs = 30000 } = {}) {
  const url = `${serviceUrl()}${path}`
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(url, {
      method,
      headers: body ? { 'content-type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
    const text = await response.text()
    if (!response.ok) {
      throw new NavLiveError(`navlive 返回 ${response.status}：${text.slice(0, 300)}`, {
        status: response.status,
        url,
      })
    }
    try {
      return JSON.parse(text)
    } catch {
      return { raw: text }
    }
  } catch (error) {
    if (error instanceof NavLiveError) throw error
    const hint =
      error?.name === 'AbortError'
        ? `navlive 请求超时（${timeoutMs} ms）`
        : `连不上 navlive 服务（${url}）：${error?.message || error}`
    throw new NavLiveError(
      `${hint}\n排查：服务器上 tmux 里有没有 navlive 会话；本机到 43121 的端口转发是否在。`,
      { url },
    )
  } finally {
    clearTimeout(timer)
  }
}

export function health() {
  return request('/health', { timeoutMs: 8000 })
}

export function scenes() {
  return request('/scenes', { timeoutMs: 180000 })
}

export function open({ scene, episode, instruction }) {
  return request('/open', {
    method: 'POST',
    body: { scene, episode: episode ?? null, instruction: instruction ?? null },
    timeoutMs: 300000,
  })
}

export function state(session) {
  return request(`/state${session ? `?session=${encodeURIComponent(session)}` : ''}`, { timeoutMs: 10000 })
}

export function action(payload) {
  return request('/action', { method: 'POST', body: payload, timeoutMs: 60000 })
}

/** 地名册：列出某场景可用的地标（不传就用当前会话的场景）。 */
export function landmarks({ scene, session } = {}) {
  const params = new URLSearchParams()
  if (scene) params.set('scene', scene)
  if (session) params.set('session', session)
  const query = params.toString()
  return request(`/landmarks${query ? `?${query}` : ''}`, { timeoutMs: 15000 })
}

/** 名字 → 坐标（可多个）。解析不出会返回 404 与候选，不猜。 */
export function resolveLandmarks({ scene, session, names }) {
  return request('/resolve', {
    method: 'POST',
    body: { scene: scene ?? null, session: session ?? null, names },
    timeoutMs: 15000,
  })
}

/** `/state` 返回的是一整个 JSON（不是 `{state: …}` 包装），这里统一取出来。 */
export function stateOf(payload) {
  return payload?.state || payload
}

export function envelope(extra = {}) {
  return {
    schema: 'agh-navlive/tool-result/1',
    service: serviceUrl(),
    generated_at: new Date().toISOString(),
    ...extra,
  }
}

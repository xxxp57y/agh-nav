/**
 * 实时导航面板（浏览器侧）—— 对应作品里"实时仿真画面 + 地图 + 机器人状态"那一栏。
 *
 * 数据来源是 navlive 服务（HTTP 直连），与 AGH 的 BFF 通道是两条独立的路。
 * 面板本身**只读**：它不给机器人下指令，下指令是 Agent 的事（聊天里说人话）。
 * 这样分工才符合赛题 —— 自然语言 → Agent 决策 → 工具执行 → 这里看到结果。
 *
 * 降级：服务连不上时显式说明原因与该做的事，不显示假数据。
 */

import { createElement, useEffect, useMemo, useRef, useState } from 'react'
import {
  fetchHealth,
  fetchScenes,
  fetchState,
  imageUrl,
  worldToMap,
} from './navlive.js'

const POLL_MS = 400

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(digits)
}

function Badge({ tone = 'muted', children }) {
  return createElement('span', { className: `agh-nl-badge agh-nl-badge-${tone}` }, children)
}

/** 楼层图 + 轨迹 + 机器人：用 <img> 铺底，上面盖一层同尺寸 SVG 画折线与机器人。 */
function MapView({ base, session, state }) {
  const affine = state?.map || {}
  const size = affine.map_size || [0, 0]
  const stamp = Math.round(Number(state?.map_ts || 0) * 5) / 5
  const trail = state?.trail || []
  const points = useMemo(() => {
    if (!size[0] || !size[1] || trail.length < 2) return ''
    const mapped = []
    for (const point of trail) {
      const pixel = worldToMap(affine, point[0], point[2])
      if (pixel) mapped.push(`${pixel[0].toFixed(1)},${pixel[1].toFixed(1)}`)
    }
    return mapped.join(' ')
  }, [affine, trail, size[0], size[1]])

  const robot = state?.pose?.position
    ? worldToMap(affine, state.pose.position[0], state.pose.position[2])
    : null

  return createElement(
    'div',
    { className: 'agh-nl-mapwrap' },
    createElement('img', {
      className: 'agh-nl-map',
      src: imageUrl(base, '/map.png', session, stamp),
      alt: '楼层图',
    }),
    size[0] && size[1]
      ? createElement(
          'svg',
          {
            className: 'agh-nl-mapoverlay',
            viewBox: `0 0 ${size[0]} ${size[1]}`,
            preserveAspectRatio: 'xMidYMid meet',
          },
          points
            ? createElement('polyline', {
                points,
                fill: 'none',
                stroke: '#e85a14',
                strokeWidth: 4,
                strokeLinejoin: 'round',
                strokeLinecap: 'round',
              })
            : null,
          affine.goal_map
            ? createElement('circle', {
                cx: affine.goal_map[0],
                cy: affine.goal_map[1],
                r: 9,
                fill: '#12a150',
                stroke: '#ffffff',
                strokeWidth: 2,
              })
            : null,
          robot
            ? createElement('circle', {
                cx: robot[0],
                cy: robot[1],
                r: 11,
                fill: '#f0780a',
                stroke: '#7a3b00',
                strokeWidth: 3,
              })
            : null,
        )
      : null,
  )
}

function Telemetry({ state }) {
  const metrics = state?.metrics || {}
  const position = state?.pose?.position || []
  const rows = [
    ['目标', state?.target_label || '—'],
    ['位置 (x, z)', `${fmt(position[0])}, ${fmt(position[2])}`],
    ['剩余距离', `${fmt(metrics.distance_to_goal)} m`],
    ['已走步数', String(state?.step_count ?? '—')],
    ['碰撞次数', String(metrics.collisions ?? '—')],
    ['plan_age', String(metrics.plan_age ?? '—')],
    ['回合判定', metrics.success === 1 ? '成功' : state?.episode_finished ? '已结束' : '进行中'],
    ['计划队列', (state?.plan || []).map((item) => item.label).join(' → ') || '—'],
  ]
  return createElement(
    'div',
    { className: 'agh-nl-telemetry' },
    ...rows.map(([label, value]) =>
      createElement(
        'div',
        { className: 'agh-nl-kv', key: label },
        createElement('span', { className: 'agh-nl-k' }, label),
        createElement('span', { className: 'agh-nl-v' }, value),
      ),
    ),
  )
}

function EventList({ state }) {
  const events = state?.events_tail || []
  const kindLabel = {
    session_opened: '打开场景',
    plan_changed: '改派目标',
    waypoint_reached: '到达航点',
    arrived: '到达终点',
    episode_over: '回合结束',
    paused: '暂停',
    resumed: '恢复',
    planner_error: '规划失败',
    stalled: '卡住',
    note: '备注',
  }
  return createElement(
    'ul',
    { className: 'agh-nl-events' },
    ...events
      .slice(-12)
      .reverse()
      .map((event) =>
        createElement(
          'li',
          { key: event.seq, className: `agh-nl-event agh-nl-event-${event.kind}` },
          createElement('span', { className: 'agh-nl-event-kind' }, kindLabel[event.kind] || event.kind),
          createElement(
            'span',
            { className: 'agh-nl-event-detail' },
            event.payload?.label || event.payload?.reason || event.payload?.note || '',
          ),
        ),
      ),
  )
}

function LiveBody({ base }) {
  const [health, setHealth] = useState(null)
  const [scenes, setScenes] = useState(null)
  const [state, setState] = useState(null)
  const [error, setError] = useState(null)
  const [pickedScene, setPickedScene] = useState('')
  const sessionRef = useRef(null)

  // 健康检查：失败就只显示一条明确的排查提示，不再空转轮询
  useEffect(() => {
    let alive = true
    fetchHealth(base)
      .then((info) => {
        if (!alive) return
        setHealth(info)
        setError(null)
        return fetchScenes(base)
      })
      .then((payload) => {
        if (!alive || !payload) return
        setScenes(payload.scenes || [])
        setPickedScene((current) => current || payload.scenes?.[0]?.scene_id || '')
      })
      .catch((err) => {
        if (alive) setError(String(err?.message || err))
      })
    return () => {
      alive = false
    }
  }, [base])

  // 状态轮询：navlive 的仿真线程一直跑，这里只读最新快照
  useEffect(() => {
    if (!health) return undefined
    let alive = true
    const tick = async () => {
      try {
        const payload = await fetchState(base, sessionRef.current)
        if (!alive) return
        setState(payload)
        sessionRef.current = payload?.session_id || sessionRef.current
        setError(null)
      } catch (err) {
        if (alive) setError(String(err?.message || err))
      }
    }
    tick()
    const timer = setInterval(tick, POLL_MS)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [base, health])

  if (error && !health) {
    return createElement(
      'div',
      { className: 'agh-nl-offline' },
      createElement('div', { className: 'agh-nl-offline-title' }, '实时导航服务未连接'),
      createElement('div', { className: 'agh-nl-offline-addr' }, base),
      createElement('div', { className: 'agh-nl-offline-hint' }, error),
      createElement(
        'div',
        { className: 'agh-nl-offline-steps' },
        '排查顺序：① 服务器上 `tmux ls` 看 navlive 会话在不在；② 本机到 43121 的端口转发；③ `curl ',
        base,
        '/health`。',
      ),
    )
  }

  const frameStamp = Math.round(Number(state?.frame_ts || 0) * 5) / 5
  const scene = scenes?.find((item) => item.scene_id === state?.scene_id)

  return createElement(
    'div',
    { className: 'agh-nl-live' },
    createElement(
      'div',
      { className: 'agh-nl-row' },
      createElement(
        'select',
        {
          className: 'agh-nl-select',
          value: pickedScene,
          onChange: (event) => setPickedScene(event.target.value),
        },
        ...(scenes || []).map((item) =>
          createElement('option', { key: item.scene_id, value: item.scene_id }, `${item.scene_id}（${item.episode_count} 回合）`),
        ),
      ),
      createElement(Badge, { tone: health?.ok ? 'ok' : 'warn' }, health?.ok ? `${health.tick_hz} Hz 运行中` : '服务异常'),
      createElement(Badge, { tone: 'muted' }, `${health?.scenes ?? 0} 场景`),
    ),
    createElement(
      'div',
      { className: 'agh-nl-hint' },
      '派活请直接在聊天里说人话（例如「在这个场景里，把客人从门口带到客厅沙发」）。这里只做实时查看。',
    ),
    createElement(
      'div',
      { className: 'agh-nl-frame' },
      state?.session_id
        ? createElement('img', {
            className: 'agh-nl-frame-img',
            src: imageUrl(base, '/frame.jpg', state.session_id, frameStamp),
            alt: '第一人称画面',
          })
        : createElement('div', { className: 'agh-nl-placeholder' }, '还没有打开场景 —— 让 Agent 调 navlive_open'),
    ),
    createElement(
      'div',
      { className: 'agh-nl-caption' },
      state?.session_id
        ? `${state.scene_id} · 回合 ${state.episode_id} · 第 ${state.step_count} 步`
        : '等待会话',
    ),
    createElement(MapView, { base, session: state?.session_id, state }),
    createElement(Telemetry, { state }),
    scene?.episodes?.length
      ? createElement(
          'details',
          { className: 'agh-nl-details' },
          createElement('summary', null, `本场景的 ${scene.episodes.length} 个回合（自然语言指令）`),
          createElement(
            'ul',
            { className: 'agh-nl-episodes' },
            ...scene.episodes.map((episode) =>
              createElement(
                'li',
                { key: episode.episode_id },
                createElement('span', { className: 'agh-nl-ep-id' }, `#${episode.episode_id}`),
                createElement('span', null, String(episode.instruction || '').trim()),
              ),
            ),
          ),
        )
      : null,
    createElement('div', { className: 'agh-nl-section-title' }, '决策 / 事件流'),
    createElement(EventList, { state }),
  )
}

export function LivePanel({ url }) {
  const base = useMemo(() => String(url || '').replace(/\/+$/, ''), [url])
  return createElement(LiveBody, { base })
}

export default LivePanel

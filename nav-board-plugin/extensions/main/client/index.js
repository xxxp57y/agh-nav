/**
 * 导航工作台（浏览器侧，左栏）。
 *
 * 三个页签，对应三种不同的问题（这个区分是刻意的，别合并）：
 *
 *   实时导航 —— 在一个**已跑好的场景**里，机器人按自然语言指令走，
 *               第一人称画面 + 2D 楼层图与轨迹 + 位姿/遥测 + 事件流。数据来自 navlive 服务。
 *   对照分析 —— 已跑完的多轮评估：8 场景 × 5 套恢复配置的对照表。
 *               数据来自 AGH 服务通道 `nav.board`（同源 BFF → 服务器数据面）。
 *   事件流   —— 逐决策证据（触发点、恢复动作、卡住/暂停），演示与复盘用。
 *
 * 为什么分成两个数据源：实时那条是**高频、每帧都变**的（几百毫秒一次），
 * 走 AGH 的服务桥会让会话事件流被淹没；离线的对照表是**低频、要授权**的，适合走服务桥。
 */

import { createElement, useEffect, useState } from 'react'
import { BASELINE } from './baseline.js'
import { LivePanel } from './live.js'

const CONFIG_LABELS = {
  R2A: 'R2A（原来那套）',
  original: 'original（闭环全关）',
  recovery_only: 'recovery_only（最稳）',
  guarded: 'guarded（两条通道+护栏）',
  collision_only: 'collision_only',
}

const CONFIG_ORDER = ['R2A', 'original', 'recovery_only', 'guarded', 'collision_only']

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

function num(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(digits)
}

function bestOf(configs, pick) {
  let best = null
  for (const name of CONFIG_ORDER) {
    const value = configs?.[name] ? pick(configs[name]) : undefined
    if (value === null || value === undefined) continue
    if (best === null || value > best.value) best = { name, value }
  }
  return best
}

function weighted(config, key) {
  const scenes = Object.values(config?.per_scene || {})
  const total = scenes.reduce((acc, scene) => acc + Number(scene.episodes || 0), 0)
  if (!total) return null
  return scenes.reduce((acc, scene) => acc + Number(scene[key] || 0) * Number(scene.episodes || 0), 0) / total
}

function successCell(value, isBest) {
  const width = `${Math.max(0, Math.min(100, Math.round(Number(value || 0) * 100)))}%`
  return createElement(
    'td',
    { className: isBest ? 'agh-nav-td agh-nav-best' : 'agh-nav-td' },
    createElement(
      'div',
      { className: 'agh-nav-cell' },
      createElement('span', { className: 'agh-nav-pct' }, pct(value)),
      createElement(
        'span',
        { className: 'agh-nav-bar' },
        createElement('span', { className: 'agh-nav-bar-fill', style: { width } }),
      ),
    ),
  )
}

function headerRow(configs) {
  return createElement(
    'tr',
    null,
    createElement('th', { className: 'agh-nav-th agh-nav-th-scene' }, '场景'),
    createElement('th', { className: 'agh-nav-th' }, '回合'),
    ...CONFIG_ORDER.filter((name) => configs?.[name]).map((name) =>
      createElement('th', { className: 'agh-nav-th', title: CONFIG_LABELS[name] || name }, name),
    ),
  )
}

function sceneRows(payload) {
  const configs = payload?.configs || {}
  const scenes = payload?.scenes || []
  const shown = CONFIG_ORDER.filter((name) => configs[name])
  return scenes.map((scene) => {
    const values = shown.map((name) => configs[name]?.per_scene?.[scene]?.success ?? null)
    const max = Math.max(...values.filter((value) => value !== null).map(Number), -1)
    const episodes = configs[shown[0]]?.per_scene?.[scene]?.episodes ?? 13
    return createElement(
      'tr',
      { key: scene },
      createElement('td', { className: 'agh-nav-td agh-nav-scene', title: scene }, scene),
      createElement('td', { className: 'agh-nav-td agh-nav-num' }, String(episodes)),
      ...values.map((value) => successCell(value, Number(value) === max && max >= 0)),
    )
  })
}

function summaryRows(payload) {
  const configs = payload?.configs || {}
  const shown = CONFIG_ORDER.filter((name) => configs[name])
  const row = (label, pick, digits = 4, best = true, title) => {
    const bestValue = best ? bestOf(configs, pick) : null
    return createElement(
      'tr',
      { key: label },
      createElement('td', { className: 'agh-nav-td agh-nav-scene', title }, label),
      createElement('td', { className: 'agh-nav-td agh-nav-num' }, '—'),
      ...shown.map((name) =>
        createElement(
          'td',
          {
            className:
              bestValue && bestValue.name === name
                ? 'agh-nav-td agh-nav-num agh-nav-best'
                : 'agh-nav-td agh-nav-num',
          },
          num(pick(configs[name]), digits),
        ),
      ),
    )
  }
  return [
    row('池化 Success', (c) => c.pooled_success, 4, true, '8 场景 104 回合的成功率'),
    row('池化 SPL', (c) => c.pooled_spl, 4, true, 'Success weighted by Path Length'),
    row('最差场景', (c) => c.spread?.min, 4, true, '场景间最小 success —— 稳定性最关键的一列'),
    row('场景间 std', (c) => c.spread?.std, 4, false, '越小越稳'),
    row('场景间 range', (c) => c.spread?.range, 4, false, '最大 - 最小'),
    row('代价：碰撞/回合', (c) => weighted(c, 'collision'), 2, false, '越低越好（不是"越大越好"）'),
    row('代价：S2/回合', (c) => weighted(c, 'system2_calls'), 2, false, 'System-2 调用次数，越低越省'),
  ]
}

function runsList(payload) {
  const runs = payload?.recent_runs || []
  if (!runs.length) return null
  return createElement(
    'details',
    { className: 'agh-nav-runs' },
    createElement('summary', null, `最近跑的 ${runs.length} 个 run（点开看逐 run 指标）`),
    createElement(
      'table',
      { className: 'agh-nav-table agh-nav-table-small' },
      createElement(
        'thead',
        null,
        createElement(
          'tr',
          null,
          createElement('th', { className: 'agh-nav-th' }, 'run'),
          createElement('th', { className: 'agh-nav-th' }, 'mode'),
          createElement('th', { className: 'agh-nav-th' }, '回合'),
          createElement('th', { className: 'agh-nav-th' }, 'Success'),
          createElement('th', { className: 'agh-nav-th' }, '最差场景'),
        ),
      ),
      createElement(
        'tbody',
        null,
        ...runs.map((run) =>
          createElement(
            'tr',
            { key: run.label },
            createElement('td', { className: 'agh-nav-td' }, run.label),
            createElement('td', { className: 'agh-nav-td' }, run.mode || '—'),
            createElement('td', { className: 'agh-nav-td agh-nav-num' }, `${run.episodes}/${run.expected_episodes}`),
            createElement('td', { className: 'agh-nav-td agh-nav-num' }, pct(run.pooled_success)),
            createElement('td', { className: 'agh-nav-td agh-nav-num' }, pct(run.worst_scene_success)),
          ),
        ),
      ),
    ),
  )
}

function AnalysisPanel({ state }) {
  const payload = state.payload || {}
  const badge =
    state.status === 'ok'
      ? `实时数据 · ${state.generated_at || ''}`
      : state.status === 'loading'
        ? '读取中…'
        : '静态快照（打包时生成）'
  const configs = payload.configs || {}
  return createElement(
    'div',
    { className: 'agh-nav-panel' },
    state.error || state.degraded
      ? createElement('div', { className: 'agh-nav-warn' }, `降级说明：${state.error || state.degraded?.reason || ''}`)
      : null,
    createElement(
      'div',
      { className: 'agh-nav-scroll' },
      createElement(
        'table',
        { className: 'agh-nav-table' },
        createElement('thead', null, headerRow(configs)),
        createElement('tbody', null, ...sceneRows(payload), ...summaryRows(payload)),
      ),
    ),
    createElement(
      'div',
      { className: 'agh-nav-note' },
      '口径：8 场景 × 13 回合 = 104 回合/配置（4 个调参场景 + 4 个模型没见过的场景）；每格 SE≈14pt。',
    ),
    createElement('span', { className: 'agh-nav-badge' }, badge),
    runsList(payload),
    createElement(
      'div',
      { className: 'agh-nav-footer' },
      createElement('button', { className: 'agh-nav-button', onClick: () => window.location.reload() }, '刷新'),
      createElement(
        'span',
        { className: 'agh-nav-footnote' },
        '细节请让 Agent 调 nav_run_summary / nav_recovery_options',
      ),
    ),
  )
}

function baselineState(reason) {
  return {
    status: 'fallback',
    source: 'baseline',
    generated_at: BASELINE?.generated_at_iso || null,
    payload: BASELINE?.payload || null,
    degraded: reason ? { reason } : null,
  }
}

const HOOKS_AVAILABLE = typeof useState === 'function' && typeof useEffect === 'function'

function AnalysisLive({ service }) {
  const [state, setState] = useState({ status: 'loading', payload: BASELINE?.payload || null })
  useEffect(() => {
    let alive = true
    if (!service || typeof service.call !== 'function') {
      setState(baselineState('当前会话没有可用的服务通道（nav.board 不可调用）'))
      return () => {
        alive = false
      }
    }
    service
      .call('nav.board', {})
      .then((result) => {
        if (!alive) return
        setState({
          status: 'ok',
          source: result?.source || 'live',
          generated_at: result?.generated_at || null,
          payload: result?.payload || null,
          degraded: result?.degraded || null,
          error: null,
        })
      })
      .catch((error) => {
        if (!alive) return
        setState(baselineState(String(error?.message || error)))
      })
    return () => {
      alive = false
    }
  }, [service])
  return createElement(AnalysisPanel, { state })
}

function StaticAnalysis() {
  return createElement(AnalysisPanel, { state: baselineState('宿主未暴露 React hooks，显示打包时的静态快照') })
}

const TABS = [
  { id: 'live', label: '实时导航' },
  { id: 'analysis', label: '对照分析' },
]

function Workbench({ service, navliveUrl }) {
  const [tab, setTab] = useState('live')
  const Analysis = HOOKS_AVAILABLE ? () => createElement(AnalysisLive, { service }) : StaticAnalysis
  return createElement(
    'div',
    { className: 'agh-nav-root' },
    createElement(
      'header',
      { className: 'agh-nav-header' },
      createElement('div', { className: 'agh-nav-title' }, '导航工作台'),
      createElement(
        'div',
        { className: 'agh-nav-sub' },
        tab === 'live'
          ? '选一个已跑好的场景，让 Agent 按自然语言指令把机器人走过去'
          : '已跑完的评估：五套恢复配置 × 8 场景对照',
      ),
      createElement(
        'div',
        { className: 'agh-nav-tabs' },
        ...TABS.map((item) =>
          createElement(
            'button',
            {
              key: item.id,
              className: item.id === tab ? 'agh-nav-tab agh-nav-tab-active' : 'agh-nav-tab',
              onClick: () => setTab(item.id),
            },
            item.label,
          ),
        ),
      ),
    ),
    tab === 'live'
      ? createElement(LivePanel, { url: navliveUrl })
      : createElement(Analysis, null),
  )
}

export function apply(ctx, config) {
  const service = ctx?.agnes?.services
  // navlive 的地址是**展示元数据**（不是凭据），放在 agnes.client.json 的 publicConfig 里，
  // 部署时改一个地方即可；拿不到就用默认值。
  const navliveUrl = config?.publicConfig?.navliveUrl || 'http://127.0.0.1:43121'
  const Component = () => createElement(Workbench, { service, navliveUrl })
  ctx.slots.register('ui:sidebar', Component, { priority: -1 })
}

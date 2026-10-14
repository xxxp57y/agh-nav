/**
 * 实时导航（navlive）工具集 —— "在一个已跑好的场景里真的把机器人走起来"。
 *
 * 与 `tools.mjs` 的分工：
 * - `nav_*`：离线数据面。看某一轮跑得怎么样、失败在哪、恢复通道会怎么动。只读、不改变世界。
 * - `navlive_*`：在线数据面。开一个场景、下一条指令、看机器人现在在哪。**会真的动**。
 *
 * 审批边界（有意设计）：
 * - 只读的看（场景清单 / 当前状态）→ `requiresApproval: 'never'`。
 * - 会让机器人动起来的（开场景、改目标、暂停/恢复）→ `requiresApproval: 'always'`。
 *   理由很实在：这些操作会占用 GPU、改变服务器状态，而且演示时**误触一次就要重新跑**。
 */

import {
  action as navliveAction,
  envelope,
  health,
  landmarks as navliveLandmarks,
  open as navliveOpen,
  scenes as navliveScenes,
  state as navliveState,
  stateOf,
} from './navlive.mjs'
import { Arr, Num, Obj, Str } from './schema.mjs'

const READ_ONLY = Object.freeze({
  isReadOnly: true,
  isDestructive: false,
  isConcurrencySafe: true,
  isOpenWorld: true,
  replay: 'safe',
  costHint: undefined,
  deferLoading: false,
  requiresApproval: 'never',
})

const WRITE = Object.freeze({
  isReadOnly: false,
  isDestructive: false,
  isConcurrencySafe: false,
  isOpenWorld: true,
  // 让机器人在仿真里真走一段：同一句话重放会得到不同的位置，所以不可重放。
  replay: 'never',
  costHint: { wallMs: 30000 },
  deferLoading: false,
  requiresApproval: 'always',
})

function result(summary, payload) {
  return { content: [{ type: 'text', text: summary }], structured: payload }
}

function round(value, digits = 2) {
  return value === null || value === undefined ? null : Number(Number(value).toFixed(digits))
}

/** 把快照整理成"模型能直接讲给用户听"的紧凑结构。 */
function digest(snapshot) {
  const metrics = snapshot.metrics || {}
  return {
    session_id: snapshot.session_id,
    scene_id: snapshot.scene_id,
    episode_id: snapshot.episode_id,
    instruction: snapshot.instruction,
    target_label: snapshot.target_label,
    target: snapshot.target,
    queue: (snapshot.plan || []).map((item) => item.label),
    pose: snapshot.pose?.position,
    distance_to_goal_m: round(metrics.distance_to_goal),
    collisions: metrics.collisions,
    plan_age: metrics.plan_age,
    success: metrics.success,
    step_count: snapshot.step_count,
    paused: Boolean(snapshot.paused),
    pause_reason: snapshot.pause_reason || null,
    goal_reached: Boolean(snapshot.goal_reached),
    episode_finished: Boolean(snapshot.episode_finished),
    trail_points: Array.isArray(snapshot.trail) ? snapshot.trail.length : 0,
    last_events: (snapshot.events_tail || []).slice(-6).map((event) => ({
      kind: event.kind,
      at: event.ts,
      detail: event.payload,
    })),
  }
}

function statusLine(snapshot) {
  const metrics = snapshot.metrics || {}
  if (!snapshot.session_id) return '还没有打开任何场景会话。'
  const position = (snapshot.pose?.position || []).map((value) => Number(value).toFixed(2)).join(', ')
  const parts = [
    `场景 ${snapshot.scene_id} · 回合 ${snapshot.episode_id}`,
    `目标「${snapshot.target_label || '—'}」`,
    `位置 (x,z)=(${Number(snapshot.pose?.position?.[0] || 0).toFixed(2)}, ${Number(snapshot.pose?.position?.[2] || 0).toFixed(2)})`,
    `剩余直线距离 ${metrics.distance_to_goal === null || metrics.distance_to_goal === undefined ? '—' : Number(metrics.distance_to_goal).toFixed(2)} m`,
    `已走 ${snapshot.step_count} 步`,
    `碰撞 ${metrics.collisions ?? '—'} 次`,
  ]
  if (snapshot.paused) parts.push(`**已暂停**（${snapshot.pause_reason || '未说明原因'}）`)
  if (snapshot.goal_reached) parts.push('**已到达**')
  if (snapshot.episode_finished && !snapshot.goal_reached) parts.push('本回合已结束（未判定成功）')
  parts.push(`坐标 ${position}`)
  return parts.join('｜')
}

function waypointArray() {
  return Arr(
    Obj(
      {
        label: Str({ description: '这一步叫什么（会显示在工作台的目标上），如「A 护士站」' }),
        position: Arr(Num(), {
          description: '目标点世界坐标 [x, y, z]（米）。可以只给 [x, z]，y 默认 0。',
          minItems: 2,
          maxItems: 3,
        }),
      },
      ['position'],
    ),
    { description: '按顺序走的航点；第一个是当前目标，其余排队' },
  )
}

export function createNavLiveTools() {
  return [
    {
      name: 'navlive_status',
      description:
        '实时导航服务（navlive）是否在线、载入了哪些场景、当前有没有打开的会话。在做任何实时导航动作之前先看这个；它同时也说明工作台右侧画面用的是哪个地址。',
      parameters: Obj({}),
      meta: READ_ONLY,
      async execute() {
        const info = await health()
        const summary = [
          `navlive 服务：${info.ok ? '在线' : '异常'}（${envelope().service}）`,
          `- 已载入场景 ${info.scenes} 个 / 回合 ${info.episodes} 个`,
          `- 仿真频率 ${info.tick_hz} Hz，当前会话 ${info.session_id || '（无）'}`,
          info.last_error ? `- ⚠️ 最近一次仿真错误：${info.last_error}` : '- 仿真循环正常',
        ].join('\n')
        return result(summary, envelope({ data: info }))
      },
    },
    {
      name: 'navlive_scenes',
      description:
        '列出可交互导航的场景与每个场景的回合（含自然语言指令、起点、终点坐标）。用户说"我想在某个场景里做任务"时先用这个把选项摆出来。',
      parameters: Obj({}),
      meta: READ_ONLY,
      async execute() {
        const payload = await navliveScenes()
        const list = payload.scenes || []
        const lines = list.map((scene) => {
          const sample = scene.episodes?.[0]
          return `- \`${scene.scene_id}\`：${scene.episode_count} 个回合｜示例指令“${String(
            sample?.instruction || '',
          ).trim().slice(0, 60)}”`
        })
        return result(
          [`可选场景（共 ${list.length} 个）：`, ...lines, '', '下一步：`navlive_open` 选一个场景与回合。'].join('\n'),
          envelope({ data: payload }),
        )
      },
    },
    {
      name: 'navlive_open',
      description:
        '打开一个场景的一个回合，得到一个"活的"导航会话：之后可以用 navlive_goto 让机器人真的走过去，工作台右侧会实时显示第一人称画面、2D 楼层图与轨迹、位姿与逐决策遥测。换场景要重新载入（约 20 秒），换回合只要几秒。',
      parameters: Obj(
        {
          scene: Str({ description: '场景 id，如 2azQ1b91cZZ（用 navlive_scenes 取）' }),
          episode: Str({ description: '回合 id，如 "10"；不填就用该场景第一个回合' }),
          instruction: Str({ description: '可选：开局就照着这句自然语言指令走' }),
        },
        ['scene'],
      ),
      meta: WRITE,
      async execute(args) {
        const payload = await navliveOpen({
          scene: args.scene,
          episode: args.episode,
          instruction: args.instruction,
        })
        const snapshot = stateOf(payload)
        const summary = [
          `已打开场景 \`${snapshot.scene_id}\` 回合 ${snapshot.episode_id}。`,
          `- 这个回合原本的指令：“${String(snapshot.instruction || '').trim()}”`,
          `- 起点 (${(snapshot.start_position || []).map((v) => Number(v).toFixed(2)).join(', ')})`,
          `- 终点 (${(snapshot.episode_goal || []).map((v) => Number(v).toFixed(2)).join(', ')})`,
          '',
          '工作台右侧现在应该已经有实时画面与楼层图了。要让它走，调 `navlive_goto`。',
        ].join('\n')
        return result(summary, envelope({ state: digest(snapshot) }))
      },
    },
    {
      name: 'navlive_goto',
      description:
        '最重要的一条：把机器人当前的导航目标改成给定的航点（世界坐标），它就会真的在 Habitat 里一步步走过去。用户中途改主意（"先送 C 站""A 站不能接收了，改去 B"）时就用这个，工作台的画面与轨迹会跟着实时变化。',
      parameters: Obj(
        {
          waypoints: waypointArray(),
          reason: Str({ description: '为什么改目标（记进事件流，便于事后复盘）' }),
        },
        ['waypoints'],
      ),
      meta: WRITE,
      async execute(args) {
        const payload = await navliveAction({
          kind: 'goto',
          waypoints: args.waypoints,
          note: args.reason || null,
        })
        const snapshot = stateOf(payload)
        const labels = (args.waypoints || []).map((w) => w.label || '航点').join(' → ')
        const summary = [
          `已改派：${labels}`,
          `- ${statusLine(snapshot)}`,
          args.reason ? `- 原因：${args.reason}` : null,
        ]
          .filter(Boolean)
          .join('\n')
        return result(summary, envelope({ state: digest(snapshot), result: payload.result }))
      },
    },
    {
      name: 'navlive_state',
      description:
        '读取当前导航会话的实时状态：位姿、目标、剩余距离、碰撞次数、plan_age、走到第几步、最近事件流，以及画面/楼层图的更新时间戳。用它向用户汇报"机器人现在到哪了"。',
      parameters: Obj({ session: Str({ description: '可选：指定会话 id；不填用当前会话' }) }),
      meta: READ_ONLY,
      async execute(args) {
        const payload = await navliveState(args.session)
        const snapshot = stateOf(payload)
        return result(statusLine(snapshot), envelope({ state: digest(snapshot) }))
      },
    },
    {
      name: 'navlive_landmarks',
      description:
        '列出当前场景里"叫得出名字的地方"（地名册）：每个地标有中文名、英文短语、以及它真实对应的坐标。用户说"带客人去客厅""去厨房"这类**地方名**时，先用这个看有没有，再用 navlive_goto_landmark 走过去。地名册来自场景自带 R2R 指令里提到的地点，坐标是该短语真正指代的回合端点。',
      parameters: Obj(
        {
          scene: Str({ description: '场景 id；不填就用当前会话的场景' }),
          limit: Num({ description: '最多列几个（默认全列）' }),
        },
        [],
      ),
      meta: READ_ONLY,
      async execute(args) {
        const payload = await navliveLandmarks({ scene: args.scene })
        const list = payload.landmarks || []
        const limit = Number(args.limit || list.length)
        const shown = list.slice(0, limit)
        const lines = shown.map(
          (item) =>
            `- ${item.name}（${item.name_en}）→ (${Number(item.position[0]).toFixed(2)}, ${Number(item.position[2]).toFixed(2)})` +
            `${item.episode_id ? ` · 来自回合 ${item.episode_id}` : ''}`,
        )
        const head = payload.scene_id
          ? `场景 ${payload.scene_id} 的地名册（共 ${list.length} 个，列出 ${shown.length} 个）：`
          : `各地名册概况：${(payload.scenes || []).map((s) => `${s.scene_id}(${s.count})`).join('、')}`
        return result(
          [head, ...lines, '', '下一步：`navlive_goto_landmark`（给名字，服务端换坐标）。'].join('\n'),
          envelope({ data: payload }),
        )
      },
    },
    {
      name: 'navlive_goto_landmark',
      description:
        '**用地方名让机器人走过去**：传地名（"客厅""阳台下的入口""厨房"），服务端在地名册里解析成真实坐标再走。名字解析不出会明确失败并给出候选，不会瞎走。支持一串地名当航点（先 A 后 B）。',
      parameters: Obj(
        {
          names: Arr(Str({ description: '地标名，如「客厅」或「kitchen」' }), {
            description: '按顺序走的地标名；第一个是当前目标',
          }),
          reason: Str({ description: '为什么让它去（记进事件流）' }),
        },
        ['names'],
      ),
      meta: WRITE,
      async execute(args) {
        const payload = await navliveAction({
          kind: 'goto_landmark',
          names: args.names,
          note: args.reason || null,
        })
        const snapshot = stateOf(payload)
        const resolved = payload.result?.resolved || []
        const summary = [
          `已按地名改派：${resolved.map((item) => item.label).join(' → ')}`,
          ...resolved.map(
            (item) =>
              `- ${item.label} → (${Number(item.position[0]).toFixed(2)}, ${Number(item.position[2]).toFixed(2)})` +
              `${item.name_en && item.name_en !== item.label ? `（指令原文：${item.name_en}）` : ''}`,
          ),
          `- ${statusLine(snapshot)}`,
        ].join('\n')
        return result(summary, envelope({ state: digest(snapshot), resolved, result: payload.result }))
      },
    },
    {
      name: 'navlive_pause',
      description:
        '让机器人停在原地（保持场景与画面，随时可以 navlive_resume 继续）。适合"先别走，我要看一下"或演示"接收方暂时不能接收"。',
      parameters: Obj({ reason: Str({ description: '暂停原因' }) }),
      meta: WRITE,
      async execute(args) {
        const payload = await navliveAction({ kind: 'pause', reason: args.reason || '人工暂停' })
        const snapshot = stateOf(payload)
        return result(
          `已暂停：${snapshot.pause_reason}｜${statusLine(snapshot)}`,
          envelope({ state: digest(snapshot) }),
        )
      },
    },
    {
      name: 'navlive_resume',
      description: '恢复被 navlive_pause 停下的机器人，继续走向当前目标。',
      parameters: Obj({}),
      meta: WRITE,
      async execute() {
        const payload = await navliveAction({ kind: 'resume' })
        const snapshot = stateOf(payload)
        return result(`已恢复｜${statusLine(snapshot)}`, envelope({ state: digest(snapshot) }))
      },
    },
  ]
}

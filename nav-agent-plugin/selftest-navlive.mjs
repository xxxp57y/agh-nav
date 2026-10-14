#!/usr/bin/env node
/**
 * navlive 工具自检（不需要 AGH，也不需要模型）。
 *
 * 它直接 import 插件里的工具定义，逐个调用 **在线只读工具**，断言：
 * 返回的是结构化 JSON、字段齐全、数字合理。写工具（会真让机器人动）**不**在这里调，
 * 只核对它们的元数据（审批要求、replay）——那是安全边界，必须显式验证。
 *
 * 用法：
 *   node selftest.mjs                 # 只读工具 + 元数据
 *   NAVLIVE_URL=http://127.0.0.1:43121 node selftest.mjs
 */

import { createNavLiveTools } from './lib/navlive-tools.mjs'
import { serviceUrl } from './lib/navlive.mjs'

let passed = 0
let failed = 0
const failures = []

function check(name, condition, detail) {
  if (condition) {
    passed += 1
    console.log(`  ok   ${name}`)
  } else {
    failed += 1
    failures.push(`${name}${detail ? ` — ${detail}` : ''}`)
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

function byName(tools, name) {
  const tool = tools.find((item) => item.name === name)
  if (!tool) throw new Error(`工具 ${name} 不在清单里`)
  return tool
}

const EXPECTED = [
  'navlive_status',
  'navlive_scenes',
  'navlive_open',
  'navlive_goto',
  'navlive_landmarks',
  'navlive_goto_landmark',
  'navlive_state',
  'navlive_pause',
  'navlive_resume',
]

//: 会真的让机器人动的工具：必须逐条检查审批边界
const WRITE_TOOLS = ['navlive_open', 'navlive_goto', 'navlive_goto_landmark', 'navlive_pause', 'navlive_resume']

async function main() {
  console.log(`navlive 服务：${serviceUrl()}`)
  const tools = createNavLiveTools()

  console.log('\n== 工具清单与 Schema ==')
  check('9 个 navlive 工具', tools.length === EXPECTED.length, `实际 ${tools.length}`)
  for (const name of EXPECTED) {
    const tool = byName(tools, name)
    check(`${name} 有描述`, typeof tool.description === 'string' && tool.description.length > 12)
    check(`${name} 有 TypeBox Schema`, tool.parameters?.[Symbol.for('TypeBox.Kind')] === 'Object')
    check(`${name} 有 execute`, typeof tool.execute === 'function')
  }

  console.log('\n== 审批边界（写操作必须 always） ==')
  for (const name of WRITE_TOOLS) {
    const meta = byName(tools, name).meta
    check(`${name} requiresApproval=always`, meta.requiresApproval === 'always', String(meta.requiresApproval))
    check(`${name} isReadOnly=false`, meta.isReadOnly === false)
    check(`${name} replay=never`, meta.replay === 'never', String(meta.replay))
  }
  for (const name of ['navlive_status', 'navlive_scenes', 'navlive_state', 'navlive_landmarks']) {
    const meta = byName(tools, name).meta
    check(`${name} requiresApproval=never`, meta.requiresApproval === 'never')
    check(`${name} isReadOnly=true`, meta.isReadOnly === true)
  }

  console.log('\n== 只读工具真调（在线数据面） ==')
  const status = await byName(tools, 'navlive_status').execute({})
  check('navlive_status 返回结构化', status.structured?.schema === 'agh-navlive/tool-result/1')
  check('navlive_status 报告在线', status.structured?.data?.ok === true, JSON.stringify(status.structured?.data))
  check('navlive_status 有场景数', Number(status.structured?.data?.scenes) >= 1)
  check('navlive_status 有人可读文本', typeof status.content?.[0]?.text === 'string' && status.content[0].text.length > 10)

  const scenes = await byName(tools, 'navlive_scenes').execute({})
  const list = scenes.structured?.data?.scenes || []
  check('navlive_scenes 至少 1 个场景', list.length >= 1, `实际 ${list.length}`)
  const first = list[0] || {}
  check('场景有 episode_count', Number(first.episode_count) >= 1)
  check('场景有自然语言指令', String(first.episodes?.[0]?.instruction || '').trim().length > 5)
  check('场景有起点与终点', Array.isArray(first.episodes?.[0]?.start_position) && Array.isArray(first.episodes?.[0]?.goal))

  const state = await byName(tools, 'navlive_state').execute({})
  check('navlive_state 返回结构化', state.structured?.schema === 'agh-navlive/tool-result/1')
  const digest = state.structured?.state || {}
  check('状态里有会话 id', typeof digest.session_id === 'string' && digest.session_id.startsWith('nl-'), String(digest.session_id))
  check('状态里有位姿', Array.isArray(digest.pose) && digest.pose.length === 3, JSON.stringify(digest.pose))
  check('状态里有剩余距离', typeof digest.distance_to_goal_m === 'number', String(digest.distance_to_goal_m))
  check('状态里有轨迹点数', Number(digest.trail_points) >= 1, String(digest.trail_points))

  console.log(`\n结果：${passed} 通过 / ${failed} 失败`)
  if (failures.length) {
    console.log('失败项：')
    for (const item of failures) console.log(`  - ${item}`)
  }
  return failed === 0 ? 0 : 1
}

main().then(
  (code) => process.exit(code),
  (error) => {
    console.error(`自检异常：${error?.stack || error}`)
    process.exit(2)
  },
)

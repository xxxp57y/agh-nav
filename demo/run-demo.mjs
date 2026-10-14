#!/usr/bin/env node
/**
 * 路线① 导航侧的**一键演示**：不接模型，直接把"自然语言请求 → AGH 工具调用"这条链路
 * 按固定剧本跑一遍，并把每一步的真实输出落成 markdown。
 *
 * 为什么要有它：
 * - 比赛现场最怕的是"模型没选中工具"这种随机性。这个脚本用**同一批工具**跑确定性剧本，
 *   用来证明"链路本身是通的"，并且给作品文档提供可核对的逐幕记录。
 * - 需要审批的写操作（`nav_run_eval`）在这里只打印 meta 与将要下发的参数，不真启动。
 *
 * 用法::
 *
 *     NAV_AGENT_EXEC=local NAV_AGENT_SRC=/srv/agh-demo/src \
 *       node demo/run-demo.mjs --run r1grd4 --episode 43 --out demo/out
 */

import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import process from 'node:process'
import { createTools } from '../nav-agent-plugin/lib/tools.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..')
const args = process.argv.slice(2)
const arg = (name, fallback) => {
  const index = args.indexOf(`--${name}`)
  if (index === -1) return fallback
  const next = args[index + 1]
  return next && !next.startsWith('--') ? next : true
}

const RUN = String(arg('run', 'r1grd4'))
const EPISODE = Number(arg('episode', 43))
const OUT_DIR = resolve(ROOT, String(arg('out', 'demo/out')))

const tools = Object.fromEntries(createTools().map((tool) => [tool.name, tool]))
const steps = []

async function call(name, input, note) {
  const tool = tools[name]
  if (!tool) throw new Error(`没有这个工具：${name}`)
  const started = Date.now()
  const output = await tool.execute(input)
  const record = {
    tool: name,
    input,
    ms: Date.now() - started,
    source: output.structured?.source,
    text: output.content?.[0]?.text || '',
    note: note || '',
    meta: {
      isReadOnly: tool.meta.isReadOnly,
      requiresApproval: tool.meta.requiresApproval,
      costHint: tool.meta.costHint || null,
    },
  }
  steps.push(record)
  console.log(`\n===== ${name} =====`)
  console.log(record.text.split('\n').slice(0, 18).join('\n'))
  console.log(`(${record.ms} ms, source=${record.source})`)
  return output
}

// --------------------------------------------------------------------------- #
// 剧本：五幕 + 一幕"需要审批的写操作"
// --------------------------------------------------------------------------- #

const SCRIPT = [
  { act: '第一幕｜先看环境能不能跑', request: '导航服务器现在什么状态？有没有别人占着 GPU？' },
  { act: '', request: '列一下我们跑过哪些导航评估，哪套恢复配置最好？' },
  { act: '第二幕｜看这个 run 到底差在哪', request: `帮我细看 ${RUN} 这个 run：逐场景成功率、失败是怎么结束的。` },
  { act: '', request: `第 ${EPISODE} 回合的智能体经历了什么？把逐决策的距离、碰撞、漂移都列出来。` },
  { act: '第三幕｜把失败变成可分析的事件', request: `把 ${RUN} 的失败转成统一失败事件，标注类型和归因。` },
  { act: '', request: '现在的重规划触发判据到底有没有信息量？给我 P@k 和 AUROC。' },
  { act: '第四幕｜失败之后该怎么调整任务', request: `如果 ${RUN} 用碰撞密度和进度停滞这两条通道，会在哪些点要求重规划？覆盖多少失败回合？` },
  { act: '', request: '五套恢复配置在 8 个场景上的最终对照表给我。' },
  { act: '第五幕｜跑到哪里了 / 要不要再来一轮', request: `看一下 ${RUN} 的进度和日志尾部。` },
]

async function main() {
  console.log(`路线① 导航侧演示（run=${RUN}, episode=${EPISODE}）`)
  await call('nav_env_status', {}, SCRIPT[0].request)
  await call('nav_list_runs', { limit: 6 }, SCRIPT[1].request)
  await call('nav_run_summary', { run: RUN }, SCRIPT[2].request)
  await call('nav_observe_episode', { run: RUN, episode: EPISODE, max_decisions: 25 }, SCRIPT[3].request)
  await call('nav_failure_events', { run: RUN, limit: 5 }, SCRIPT[4].request)
  await call('nav_diagnose_triggers', { run: RUN }, SCRIPT[5].request)
  await call('nav_recovery_options', { run: RUN, channels: 'both' }, SCRIPT[6].request)
  await call('nav_compare_configs', {}, SCRIPT[7].request)
  await call('nav_run_progress', { run: RUN }, SCRIPT[8].request)

  // 需要审批的写操作：只展示元数据与将要下发的参数，不真启动。
  const launch = tools.nav_run_eval
  const planned = {
    label: 'r1demo4',
    budget_mode: 'recovery_only',
    watchdog: 'off',
    channels: 'both',
    episodes: 52,
  }
  steps.push({
    tool: 'nav_run_eval',
    input: planned,
    ms: 0,
    source: 'plan-only',
    note: '演示里不真启动：该工具 requiresApproval=always，需要人工在 AGH 里批准',
    meta: {
      isReadOnly: launch.meta.isReadOnly,
      requiresApproval: launch.meta.requiresApproval,
      costHint: launch.meta.costHint || null,
    },
    text: [
      '（未执行）准备下发：',
      `- 工具：nav_run_eval`,
      `- 参数：${JSON.stringify(planned)}`,
      `- 审批：${launch.meta.requiresApproval}（必须人工批准）`,
      `- 代价：${launch.meta.costHint}`,
    ].join('\n'),
  })

  // ------------------------------------------------------------------ #
  const transcript = [
    '# 路线① 导航侧：AGH 工具链演示记录（确定性剧本）',
    '',
    `- 生成时间：${new Date().toISOString()}`,
    `- run：\`${RUN}\`；episode：\`${EPISODE}\``,
    '- 说明：本记录由 `demo/run-demo.mjs` 直接调用 AGH 工具对象生成，**没有模型参与**；',
    '  因此它证明的是"工具链路 + 数据面"，不证明"模型一定会选中工具"。',
    '',
  ]
  for (const step of steps) {
    transcript.push(`## ${step.tool}${step.note ? `（${step.note}）` : ''}`)
    transcript.push('')
    transcript.push(`- 输入：\`${JSON.stringify(step.input)}\``)
    transcript.push(`- 数据来源：\`${step.source}\`；耗时：${step.ms} ms`)
    transcript.push(`- 审批：\`${step.meta.requiresApproval}\`${step.meta.costHint ? `；代价提示：${step.meta.costHint}` : ''}`)
    transcript.push('')
    transcript.push('```text')
    transcript.push(step.text)
    transcript.push('```')
    transcript.push('')
  }
  mkdirSync(OUT_DIR, { recursive: true })
  const transcriptPath = join(OUT_DIR, 'demo-transcript.md')
  writeFileSync(transcriptPath, `${transcript.join('\n')}\n`, 'utf8')
  console.log(`\n[run-demo] 写入 ${transcriptPath}`)

  const keyNumbers = {
    schema: 'agh-nav/demo-key-numbers/1',
    generated_at: new Date().toISOString(),
    run: RUN,
    episode: EPISODE,
    steps: steps.map((step) => ({ tool: step.tool, ms: step.ms, source: step.source })),
  }
  const keyPath = join(OUT_DIR, 'demo-key-numbers.json')
  writeFileSync(keyPath, JSON.stringify(keyNumbers, null, 2), 'utf8')
  console.log(`[run-demo] 写入 ${keyPath}`)
}

main().catch((error) => {
  console.error(error?.stack || error)
  process.exit(1)
})

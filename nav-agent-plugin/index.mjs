/**
 * @agh-nav/nav-agent-plugin —— 导航侧的 AGH 能力包（后端行）。
 *
 * 两条数据线，别混：
 *
 * 1. **离线数据面 `nav_*`**（`lib/tools.mjs`）：读已跑完的 run 日志，算指标、找失败、
 *    复算恢复通道。回答"这一轮跑得怎么样、失败在哪"。只读，不改世界。
 * 2. **在线数据面 `navlive_*`**（`lib/navlive-tools.mjs`）：连 navlive 服务，
 *    在一个已跑好的场景里真的把机器人走起来。回答"现在这个场景里，机器人去哪、到哪了"。
 *    会动的操作要审批。
 *
 * 外加一个 Skill `nav-failure-diagnosis`（离线侧的诊断方法论）。
 */

import { NAV_SKILL } from './lib/skill.mjs'
import { createNavLiveTools } from './lib/navlive-tools.mjs'
import { createTools } from './lib/tools.mjs'

export const navAgent = {
  inject: ['extension', 'skills'],
  apply(ctx) {
    const agnes = ctx.extension()
    const offline = createTools()
    const live = createNavLiveTools()
    const tools = [...offline, ...live]
    for (const tool of tools) {
      agnes.registerTool(tool)
    }
    let skillRegistered = false
    try {
      if (ctx.skills && typeof ctx.skills.register === 'function') {
        ctx.skills.register(NAV_SKILL)
        skillRegistered = true
      }
    } catch (error) {
      // Skill 注册失败不能让工具也起不来：这是两条独立的能力线。
      if (agnes.ctx?.log?.warn) {
        agnes.ctx.log.warn(`nav-failure-diagnosis Skill 注册失败：${error.message}`)
      }
    }
    if (agnes.ctx?.log?.info) {
      agnes.ctx.log.info(
        `@agh-nav/nav-agent-plugin 已就绪：${offline.length} 个离线诊断工具 + ${live.length} 个实时导航工具${
          skillRegistered ? ' + 1 个 Skill' : ''
        }`,
      )
    }
  },
}

export const meta = {
  tools: [
    'nav_env_status',
    'nav_list_runs',
    'nav_run_summary',
    'nav_observe_episode',
    'nav_failure_events',
    'nav_diagnose_triggers',
    'nav_recovery_options',
    'nav_compare_configs',
    'nav_run_progress',
    'nav_run_eval',
    'nav_tail_log',
    'navlive_status',
    'navlive_scenes',
    'navlive_open',
    'navlive_goto',
    'navlive_landmarks',
    'navlive_goto_landmark',
    'navlive_state',
    'navlive_pause',
    'navlive_resume',
  ],
  skill: NAV_SKILL.name,
}

export default navAgent

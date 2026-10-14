#!/usr/bin/env node
/**
 * 无模型自检：验证 `@agh-nav/nav-agent-plugin` 在**不启动 AGH** 的情况下
 * 也能证明自己是对的。检查四层：
 *
 * 1. **形状**：mock 一个 Cordis Context（只有 `extension()` / `skills`），
 *    `apply()` 后工具与 Skill 是否按声明注册、meta 是否自洽、参数 Schema 是否带 TypeBox 标记。
 * 2. **契约**：每个 tool.execute 返回 `{content, structured}`，structured 带 source/transport。
 * 3. **真数据**：用真实 run 跑一遍只读工具，核对关键字段（不是"没报错"就算过）。
 * 4. **回退**：强制 ssh 模式指向一个不存在的域名 + 已有缓存，
 *    验证"实时失败 → 缓存快照 + degraded 标注"这条路径真的生效。
 *
 * 用法（在装了 Node 24 的机器上，本地或服务器都行）::
 *
 *     NAV_AGENT_EXEC=local NAV_AGENT_SRC=/srv/agh-demo/src \
 *       node selftest.mjs --run r1grd4 --episode 43
 *     node selftest.mjs --offline --json
 *
 * 退出码：0 = 全过；1 = 有失败项（打印每个失败项的原始证据）。
 */

import { existsSync } from 'node:fs'
import process from 'node:process'

const args = process.argv.slice(2)
const flag = (name, fallback = undefined) => {
  const index = args.indexOf(`--${name}`)
  if (index === -1) return fallback
  const next = args[index + 1]
  return next && !next.startsWith('--') ? next : true
}
const RUN = String(flag('run', 'r1grd4'))
const EPISODE = Number(flag('episode', 43))
const OFFLINE = Boolean(flag('offline', false))
const AS_JSON = Boolean(flag('json', false))
const INCLUDE_LAUNCH = Boolean(flag('include-launch', false))

if (OFFLINE) {
  process.env.NAV_AGENT_EXEC = 'ssh'
  process.env.NAV_AGENT_SSH_HOST = 'offline.invalid'
  process.env.NAV_AGENT_SSH_BIN = existsSync('C:\\Windows\\System32\\OpenSSH\\ssh.exe')
    ? 'C:\\Windows\\System32\\OpenSSH\\ssh.exe'
    : 'ssh'
  process.env.NAV_AGENT_TIMEOUT_MS = '20000'
}

const checks = []
function check(name, ok, evidence) {
  checks.push({ name, ok: Boolean(ok), evidence })
  if (!AS_JSON) {
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : `\n      ↳ ${typeof evidence === 'string' ? evidence : JSON.stringify(evidence)}`}`)
  }
}

function mockContext() {
  const tools = []
  const skills = []
  const logs = []
  const agnes = {
    registerTool(tool) {
      tools.push(tool)
    },
    on() {},
    ctx: {
      log: {
        info: (message) => logs.push(`info: ${message}`),
        warn: (message) => logs.push(`warn: ${message}`),
        error: (message) => logs.push(`error: ${message}`),
      },
    },
  }
  const ctx = {
    extension: () => agnes,
    skills: { register: (skill) => skills.push(skill) },
  }
  return { ctx, tools, skills, logs }
}

function hasKindTag(schema, seen = new Set()) {
  if (!schema || typeof schema !== 'object') return true
  if (seen.has(schema)) return true
  seen.add(schema)
  const kind = schema[Symbol.for('TypeBox.Kind')]
  if (Array.isArray(schema.anyOf)) {
    return schema.anyOf.every((item) => hasKindTag(item, seen))
  }
  if (schema.items) return hasKindTag(schema.items, seen)
  if (schema.properties) {
    return Boolean(kind) && Object.values(schema.properties).every((item) => hasKindTag(item, seen))
  }
  return Boolean(kind)
}

async function main() {
  const { navAgent, meta } = await import('./index.mjs')
  const { buildRemoteCommand } = await import('./lib/probe.mjs')
  const { ctx, tools, skills, logs } = mockContext()

  // 回归：ssh 远端命令必须写成 `cd <root> && PYTHONPATH=… python …`。
  // 少了 `&&` 时 bash 会把 `cd /x PYTHONPATH=/y python` 当成 cd 的三个参数，
  // 于是**所有工具都静默回退到缓存**（这个 bug 在 Windows→服务器的实测里抓到过）。
  const remote = buildRemoteCommand(
    { remote: { root: '/r', src: '/s', python: 'py', probe: 'p' } },
    ['status'],
  )
  check('ssh 远端命令含 `cd … && PYTHONPATH=…`', remote.includes('cd /r && PYTHONPATH=/s'), remote)

  // ---- 1. 形状 ---------------------------------------------------------- #
  navAgent.apply(ctx)
  check('index.mjs 导出 navAgent.inject = [extension, skills]',
    Array.isArray(navAgent.inject) && navAgent.inject.includes('extension') && navAgent.inject.includes('skills'),
    navAgent.inject)
  check(`注册工具数 = ${meta.tools.length}`, tools.length === meta.tools.length, `实际 ${tools.length}`)
  check('工具名与 meta 声明一致', meta.tools.every((name) => tools.some((tool) => tool.name === name)), meta.tools.filter((name) => !tools.some((t) => t.name === name)))
  check('注册了 1 个 Skill', skills.length === 1 && skills[0].name === meta.skill, skills.map((s) => s.name))
  check('Skill 有非空 body 与 description', Boolean(skills[0]?.body?.length > 500) && Boolean(skills[0]?.description), {
    bodyLength: skills[0]?.body?.length,
  })
  for (const tool of tools) {
    check(
      `[${tool.name}] 参数 Schema 带 TypeBox 标记且封闭`,
      hasKindTag(tool.parameters) && tool.parameters.additionalProperties === false,
      tool.parameters,
    )
    check(
      `[${tool.name}] meta 完整`,
      typeof tool.meta?.isReadOnly === 'boolean' &&
        typeof tool.meta?.requiresApproval === 'string' &&
        typeof tool.meta?.replay === 'string',
      tool.meta,
    )
  }
  const launch = tools.find((tool) => tool.name === 'nav_run_eval')
  check(
    'nav_run_eval 声明了审批与代价提示',
    launch?.meta?.requiresApproval === 'always' &&
      Number(launch?.meta?.costHint?.wallMs) > 0 &&
      launch?.meta?.replay === 'never',
    launch?.meta,
  )

  // ---- 2/3. 真数据 ------------------------------------------------------ #
  const byName = Object.fromEntries(tools.map((tool) => [tool.name, tool]))
  const cases = [
    ['nav_env_status', {}],
    ['nav_list_runs', { limit: 3 }],
    ['nav_run_summary', { run: RUN }],
    ['nav_observe_episode', { run: RUN, episode: EPISODE, max_decisions: 10 }],
    ['nav_failure_events', { run: RUN, limit: 3 }],
    ['nav_diagnose_triggers', { run: RUN }],
    ['nav_recovery_options', { run: RUN }],
    ['nav_compare_configs', {}],
    ['nav_run_progress', { run: RUN }],
    ['nav_tail_log', { path: '/srv/agh-demo/logs/route1/FINAL_TABLE.txt', lines: 5 }],
  ]
  const observed = {}
  for (const [name, input] of cases) {
    const tool = byName[name]
    try {
      const started = Date.now()
      const output = await tool.execute(input)
      const structured = output?.structured
      const text = output?.content?.[0]?.text || ''
      observed[name] = { ms: Date.now() - started, source: structured?.source, textLength: text.length }
      check(
        `[${name}] 返回 content + structured`,
        Boolean(text.length > 40) && Boolean(structured) && Boolean(structured.schema),
        { textLength: text.length, structuredKeys: Object.keys(structured || {}).slice(0, 8) },
      )
      check(
        `[${name}] structured 标注数据来源`,
        structured?.source === 'live' || structured?.source === 'cache',
        structured?.source,
      )
    } catch (error) {
      if (OFFLINE) {
        // 离线模式：没有对应缓存样本的工具**必须显式失败**（fail-closed），
        // 而不是给一个看起来像真的的空结果。
        check(
          `[${name}] 离线且无缓存样本时 fail-closed`,
          /数据面|超时|无法启动|命令失败|JSON/.test(String(error.message)),
          error.message,
        )
      } else {
        check(`[${name}] 执行成功`, false, error.message)
      }
    }
  }

  // 关键字段核对（不是"没抛异常"就算通过）。
  // 离线且没有对应缓存样本时工具会 fail-closed，对应核对跳过（上面已断言 fail-closed）。
  const available = (name) => Boolean(observed[name])
  try {
    if (available('nav_compare_configs')) {
      const board = await byName.nav_compare_configs.execute({})
      const configs = Object.keys(board.structured.data.configs || {})
      const scenes = board.structured.data.scenes || []
      check('nav_compare_configs 覆盖 5 个配置', configs.length === 5, configs)
      check('nav_compare_configs 覆盖 8 个场景', scenes.length === 8, scenes)
      const recoveryOnly = board.structured.data.configs.recovery_only
      check(
        '对照表里的 recovery_only 数字与 FINAL_TABLE 一致（0.6635 / std 0.0914）',
        Math.abs(Number(recoveryOnly?.pooled_success) - 0.6635) < 0.0002 &&
          Math.abs(Number(recoveryOnly?.spread?.std) - 0.0914) < 0.0002,
        { pooled: recoveryOnly?.pooled_success, std: recoveryOnly?.spread?.std },
      )
    }

    if (available('nav_diagnose_triggers')) {
      const triggers = await byName.nav_diagnose_triggers.execute({ run: RUN })
      const auroc = Number(triggers.structured.data.auroc_feedback_score_h1)
      check(`nav_diagnose_triggers 的 AUROC 落在 [0,1]（实测 ${auroc}）`, auroc >= 0 && auroc <= 1, auroc)
    }

    if (available('nav_recovery_options')) {
      const rec = await byName.nav_recovery_options.execute({ run: RUN })
      check(
        'nav_recovery_options 用的是在线同一份通道实现',
        String(rec.structured.data.channels?.implementation || '').includes('route1.nav.channels'),
        rec.structured.data.channels?.implementation,
      )
      check(
        'nav_recovery_options 给出回合级覆盖/精度',
        Number.isFinite(Number(rec.structured.data.episode_level_precision)) &&
          Number.isFinite(Number(rec.structured.data.base_rate)),
        { precision: rec.structured.data.episode_level_precision, base: rec.structured.data.base_rate },
      )
    }

    if (available('nav_failure_events')) {
      const events = await byName.nav_failure_events.execute({ run: RUN, limit: 3 })
      check(
        'nav_failure_events 产出 FailureEvent 契约字段',
        (events.structured.data.events || []).every(
          (event) => event.id && event.source === 'nav' && event.signals && event.attribution && event.recovery,
        ),
        (events.structured.data.events || []).map((event) => event.id),
      )
    }
  } catch (error) {
    check('关键字段核对未因异常中断', false, error.message)
  }

  // ---- 4. 回退路径 ------------------------------------------------------ #
  if (OFFLINE) {
    if (available('nav_compare_configs')) {
      const offline = await byName.nav_compare_configs.execute({})
      check('离线模式下走缓存回退（source=cache）', offline.structured.source === 'cache', offline.structured.source)
      check(
        '离线模式显式标注 degraded 原因',
        Boolean(offline.structured.degraded?.reason),
        offline.structured.degraded?.reason?.split('\n')[0],
      )
      check(
        '缓存回退仍能给出 5×8 对照表',
        Object.keys(offline.structured.data.configs || {}).length >= 5,
        Object.keys(offline.structured.data.configs || {}),
      )
    } else {
      check('离线模式下确实读到了缓存文件', false, '缓存里没有 board 数据（先跑 cache-all 生成 data/cache/nav-cache.json）')
    }
  }

  if (INCLUDE_LAUNCH) {
    try {
      const launched = await byName.nav_run_eval.execute({
        label: `aghselftest${Date.now().toString().slice(-5)}`,
        budget_mode: 'recovery_only',
        watchdog: 'off',
        channels: 'both',
        episodes: 13,
      })
      check(
        '[nav_run_eval] 真启动一次评估',
        Boolean(launched.structured.data.tmux_session),
        launched.structured.data.tmux_session,
      )
    } catch (error) {
      check('[nav_run_eval] 真启动一次评估', false, error.message)
    }
  } else {
    console.log('SKIP  nav_run_eval 真实启动（要跑请加 --include-launch；它会占用 GPU 约 20 分钟）')
  }

  const failed = checks.filter((item) => !item.ok)
  const summary = {
    schema: 'agh-nav/selftest/1',
    run: RUN,
    episode: EPISODE,
    offline: OFFLINE,
    tools: tools.length,
    skills: skills.length,
    checks: checks.length,
    passed: checks.length - failed.length,
    failed: failed.length,
    observed,
    plugin_logs: logs,
    failures: failed.map((item) => ({ name: item.name, evidence: item.evidence })),
  }
  if (AS_JSON) console.log(JSON.stringify(summary, null, 2))
  else {
    console.log('')
    console.log(`自检结果：${summary.passed}/${summary.checks} 通过${failed.length ? `，${failed.length} 项失败` : ''}`)
    console.log(`观测：${JSON.stringify(observed)}`)
  }
  return failed.length === 0 ? 0 : 1
}

main()
  .then((code) => process.exit(code))
  .catch((error) => {
    console.error(`自检自身崩溃：${error?.stack || error}`)
    process.exit(1)
  })

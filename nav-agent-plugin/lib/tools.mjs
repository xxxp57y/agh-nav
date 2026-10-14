/**
 * 路线① 导航侧的 AGH 工具集。
 *
 * 工具边界（有意设计，别乱扩）：
 * - **只读工具**默认 `requiresApproval: 'never'`：读日志/算指标/给建议不改任何东西。
 * - **唯一的写工具**是 `nav_run_eval`（在服务器上起一次 Habitat 闭环评估），
 *   它显式声明 `requiresApproval: 'always'` 与 `costHint`，因为一次 52 回合 ≈ 1 小时 GPU。
 * - 所有工具返回 `structured`（结构化 JSON，供面板/报告复用）+ 一段**人可读**的 text。
 */

import { describeConfig, loadConfig } from './config.mjs'
import { envelope, probe, runShell } from './probe.mjs'
import { Arr, Int, Lit, Num, Obj, Str, Union } from './schema.mjs'

const READ_ONLY_META = Object.freeze({
  isReadOnly: true,
  isDestructive: false,
  isConcurrencySafe: true,
  isOpenWorld: true,
  replay: 'safe',
  costHint: undefined,
  deferLoading: false,
  requiresApproval: 'never',
})

const LAUNCH_META = Object.freeze({
  isReadOnly: false,
  isDestructive: false,
  isConcurrencySafe: false,
  isOpenWorld: true,
  // AGH 的 replay 只能是 safe | never | idempotent：起一次评估不可重放，选 never。
  replay: 'never',
  // costHint 是结构化的 { credits?, wallMs? }，不是给人看的字符串；
  // 52 回合实测 ≈ 60–70 分钟，取上限 70 分钟。
  costHint: { wallMs: 70 * 60 * 1000 },
  deferLoading: false,
  requiresApproval: 'always',
})

const RUN_PARAM = Str({ description: 'run 标签，如 r1grd4 / r1ro4c / r2adapt_capture（也可给完整目录名）' })

function result(summary, payload) {
  return {
    content: [{ type: 'text', text: summary }],
    structured: payload,
  }
}

function pct(value) {
  if (value === null || value === undefined) return '—'
  return `${(Number(value) * 100).toFixed(1)}%`
}

function num(value, digits = 2) {
  if (value === null || value === undefined) return '—'
  return Number(value).toFixed(digits)
}

async function callProbe(argv, cacheSample, required = []) {
  const config = loadConfig()
  const live = await probe(config, argv, { cacheSample })
  const payload = envelope(live, { command: argv[0], probe_argv: argv })
  const missing = required.filter((key) => payload.data?.[key] === undefined)
  if (missing.length) {
    // 结构异常必须显式失败：绝不让"字段缺失"退化成 NaN/undefined 混进结论。
    throw new Error(
      `数据面返回结构异常（${argv[0]} 缺 ${missing.join(', ')}）：${JSON.stringify(payload.data).slice(0, 240)}`,
    )
  }
  return { config, payload }
}

function cacheNote(payload) {
  if (payload.source !== 'cache') return ''
  return `\n⚠️ 实时数据面不可用，以下为**缓存快照**（生成于 ${payload.cache_generated_at}）：${
    payload.degraded?.reason?.split('\n')[0] || ''
  }`
}

/** 从 cache-all 快照里挑样本；挑不到就返回 null（让上层抛真实错误）。 */
function sample(picker) {
  return picker
}

function pickSample(payload, key, match = () => true) {
  const samples = payload?.samples || {}
  const value = samples[key]
  if (!value) return null
  return match(value) ? value : null
}

// --------------------------------------------------------------------------- #
// 工具定义
// --------------------------------------------------------------------------- #

export function createTools() {
  return [
    {
      name: 'nav_env_status',
      description:
        '查看导航服务器（Habitat/InternNav 闭环）的当前状态：GPU 占用、tmux 会话、正在跑的评估、磁盘余量、conda 环境与 route1 代码是否可导入。做任何导航决策前先看这个。',
      parameters: Obj({}),
      meta: READ_ONLY_META,
      async execute() {
        const { payload } = await callProbe(['status'], (cached) => cached.status, ['server', 'disk'])
        const d = payload.data
        const gpu = (d.gpu || []).map((line) => `  - ${line}`).join('\n')
        const summary = [
          `导航服务器状态（${payload.source === 'live' ? '实时' : '缓存'}）：`,
          `- 主机：${d.server.host}（route1 可导入：${d.route1_importable ? '是' : '否'}）`,
          `- 正在跑的评估：${(d.eval_running || []).length ? d.eval_running.join('; ') : '无'}`,
          `- tmux：${(d.tmux_sessions || []).join(', ') || '无'}`,
          `- 磁盘 /mnt/sda：剩余 ${d.disk.free_gb} GB（已用 ${d.disk.used_pct}%）`,
          `- 可用 run 数：${d.runs_available}`,
          gpu ? `- GPU：\n${gpu}` : '- GPU：不可读',
          cacheNote(payload),
        ].join('\n')
        return result(summary, payload)
      },
    },

    {
      name: 'nav_list_runs',
      description:
        '列出服务器上所有导航评估 run（标签、模式、回合数、池化 Success/SPL、最差场景、碰撞与 System-2 代价）。想比较"哪套恢复配置更好"时先看这个。',
      parameters: Obj({
        limit: Int({ minimum: 1, maximum: 60, description: '最多返回多少个 run（默认 10，按时间倒序）' }),
      }),
      meta: READ_ONLY_META,
      async execute(args = {}) {
        const limit = Number(args.limit || 10)
        const { payload } = await callProbe(['runs'], (cached) => cached.runs, ['runs'])
        const all = payload.data.runs || []
        const rows = all.slice(-limit).reverse()
        payload.data = { ...payload.data, shown: rows.length, total: all.length, runs: rows }
        const table = rows
          .map(
            (r) =>
              `| ${r.label} | ${r.mode} | ${r.episodes}/${r.expected_episodes} | ${r.n_scenes} | ${pct(
                r.pooled_success,
              )} | ${pct(r.worst_scene_success)} | ${num(r.collision_per_ep)} | ${num(r.system2_per_ep)} |`,
          )
          .join('\n')
        const summary = [
          `服务器上共 ${all.length} 个 run（下表显示最近 ${rows.length} 个；8 场景口径请用 nav_compare_configs）：`,
          '| run | mode | 回合 | 场景 | 池化 Success | 最差场景 | 碰撞/回合 | S2/回合 |',
          '| --- | --- | --- | --- | --- | --- | --- | --- |',
          table,
          cacheNote(payload),
        ].join('\n')
        return result(summary, payload)
      },
    },

    {
      name: 'nav_run_summary',
      description:
        '单个 run 的逐场景指标：Success/SPL/NE/碰撞/步数/System-2、终止方式分布、重规划原因直方图、无动作事件与官方失败标签覆盖率。',
      parameters: Obj({ run: RUN_PARAM }, ['run']),
      meta: READ_ONLY_META,
      async execute(args) {
        const { payload } = await callProbe(
          ['run-summary', '--run', args.run],
          (cached) =>
            pickSample(cached, 'run-summary', (value) => (value.summary ?? value).label === args.run),
          ['summary'],
        )
        const s = payload.data.summary ?? payload.data
        if (!s || !Array.isArray(s.per_scene)) {
          throw new Error(`run-summary 返回结构异常（缺 per_scene）：${JSON.stringify(payload.data).slice(0, 200)}`)
        }
        const rows = (s.per_scene || [])
          .map(
            (sc) =>
              `| ${sc.scene_id} | ${sc.episodes} | ${pct(sc.success)} | ${num(sc.spl, 3)} | ${num(sc.ne, 2)} | ${num(
                sc.collision,
              )} | ${num(sc.steps, 1)} | ${num(sc.system2_calls)} | ${JSON.stringify(sc.terminations)} |`,
          )
          .join('\n')
        const summary = [
          `run ${s.label}（mode=${s.mode}，${s.episodes}/${s.expected_episodes} 回合）：池化 Success ${pct(
            s.pooled_success,
          )}、SPL ${num(s.pooled_spl, 4)}`,
          `场景间 Success：min ${pct(s.cross_scene_success_spread?.min)} / max ${pct(
            s.cross_scene_success_spread?.max,
          )} / std ${num(s.cross_scene_success_spread?.std, 4)} / range ${num(
            s.cross_scene_success_spread?.range,
            4,
          )}`,
          `代价：碰撞 ${num(s.cost.collision_per_ep)}/回合、步数 ${num(s.cost.steps_per_ep)}、System-2 ${num(
            s.cost.system2_per_ep,
          )}、重规划 ${num(s.cost.replans_per_ep)}`,
          `重规划原因：${JSON.stringify(s.triggers.replan_reasons)}`,
          `无动作事件 ${s.triggers.no_action_events}，其中官方标签口径 ${s.triggers.official_label_events}（覆盖 ${pct(
            s.triggers.official_label_coverage,
          )}）`,
          '| scene | 回合 | Success | SPL | NE | 碰撞 | 步数 | S2 | 终止方式 |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
          rows,
          cacheNote(payload),
        ].join('\n')
        return result(summary, payload)
      },
    },

    {
      name: 'nav_observe_episode',
      description:
        '环境反馈：取某个回合的**逐决策观测序列**（距目标距离、碰撞计数、plan_age、轨迹漂移、几何分、是否重规划、原因、是否有可执行动作）。这是"看智能体到底经历了什么"的工具。',
      parameters: Obj(
        {
          run: RUN_PARAM,
          episode: Int({ minimum: 0, description: 'episode_id（数字，如 220）' }),
          max_decisions: Int({ minimum: 5, maximum: 400, description: '最多返回多少个决策点（默认 120）' }),
        },
        ['run', 'episode'],
      ),
      meta: READ_ONLY_META,
      async execute(args) {
        const { payload } = await callProbe(
          ['series', '--run', args.run, '--episode', String(args.episode)],
          (cached) =>
            pickSample(
              cached,
              'series',
              (value) => value.run === args.run && Number(value.episode) === Number(args.episode),
            ),
          ['decisions'],
        )
        const max = Number(args.max_decisions || 120)
        if (!Array.isArray(payload.data.decisions)) {
          throw new Error(`series 返回结构异常（缺 decisions）：${JSON.stringify(payload.data).slice(0, 200)}`)
        }
        const all = payload.data.decisions || []
        const shown = all.slice(0, max)
        payload.data = { ...payload.data, decisions_total: all.length, decisions: shown }
        const meta = payload.data.episode_meta || {}
        const rows = shown
          .map(
            (d) =>
              `| ${d.decision_index} | ${d.step} | ${num(d.distance_to_goal)} | ${d.collision_count} | ${d.plan_age} | ${num(
                d.trajectory_drift,
                3,
              )} | ${num(d.feedback_score, 3)} | ${d.should_replan ? '是' : ''} | ${d.replan_reason || ''} | ${
                d.has_executable_action === false ? '否' : '是'
              } |`,
          )
          .join('\n')
        const summary = [
          `run ${payload.data.run} / episode ${payload.data.episode}（场景 ${meta.scene_id}，${
            meta.termination || '?'
          }，success=${meta.success}）`,
          meta.instruction ? `指令：${String(meta.instruction).trim()}` : '',
          `该回合：步数 ${meta.steps}、碰撞 ${meta.collision_count}、重规划 ${meta.replans}、System-2 ${
            meta.system2_calls
          }、无动作 ${meta.no_action_count}、NE ${num(meta.ne, 2)}`,
          `重规划原因：${JSON.stringify(meta.replan_reasons)}`,
          `（共 ${all.length} 个决策点，下表显示前 ${shown.length} 个）`,
          '| # | step | 距目标 | 碰撞 | plan_age | 漂移 | 几何分 | 重规划 | 原因 | 有可执行动作 |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
          rows,
          cacheNote(payload),
        ]
          .filter(Boolean)
          .join('\n')
        return result(summary, payload)
      },
    },

    {
      name: 'nav_failure_events',
      description:
        '把日志转成**统一 FailureEvent 契约**的事件流（route1.contracts：type N1/N2/N4/N5、signals、REFLECT 式归因、恢复动作、结果）。跨仿真器/跨模块都用这一份契约。',
      parameters: Obj(
        {
          run: RUN_PARAM,
          limit: Int({ minimum: 1, maximum: 200, description: '最多返回多少条事件（默认 20）' }),
          episode: Int({ minimum: 0, description: '只看某个 episode（可选）' }),
        },
        ['run'],
      ),
      meta: READ_ONLY_META,
      async execute(args) {
        const limit = Number(args.limit || 20)
        const argv = ['failures', '--run', args.run, '--limit', String(limit)]
        if (args.episode !== undefined && args.episode !== null) {
          argv.push('--episode', String(args.episode))
        }
        const { payload } = await callProbe(
          argv,
          (cached) => pickSample(cached, 'failures', (value) => value.run === args.run),
          ['events'],
        )
        const d = payload.data
        const head = (d.events || []).slice(0, 12)
        const rows = head
          .map(
            (e) =>
              `| ${e.type} | ${e.scene_id} | ${e.episode_id} | ${e.step} | ${e.attribution.summary_l1} | ${
                e.recovery.action
              } | ${JSON.stringify(e.attribution.evidence)} |`,
          )
          .join('\n')
        const summary = [
          `run ${d.run} 的失败事件（共 ${d.count} 条，类型分布 ${JSON.stringify(d.type_histogram)}，恢复动作 ${
            JSON.stringify(d.recovery_histogram)
          }）`,
          '| 类型 | 场景 | 回合 | 步 | 归因 | 恢复动作 | 证据 |',
          '| --- | --- | --- | --- | --- | --- | --- |',
          rows,
          cacheNote(payload),
        ].join('\n')
        return result(summary, payload)
      },
    },

    {
      name: 'nav_diagnose_triggers',
      description:
        '触发质量诊断：现有几何判据的 P@1/P@3/P@5（两种标签口径都算：any_replan 与官方 model_invoked 口径）、几何分对真失败的一步 AUROC、以及基率。用来回答"现在的重规划触发到底有没有信息量"。',
      parameters: Obj({ run: RUN_PARAM }, ['run']),
      meta: READ_ONLY_META,
      async execute(args) {
        const { payload } = await callProbe(
          ['triggers', '--run', args.run],
          (cached) => pickSample(cached, 'triggers', (value) => value.run === args.run),
          ['decisions', 'any_replan_p@1'],
        )
        const d = payload.data
        const summary = [
          `run ${d.run}（${d.decisions} 个决策点）触发质量：`,
          `- any_replan 口径：P@1 ${num(d['any_replan_p@1']?.p_at_k, 4)}（${d['any_replan_p@1']?.triggers} 次触发）｜P@3 ${num(
            d['any_replan_p@3']?.p_at_k,
            4,
          )}｜P@5 ${num(d['any_replan_p@5']?.p_at_k, 4)}`,
          `- 官方标签口径（model_invoked=True 且无可执行动作）：P@1 ${num(
            d['official_label_p@1']?.p_at_k,
            4,
          )}｜P@3 ${num(d['official_label_p@3']?.p_at_k, 4)}｜P@5 ${num(d['official_label_p@5']?.p_at_k, 4)}`,
          `- 几何分对"下一步真失败"的 AUROC：${num(d.auroc_feedback_score_h1, 4)}（n=${d.auroc_n}，基率 ${num(
            d.base_rate_h1_any,
            4,
          )}）`,
          `- 官方失败标签基率：${num(d.base_rate_official, 4)}`,
          `口径说明：${d.note}`,
          cacheNote(payload),
        ].join('\n')
        return result(summary, payload)
      },
    },

    {
      name: 'nav_recovery_options',
      description:
        '恢复决策：用**在线那两条通道的同一份实现**（碰撞密度 + 进度停滞，route1.nav.channels）离线复算它们会在哪些决策点要求重规划，并给出回合级覆盖/精度与提前量。这是"导航失败后自主调整"的核心工具。',
      parameters: Obj(
        {
          run: RUN_PARAM,
          channels: Union([Lit('both'), Lit('collision'), Lit('progress'), Lit('none')]),
          episode: Int({ minimum: 0, description: '只算某个 episode（可选）' }),
        },
        ['run'],
      ),
      meta: READ_ONLY_META,
      async execute(args) {
        const argv = ['recovery', '--run', args.run, '--channels', args.channels || 'both']
        if (args.episode !== undefined && args.episode !== null) {
          argv.push('--episode', String(args.episode))
        }
        const { payload } = await callProbe(
          argv,
          (cached) => pickSample(cached, 'recovery', (value) => value.run === args.run),
          ['fires_total', 'channels'],
        )
        const d = payload.data
        const detail = (d.detail || []).filter((e) => e.n_fires > 0).slice(0, 10)
        payload.data = { ...d, detail: (d.detail || []).slice(0, 30) }
        const rows = detail
          .map(
            (e) =>
              `| ${e.scene_id} | ${e.episode_id} | ${e.success ? '成功' : '失败'} | ${e.steps} | ${e.n_fires} | ${(
                e.fires || []
              )
                .map((f) => `step${f.step}:${f.reason.replace('_replan', '')}`)
                .join(', ')} |`,
          )
          .join('\n')
        const summary = [
          `run ${d.run} 的恢复通道复算（实现：${d.channels.implementation}）`,
          `- 参数：碰撞窗口 ${d.channels.collision_window_actions} 动作内 ≥${d.channels.collision_min_in_window} 次；进度窗口 ${
            d.channels.progress_window_actions
          } 动作内改善 <${d.channels.progress_min_improvement_m} m；每回合每通道封顶 ${d.channels.max_per_episode} 次`,
          `- 触发 ${d.fires_total} 次（${num(d.fires_per_episode)} 次/回合），分布在 ${d.episodes_with_fire}/${
            d.episodes
          } 个回合；原因分布 ${JSON.stringify(d.reason_histogram)}`,
          `- 回合级：失败回合 ${d.failed_episodes} 个，其中被通道覆盖 ${d.failed_episodes_covered} 个；精度 ${num(
            d.episode_level_precision,
            4,
          )}（基率 ${num(d.base_rate, 4)}）`,
          detail.length ? '触发了通道的回合（样例）：' : '（没有回合触发通道）',
          detail.length ? '| 场景 | 回合 | 结果 | 步数 | 触发次数 | 触发点 |' : '',
          detail.length ? '| --- | --- | --- | --- | --- | --- |' : '',
          rows,
          cacheNote(payload),
        ]
          .filter(Boolean)
          .join('\n')
        return result(summary, payload)
      },
    },

    {
      name: 'nav_compare_configs',
      description:
        '五套恢复配置 × 8 场景的最终对照表（每格 13 回合、每配置 104 回合）：R2A / original / recovery_only / guarded / collision_only，含池化 Success、SPL、场景间 min/std/range。回答"到底该用哪套恢复策略"。',
      parameters: Obj({
        configs: Arr(Str(), { description: '只显示这些配置（默认全部五项）' }),
      }),
      meta: READ_ONLY_META,
      async execute(args = {}) {
        const { payload } = await callProbe(['board'], (cached) => cached.board, ['configs', 'scenes'])
        const d = payload.data
        const wanted = (args.configs && args.configs.length ? args.configs : Object.keys(d.configs)).filter(
          (name) => d.configs[name],
        )
        if (args.configs && args.configs.length && wanted.length === 0) {
          throw new Error(`未知配置：${args.configs.join(', ')}；可用：${Object.keys(d.configs).join(', ')}`)
        }
        const header = `| 场景 | 回合 | ${wanted.join(' | ')} |`
        const divider = `| --- | --- | ${wanted.map(() => '---').join(' | ')} |`
        const rows = d.scenes
          .map((scene) => {
            const cells = wanted.map((name) => pct(d.configs[name]?.per_scene?.[scene]?.success))
            const episodes = d.configs[wanted[0]]?.per_scene?.[scene]?.episodes ?? 13
            return `| ${scene} | ${episodes} | ${cells.join(' | ')} |`
          })
          .join('\n')
        const pooled = `| **池化 Success（104）** | - | ${wanted
          .map((name) => pct(d.configs[name]?.pooled_success))
          .join(' | ')} |`
        const spl = `| **池化 SPL** | - | ${wanted.map((name) => num(d.configs[name]?.pooled_spl, 4)).join(' | ')} |`
        const stat = (key, label) =>
          `| **${label}** | - | ${wanted
            .map((name) => num(d.configs[name]?.spread?.[key], 4))
            .join(' | ')} |`
        const cost = (key, label, digits = 2) =>
          `| ${label} | - | ${wanted
            .map((name) => {
              const scenes = Object.values(d.configs[name]?.per_scene || {})
              if (!scenes.length) return '—'
              const total = scenes.reduce((acc, s) => acc + Number(s.episodes || 0), 0)
              const weighted =
                scenes.reduce((acc, s) => acc + Number(s[key] || 0) * Number(s.episodes || 0), 0) / total
              return num(weighted, digits)
            })
            .join(' | ')} |`
        const summary = [
          `路线① 导航侧最终对照（${d.evaluation_protocol.episodes_per_config} 回合/配置，${d.evaluation_protocol.manifest}）：`,
          header,
          divider,
          rows,
          pooled,
          spl,
          stat('min', '最差场景'),
          stat('std', '场景间 std'),
          stat('range', '场景间 range'),
          cost('collision', '代价：碰撞/回合'),
          cost('system2_calls', '代价：S2/回合'),
          '',
          '结论口径：`recovery_only` 最稳（std 0.0914 / range 0.2308 / 最差场景 0.5385）；`guarded` 均值与碰撞代价最好（0.7115 / 8.18 次）；原 R2A 最差场景只有 0.1538。',
          cacheNote(payload),
        ].join('\n')
        return result(summary, payload)
      },
    },

    {
      name: 'nav_run_progress',
      description: '某个正在跑的评估到哪了：完成回合数、逐场景计数、当前池化 Success、tmux 会话与日志尾部。',
      parameters: Obj({ run: RUN_PARAM }, ['run']),
      meta: READ_ONLY_META,
      async execute(args) {
        const { payload } = await callProbe(['progress', '--run', args.run], undefined, ['episodes'])
        const d = payload.data
        const summary = [
          `run ${d.run}：${d.episodes}/${d.expected_episodes} 回合（${d.complete ? '已跑完' : '进行中'}），当前 Success ${pct(
            d.success_so_far,
          )}`,
          `逐场景：${JSON.stringify(d.scenes)}`,
          `tmux：${(d.tmux_sessions || []).join(', ') || '无'}`,
          d.log_tail && d.log_tail.length ? `日志尾部：\n${d.log_tail.slice(-6).join('\n')}` : '',
        ]
          .filter(Boolean)
          .join('\n')
        return result(summary, payload)
      },
    },

    {
      name: 'nav_run_eval',
      description:
        '在导航服务器上启动一次新的 Habitat/InternNav 闭环评估（tmux 后台）。参数决定"恢复策略"：budget_mode 选恢复策略族，watchdog 决定碰撞看门狗是判死还是关掉，channels 决定两条 out-of-tree 通道是否参与触发。**需要人工审批**，代价约 70 分钟 GPU（52 回合），会真实改动服务器状态。',
      parameters: Obj(
        {
          label: Str({
            pattern: '^[A-Za-z0-9_]{2,32}$',
            description: '本次 run 的标签（字母数字下划线），产物落在 logs/habitat/route1_<label>_seed<SEED>',
          }),
          budget_mode: Union([Lit('recovery_only'), Lit('original'), Lit('drift_recovery_periodic')]),
          watchdog: Union([Lit('off'), Lit('on')]),
          channels: Union([Lit('both'), Lit('collision'), Lit('progress'), Lit('none')]),
          episodes: Int({ minimum: 13, maximum: 104, description: '回合数（多场景清单默认 52）' }),
        },
        ['label'],
      ),
      meta: LAUNCH_META,
      async execute(args) {
        const argv = [
          'launch',
          '--label',
          args.label,
          '--budget-mode',
          args.budget_mode || 'recovery_only',
          '--watchdog',
          args.watchdog || 'off',
          '--channels',
          args.channels || 'both',
          '--episodes',
          String(args.episodes || 52),
        ]
        const { payload } = await callProbe(argv)
        const d = payload.data
        const summary = [
          `已在 tmux 会话 ${d.tmux_session} 启动评估 ${d.label}（${JSON.stringify(d.env)}）`,
          `- 产物：${d.run_dir}`,
          `- 日志：${d.log}`,
          `- 预计耗时：约 ${d.expected_minutes} 分钟`,
          `- 参考值（8 场景 104 回合口径）：${JSON.stringify(d.expected_result.reference_pooled_success)}`,
          `下一步：${(d.next_steps || []).join('；')}`,
        ].join('\n')
        return result(summary, payload)
      },
    },

    {
      name: 'nav_tail_log',
      description: '读取服务器上某个导航日志/产物文件的尾部（只允许 logs/route1、out/route1、InternNav/logs/habitat 下的文本文件）。',
      parameters: Obj(
        {
          path: Str({ description: '服务器上的绝对路径' }),
          lines: Int({ minimum: 1, maximum: 400, description: '读取尾部多少行（默认 40）' }),
        },
        ['path'],
      ),
      meta: READ_ONLY_META,
      async execute(args) {
        const allowed = [
          '/srv/agh-demo/logs/route1/',
          '/srv/agh-demo/out/route1/',
          '/srv/agh-demo/InternNav/logs/habitat/',
          '/srv/agh-demo/agh-nav/',
        ]
        const path = String(args.path || '')
        if (!allowed.some((prefix) => path.startsWith(prefix))) {
          throw new Error(`路径不在白名单内（只允许：${allowed.join(' , ')}）`)
        }
        if (!/^[A-Za-z0-9._/@+-]*$/.test(path)) {
          throw new Error('路径包含不允许的字符')
        }
        const lines = Number(args.lines || 40)
        const config = loadConfig()
        const { stdout, ms } = await runShell(config, `tail -n ${lines} ${path}`)
        const payload = envelope(
          { source: 'live', transport: config.exec, ms, data: { path, lines } },
          { command: 'tail' },
        )
        return result(`${path} 的尾部 ${lines} 行：\n${stdout}\n\n（传输：${describeConfig(config)}）`, {
          ...payload,
          stdout,
        })
      },
    },
  ]
}

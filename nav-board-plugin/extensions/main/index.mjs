/**
 * `@agh-nav/nav-board-plugin` 的后端行。
 *
 * 只注册**一个查询服务** `nav.board`（kind: query，只读、无副作用），
 * 浏览器侧的 `extensions/main/client/index.js` 通过 AGH 的同源 BFF 调用它：
 *
 *   浏览器 → 当前 client row 的 service allow-list → BFF → daemon 校验
 *          → Host 行服务授权 → 本 handler
 *
 * 注意：这与 `nav-agent-plugin` 的工具是两条独立能力线——看板插件被禁用时，
 * 工具仍然可用（反之亦然）。数据面（服务器上的 agh_nav_probe.py）是共享的。
 */

import { loadBoard } from '../../lib/remote.mjs'

export const navBoard = {
  apply(ctx) {
    ctx.services.register({
      name: 'nav.board',
      kind: 'query',
      inputSchema: {
        type: 'object',
        properties: { force: { type: 'boolean' } },
        additionalProperties: false,
      },
      outputSchema: {
        type: 'object',
        properties: {
          source: { type: 'string' },
          generated_at: { type: ['string', 'null'] },
          degraded: { type: ['object', 'null'] },
          payload: { type: 'object' },
        },
        required: ['source', 'payload'],
        additionalProperties: false,
      },
      timeoutMs: 20000,
      maxResultBytes: 524288,
      async handler(input) {
        const board = await loadBoard({ force: Boolean(input?.force) })
        // 只把面板真正要用的字段送过桥，避免浏览器拿到整份日志派生物。
        const payload = board.payload || {}
        const configs = {}
        for (const [name, value] of Object.entries(payload.configs || {})) {
          if (!value) continue
          configs[name] = {
            pooled_success: value.pooled_success,
            pooled_spl: value.pooled_spl,
            spread: value.spread,
            episodes_total: value.episodes_total,
            per_scene: value.per_scene,
          }
        }
        return {
          source: board.source,
          generated_at: board.generated_at,
          degraded: board.degraded || null,
          payload: {
            schema: payload.schema,
            command: payload.command,
            generated_at_iso: payload.generated_at_iso,
            evaluation_protocol: payload.evaluation_protocol,
            scenes: payload.scenes || [],
            configs,
            recent_runs: payload.recent_runs || [],
          },
        }
      },
    })
  },
}

export default navBoard

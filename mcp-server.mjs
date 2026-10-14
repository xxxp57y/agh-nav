#!/usr/bin/env node
/**
 * navlive MCP 服务端（stdio）—— 让 AGH 用 MCP 拿到实时导航工具。
 *
 * 为什么还要有 MCP 这一条路
 * ------------------------
 * 把工具做成 AGH 插件包（`@agh-nav/nav-agent-plugin`）是最"正统"的接法，但它要走
 * 安装 → 信任 → 启用 → 客户端模块物化这条链；而这套 AGH 是 pre-alpha，**服务在跑时
 * 包操作会被状态机拒掉**（`E_PACKAGE_STATE`），于是每次改工具都要停一次工作台。
 * MCP 是 AGH 已经支持的另一条公开接法：外部进程即插即用，不碰包状态，改完重启进程即可。
 *
 * 协议：MCP over stdio（换行分隔的 JSON-RPC 2.0）。只实现 `initialize`、
 * `tools/list`、`tools/call` 与 `notifications/initialized` —— 客户端要的刚好是这些。
 *
 * 用法：
 *   NAVLIVE_URL=http://127.0.0.1:43121 node mcp-server.mjs
 *
 * 在 AGH 里注册（CLI）：
 *   agnes mcp add navlive --stdio node --arg <本文件绝对路径>
 *   agnes mcp get navlive            # 取 revision
 *   agnes mcp trust navlive --expected-revision <rev>
 *   agnes mcp enable navlive
 *   agnes mcp test navlive
 *   agnes mcp tools navlive
 */

import { createNavLiveTools } from './nav-agent-plugin/lib/navlive-tools.mjs'
import { serviceUrl } from './nav-agent-plugin/lib/navlive.mjs'

const PROTOCOL_VERSION = '2025-06-18'
const SERVER_INFO = { name: 'agh-navlive', version: '0.1.0' }

const TOOLS = createNavLiveTools()
const BY_NAME = new Map(TOOLS.map((tool) => [tool.name, tool]))

/** TypeBox 形状 → 普通 JSON Schema（MCP 只吃普通 Schema）。 */
function plainSchema(node) {
  if (node === null || typeof node !== 'object') return node
  if (Array.isArray(node)) return node.map(plainSchema)
  const out = {}
  for (const [key, value] of Object.entries(node)) {
    if (key === 'Symbol(TypeBox.Kind)' || key.startsWith('Symbol(')) continue
    out[key] = plainSchema(value)
  }
  // TypeBox 的 kind 标记在 Symbol 上，普通 JSON 里已经没有了；type 字段本来就带
  return out
}

function toolList() {
  return TOOLS.map((tool) => ({
    name: tool.name,
    description: tool.description,
    inputSchema: plainSchema(tool.parameters),
    annotations: {
      readOnlyHint: tool.meta?.isReadOnly === true,
      destructiveHint: tool.meta?.isDestructive === true,
      idempotentHint: tool.meta?.replay === 'idempotent',
      openWorldHint: tool.meta?.isOpenWorld === true,
    },
  }))
}

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`)
}

function reply(id, result) {
  send({ jsonrpc: '2.0', id, result })
}

function fail(id, code, message, data) {
  send({ jsonrpc: '2.0', id, error: { code, message, ...(data ? { data } : {}) } })
}

async function handle(message) {
  const { id, method, params } = message
  if (method === 'initialize') {
    reply(id, {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: { tools: { listChanged: false } },
      serverInfo: SERVER_INFO,
      instructions:
        `实时导航服务：${serviceUrl()}。用 navlive_scenes 看有哪些场景，` +
        'navlive_open 打开一个场景，navlive_goto 让机器人真的走过去（会动的调用需审批），' +
        'navlive_state 汇报它现在在哪。',
    })
    return
  }
  if (method === 'notifications/initialized' || method === 'initialized') return
  if (method === 'ping') {
    reply(id, {})
    return
  }
  if (method === 'tools/list') {
    reply(id, { tools: toolList() })
    return
  }
  if (method === 'tools/call') {
    const name = params?.name
    const tool = BY_NAME.get(name)
    if (!tool) {
      fail(id, -32602, `未知工具 ${name}；可用：${[...BY_NAME.keys()].join(', ')}`)
      return
    }
    try {
      const result = await tool.execute(params?.arguments || {})
      const content = (result.content || []).map((block) =>
        block.type === 'text'
          ? { type: 'text', text: block.text }
          : { type: 'text', text: JSON.stringify(block) },
      )
      const structured = result.structured
      reply(id, {
        content: content.length ? content : [{ type: 'text', text: '(无输出)' }],
        ...(structured
          ? { structuredContent: structured, content: [...content, { type: 'text', text: '```json\n' + JSON.stringify(structured, null, 2).slice(0, 20000) + '\n```' }] }
          : {}),
        isError: false,
      })
    } catch (error) {
      reply(id, {
        content: [{ type: 'text', text: `${name} 失败：${error?.message || error}` }],
        isError: true,
      })
    }
    return
  }
  if (typeof id !== 'undefined') fail(id, -32601, `未实现的方法 ${method}`)
}

let buffer = ''
//: 正在处理的请求数：stdin 结束时不能立刻退出，否则会把还没回完的 tools/call 掐掉
//: （实测症状：tools/list 正常，tools/call 一个字都没有）。
let pending = 0
let stdinEnded = false

function maybeExit() {
  if (stdinEnded && pending === 0) process.exit(0)
}

process.stdin.setEncoding('utf8')
process.stdin.on('data', (chunk) => {
  buffer += chunk
  let index = buffer.indexOf('\n')
  while (index >= 0) {
    const line = buffer.slice(0, index).trim()
    buffer = buffer.slice(index + 1)
    index = buffer.indexOf('\n')
    if (!line) continue
    let message
    try {
      message = JSON.parse(line)
    } catch {
      process.stderr.write(`[navlive-mcp] 无法解析的一行：${line.slice(0, 200)}\n`)
      continue
    }
    pending += 1
    Promise.resolve(handle(message))
      .catch((error) => {
        process.stderr.write(`[navlive-mcp] 处理出错：${error?.stack || error}\n`)
        if (typeof message?.id !== 'undefined') fail(message.id, -32603, String(error?.message || error))
      })
      .finally(() => {
        pending -= 1
        maybeExit()
      })
  }
})
process.stdin.on('end', () => {
  stdinEnded = true
  maybeExit()
})
process.stderr.write(`[navlive-mcp] 就绪，工具 ${TOOLS.length} 个，服务 ${serviceUrl()}\n`)

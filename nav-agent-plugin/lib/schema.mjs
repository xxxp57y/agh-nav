/**
 * 极小 TypeBox 形状助手。
 *
 * AGH 的工具参数要求**带 TypeBox kind 标记**的 Schema（见 docs/develop/backend.zh-CN.md）。
 * 这里手写 `Symbol.for('TypeBox.Kind')` 标记，避免插件对 `@sinclair/typebox` 产生
 * 构建期依赖——插件包因此可以是一个"零依赖 ESM 快照"，装进 AGH 不需要打包。
 *
 * 用法与 `Type.Object({...})` 等价：`Obj({ run: Str({ description: '...' }) }, ['run'])`。
 */

const Kind = Symbol.for('TypeBox.Kind')

export const Obj = (properties, required = []) => ({
  [Kind]: 'Object',
  type: 'object',
  properties,
  required,
  additionalProperties: false,
})

export const Str = (extra = {}) => ({ [Kind]: 'String', type: 'string', ...extra })
export const Num = (extra = {}) => ({ [Kind]: 'Number', type: 'number', ...extra })
export const Int = (extra = {}) => ({ [Kind]: 'Integer', type: 'integer', ...extra })
export const Bool = (extra = {}) => ({ [Kind]: 'Boolean', type: 'boolean', ...extra })
export const Arr = (items, extra = {}) => ({ [Kind]: 'Array', type: 'array', items, ...extra })
export const Lit = (value) => ({
  [Kind]: 'Literal',
  const: value,
  type: typeof value === 'number' ? 'number' : typeof value === 'boolean' ? 'boolean' : 'string',
})
export const Union = (anyOf) => ({ [Kind]: 'Union', anyOf })

export { Kind }

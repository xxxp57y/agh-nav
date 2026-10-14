/**
 * 打印已安装包在 AGH 里**真实计算出来的** capabilityHash（trust 需要它）。
 * 用的是 PackageManager 自己的 capabilityHash()，不是自己拍的哈希。
 *
 * 用法：corepack pnpm exec tsx agh-capability-hash.mts <lockfile> <packageId>
 */
import { readFileSync } from 'node:fs'
// 放在 <repo>/agh-nav/tools/ 下，所以仓库根是上两级
import { capabilityHash } from '../../packages/package-manager/src/integrity.js'

const [lockPath, packageId] = process.argv.slice(2)
if (!lockPath || !packageId) {
  console.error('usage: tsx agh-nav/tools/agh-capability-hash.mts <lockfile> <packageId>')
  process.exit(2)
}
const lock = JSON.parse(readFileSync(lockPath, 'utf8'))
const entry = lock.packages?.[packageId]
if (!entry) {
  console.error(`lock 里没有 ${packageId}`)
  process.exit(1)
}
console.log(
  JSON.stringify(
    {
      id: packageId,
      integrity: entry.integrity,
      capabilityHash: capabilityHash(entry),
      contributions: entry.contributions ?? [],
      dependencies: entry.dependencies ?? {},
    },
    null,
    2,
  ),
)

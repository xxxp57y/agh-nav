/**
 * 诊断脚本：直接调 PackageManager.trustWorkspace，把被 daemon 吞掉的真实错误码打出来。
 * 用法：corepack pnpm exec tsx agh-diag-trust.mts <profileDir> <deployDir>
 */
import { createPackageManager } from './packages/package-manager/src/index.js'

const [profileDir, deployDir] = process.argv.slice(2)
if (!profileDir || !deployDir) {
  console.error('usage: tsx agh-diag-trust.mts <profileDir> <deployDir>')
  process.exit(2)
}
const manager = createPackageManager({
  dataDir: `${profileDir}/../data`,
  cwd: process.cwd(),
  agnesVersion: '0.1.0',
})
try {
  const result = await manager.trustWorkspace(profileDir, deployDir)
  console.log('OK', JSON.stringify(result))
} catch (error) {
  console.log('FAILED')
  console.log('name:', (error as Error).name)
  console.log('message:', (error as Error).message)
  console.log('code:', (error as { code?: string }).code)
  console.log('detail:', JSON.stringify((error as { detail?: unknown }).detail, null, 2))
  process.exit(1)
}

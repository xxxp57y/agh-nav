/** Uses AGH's own configuration service and package integrity function. Never prints credentials. */
import { readFileSync, writeFileSync, mkdirSync, cpSync, chmodSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { spawnSync } from 'node:child_process'

const base = process.env.AGH_MHS_BASE
if (!base) throw new Error('Set AGH_MHS_BASE to your isolated deployment directory')
const runtime = process.env.AGH_MHS_RUNTIME ?? join(base, 'agh-mhs-runtime')
const home = process.env.AGH_MHS_HOME ?? join(base, 'agh-mhs-home')
const profile = join(home, 'profiles/local-dev')
const work = process.env.AGH_MHS_WORKSPACE ?? join(base, 'agh-mhs-workspace')
const cli = join(runtime, 'packages/cli/dist/local/agnes.mjs')
const env = { ...process.env, AGH_HOME: home, AGNES_PROFILE: 'local-dev',
  AGNES_HUB_LISTEN: '127.0.0.1:43132', AGNES_WEB_ORIGIN: 'http://127.0.0.1:43130' }
const run = (args: string[], tty = false) => {
  // Only constant package commands use the PTY. No prompt or secret is passed through a shell.
  const result = tty
    ? spawnSync('script', ['-qec', `${process.execPath} ${cli} ${args.join(' ')}`, '/dev/null'],
        { env, cwd: work, input: 'y\n', encoding: 'utf8' })
    : spawnSync(process.execPath, [cli, ...args], { env, cwd: work, encoding: 'utf8' })
  if (result.status !== 0) throw new Error(`AGH ${args[0]} failed: ${result.stderr?.slice(-1500)} ${result.stdout?.slice(-1500)}`)
  if (result.stdout) console.log(result.stdout.trim())
}
mkdirSync(home, { recursive: true, mode: 0o700 })
chmodSync(home, 0o700)
mkdirSync(profile, { recursive: true })
mkdirSync(work, { recursive: true })
const { createConfigurationService } = await import(pathToFileURL(join(runtime, 'packages/host/src/configuration.ts')).href)
const config = createConfigurationService({ home, profile: 'local-dev' })
if (!(await config.get()).configured) {
  const key = readFileSync(process.env.AGH_MHS_KEY_FILE ?? join(base, 'agh-nav/agent/.llm-key'), 'utf8').trim()
  await config.save({ providerId: 'agnes-ai', baseUrl: 'https://api.agnes-ai.cn/v1',
    apiKey: key, model: 'agnes-3.0-flash', expectedRevision: 0,
    defaultSettings: { contextWindow: 32000 } })
  console.log('Agnes 3.0 Flash configured in isolated home; credential omitted from output')
}
const plugin = join(work, 'mhs-plugin')
cpSync(join(runtime, 'packages/mhs/plugin'), plugin, { recursive: true })
const descriptor = join(plugin, 'client/agnes.client.json')
const client = JSON.parse(readFileSync(descriptor, 'utf8'))
client.client.publicConfig.hubUrl = 'ws://127.0.0.1:43132/ws/hub'
writeFileSync(descriptor, JSON.stringify(client, null, 2))
run(['package', 'inspect', 'file:./mhs-plugin'])
const lockPath = join(profile, 'agnes-lock.json')
if (existsSync(lockPath) && JSON.parse(readFileSync(lockPath, 'utf8')).packages?.['@agnes/mhs']) {
  // CLI add is install-only. Updates use the same supported SDK operation as Plugin management.
  const { bootLocalConnect } = await import(pathToFileURL(join(runtime, 'packages/cli/src/boot/connect.ts')).href)
  const { inspectPackage, waitForPackageOperation, newPackageCommandId } =
    await import(pathToFileURL(join(runtime, 'packages/cli-tui/src/package-admin.ts')).href)
  const connected = await bootLocalConnect({ profile: 'local-dev', cwd: work }, { env, cwd: work },
    join(home, 'data/daemon/agnesd.sock'))
  try {
    const preview = await inspectPackage(connected.client, 'local-dev', { type: 'file', ref: 'file:./mhs-plugin' })
    const receipt = await connected.client.packages.update({ profile: 'local-dev',
      clientId: await connected.client.clientId(), commandId: newPackageCommandId('update'),
      id: '@agnes/mhs', source: { type: 'file', ref: 'file:./mhs-plugin' },
      expectedIntegrity: preview.integrity })
    const outcome = await waitForPackageOperation(connected.client, receipt)
    console.log(`MHS ${outcome.operation}: ${outcome.state}`)
  } finally { await connected.close() }
} else {
  run(['package', 'add', 'file:./mhs-plugin'], true)
}
const lock = JSON.parse(readFileSync(lockPath, 'utf8'))
const entry = lock.packages['@agnes/mhs']
if (!entry) throw new Error('MHS was not installed')
const { capabilityHash } = await import(pathToFileURL(join(runtime, 'packages/package-manager/src/integrity.ts')).href)
run(['package', 'trust', '@agnes/mhs', entry.integrity, capabilityHash(entry)], true)
run(['package', 'enable', '@agnes/mhs'], true)
run(['package', 'status'])

param(
    [string]$SshHost = $env:AGH_MHS_SSH_HOST,
    [int]$SshPort = $(if ($env:AGH_MHS_SSH_PORT) { [int]$env:AGH_MHS_SSH_PORT } else { 22 }),
    [string]$SshKey = $env:AGH_MHS_SSH_KEY
)
$ErrorActionPreference = 'Stop'
$taskWeb = 'http://127.0.0.1:43130/'
$taskMissing = @()
foreach ($taskPort in @(43130, 43131, 43132, 43134)) {
    $taskSocket = [System.Net.Sockets.TcpClient]::new()
    try {
        if (-not $taskSocket.ConnectAsync('127.0.0.1', $taskPort).Wait(1000)) { $taskMissing += $taskPort }
    } catch { $taskMissing += $taskPort }
    finally { $taskSocket.Dispose() }
}
if ($taskMissing.Count) {
    if (-not $SshHost -or $SshHost.StartsWith('-')) { throw '请设置 AGH_MHS_SSH_HOST 为自己的 user@host。' }
    $taskArguments = @('-p', "$SshPort", '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30', '-N')
    if ($SshKey) {
        if (-not (Test-Path -LiteralPath $SshKey)) { throw 'SSH 私钥文件不存在。' }
        $taskArguments += @('-i', ('"' + $SshKey + '"'))
    }
    foreach ($taskPort in $taskMissing) { $taskArguments += @('-L', "127.0.0.1:${taskPort}:127.0.0.1:${taskPort}") }
    $taskArguments += $SshHost
    $taskTunnel = Start-Process -FilePath 'ssh.exe' -ArgumentList $taskArguments -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 3
    if ($taskTunnel.HasExited) { throw '端口转发失败，请检查 SSH 配置和端口占用。' }
}
try { Invoke-WebRequest -Uri $taskWeb -TimeoutSec 5 | Out-Null }
catch { throw '工作台未就绪，请在部署目录运行 bash mhs/start-mhs.sh start。' }
Start-Process $taskWeb

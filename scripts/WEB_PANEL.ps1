param(
    [ValidateSet('Start','Status','Stop')][string]$Action = 'Start',
    [ValidateRange(1024,65535)][int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$basePython = (& $pythonPath -c 'import sys; print(sys._base_executable)').Trim()
$runtimeDir = Join-Path $projectRoot 'data\runtime\web'
$statePath = Join-Path $runtimeDir 'server.json'
$serverProcess = $null
if (Test-Path -LiteralPath $statePath) {
    try {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $candidate = Get-CimInstance Win32_Process -Filter ('ProcessId={0}' -f [int]$state.pid)
        if ($candidate -and $candidate.ExecutablePath -in @($pythonPath, $basePython) -and
            $candidate.CommandLine -match '-m (?:midas|portagent) .*web') { $serverProcess = $candidate }
    } catch { }
}
if ($Action -eq 'Stop') {
    if ($serverProcess) { Stop-Process -Id $serverProcess.ProcessId; Write-Output 'Panel stopped; active job may finish in its own process.' }
    else { Write-Output 'No owned panel process found.' }
    exit 0
}
if ($serverProcess) { Write-Output $state.url; exit 0 }
if ($Action -eq 'Status') { Write-Output 'Panel is stopped.'; exit 1 }
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$configPath = Join-Path $projectRoot 'configs\default.yaml'
$arguments = '-u -m midas --config "{0}" web --port {1}' -f $configPath, $Port
$env:PYTHONUTF8 = '1'
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeDir 'server.log') -RedirectStandardError (Join-Path $runtimeDir 'server-error.log') -PassThru
Write-Output ('Panel starting: http://127.0.0.1:{0} (PID {1})' -f $Port, $process.Id)

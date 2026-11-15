param(
    [string]$Config = (Join-Path $PSScriptRoot '..\configs\default.yaml'),
    [ValidatePattern('^(?:[01][0-9]|2[0-3]):[0-5][0-9]$')]
    [string]$AtTime = '07:00'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$Config = (Resolve-Path -LiteralPath $Config).Path
$bootstrapDir = Join-Path $projectRoot 'data\runtime\scheduler'
New-Item -ItemType Directory -Path $bootstrapDir -Force | Out-Null
$bootstrapLog = Join-Path $bootstrapDir 'bootstrap.log'
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
try {
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
    # Native stderr is diagnostic output, not a PowerShell terminating error.
    $ErrorActionPreference = 'Continue'
    & $pythonPath -u -m midas --config $Config scheduled-run --due-at $AtTime *>> $bootstrapLog
    $runExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    exit $runExitCode
} catch {
    $_ | Out-String | Add-Content -LiteralPath $bootstrapLog
    exit 1
}

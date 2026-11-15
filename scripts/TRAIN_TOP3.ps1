param(
    [string]$Date = (Get-Date -Format 'yyyy-MM-dd'),
    [string]$TrainingPython = '',
    [switch]$SkipFetch
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $TrainingPython) { $TrainingPython = Join-Path $projectRoot '.venv-research\Scripts\python.exe' }
if ($Date -notmatch '^\d{4}-\d{2}-\d{2}$') { throw 'Expected date YYYY-MM-DD' }
[datetime]::ParseExact($Date, 'yyyy-MM-dd', [cultureinfo]::InvariantCulture) | Out-Null
if (-not (Test-Path -LiteralPath $TrainingPython)) { throw 'Training environment not found; install the research extra into a separate environment.' }
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$marketRoot = Join-Path $projectRoot ('data\research\top3\market-{0}-2700' -f $Date)
$env:PYTHONUTF8 = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '4'
Push-Location -LiteralPath $projectRoot
try {
    if (-not $SkipFetch) {
        & $projectPython -u -m portagent.research.prepare --date $Date
        if ($LASTEXITCODE) { throw 'Market snapshot collection failed; rerun resumes successful assets.' }
    }
    & $TrainingPython -u -m portagent.research.benchmark --market-root $marketRoot
    if ($LASTEXITCODE) { throw 'Model benchmark failed; see output.' }
    & $TrainingPython -u -m portagent.research.modalities --market-root $marketRoot
    if ($LASTEXITCODE) { throw 'Source ablation failed; see output.' }
} finally { Pop-Location }

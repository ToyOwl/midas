param(
    [ValidateSet('Install', 'Status', 'Remove', 'Export')]
    [string]$Action = 'Status',
    [ValidatePattern('^(?:[01][0-9]|2[0-3]):[0-5][0-9]$')]
    [string]$AtTime = '07:00',
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$TaskName = 'PortagentDaily',
    [string]$Config = (Join-Path $PSScriptRoot '..\configs\default.yaml')
)
$ErrorActionPreference = 'Stop'
if ($Action -eq 'Status') {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) { Write-Output 'Task is not installed.'; exit 1 }
    $task | Select-Object TaskName, State, Description
    $task | Get-ScheduledTaskInfo | Select-Object LastRunTime, LastTaskResult, NextRunTime, NumberOfMissedRuns
    exit 0
}
if ($Action -eq 'Remove') {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    exit 0
}
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Config = (Resolve-Path -LiteralPath $Config).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
$env:PYTHONUTF8 = '1'
& $pythonPath -c 'import sys; from portagent.config import load_config; load_config(sys.argv[1])' $Config
if ($LASTEXITCODE -ne 0) { throw 'Configuration validation failed.' }
$userId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$runner = Join-Path $PSScriptRoot 'RUN_SCHEDULED.ps1'
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}" -Config "{1}" -AtTime "{2}"' -f $runner, $Config, $AtTime
$powershellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
# Explicit UTC+03 boundary keeps 07:00 Moscow when the computer changes time zone.
$moscowNow = [DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(3))
$firstDay = $moscowNow
if ($moscowNow.ToString('HH:mm') -ge $AtTime) { $firstDay = $moscowNow.AddDays(1) }
$boundary = '{0}T{1}:00+03:00' -f $firstDay.ToString('yyyy-MM-dd'), $AtTime
# Interactive token uses the owner's network/GPU session; no passwords stored.
$description = 'Midas: daily at {0} Europe/Moscow; catch up after logon; local LLM, reports and health log. Requires signed-in user (screen may be locked).' -f $AtTime
$xmlUser = [System.Security.SecurityElement]::Escape($userId)
$xmlCommand = [System.Security.SecurityElement]::Escape($powershellPath)
$xmlArgs = [System.Security.SecurityElement]::Escape($arguments)
$xmlDirectory = [System.Security.SecurityElement]::Escape($projectRoot)
$taskXml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>$description</Description></RegistrationInfo>
  <Triggers>
    <CalendarTrigger><StartBoundary>$boundary</StartBoundary><Enabled>true</Enabled><ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger>
    <LogonTrigger><Enabled>true</Enabled><Delay>PT1M</Delay><UserId>$xmlUser</UserId></LogonTrigger>
  </Triggers>
  <Principals><Principal id="Author"><UserId>$xmlUser</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable><RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle></IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand><Enabled>true</Enabled><Hidden>false</Hidden><RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>true</WakeToRun><ExecutionTimeLimit>PT4H</ExecutionTimeLimit><Priority>7</Priority>
    <RestartOnFailure><Interval>PT30M</Interval><Count>3</Count></RestartOnFailure>
  </Settings>
  <Actions Context="Author"><Exec><Command>$xmlCommand</Command><Arguments>$xmlArgs</Arguments><WorkingDirectory>$xmlDirectory</WorkingDirectory></Exec></Actions>
</Task>
"@
if ($Action -eq 'Export') {
    $destination = Join-Path $projectRoot 'data\runtime\scheduler\task.xml'
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    $taskXml | Set-Content -LiteralPath $destination -Encoding Unicode
    Write-Output $destination
} else {
    Register-ScheduledTask -TaskName $TaskName -Xml $taskXml -Force | Select-Object TaskName, State
    Get-ScheduledTaskInfo -TaskName $TaskName | Select-Object NextRunTime, LastTaskResult
}

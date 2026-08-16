param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunScript = Join-Path $ProjectRoot "run.ps1"
$AppUrl = "http://127.0.0.1:8000"

function Test-DublineReady {
    try {
        $Response = Invoke-WebRequest -UseBasicParsing -Uri $AppUrl -TimeoutSec 2
        return $Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500
    }
    catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $RunScript)) {
    throw "Dubline's run script was not found at $RunScript"
}

if (Test-DublineReady) {
    Write-Host "Dubline is already running." -ForegroundColor Green
    if (-not $NoBrowser) {
        Start-Process $AppUrl
    }
    return
}

Write-Host "Starting Dubline..." -ForegroundColor Green
Write-Host "Keep this window open while you use the app." -ForegroundColor DarkGray
Write-Host "Press Ctrl+C here when you want to stop the server." -ForegroundColor DarkGray
Write-Host

$BrowserOpener = $null
if (-not $NoBrowser) {
    $BrowserOpener = Start-Job -ArgumentList $AppUrl -ScriptBlock {
        param($Url)
        for ($Attempt = 0; $Attempt -lt 180; $Attempt++) {
            try {
                $Response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
                if ($Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500) {
                    Start-Process $Url
                    return
                }
            }
            catch {
                Start-Sleep -Seconds 1
            }
        }
    }
}

try {
    Set-Location -LiteralPath $ProjectRoot
    & $RunScript
    if ($LASTEXITCODE -ne 0) {
        throw "Dubline stopped with exit code $LASTEXITCODE"
    }
}
finally {
    if ($BrowserOpener) {
        Stop-Job -Job $BrowserOpener -ErrorAction SilentlyContinue
        Remove-Job -Job $BrowserOpener -Force -ErrorAction SilentlyContinue
    }
}

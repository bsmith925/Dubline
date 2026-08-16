$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $ProjectRoot "vendor\index-tts\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Runtime)) {
    throw "The CUDA runtime is not installed. Run .\setup.ps1 once first."
}
Set-Location -LiteralPath $ProjectRoot
& $Runtime -m uvicorn app.main:app --host 127.0.0.1 --port 8000

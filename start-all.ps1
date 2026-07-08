# Starts the shared Postgres container plus all three services, each in its own
# terminal window (so logs stay separate and readable). Run once from the repo root:
#   powershell -File start-all.ps1

$root = $PSScriptRoot

Write-Host "Starting shared Postgres..." -ForegroundColor Cyan
Push-Location (Join-Path $root "Shared-Postgres")
docker compose up -d
Pop-Location

Start-Sleep -Seconds 2

Write-Host "Starting Agent 1 (Schema Intelligence Layer) on :8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root\Schema-Intelligence-Layer'; .\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --reload"

Write-Host "Starting Agent 2 (Data Profiling Layer) on :8001..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root\MVA-use-case-latest-one'; .\venv\Scripts\python.exe -m uvicorn app.main:app --port 8001 --reload"

Write-Host "Starting Orchestrator on :8002..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root\Agent-Orchestrator'; .\venv\Scripts\python.exe -m uvicorn app.main:app --port 8002 --reload"

Write-Host ""
Write-Host "All services launching. Give them a few seconds, then check:" -ForegroundColor Green
Write-Host "  Agent 1:      http://127.0.0.1:8000/health"
Write-Host "  Agent 2:      http://127.0.0.1:8001/api/v1/health"
Write-Host "  Orchestrator: http://127.0.0.1:8002/health"
Write-Host "  Full pipeline: http://127.0.0.1:8002/docs"

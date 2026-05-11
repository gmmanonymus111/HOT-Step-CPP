# tools/smoke-test.ps1 — Local GPU smoke test for pre-release validation
#
# Usage:
#   .\tools\smoke-test.ps1 -BuildPath "release\out\HOT-Step-CPP-v1.1.0-win-x64-cuda.zip"
#   .\tools\smoke-test.ps1 -Dev    # Test current dev environment
#
# Tests:
#   1. Server starts and responds to health check
#   2. A short instrumental generation completes successfully
#   3. Audio output file is produced
#
# Requirements:
#   - Models must be available (copies from dev models/ dir if present)
#   - GPU (CUDA/Vulkan) for meaningful performance testing

param(
    [string]$BuildPath,              # Path to release zip or extracted directory
    [switch]$Dev,                    # Test current dev environment directly
    [int]$TimeoutSeconds = 600,      # Overall timeout (10 min default)
    [int]$Port = 3199,               # Non-default port to avoid conflicts
    [string]$ModelsPath              # Custom models directory to copy from
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# ── Colours ──────────────────────────────────────────────────────────────
function Write-Step  { param($msg) Write-Host "  [$((Get-Date).ToString('HH:mm:ss'))] $msg" }
function Write-Pass  { param($msg) Write-Host "  ✅ $msg" -ForegroundColor Green }
function Write-Fail  { param($msg) Write-Host "  ❌ $msg" -ForegroundColor Red }
function Write-Info  { param($msg) Write-Host "  ℹ️  $msg" -ForegroundColor Cyan }

# ── Resolve test directory ───────────────────────────────────────────────

$cleanup = $false
$testDir = $null
$serverProcess = $null
$startTime = Get-Date
$baseUrl = "http://localhost:$Port"

try {
    Write-Host "`n════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  HOT-Step Smoke Test" -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════`n" -ForegroundColor Cyan

    if ($Dev) {
        # Dev mode: test the current project in-place
        Write-Info "Mode: Development environment"
        $testDir = $ProjectRoot
        $serverCmd = "npx"
        $serverArgs = @("tsx", "server/src/index.ts")
    }
    elseif ($BuildPath) {
        if ($BuildPath -match '\.zip$') {
            # Extract zip to temp directory
            $testDir = Join-Path $env:TEMP "hot-step-smoke-test-$(Get-Random)"
            Write-Info "Extracting $BuildPath -> $testDir"
            Expand-Archive -Path $BuildPath -DestinationPath $testDir -Force
            $cleanup = $true
        }
        elseif (Test-Path $BuildPath -PathType Container) {
            $testDir = $BuildPath
        }
        else {
            throw "BuildPath must be a .zip file or directory: $BuildPath"
        }

        # Portable mode: use bundled Node.js
        $serverCmd = Join-Path $testDir "runtime\node.exe"
        $serverArgs = @((Join-Path $testDir "server\server.mjs"))

        if (-not (Test-Path $serverCmd)) {
            throw "Node.js not found at $serverCmd — is this a valid release build?"
        }

        Write-Info "Mode: Release build at $testDir"
    }
    else {
        throw "Specify -BuildPath <zip or dir> or -Dev"
    }

    # ── Check models ─────────────────────────────────────────────────
    $modelsDir = Join-Path $testDir "models"
    if (-not (Test-Path $modelsDir) -or (Get-ChildItem $modelsDir -Filter "*.gguf" -ErrorAction SilentlyContinue).Count -eq 0) {
        # Try to symlink/copy from dev environment
        $devModels = $ModelsPath
        if (-not $devModels) {
            $devModels = Join-Path $ProjectRoot "models"
        }
        if (Test-Path $devModels) {
            Write-Info "Linking models from $devModels"
            if (Test-Path $modelsDir) { Remove-Item $modelsDir -Force -Recurse -ErrorAction SilentlyContinue }
            # Try symlink first, fall back to junction
            try {
                New-Item -ItemType SymbolicLink -Path $modelsDir -Target $devModels -ErrorAction Stop | Out-Null
            }
            catch {
                cmd /c "mklink /J `"$modelsDir`" `"$devModels`""
            }
        }
        else {
            Write-Fail "No models found at $modelsDir and no dev models at $devModels"
            Write-Host "  Download models first, or pass -ModelsPath <path>" -ForegroundColor Yellow
            exit 1
        }
    }

    $modelCount = (Get-ChildItem $modelsDir -Filter "*.gguf" -ErrorAction SilentlyContinue).Count
    Write-Info "Models directory: $modelsDir ($modelCount .gguf files)"

    # ── Start server ─────────────────────────────────────────────────
    Write-Step "Starting server on port $Port..."

    $env:PORT = $Port
    $env:HOT_STEP_ROOT = $testDir

    if ($Dev) {
        $serverProcess = Start-Process -FilePath $serverCmd -ArgumentList $serverArgs `
            -WorkingDirectory $testDir -PassThru -NoNewWindow `
            -RedirectStandardOutput "$env:TEMP\smoke-test-stdout.log" `
            -RedirectStandardError "$env:TEMP\smoke-test-stderr.log"
    }
    else {
        $serverProcess = Start-Process -FilePath $serverCmd -ArgumentList $serverArgs `
            -WorkingDirectory $testDir -PassThru -NoNewWindow `
            -RedirectStandardOutput "$env:TEMP\smoke-test-stdout.log" `
            -RedirectStandardError "$env:TEMP\smoke-test-stderr.log"
    }

    Write-Info "Server PID: $($serverProcess.Id)"

    # ── Wait for health ──────────────────────────────────────────────
    Write-Step "Waiting for server health..."
    $healthTimeout = 120  # 2 minutes for model loading
    $healthStart = Get-Date

    while ($true) {
        $elapsed = ((Get-Date) - $healthStart).TotalSeconds
        if ($elapsed -gt $healthTimeout) {
            throw "Server failed to become healthy within ${healthTimeout}s"
        }

        try {
            $response = Invoke-RestMethod -Uri "$baseUrl/api/models/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
            Write-Pass "Server healthy (${elapsed:N0}s)"
            break
        }
        catch {
            if ($serverProcess.HasExited) {
                throw "Server process exited with code $($serverProcess.ExitCode)"
            }
            Start-Sleep -Seconds 2
        }
    }

    # ── Submit generation ────────────────────────────────────────────
    Write-Step "Submitting test generation (5s instrumental, turbo, 4 steps)..."

    $genParams = @{
        prompt          = "acoustic guitar folk ballad, warm and gentle"
        instrumental    = $true
        duration        = 5
        inferenceSteps  = 4
        randomSeed      = $true
        batchSize       = 1
        skipLm          = $true  # Skip LM for speed — only tests DiT + VAE
    } | ConvertTo-Json

    $genResponse = Invoke-RestMethod -Uri "$baseUrl/api/generate" `
        -Method POST -ContentType "application/json" -Body $genParams -TimeoutSec 30

    $jobId = $genResponse.jobId
    if (-not $jobId) {
        throw "Generation failed to start: $($genResponse | ConvertTo-Json -Compress)"
    }
    Write-Info "Job ID: $jobId"

    # ── Poll for completion ──────────────────────────────────────────
    Write-Step "Polling for completion..."
    $genStart = Get-Date

    while ($true) {
        $elapsed = ((Get-Date) - $genStart).TotalSeconds
        if ($elapsed -gt $TimeoutSeconds) {
            throw "Generation timed out after ${TimeoutSeconds}s"
        }

        $status = Invoke-RestMethod -Uri "$baseUrl/api/generate/status/$jobId" -Method GET -TimeoutSec 10

        switch ($status.status) {
            "succeeded" {
                $genTime = [math]::Round($elapsed, 1)
                Write-Pass "Generation succeeded in ${genTime}s"
                break
            }
            "failed" {
                throw "Generation failed: $($status.error)"
            }
            "cancelled" {
                throw "Generation was cancelled"
            }
            default {
                $stage = $status.stage ?? $status.status
                $progress = $status.progress ?? 0
                Write-Host "`r  ⏳ [$progress%] $stage" -NoNewline
            }
        }

        if ($status.status -eq "succeeded") { break }
        Start-Sleep -Seconds 2
    }
    Write-Host ""  # Clear the progress line

    # ── Validate result ──────────────────────────────────────────────
    Write-Step "Validating result..."

    if (-not $status.result) {
        throw "No result object in completed job"
    }

    $audioUrls = $status.result.audioUrls
    if (-not $audioUrls -or $audioUrls.Count -eq 0) {
        throw "No audio URLs in result"
    }

    Write-Pass "Audio produced: $($audioUrls.Count) file(s)"
    Write-Pass "BPM: $($status.result.bpm), Duration: $($status.result.duration)s"

    # ── Shutdown ─────────────────────────────────────────────────────
    Write-Step "Shutting down server..."
    try {
        Invoke-RestMethod -Uri "$baseUrl/api/shutdown" -Method POST -TimeoutSec 10 -ErrorAction SilentlyContinue
    }
    catch {
        # Shutdown endpoint may close connection before responding
    }
    Start-Sleep -Seconds 2

    # ── Summary ──────────────────────────────────────────────────────
    $totalTime = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 1)

    Write-Host "`n════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  ✅ SMOKE TEST PASSED" -ForegroundColor Green
    Write-Host "  Total time: ${totalTime}s" -ForegroundColor Green
    Write-Host "  Generation time: ${genTime}s" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════`n" -ForegroundColor Green

    exit 0
}
catch {
    $totalTime = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 1)

    Write-Host "`n════════════════════════════════════════════════════════" -ForegroundColor Red
    Write-Host "  ❌ SMOKE TEST FAILED" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  Total time: ${totalTime}s" -ForegroundColor Red
    Write-Host "════════════════════════════════════════════════════════`n" -ForegroundColor Red

    # Dump last 20 lines of server output for debugging
    $logFile = "$env:TEMP\smoke-test-stderr.log"
    if (Test-Path $logFile) {
        Write-Host "  Last 20 lines of server stderr:" -ForegroundColor Yellow
        Get-Content $logFile -Tail 20 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }

    exit 1
}
finally {
    # Kill server if still running
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Write-Step "Killing server process..."
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }

    # Clean up temp directory
    if ($cleanup -and $testDir -and (Test-Path $testDir)) {
        Write-Step "Cleaning up $testDir"
        Remove-Item -Recurse -Force $testDir -ErrorAction SilentlyContinue
    }

    # Clean up env
    Remove-Item Env:\PORT -ErrorAction SilentlyContinue
    Remove-Item Env:\HOT_STEP_ROOT -ErrorAction SilentlyContinue
}

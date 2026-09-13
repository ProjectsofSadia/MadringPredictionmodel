# run_madring.ps1
# One-shot: fix duplicate downloads, verify required files, run tests, run the pipeline.
# Put this in the folder with the project files and run:
#   powershell -ExecutionPolicy Bypass -File .\run_madring.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "`nWorking in: $PSScriptRoot" -ForegroundColor Cyan

# --- 1. collapse "name (1).py" duplicates onto "name.py" -------------------
Get-ChildItem -File -Filter "* (*).*" | ForEach-Object {
    $clean = $_.Name -replace '\s*\(\d+\)', ''
    $target = Join-Path $PSScriptRoot $clean
    if ((-not (Test-Path $target)) -or ($_.LastWriteTime -gt (Get-Item $target).LastWriteTime)) {
        Write-Host "  using newer download: $($_.Name) -> $clean" -ForegroundColor Yellow
        Move-Item $_.FullName $target -Force
    }
}

# --- 2. required files -----------------------------------------------------
$code = @("run_pipeline.py", "config.py", "fetch_data.py", "build_dataset.py", "features.py",
          "priors.py", "train_baseline.py", "train_xgboost.py", "evaluate.py",
          "predict_madring.py", "race_simulator.py", "plots.py", "simulation_assumptions.yaml")
$data = @("madring_grid.csv", "madring_race_inputs.yaml")

function Find-Here([string]$n) {
    foreach ($p in @((Join-Path $PSScriptRoot $n),
                     (Join-Path $PSScriptRoot "src\$n"),
                     (Join-Path $PSScriptRoot "data\$n"))) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

$missing = @()
foreach ($f in ($code + $data)) { if (-not (Find-Here $f)) { $missing += $f } }

if ($missing) {
    Write-Host "`nMISSING FILES - download these from the chat into this folder:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "  all required files present" -ForegroundColor Green

# --- 3. confirm the patched config is the one in use -----------------------
$cfg = Find-Here "config.py"
if (-not (Select-String -Path $cfg -Pattern "ASSUMPTIONS_PATH" -Quiet)) {
    Write-Host "`nOLD config.py IS STILL IN PLACE at $cfg" -ForegroundColor Red
    Write-Host "Re-download config.py from the chat and overwrite it, then run this again."
    exit 1
}

Write-Host "`nPath resolution:" -ForegroundColor Cyan
python -c "import config; print('  ROOT      ', config.ROOT); print('  DATA      ', config.DATA); print('  ASSUMPTION', config.ASSUMPTIONS_PATH); print('  CACHE     ', config.CACHE, '(exists:', config.CACHE.exists(), ')')"
if ($LASTEXITCODE -ne 0) { Write-Host "config.py failed to import - paste the error." -ForegroundColor Red; exit 1 }

# --- 4. tests (skipped if the tests folder was not downloaded) -------------
if (Test-Path (Join-Path $PSScriptRoot "tests")) {
    Write-Host "`nRunning tests..." -ForegroundColor Cyan
    python -m pytest tests -q
    if ($LASTEXITCODE -ne 0) { Write-Host "Tests failed - paste the output." -ForegroundColor Red; exit 1 }
}
else { Write-Host "`n(no tests folder - skipping)" -ForegroundColor DarkGray }

# --- 5. the pipeline -------------------------------------------------------
Write-Host "`nRunning pipeline. Full log -> pipeline_log.txt" -ForegroundColor Cyan
python run_pipeline.py 2>&1 | Tee-Object -FilePath pipeline_log.txt
$code_exit = $LASTEXITCODE

Write-Host "`n----- SUMMARY -----" -ForegroundColor Cyan
Get-Content pipeline_log.txt | Select-String -Pattern "^\[|^  [A-Z]|^  selected|^  LOW|^  BASE|^  HIGH|STOPPING|MAE=" |
    ForEach-Object { Write-Host $_.Line }

if ($code_exit -eq 0) {
    Write-Host "`nOutputs:" -ForegroundColor Green
    Get-ChildItem outputs, figures -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "  $($_.FullName)" }
    Write-Host "`nFreeze it before the race:" -ForegroundColor Yellow
    Write-Host '  git init; git add -A; git commit -m "Pre-race MADRING prediction, frozen before lights out"'
}
else { Write-Host "`nPipeline exited with code $code_exit - paste pipeline_log.txt." -ForegroundColor Red }

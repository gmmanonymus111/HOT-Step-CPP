# verify-hooks.ps1 — Post-sync verification of HOT-Step integration hooks
#
# Run after any upstream sync to verify all HOT-Step hooks are intact.
# Exit code 0 = all good, 1 = broken hooks detected.
#
# Usage: powershell -File engine\verify-hooks.ps1

$src   = "$PSScriptRoot\src"
$tools = "$PSScriptRoot\tools"
$errors = 0

Write-Host "`n=== HOT-Step Hook Verification ===" -ForegroundColor Cyan
Write-Host ""

# ── Hook 1: pipeline-synth-ops.cpp must include hot-step-sampler.h ────
$content = Get-Content "$src\pipeline-synth-ops.cpp" -Raw
if ($content -match '#include\s+"hot-step-sampler\.h"') {
    Write-Host "  [OK] pipeline-synth-ops.cpp -> hot-step-sampler.h" -ForegroundColor Green
} elseif ($content -match '#include\s+"dit-sampler\.h"') {
    Write-Host "  [FAIL] pipeline-synth-ops.cpp includes dit-sampler.h (should be hot-step-sampler.h)" -ForegroundColor Red
    Write-Host "         Fix: change #include `"dit-sampler.h`" to #include `"hot-step-sampler.h`"" -ForegroundColor Yellow
    $errors++
} else {
    Write-Host "  [WARN] pipeline-synth-ops.cpp: no sampler include found" -ForegroundColor Yellow
    $errors++
}

# ── Hook 2: model-store.h must include hot-step-params.h ──────────────
$content = Get-Content "$src\model-store.h" -Raw
if ($content -match '#include\s+"hot-step-params\.h"') {
    Write-Host "  [OK] model-store.h -> hot-step-params.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] model-store.h missing hot-step-params.h include" -ForegroundColor Red
    $errors++
}

# ── Hook 3: dit.h must include adapter-merge.h and adapter-runtime.h ──
$content = Get-Content "$src\dit.h" -Raw
if ($content -match '#include\s+"adapter-merge\.h"') {
    Write-Host "  [OK] dit.h -> adapter-merge.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] dit.h missing adapter-merge.h include" -ForegroundColor Red
    $errors++
}
if ($content -match '#include\s+"adapter-runtime\.h"') {
    Write-Host "  [OK] dit.h -> adapter-runtime.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] dit.h missing adapter-runtime.h include" -ForegroundColor Red
    $errors++
}

# ── Hook 4: hot-step-server.cpp must include hot-step-params.h ────────
$content = Get-Content "$tools\hot-step-server.cpp" -Raw
if ($content -match '#include\s+"hot-step-params\.h"') {
    Write-Host "  [OK] hot-step-server.cpp -> hot-step-params.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] hot-step-server.cpp missing hot-step-params.h include" -ForegroundColor Red
    $errors++
}

# ── Hook 5: fsq-detok.h must include fsq-quant.h, and neither fsq-detok.h
#            nor fsq-tok.h may carry upstream's own FSQ quantizer copies ──
$content = Get-Content "$src\fsq-detok.h" -Raw
if ($content -match '#include\s+"fsq-quant\.h"') {
    Write-Host "  [OK] fsq-detok.h -> fsq-quant.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] fsq-detok.h missing fsq-quant.h include" -ForegroundColor Red
    Write-Host "         Upstream's FSQ encode/decode are NOT reference-conformant" -ForegroundColor Yellow
    Write-Host "         (plain tanh, no ResidualFSQ soft clamp = 40% index match)." -ForegroundColor Yellow
    $errors++
}
if ($content -match 'static\s+void\s+fsq_decode_index' -or $content -match 'static\s+const\s+int\s+FSQ_LEVELS') {
    Write-Host "  [FAIL] fsq-detok.h re-declares FSQ_LEVELS/fsq_decode_index (upstream copy is back)" -ForegroundColor Red
    $errors++
}
$content = Get-Content "$src\fsq-tok.h" -Raw
if ($content -match 'static\s+int\s+fsq_encode_index') {
    Write-Host "  [FAIL] fsq-tok.h re-declares fsq_encode_index (upstream copy is back)" -ForegroundColor Red
    Write-Host "         Delete it; the conformant one lives in src/fsq-quant.h" -ForegroundColor Yellow
    $errors++
} else {
    Write-Host "  [OK] fsq-tok.h uses shared fsq_encode_index" -ForegroundColor Green
}

# ── Hook 6: linker sentinel present in hot-step-sampler.h ─────────────
$content = Get-Content "$src\hot-step-sampler.h" -Raw
if ($content -match 'hotstep_sampler_linked_') {
    Write-Host "  [OK] hot-step-sampler.h has linker sentinel" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] hot-step-sampler.h missing linker sentinel (hotstep_sampler_linked_)" -ForegroundColor Red
    $errors++
}

# ── Summary ───────────────────────────────────────────────────────────
Write-Host ""
if ($errors -gt 0) {
    Write-Host "  $errors hook(s) broken! Fix before building." -ForegroundColor Red
    Write-Host "" 
    exit 1
} else {
    Write-Host "  All hooks intact." -ForegroundColor Green
    Write-Host ""
    exit 0
}

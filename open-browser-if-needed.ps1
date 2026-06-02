# open-browser-if-needed.ps1
# Opens the given URL in the default browser ONLY if no existing browser tab
# with the HOT-Step window title is already open.
#
# Usage: powershell -ExecutionPolicy Bypass -File open-browser-if-needed.ps1 <url> [delay_seconds]

param(
    [Parameter(Mandatory=$true)]
    [string]$Url,

    [int]$DelaySeconds = 4
)

# Wait for the server to start
if ($DelaySeconds -gt 0) {
    Start-Sleep -Seconds $DelaySeconds
}

# Check if any window title contains our app identifier.
# Browser tabs show the page <title> in the window title bar.
# The title set in ui/index.html is "HOT-Step 9000 ⚡ CPP".
# We search for "HOT-Step 9000" which is distinctive enough to avoid false positives
# but loose enough to survive minor title changes.
$searchString = "HOT-Step 9000"

$existingTab = Get-Process | Where-Object {
    $_.MainWindowTitle -and $_.MainWindowTitle -like "*$searchString*"
} | Select-Object -First 1

if ($existingTab) {
    Write-Host "[HOT-Step] Browser tab already open ($($existingTab.MainWindowTitle)) - skipping."
} else {
    Write-Host "[HOT-Step] Opening browser: $Url"
    Start-Process $Url
}

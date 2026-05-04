@echo off
REM HOT-Step engine build (CUDA, native arch only)
REM Compiles ONLY for the local GPU — fast dev builds.
REM
REM Automatically finds Visual Studio / Build Tools via vswhere.
REM Automatically downloads ONNX Runtime GPU SDK for SuperSep support.

REM --- Find vcvars64.bat dynamically ---
REM vswhere ships with VS 2017+ and VS BuildTools.
REM
REM IMPORTANT: %ProgramFiles(x86)% contains parentheses which break
REM batch for-loop parsing. We write the vswhere output to a temp file
REM and read from that instead.

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Is Visual Studio or Build Tools installed?
    exit /b 1
)

set "VCVARS_TMP=%TEMP%\vcvars_path.txt"
"%VSWHERE%" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat" > "%VCVARS_TMP%" 2>nul

set "VCVARS="
for /f "usebackq tokens=*" %%i in ("%VCVARS_TMP%") do set "VCVARS=%%i"
del "%VCVARS_TMP%" 2>nul

if not defined VCVARS (
    echo ERROR: Could not find vcvars64.bat via vswhere.
    echo        Install the "Desktop development with C++" workload.
    exit /b 1
)

REM Skip vcvars if already sourced (prevents PATH overflow on repeated runs)
if defined VSCMD_VER (
    echo Using cached VS environment ^(VSCMD_VER=%VSCMD_VER%^)
) else (
    echo Using: %VCVARS%
    call "%VCVARS%"
)

REM ── ONNX Runtime GPU SDK (for SuperSep stem separation) ────────────
REM Auto-downloads from Microsoft's GitHub Releases if not present.
REM Users can skip this by setting ONNXRUNTIME_ROOT env var.

set "ORT_VERSION=1.25.1"
set "ORT_DIR=%~dp0deps\onnxruntime"
set "ORT_MARKER=%ORT_DIR%\include\onnxruntime_cxx_api.h"

if defined ONNXRUNTIME_ROOT (
    echo [ORT] Using ONNXRUNTIME_ROOT=%ONNXRUNTIME_ROOT%
    goto :build
)

if exist "%ORT_MARKER%" (
    echo [ORT] Found at %ORT_DIR%
    goto :build
)

echo.
echo [ORT] ONNX Runtime GPU SDK not found. Downloading v%ORT_VERSION%...
echo [ORT] (one-time download for SuperSep stem separation)
echo.

set "ORT_ZIP=%TEMP%\onnxruntime-win-x64-gpu-%ORT_VERSION%.zip"
set "ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ORT_VERSION%/onnxruntime-win-x64-gpu-%ORT_VERSION%.zip"

echo [ORT] Downloading from %ORT_URL%
curl -L -o "%ORT_ZIP%" "%ORT_URL%"
if errorlevel 1 (
    echo [ORT] WARNING: Download failed. Building without SuperSep support.
    goto :build
)

echo [ORT] Extracting...
mkdir "%~dp0deps" 2>nul
powershell -NoProfile -Command "Expand-Archive -Path '%ORT_ZIP%' -DestinationPath '%~dp0deps' -Force"
if errorlevel 1 (
    echo [ORT] WARNING: Extraction failed. Building without SuperSep support.
    goto :build
)

REM Rename extracted folder (it has version in the name)
if exist "%~dp0deps\onnxruntime-win-x64-gpu-%ORT_VERSION%" (
    ren "%~dp0deps\onnxruntime-win-x64-gpu-%ORT_VERSION%" onnxruntime
)

del "%ORT_ZIP%" 2>nul

if exist "%ORT_MARKER%" (
    echo [ORT] Successfully installed to %ORT_DIR%
) else (
    echo [ORT] WARNING: Installation may have failed. Check %ORT_DIR%
)

:build
cd /d "%~dp0"
mkdir build 2>nul
cd build

REM Only run cmake configure if not yet configured (avoids invalidating incremental builds)
if not exist "CMakeCache.txt" (
    cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
)
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

cd ..

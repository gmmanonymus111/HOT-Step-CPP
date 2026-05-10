#!/bin/bash
# HOT-Step CPP — macOS launcher
#
# Starts the Node.js server, which in turn spawns ace-server as a child process.
# Opens the browser automatically after the server is ready.
#
# Usage:
#   ./launch.sh              # Normal launch
#   HOT_STEP_ROOT=$(pwd) ./launch.sh  # Portable mode (if needed)

set -e

cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════╗"
echo "║         HOT-Step 9000 ⚡ CPP            ║"
echo "║    High-Performance Music Generation     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check that the engine binary exists
ENGINE_BIN="engine/build/ace-server"
if [ ! -f "${ENGINE_BIN}" ]; then
    ENGINE_BIN="engine/ace-server"
fi
if [ ! -f "${ENGINE_BIN}" ]; then
    echo "⚠️  ace-server not found!"
    echo "   Run: cd engine && ./build-mac.sh"
    echo ""
fi

# Check that server deps are installed
if [ ! -d "server/node_modules" ]; then
    echo "📦 Installing server dependencies..."
    cd server && npm install && cd ..
fi

# Check that UI is built (for production mode)
if [ ! -d "ui/dist" ]; then
    echo "⚠️  UI not built — run: cd ui && npm install && npm run build"
    echo "   Or use dev.sh for development mode with hot-reload."
    echo ""
fi

# Gatekeeper note for first launch
if [[ "$(uname)" == "Darwin" ]] && ! xattr -l "${ENGINE_BIN}" 2>/dev/null | grep -q "com.apple.quarantine" 2>/dev/null; then
    : # No quarantine flag — good
else
    echo "🔒 If macOS blocks the app, run: xattr -cr $(pwd)"
    echo ""
fi

# Start the server
cd server
exec npm start

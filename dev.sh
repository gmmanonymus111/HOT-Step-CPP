#!/bin/bash
# HOT-Step CPP — macOS development mode
#
# Starts both the Vite dev server (hot-reload UI) and the Node.js server.
# The Node.js server spawns ace-server as a child process.
#
# Usage:
#   ./dev.sh

set -e

cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════╗"
echo "║     HOT-Step 9000 ⚡ DEV MODE           ║"
echo "║       Vite HMR + tsx watch               ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Install deps if needed
if [ ! -d "server/node_modules" ]; then
    echo "📦 Installing server dependencies..."
    cd server && npm install && cd ..
fi
if [ ! -d "ui/node_modules" ]; then
    echo "📦 Installing UI dependencies..."
    cd ui && npm install && cd ..
fi

# Cleanup function: kill Vite when the script exits
cleanup() {
    echo ""
    echo "[dev.sh] Shutting down..."
    if [ -n "${VITE_PID}" ] && kill -0 "${VITE_PID}" 2>/dev/null; then
        kill "${VITE_PID}" 2>/dev/null || true
        echo "[dev.sh] Vite stopped"
    fi
}
trap cleanup EXIT INT TERM

# Start Vite dev server (UI hot-reload) in background
echo "🚀 Starting Vite dev server..."
cd ui
npm run dev &
VITE_PID=$!
cd ..

# Give Vite a moment to start
sleep 2

# Start Node.js server (spawns ace-server) in foreground
echo "🚀 Starting Node.js server..."
cd server
npm run dev

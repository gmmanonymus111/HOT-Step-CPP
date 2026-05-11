#!/bin/bash
# tools/smoke-test.sh — Local GPU smoke test for pre-release validation (macOS)
#
# Usage:
#   ./tools/smoke-test.sh <build.tar.gz or extracted dir>
#   ./tools/smoke-test.sh --dev    # Test current dev environment
#
# Tests:
#   1. Server starts and responds to health check
#   2. A short instrumental generation completes successfully
#   3. Audio output file is produced
#
# Requirements:
#   - Models must be available (symlinks from dev models/ dir if present)
#   - curl and jq (falls back to grep if jq unavailable)

set -euo pipefail

PORT="${SMOKE_TEST_PORT:-3199}"
TIMEOUT="${SMOKE_TEST_TIMEOUT:-600}"
BASE_URL="http://localhost:$PORT"
CLEANUP=false
TEST_DIR=""
SERVER_PID=""
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
START_TIME=$(date +%s)

# ── Helpers ──────────────────────────────────────────────────────────────
log()  { echo "  [$(date '+%H:%M:%S')] $*"; }
pass() { echo "  ✅ $*"; }
fail() { echo "  ❌ $*"; }
info() { echo "  ℹ️  $*"; }

json_get() {
    # Extract a JSON field value. Uses jq if available, falls back to grep.
    local json="$1" field="$2"
    if command -v jq &>/dev/null; then
        echo "$json" | jq -r ".$field // empty"
    else
        echo "$json" | grep -oP "\"$field\"\s*:\s*\"?\K[^\",}]+" | head -1
    fi
}

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Killing server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ "$CLEANUP" = true ] && [ -n "$TEST_DIR" ] && [ -d "$TEST_DIR" ]; then
        log "Cleaning up $TEST_DIR"
        rm -rf "$TEST_DIR"
    fi
}
trap cleanup EXIT

# ── Parse args ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  HOT-Step Smoke Test (macOS)"
echo "════════════════════════════════════════════════════════"
echo ""

DEV_MODE=false
BUILD_PATH=""

if [ "${1:-}" = "--dev" ]; then
    DEV_MODE=true
elif [ -n "${1:-}" ]; then
    BUILD_PATH="$1"
else
    echo "Usage: $0 <build.tar.gz or dir> | --dev"
    exit 1
fi

# ── Resolve test directory ───────────────────────────────────────────────
if [ "$DEV_MODE" = true ]; then
    info "Mode: Development environment"
    TEST_DIR="$PROJECT_ROOT"
else
    if [[ "$BUILD_PATH" == *.tar.gz ]]; then
        TEST_DIR="$(mktemp -d)/hot-step-smoke"
        info "Extracting $BUILD_PATH -> $TEST_DIR"
        mkdir -p "$TEST_DIR"
        tar -xzf "$BUILD_PATH" -C "$TEST_DIR"
        CLEANUP=true
    elif [ -d "$BUILD_PATH" ]; then
        TEST_DIR="$BUILD_PATH"
    else
        fail "BuildPath must be a .tar.gz or directory: $BUILD_PATH"
        exit 1
    fi
    info "Mode: Release build at $TEST_DIR"
fi

# ── Check models ─────────────────────────────────────────────────────────
MODELS_DIR="$TEST_DIR/models"
GGUF_COUNT=$(find "$MODELS_DIR" -name "*.gguf" 2>/dev/null | wc -l | tr -d ' ')

if [ "$GGUF_COUNT" -eq 0 ]; then
    DEV_MODELS="$PROJECT_ROOT/models"
    if [ -d "$DEV_MODELS" ]; then
        info "Linking models from $DEV_MODELS"
        rm -rf "$MODELS_DIR"
        ln -sf "$DEV_MODELS" "$MODELS_DIR"
        GGUF_COUNT=$(find "$MODELS_DIR" -name "*.gguf" 2>/dev/null | wc -l | tr -d ' ')
    else
        fail "No models found at $MODELS_DIR and no dev models at $DEV_MODELS"
        exit 1
    fi
fi

info "Models directory: $MODELS_DIR ($GGUF_COUNT .gguf files)"

# ── Start server ─────────────────────────────────────────────────────────
log "Starting server on port $PORT..."

export PORT
export HOT_STEP_ROOT="$TEST_DIR"

if [ "$DEV_MODE" = true ]; then
    npx tsx server/src/index.ts > /tmp/smoke-test-stdout.log 2> /tmp/smoke-test-stderr.log &
    SERVER_PID=$!
else
    "$TEST_DIR/runtime/bin/node" "$TEST_DIR/server/server.mjs" > /tmp/smoke-test-stdout.log 2> /tmp/smoke-test-stderr.log &
    SERVER_PID=$!
fi

info "Server PID: $SERVER_PID"

# ── Wait for health ─────────────────────────────────────────────────────
log "Waiting for server health..."
HEALTH_TIMEOUT=120
HEALTH_START=$(date +%s)

while true; do
    ELAPSED=$(( $(date +%s) - HEALTH_START ))
    if [ "$ELAPSED" -gt "$HEALTH_TIMEOUT" ]; then
        fail "Server failed to become healthy within ${HEALTH_TIMEOUT}s"
        tail -20 /tmp/smoke-test-stderr.log 2>/dev/null || true
        exit 1
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        fail "Server process exited unexpectedly"
        tail -20 /tmp/smoke-test-stderr.log 2>/dev/null || true
        exit 1
    fi

    if curl -sf "$BASE_URL/api/models/health" > /dev/null 2>&1; then
        pass "Server healthy (${ELAPSED}s)"
        break
    fi

    sleep 2
done

# ── Submit generation ────────────────────────────────────────────────────
log "Submitting test generation (5s instrumental, turbo, 4 steps)..."

GEN_RESPONSE=$(curl -sf -X POST "$BASE_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "acoustic guitar folk ballad, warm and gentle",
        "instrumental": true,
        "duration": 5,
        "inferenceSteps": 4,
        "randomSeed": true,
        "batchSize": 1,
        "skipLm": true
    }')

JOB_ID=$(json_get "$GEN_RESPONSE" "jobId")
if [ -z "$JOB_ID" ]; then
    fail "Generation failed to start: $GEN_RESPONSE"
    exit 1
fi
info "Job ID: $JOB_ID"

# ── Poll for completion ─────────────────────────────────────────────────
log "Polling for completion..."
GEN_START=$(date +%s)

while true; do
    ELAPSED=$(( $(date +%s) - GEN_START ))
    if [ "$ELAPSED" -gt "$TIMEOUT" ]; then
        fail "Generation timed out after ${TIMEOUT}s"
        exit 1
    fi

    STATUS_JSON=$(curl -sf "$BASE_URL/api/generate/status/$JOB_ID")
    STATUS=$(json_get "$STATUS_JSON" "status")

    case "$STATUS" in
        succeeded)
            GEN_TIME="$ELAPSED"
            pass "Generation succeeded in ${GEN_TIME}s"
            break
            ;;
        failed)
            ERROR=$(json_get "$STATUS_JSON" "error")
            fail "Generation failed: $ERROR"
            exit 1
            ;;
        cancelled)
            fail "Generation was cancelled"
            exit 1
            ;;
        *)
            STAGE=$(json_get "$STATUS_JSON" "stage")
            PROGRESS=$(json_get "$STATUS_JSON" "progress")
            printf "\r  ⏳ [%s%%] %s" "${PROGRESS:-0}" "${STAGE:-$STATUS}"
            ;;
    esac

    sleep 2
done
echo ""  # Clear progress line

# ── Validate result ──────────────────────────────────────────────────────
log "Validating result..."

AUDIO_COUNT=$(echo "$STATUS_JSON" | grep -oP '"audioUrls"\s*:\s*\[' | wc -l | tr -d ' ')
if [ "$AUDIO_COUNT" -eq 0 ]; then
    fail "No audio URLs in result"
    exit 1
fi

pass "Audio produced"

# ── Shutdown ─────────────────────────────────────────────────────────────
log "Shutting down server..."
curl -sf -X POST "$BASE_URL/api/shutdown" > /dev/null 2>&1 || true
sleep 2

# ── Summary ──────────────────────────────────────────────────────────────
TOTAL_TIME=$(( $(date +%s) - START_TIME ))

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅ SMOKE TEST PASSED"
echo "  Total time: ${TOTAL_TIME}s"
echo "  Generation time: ${GEN_TIME}s"
echo "════════════════════════════════════════════════════════"
echo ""

exit 0

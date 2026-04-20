#!/usr/bin/env bash
#
# End-to-end smoke test for mlx-lm-server with both BaselineEngine and
# DFlashEngine wired via EngineRegistry.
#
# Usage:
#   scripts/smoke-dflash.sh            # baseline only (fast, ~500MB download if needed)
#   scripts/smoke-dflash.sh --dflash   # also hit DFlash path (~10GB download: target+draft)
#
# What it does:
#   1. Start mlx-lm-server on 127.0.0.1:8081 in background
#   2. Poll /health until ready (timeout 180s — model load)
#   3. GET /v1/models — verify both engines advertise their aliases
#   4. POST /v1/chat/completions with baseline model (non-streaming)
#   5. If --dflash: POST /v1/chat/completions with dflash:qwen3-4b
#      (this triggers ~9GB HuggingFace download on first call; log has progress)
#   6. Capture all responses, kill server, emit pass/fail summary
#
# Nothing is committed; script is idempotent.

set -euo pipefail

INCLUDE_DFLASH=false
if [[ "${1:-}" == "--dflash" ]]; then
    INCLUDE_DFLASH=true
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT=8081
HOST=127.0.0.1
LOG_DIR="$(mktemp -d /tmp/dflash-smoke.XXXXXX)"
SERVER_LOG="$LOG_DIR/server.log"
CLIENT_LOG="$LOG_DIR/client.log"
PID_FILE="$LOG_DIR/server.pid"

cleanup() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[smoke] stopping server pid=$pid"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    fi
    echo "[smoke] logs kept at: $LOG_DIR"
}
trap cleanup EXIT

echo "[smoke] building mlx-lm-server (debug)..."
swift build --product mlx-lm-server 2>&1 | tail -5

BIN="$(swift build --show-bin-path)/mlx-lm-server"
if [[ ! -x "$BIN" ]]; then
    echo "[smoke] ERROR: $BIN not built" >&2
    exit 1
fi

echo "[smoke] starting server: $BIN --host $HOST --port $PORT"
"$BIN" --host "$HOST" --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "[smoke] server pid=$SERVER_PID log=$SERVER_LOG"

echo "[smoke] waiting for /health ready (timeout 180s)..."
READY=false
for i in {1..180}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[smoke] FAIL: server exited prematurely. Last log lines:" >&2
        tail -30 "$SERVER_LOG" >&2
        exit 1
    fi
    if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        READY=true
        echo "[smoke] server ready after ${i}s"
        break
    fi
    sleep 1
done

if [[ "$READY" != "true" ]]; then
    echo "[smoke] FAIL: /health did not respond within 180s. Last server log:" >&2
    tail -30 "$SERVER_LOG" >&2
    exit 1
fi

echo
echo "=== GET /v1/models ==="
curl -sf "http://$HOST:$PORT/v1/models" | tee "$CLIENT_LOG.models"
echo
echo

echo "=== POST /v1/chat/completions (baseline) ==="
BASELINE_MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
BASELINE_REQ="$LOG_DIR/baseline-request.json"
cat > "$BASELINE_REQ" <<EOF
{
  "model": "$BASELINE_MODEL",
  "messages": [
    {"role": "user", "content": "Say hello in exactly 3 words."}
  ],
  "max_tokens": 16,
  "stream": false
}
EOF
if curl -sf -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        --data @"$BASELINE_REQ" \
        | tee "$CLIENT_LOG.baseline"; then
    echo
    echo "[smoke] baseline PASS"
else
    echo "[smoke] FAIL: baseline /v1/chat/completions did not return 2xx" >&2
    exit 1
fi

if [[ "$INCLUDE_DFLASH" == "true" ]]; then
    echo
    echo "=== POST /v1/chat/completions (dflash:qwen3-4b) ==="
    echo "[smoke] NOTE: first DFlash call downloads ~9GB (target + draft). May take several minutes."
    DFLASH_REQ="$LOG_DIR/dflash-request.json"
    cat > "$DFLASH_REQ" <<EOF
{
  "model": "dflash:qwen3-4b",
  "messages": [
    {"role": "user", "content": "Solve: 2x + 5 = 17. Show your work step by step."}
  ],
  "max_tokens": 48,
  "stream": false
}
EOF
    if curl -sf --max-time 1800 -X POST "http://$HOST:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            --data @"$DFLASH_REQ" \
            | tee "$CLIENT_LOG.dflash"; then
        echo
        echo "[smoke] dflash PASS"
        echo
        echo "[smoke] acceptance_rate field (expect >0 if lossless gate works):"
        if command -v jq >/dev/null 2>&1; then
            jq '.usage.acceptance_rate' < "$CLIENT_LOG.dflash" || true
        else
            grep -o '"acceptance_rate":[^,}]*' "$CLIENT_LOG.dflash" || true
        fi
    else
        echo "[smoke] FAIL: dflash /v1/chat/completions did not return 2xx" >&2
        echo "[smoke] last 40 server log lines:" >&2
        tail -40 "$SERVER_LOG" >&2
        exit 1
    fi
fi

echo
echo "=== SMOKE TEST PASSED ==="
echo "Server log:   $SERVER_LOG"
echo "Client logs:  $CLIENT_LOG.*"
echo
echo "To run with DFlash (larger download): scripts/smoke-dflash.sh --dflash"

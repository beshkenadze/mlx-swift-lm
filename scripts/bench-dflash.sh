#!/usr/bin/env bash
#
# Manual DFlash benchmark for mlx-lm-server.
#
# Run `scripts/smoke-dflash.sh --dflash` first so the Qwen3 target + draft
# weights are already cached and loaded once before benchmarking.
#
# This script builds and launches `mlx-lm-server` locally, then runs the same
# prompt through:
#   1. baseline `mlx-community/Qwen3-4B-bf16`
#   2. `dflash:qwen3-4b`
#
# Output: a markdown table with completion tokens, total request time,
# tokens/second, and DFlash acceptance rate.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOST=127.0.0.1
PORT=8082
PROMPT="Write a 200-word summary of the causes of World War I."
MAX_TOKENS=512
BASELINE_MODEL="mlx-community/Qwen3-4B-bf16"
DFLASH_MODEL="dflash:qwen3-4b"

LOG_DIR="$(mktemp -d /tmp/dflash-bench.XXXXXX)"
SERVER_LOG="$LOG_DIR/server.log"
PID_FILE="$LOG_DIR/server.pid"

cleanup() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[bench] stopping server pid=$pid"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    fi
    echo "[bench] logs kept at: $LOG_DIR"
}
trap cleanup EXIT

json_number_field() {
    local file="$1"
    local jq_field="$2"
    local grep_key="$3"

    if command -v jq >/dev/null 2>&1; then
        jq -r "$jq_field // empty" < "$file"
    else
        grep -o "\"$grep_key\":[0-9.]*" "$file" | head -1 | cut -d: -f2
    fi
}

format_seconds() {
    local seconds="$1"
    awk -v seconds="$seconds" 'BEGIN { printf "%.2fs", seconds }'
}

format_rate() {
    local tokens="$1"
    local seconds="$2"
    awk -v tokens="$tokens" -v seconds="$seconds" 'BEGIN {
        if (seconds <= 0) {
            print "inf"
        } else {
            printf "%.1f", tokens / seconds
        }
    }'
}

format_ratio() {
    local numerator="$1"
    local denominator="$2"
    awk -v numerator="$numerator" -v denominator="$denominator" 'BEGIN {
        if (denominator <= 0) {
            print "inf"
        } else {
            printf "%.2fx", numerator / denominator
        }
    }'
}

write_request() {
    local model="$1"
    local request_file="$2"

    cat > "$request_file" <<EOF
{
  "model": "$model",
  "messages": [
    {"role": "user", "content": "$PROMPT"}
  ],
  "max_tokens": $MAX_TOKENS,
  "stream": false
}
EOF
}

run_completion() {
    local model="$1"
    local response_file="$2"
    local request_file="$3"

    write_request "$model" "$request_file"
    curl --silent --show-error --fail-with-body \
        --max-time 1800 \
        -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        --data @"$request_file" \
        -o "$response_file" \
        -w "%{time_total}"
}

echo "[bench] building mlx-lm-server via xcodebuild (debug)..."
DERIVED_DATA="$REPO_ROOT/.build-xcode"
xcodebuild \
    -scheme mlx-lm-server \
    -configuration Debug \
    -destination 'platform=macOS' \
    -derivedDataPath "$DERIVED_DATA" \
    build 2>&1 | tail -5

BIN="$DERIVED_DATA/Build/Products/Debug/mlx-lm-server"
if [[ ! -x "$BIN" ]]; then
    echo "[bench] ERROR: $BIN not built" >&2
    exit 1
fi

echo "[bench] starting server with baseline model $BASELINE_MODEL"
"$BIN" \
    --host "$HOST" \
    --port "$PORT" \
    --model "$BASELINE_MODEL" \
    --dflash-target "$BASELINE_MODEL" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo "[bench] waiting for /health ready (timeout 180s)..."
READY=false
for i in {1..180}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[bench] FAIL: server exited prematurely. Last log lines:" >&2
        tail -30 "$SERVER_LOG" >&2
        exit 1
    fi
    if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        READY=true
        echo "[bench] server ready after ${i}s"
        break
    fi
    sleep 1
done

if [[ "$READY" != "true" ]]; then
    echo "[bench] FAIL: /health did not respond within 180s. Last server log:" >&2
    tail -30 "$SERVER_LOG" >&2
    exit 1
fi

BASELINE_REQUEST="$LOG_DIR/baseline-request.json"
BASELINE_RESPONSE="$LOG_DIR/baseline-response.json"
DFLASH_REQUEST="$LOG_DIR/dflash-request.json"
DFLASH_RESPONSE="$LOG_DIR/dflash-response.json"

echo "[bench] running baseline request..."
baseline_time="$(run_completion "$BASELINE_MODEL" "$BASELINE_RESPONSE" "$BASELINE_REQUEST")"
baseline_tokens="$(json_number_field "$BASELINE_RESPONSE" '.usage.completion_tokens' 'completion_tokens')"
baseline_tok_per_s="$(format_rate "$baseline_tokens" "$baseline_time")"

echo "[bench] running dflash request..."
dflash_time="$(run_completion "$DFLASH_MODEL" "$DFLASH_RESPONSE" "$DFLASH_REQUEST")"
dflash_tokens="$(json_number_field "$DFLASH_RESPONSE" '.usage.completion_tokens' 'completion_tokens')"
dflash_tok_per_s="$(format_rate "$dflash_tokens" "$dflash_time")"
dflash_acceptance_rate="$(json_number_field "$DFLASH_RESPONSE" '.usage.acceptance_rate' 'acceptance_rate')"

speedup="$(format_ratio "$baseline_time" "$dflash_time")"

echo
printf '| %-8s | %6s | %10s | %6s | %15s |\n' "Engine" "Tokens" "Total time" "tok/s" "acceptance_rate"
printf '|---------|-------:|-----------:|------:|----------------:|\n'
printf '| %-8s | %6s | %10s | %6s | %15s |\n' \
    "baseline" \
    "$baseline_tokens" \
    "$(format_seconds "$baseline_time")" \
    "$baseline_tok_per_s" \
    "n/a"
printf '| %-8s | %6s | %10s | %6s | %15s |\n' \
    "dflash" \
    "$dflash_tokens" \
    "$(format_seconds "$dflash_time")" \
    "$dflash_tok_per_s" \
    "$dflash_acceptance_rate"
printf '| %-8s | %6s | %10s | %6s | %15s |\n' \
    "speedup" \
    "n/a" \
    "$speedup" \
    "" \
    ""


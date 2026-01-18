#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PIDS=()

cleanup() {
  if [ "${#PIDS[@]}" -gt 0 ]; then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

python "$ROOT_DIR/src/api/api-rest.py" &
PIDS+=("$!")

python "$ROOT_DIR/src/mqtt/servidor_mqtt.py" &
PIDS+=("$!")

python "$ROOT_DIR/src/mqtt/cliente_mqtt.py" &
PIDS+=("$!")

npm --prefix "$ROOT_DIR/src/dashboard" start &
PIDS+=("$!")

wait

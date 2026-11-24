#!/usr/bin/env bash
# Останавливает ffmpeg-публикаторы, запущенные start_streams.sh.
set -uo pipefail
PIDFILE="$(dirname "$0")/.pids"
if [[ "${1:-}" == "--pidfile" ]]; then PIDFILE="$2"; fi
[[ -f "$PIDFILE" ]] || exit 0
mapfile -t PIDS < "$PIDFILE"
if [[ ${#PIDS[@]} -gt 0 ]]; then
  kill "${PIDS[@]}" 2>/dev/null
  for _ in $(seq 1 50); do
    alive=0
    for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1 && break; done
    [[ $alive -eq 0 ]] && break
    sleep 0.1
  done
  kill -9 "${PIDS[@]}" 2>/dev/null
fi
: > "$PIDFILE"
echo "потоки остановлены"

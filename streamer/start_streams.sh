#!/usr/bin/env bash
# Публикует N камер в MediaMTX: rtsp://<host>/cam0 … cam{N-1}.
# Каждая камера — ffmpeg -re -stream_loop -1 -c copy (без перекодирования) по кругу из клипов,
# камеры с одним клипом сдвинуты по времени старта. ffmpeg привязан к ядрам через taskset.
#
#   streamer/start_streams.sh N [--videos DIR] [--rtsp rtsp://127.0.0.1:8554] [--cores 0-3] [--pidfile FILE]
set -euo pipefail

N="${1:?использование: start_streams.sh N [--videos DIR] [--rtsp URL] [--cores LIST] [--pidfile FILE]}"
shift
VIDEOS="data/videos/h264"
RTSP="rtsp://127.0.0.1:8554"
CORES=""
PIDFILE="$(dirname "$0")/.pids"
OFFSET_STEP=37   # секунды сдвига между копиями одного клипа

while [[ $# -gt 0 ]]; do
  case "$1" in
    --videos) VIDEOS="$2"; shift 2 ;;
    --rtsp) RTSP="$2"; shift 2 ;;
    --cores) CORES="$2"; shift 2 ;;
    --pidfile) PIDFILE="$2"; shift 2 ;;
    *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
  esac
done

mapfile -t CLIPS < <(find "$VIDEOS" -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.ts' \) | sort)
if [[ ${#CLIPS[@]} -eq 0 ]]; then
  echo "нет клипов в $VIDEOS (запустите scripts/prepare_videos.py)" >&2
  exit 1
fi

# длительности клипов для сдвига старта
declare -a DUR
for i in "${!CLIPS[@]}"; do
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${CLIPS[$i]}" | cut -d. -f1)
  DUR[$i]=${d:-300}
done

PREFIX=()
if [[ -n "$CORES" ]]; then
  PREFIX=(taskset -c "$CORES")
fi

"$(dirname "$0")/stop_streams.sh" --pidfile "$PIDFILE" >/dev/null 2>&1 || true
: > "$PIDFILE"

NC=${#CLIPS[@]}
for ((i = 0; i < N; i++)); do
  ci=$((i % NC))
  copy=$((i / NC))
  dur=${DUR[$ci]}
  offset=$(( (copy * OFFSET_STEP) % (dur > 60 ? dur - 30 : 1) ))
  "${PREFIX[@]}" ffmpeg -nostdin -hide_banner -loglevel error \
      -re -stream_loop -1 -ss "$offset" -i "${CLIPS[$ci]}" \
      -c copy -an -f rtsp -rtsp_transport tcp "$RTSP/cam$i" \
      >/dev/null 2>>"${PIDFILE}.log" &
  echo $! >> "$PIDFILE"
done

echo "запущено $N потоков ($NC клипов) → $RTSP/cam0..cam$((N - 1))"

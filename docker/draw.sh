#!/usr/bin/env bash
# Draw an image with the arm (phase 2). The image can be anywhere on the host:
# it is copied into ros2_ws/images/, which the control container sees as /ros2_ws/images/.
#   ./draw.sh ~/Desktop/chat.png
#   ./draw.sh ../ros2_ws/images/star.png
# Needs the control container running (docker-compose up -d control viz rviz).
set -euo pipefail

[ $# -eq 1 ] || { echo "Usage: $0 <image.png|jpg>"; exit 1; }
[ -f "$1" ] || { echo "File not found: $1"; exit 1; }

cd "$(dirname "$0")"
IMAGES=../ros2_ws/images
NAME="$(basename "$1")"
mkdir -p "$IMAGES"
[ "$(cd "$(dirname "$1")" 2>/dev/null && pwd)" = "$(cd "$IMAGES" && pwd)" ] || cp "$1" "$IMAGES/$NAME"

docker-compose ps --status running control | grep -q control \
  || { echo "control container not running: docker-compose up -d control viz rviz"; exit 1; }

# --keep-alive: without it, `pub --once` exits before the message is delivered.
docker-compose exec -T control bash -lc "source /ros2_ws/install/setup.bash && \
  timeout 30 ros2 topic pub -w 1 --once --keep-alive 5 /draw_image std_msgs/msg/String \
  '{data: /ros2_ws/images/$NAME}' > /dev/null"

echo "Drawing $NAME. RViz: http://localhost:6080/vnc.html  (Ctrl+C stops following the logs, not the drawing)"
docker-compose logs -f --since 5s control | grep --line-buffered -E "Drawing|unreachable|Skipping|timeout|Cannot draw|Cannot read"

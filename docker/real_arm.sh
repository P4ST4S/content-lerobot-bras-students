#!/usr/bin/env bash
# Real arm on macOS (Colima, no USB passthrough): serial bridge + control restart + logs.
#   ./real_arm.sh           # after (re)plugging the arm
#   ./real_arm.sh --build   # also rebuild the images (after a Dockerfile change)
# Ctrl+C stops the bridge and the session.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="$HOME/.local/share/uv/tools/lerobot/bin/python"
TCP_PORT="$(grep -E '^SERIAL_TCP=' .env | cut -d: -f2)"
DEVICE="$(ls /dev/tty.usbmodem* 2>/dev/null | head -1 || true)"

[ -n "$DEVICE" ] || { echo "Arm not found: plug in the USB cable and the power supply."; exit 1; }
[ -n "$TCP_PORT" ] || { echo "SERIAL_TCP missing in docker/.env"; exit 1; }
grep -q '^USE_SIM=false' .env || echo "Warning: USE_SIM is not false in docker/.env, the sim will run."

pkill -f serial_bridge.py 2>/dev/null || true
"$PYTHON" serial_bridge.py "$DEVICE" "$TCP_PORT" &
BRIDGE_PID=$!
CAMERA_INDEX="$(grep -E '^CAMERA_INDEX=' .env | cut -d= -f2)"
pkill -f "webcam.py bridge" 2>/dev/null || true
"$PYTHON" webcam.py bridge --camera "${CAMERA_INDEX:-0}" &
CAMERA_PID=$!
trap 'kill $BRIDGE_PID $CAMERA_PID 2>/dev/null' EXIT
sleep 1
kill -0 "$BRIDGE_PID" 2>/dev/null || { echo "Bridge failed to start (port $TCP_PORT busy?)"; exit 1; }

[ "${1:-}" = "--build" ] && docker-compose build control rviz perception
docker-compose up -d control viz rviz perception
docker-compose restart control perception

echo "Hold the arm. RViz: http://localhost:6080/vnc.html  (Ctrl+C to stop)"
docker-compose logs -f --since 1s control perception

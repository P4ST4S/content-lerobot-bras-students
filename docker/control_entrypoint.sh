#!/usr/bin/env bash
# Builds the workspace, launches the ROS2 nodes, and shuts them down cleanly
# on SIGTERM so destroy_node() runs (the real arm gets a proper disconnect).
set -e
set -m  # each background job gets its own process group

source /opt/ros/jazzy/setup.bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash

pids=()
launch() {
  local pkg=$1 file=$2
  shift 2
  if [ -f "install/$pkg/share/$pkg/launch/$file" ]; then
    ros2 launch "$pkg" "$file" "$@" &
    pids+=($!)
  fi
}

# Ensure serial device nodes exist in /dev for real hardware access
for i in 0 1 2 3; do
  [ -e /dev/ttyACM$i ] || mknod /dev/ttyACM$i c 166 $i 2>/dev/null && chmod 666 /dev/ttyACM$i 2>/dev/null || true
  [ -e /dev/ttyUSB$i ] || mknod /dev/ttyUSB$i c 188 $i 2>/dev/null && chmod 666 /dev/ttyUSB$i 2>/dev/null || true
done

# Serial port bridged over TCP (hosts without USB passthrough, see serial_bridge.py)
if [ -n "${SERIAL_TCP:-}" ]; then
  socat pty,link=/dev/ttyBRIDGE0,raw,echo=0 "tcp:${SERIAL_TCP}" &
  pids+=($!)
  sleep 1
fi

launch so101_driver driver.launch.py "use_sim:=${USE_SIM:-true}" "port:=${REAL_ROBOT_PORT:-/dev/ttyACM0}"
launch so101_brain brain.launch.py

shutdown() {
  trap - TERM INT
  set +e  # wait returns the child status; errexit would abort the shutdown
  [ ${#pids[@]} -eq 0 ] && exit 0
  # Signal the whole group, as Ctrl-C does: ros2 launch ignores a SIGINT sent to it alone.
  for p in "${pids[@]}"; do kill -INT -"$p" 2>/dev/null; done
  ( sleep 5; for p in "${pids[@]}"; do kill -KILL -"$p" 2>/dev/null; done ) &
  for p in "${pids[@]}"; do wait "$p"; done
  exit 0
}

trap shutdown TERM INT

sleep infinity &
wait $!

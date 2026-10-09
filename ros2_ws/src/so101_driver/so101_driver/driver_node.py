#!/usr/bin/env python3
"""driver_node.py - ROS2 driver of the SO-ARM101.

The only node that talks to the backend: SO101Sim (MuJoCo) or SO101Follower
(real arm), chosen by the `use_sim` parameter. The rest of the stack never
knows which one runs.

TODO:
  - Publish /joint_states in RADIANS (>= 20 Hz)
  - Subscribe to /joint_command (radians, gripper 0-100 %)
"""
import json
import os
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import rclpy
import rclpy.node
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image, JointState
from so101_sim import SO101Sim
from tf2_ros import StaticTransformBroadcaster

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Gripper joint limits in the URDF (rad).
# LeRobot convention: 0 % = closed (lower), 100 % = open (upper).
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)

# Calibration file of the real arm, mounted by docker compose (CALIBRATION_FILE in docker/.env).
CALIBRATION_FILE = Path("/calibration/arm.json")
DEFAULT_PORT = "/dev/ttyACM0"

CONTROL_HZ = 50.0
JOINT_STATE_HZ = 30.0
MAX_JOINT_SPEED = 2.0
MOTOR_ACCELERATION = 50

JOINT_OFFSETS = {
    "shoulder_pan": -0.0606,
    "shoulder_lift": 0.0736,
    "elbow_flex": 0.0898,
    "wrist_flex": 0.0176,
    "wrist_roll": 0.0575,
}
CAMERA_HZ = 10.0
CAMERA_FRAME = "external_cam"
CAMERA_CALIBRATION = Path("/ros2_ws/calibration/camera.json")


def _gripper_pct_to_rad(pct: float) -> float:
    lower, higher = GRIPPER_RANGE_RAD
    pct = min(max(pct, 0.0), 100.0)
    return lower + pct / 100.0 * (higher - lower)


class DriverNode(rclpy.node.Node):
    """Driver SO-ARM101: /joint_command -> backend -> /joint_states."""

    def __init__(self):
        super().__init__("so101_driver")
        self._use_sim: bool = self.declare_parameter("use_sim", True).value
        port: str = self.declare_parameter("port", DEFAULT_PORT).value
        mode = "simulation" if self._use_sim else "hardware"
        self.get_logger().info(f"Mode: {mode} (port={port})")

        self._robot = self._connect_arm(port)
        self._joint_offsets = {} if self._use_sim else JOINT_OFFSETS
        self._passive = not self._use_sim and not self._robot.bus.read("Torque_Enable", "shoulder_pan")
        if self._passive:
            self.get_logger().warn("Torque is off: /joint_command is ignored (read-only mode)")
        self._robot_lock = threading.Lock()
        self.get_logger().info(f"Robot connected: {type(self._robot).__name__}")

        if not self._use_sim:
            for motor in self._robot.bus.motors:
                self._robot.bus.write("Acceleration", motor, MOTOR_ACCELERATION)
            self.get_logger().info(f"Motor acceleration set to {MOTOR_ACCELERATION}")

        self._joint_state_pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_timer(1.0 / JOINT_STATE_HZ, self._publish_joint_states)

        self._last_command: JointState | None = None
        self._last_obs: dict | None = None
        self._sent: dict = {}
        self.create_subscription(JointState, "joint_command", self._cb_joint_command, 10)
        self.create_timer(1.0 / CONTROL_HZ, self._control_step)

        self._bridge = CvBridge()
        self._image_pub = self.create_publisher(Image, "external_cam/image_raw", 10)
        self._camera_info_pub = self.create_publisher(CameraInfo, "external_cam/camera_info", 10)
        self._tf_static = StaticTransformBroadcaster(self)
        self._camera_running = True
        if self._use_sim:
            k = self._robot.get_camera_intrinsics()
            extrinsics = self._robot.get_camera_extrinsics()
            self._camera_info = self._build_camera_info(
                k, self._robot.camera_width, self._robot.camera_height
            )
            self._tf_static.sendTransform(
                self._build_camera_transform(
                    extrinsics[:3, :3] @ np.diag([1.0, -1.0, -1.0]), extrinsics[:3, 3]
                )
            )
            self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self._camera_thread.start()
        elif os.environ.get("CAMERA_TCP"):
            calib = json.loads(CAMERA_CALIBRATION.read_text()) if CAMERA_CALIBRATION.is_file() else {}
            self._undistort_maps = None
            self._camera_info = CameraInfo(header=self._camera_info_header())
            if "K" in calib:
                k, dist = np.array(calib["K"]), np.array(calib["dist"])
                size = (calib["width"], calib["height"])
                new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, size, 0)
                self._undistort_maps = cv2.initUndistortRectifyMap(
                    k, dist, None, new_k, size, cv2.CV_16SC2
                )
                self._camera_info = self._build_camera_info(new_k, *size)
            else:
                self.get_logger().warn("Camera not calibrated: publishing raw images (intrinsics step)")
            if "T_world_camera" in calib:
                world_camera = np.array(calib["T_world_camera"])
                self._tf_static.sendTransform(
                    self._build_camera_transform(world_camera[:3, :3], world_camera[:3, 3])
                )
            else:
                self.get_logger().warn("No camera pose yet: run 'webcam.py extrinsics'")
            self._camera_thread = threading.Thread(
                target=self._real_camera_loop, args=(os.environ["CAMERA_TCP"],), daemon=True
            )
            self._camera_thread.start()
        else:
            self._camera_running = False
            self.get_logger().warn("No camera: set CAMERA_TCP in docker/.env")

        self.get_logger().info("Driver node ready.")

    def _connect_arm(self, port: str = DEFAULT_PORT) -> SO101Sim | SO101Follower:
        """Creates and connects the backend chosen by use_sim (provided)."""
        if self._use_sim:
            robot = SO101Sim()
            robot.connect()
            return robot

        if not CALIBRATION_FILE.is_file():
            raise RuntimeError(
                f"No calibration file at {CALIBRATION_FILE}: set CALIBRATION_FILE in docker/.env"
            )
        config = SO101FollowerConfig(
            port=port,
            id=CALIBRATION_FILE.stem,
            calibration_dir=CALIBRATION_FILE.parent,
            use_radians=True,
        )
        robot = SO101Follower(config)
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            robot.disconnect()
            raise RuntimeError(
                f"{CALIBRATION_FILE} does not match the motors: wrong file for this arm? "
                "Re-run lerobot-calibrate on the host."
            )
        # robot.bus.disable_torque()  # Comment out this line to control the robot.
        self.get_logger().info(
            "Torque disabled: arm can be moved freely by hand. Remove this part to control the arm."
        )
        return robot

    def destroy_node(self):
        """Disconnects the backend (provided). On the real arm this disables the torque."""
        if getattr(self, "_camera_running", False):
            self._camera_running = False
            self._camera_thread.join()
        if hasattr(self, "_robot") and self._robot.is_connected:
            self._robot.disconnect()
            self.get_logger().info("Robot disconnected.")
        super().destroy_node()

    def _cb_joint_command(self, msg: JointState):
        self._last_command = msg
        self._sent = {}

    def _control_step(self):
        if self._passive:
            return
        if self._last_command is None or self._last_obs is None:
            return
        cmd = self._last_command
        max_step = MAX_JOINT_SPEED / CONTROL_HZ
        action = {}
        for name, target in zip(cmd.name, cmd.position):
            if name not in JOINT_NAMES:
                continue
            key = f"{name}.pos"
            if name == "gripper":
                action[key] = target
                continue
            target += self._joint_offsets.get(name, 0.0)
            current = self._sent.get(key, self._last_obs[key])
            action[key] = current + max(-max_step, min(max_step, target - current))
        try:
            with self._robot_lock:
                self._robot.send_action(action)
        except ConnectionError as e:
            self.get_logger().warn(f"Command skipped: {e}", throttle_duration_sec=1.0)
            return
        self._sent.update(action)

    def _camera_info_header(self):
        info = CameraInfo()
        info.header.frame_id = CAMERA_FRAME
        return info.header

    def _build_camera_info(self, k, width, height) -> CameraInfo:
        info = CameraInfo()
        info.header.frame_id = CAMERA_FRAME
        info.width = int(width)
        info.height = int(height)
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = np.asarray(k, dtype=float).flatten().tolist()
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(k[0, 0]), 0.0, float(k[0, 2]), 0.0, 0.0, float(k[1, 1]), float(k[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _build_camera_transform(self, rotation, translation) -> TransformStamped:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation, dtype=float).flatten())
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "world"
        tf.child_frame_id = CAMERA_FRAME
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
            np.asarray(translation, dtype=float).tolist()
        )
        tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z = (
            quat.tolist()
        )
        return tf

    def _camera_loop(self):
        renderer = mujoco.Renderer(
            self._robot.model, self._robot.camera_height, self._robot.camera_width
        )
        while self._camera_running:
            start = time.monotonic()
            with self._robot_lock:
                renderer.update_scene(self._robot.data, camera=self._robot.camera_name)
            self._publish_image(renderer.render())
            time.sleep(max(0.0, 1.0 / CAMERA_HZ - (time.monotonic() - start)))
        renderer.close()

    def _real_camera_loop(self, address):
        host, port = address.rsplit(":", 1)
        while self._camera_running:
            try:
                with socket.create_connection((host, int(port)), timeout=5) as conn:
                    self.get_logger().info(f"Camera connected: {address}")
                    stream = conn.makefile("rb")
                    last = 0.0
                    while self._camera_running:
                        (length,) = struct.unpack(">I", stream.read(4))
                        jpeg = stream.read(length)
                        if time.monotonic() - last < 1.0 / CAMERA_HZ:
                            continue
                        last = time.monotonic()
                        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        if self._undistort_maps is not None:
                            frame = cv2.remap(frame, *self._undistort_maps, cv2.INTER_LINEAR)
                        self._publish_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except (OSError, struct.error) as e:
                self.get_logger().warn(f"Camera bridge unavailable ({e}), retrying", throttle_duration_sec=5.0)
                time.sleep(1.0)

    def _publish_image(self, rgb):
        image = self._bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        image.header.stamp = self.get_clock().now().to_msg()
        image.header.frame_id = CAMERA_FRAME
        self._camera_info.header.stamp = image.header.stamp
        self._image_pub.publish(image)
        self._camera_info_pub.publish(self._camera_info)

    def _publish_joint_states(self):
        try:
            with self._robot_lock:
                obs = self._robot.get_observation()
        except ConnectionError as e:
            self.get_logger().warn(f"Read skipped: {e}", throttle_duration_sec=1.0)
            return
        self._last_obs = obs

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [
            obs[f"{name}.pos"] - self._joint_offsets.get(name, 0.0) for name in JOINT_NAMES[:-1]
        ] + [
            _gripper_pct_to_rad(obs["gripper.pos"])
        ]

        self._joint_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

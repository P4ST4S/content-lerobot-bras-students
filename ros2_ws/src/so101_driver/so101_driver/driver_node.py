#!/usr/bin/env python3
"""driver_node.py - ROS2 driver of the SO-ARM101.

The only node that talks to the backend: SO101Sim (MuJoCo) or SO101Follower
(real arm), chosen by the `use_sim` parameter. The rest of the stack never
knows which one runs.

TODO:
  - Publish /joint_states in RADIANS (>= 20 Hz)
  - Subscribe to /joint_command (radians, gripper 0-100 %)
"""
import threading
import time
from pathlib import Path

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
CAMERA_HZ = 10.0
CAMERA_FRAME = "external_cam"


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

        if self._use_sim:
            self._bridge = CvBridge()
            self._image_pub = self.create_publisher(Image, "external_cam/image_raw", 10)
            self._camera_info_pub = self.create_publisher(CameraInfo, "external_cam/camera_info", 10)
            self._camera_info = self._build_camera_info()
            self._tf_static = StaticTransformBroadcaster(self)
            self._tf_static.sendTransform(self._build_camera_transform())
            self._camera_running = True
            self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self._camera_thread.start()

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
        robot.bus.disable_torque()  # Comment out this line to control the robot.
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
            current = self._sent.get(key, self._last_obs[key])
            action[key] = current + max(-max_step, min(max_step, target - current))
        try:
            with self._robot_lock:
                self._robot.send_action(action)
        except ConnectionError as e:
            self.get_logger().warn(f"Command skipped: {e}", throttle_duration_sec=1.0)
            return
        self._sent.update(action)

    def _build_camera_info(self) -> CameraInfo:
        k = self._robot.get_camera_intrinsics()
        info = CameraInfo()
        info.header.frame_id = CAMERA_FRAME
        info.width = self._robot.camera_width
        info.height = self._robot.camera_height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = k.flatten().tolist()
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [k[0, 0], 0.0, k[0, 2], 0.0, 0.0, k[1, 1], k[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _build_camera_transform(self) -> TransformStamped:
        extrinsics = self._robot.get_camera_extrinsics()
        rotation = extrinsics[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.flatten())
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "world"
        tf.child_frame_id = CAMERA_FRAME
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
            extrinsics[:3, 3].tolist()
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
            image = self._bridge.cv2_to_imgmsg(renderer.render(), encoding="rgb8")
            image.header.stamp = self.get_clock().now().to_msg()
            image.header.frame_id = CAMERA_FRAME
            self._camera_info.header.stamp = image.header.stamp
            self._image_pub.publish(image)
            self._camera_info_pub.publish(self._camera_info)
            time.sleep(max(0.0, 1.0 / CAMERA_HZ - (time.monotonic() - start)))
        renderer.close()

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
        msg.position = [obs[f"{name}.pos"] for name in JOINT_NAMES[:-1]] + [
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

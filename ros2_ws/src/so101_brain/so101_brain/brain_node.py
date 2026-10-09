#!/usr/bin/env python3
"""brain_node.py - intelligence of the arm: kinematics, then pick & place.

Never talks to the backend: reads /joint_states, commands through /joint_command.

TODO:
  - Forward kinematics: from /joint_states, publish the TCP pose
    (e.g. /end_effector_pose, geometry_msgs/PoseStamped) and show it in RViz
  - Inverse kinematics: reach an XYZ target (ikpy and pinocchio are installed)
  - Later: pick & place state machine
"""
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
import rclpy.node
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from so101_interfaces.srv import GoToTarget

URDF_PATH = Path("/opt/so101/sim/so101_sim/assets/so101/so101_new_calib.urdf")
EE_FRAME = "gripper_frame_link"
ROBOT_FRAMES = ("", "world", "base_link")

IK_MAX_ITER = 200
IK_TOLERANCE = 1e-3
IK_DAMPING = 1e-4
IK_MAX_STEP = 0.2
IK_RESTARTS = 10

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]


class BrainNode(rclpy.node.Node):
    """Brain of the arm: kinematics, then pick & place."""

    def __init__(self):
        super().__init__("so101_brain")

        self._model = pin.buildModelFromUrdf(str(URDF_PATH))
        self._data = self._model.createData()
        self._ee_frame_id = self._model.getFrameId(EE_FRAME)
        self._q = pin.neutral(self._model)
        self._idx_q = {
            name: self._model.joints[self._model.getJointId(name)].idx_q for name in JOINT_NAMES
        }

        self._ee_pose_pub = self.create_publisher(PoseStamped, "end_effector_pose", 10)
        self._joint_command_pub = self.create_publisher(JointState, "joint_command", 10)
        self.create_subscription(JointState, "joint_states", self._cb_joint_states, 10)
        self.create_service(GoToTarget, "go_to_target", self._cb_go_to_target)
        self.create_subscription(PointStamped, "target_position", self._cb_target_position, 10)

        self.get_logger().info("Brain node ready.")

    def _cb_joint_states(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            if name in self._idx_q:
                self._q[self._idx_q[name]] = position
        self._publish_end_effector_pose(msg.header.stamp)

    def _cb_go_to_target(self, request, response):
        response.success, response.message = self._go_to(
            request.target_pose.pose.position, request.target_pose.header.frame_id
        )
        return response

    def _cb_target_position(self, msg: PointStamped):
        success, message = self._go_to(msg.point, msg.header.frame_id)
        log = self.get_logger().info if success else self.get_logger().warn
        log(f"Target ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f}): {message}")

    def _go_to(self, point, frame):
        if frame not in ROBOT_FRAMES:
            return False, f"Unsupported frame '{frame}', expected world or base_link"

        q, error = self._solve_ik(np.array([point.x, point.y, point.z]))
        if error > IK_TOLERANCE:
            return False, f"Unreachable: closest solution is {error * 1000:.1f} mm away"

        self._send_joint_command(q)
        return True, f"IK solved, predicted error {error * 1000:.2f} mm"

    def _forward_kinematics(self, q):
        pin.forwardKinematics(self._model, self._data, q)
        return pin.updateFramePlacement(self._model, self._data, self._ee_frame_id)

    def _publish_end_effector_pose(self, stamp):
        ee = self._forward_kinematics(self._q)
        quat = pin.Quaternion(ee.rotation)

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = ee.translation.tolist()
        msg.pose.orientation.x = quat.x
        msg.pose.orientation.y = quat.y
        msg.pose.orientation.z = quat.z
        msg.pose.orientation.w = quat.w

        self._ee_pose_pub.publish(msg)

    def _solve_ik(self, target_position):
        lower, upper = self._model.lowerPositionLimit, self._model.upperPositionLimit
        rng = np.random.default_rng(0)
        seeds = [self._q.copy(), pin.neutral(self._model)]
        seeds += [rng.uniform(lower, upper) for _ in range(IK_RESTARTS)]

        best_q, best_error = None, np.inf
        for seed in seeds:
            q, error = self._solve_ik_from(seed, target_position)
            if error < best_error:
                best_q, best_error = q, error
            if error < IK_TOLERANCE:
                break
        return best_q, best_error

    def _solve_ik_from(self, q, target_position):
        for _ in range(IK_MAX_ITER):
            error = target_position - self._forward_kinematics(q).translation
            if np.linalg.norm(error) < IK_TOLERANCE:
                break
            jacobian = pin.computeFrameJacobian(
                self._model, self._data, q, self._ee_frame_id, pin.LOCAL_WORLD_ALIGNED
            )[:3]
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + IK_DAMPING * np.eye(3), error
            )
            step = np.linalg.norm(dq)
            if step > IK_MAX_STEP:
                dq *= IK_MAX_STEP / step
            q = np.clip(q + dq, self._model.lowerPositionLimit, self._model.upperPositionLimit)
        error = target_position - self._forward_kinematics(q).translation
        return q, float(np.linalg.norm(error))

    def _send_joint_command(self, q, gripper_pct=None):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINT_NAMES)
        msg.position = [float(q[self._idx_q[name]]) for name in ARM_JOINT_NAMES]
        if gripper_pct is not None:
            msg.name.append("gripper")
            msg.position.append(float(gripper_pct))
        self._joint_command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
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

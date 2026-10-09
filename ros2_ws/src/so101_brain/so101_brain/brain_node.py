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

import pinocchio as pin
import rclpy
import rclpy.node
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState

URDF_PATH = Path("/opt/so101/sim/so101_sim/assets/so101/so101_new_calib.urdf")
EE_FRAME = "gripper_frame_link"

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

        self._ee_pose_pub = self.create_publisher(PoseStamped, "end_effector_pose", 10)
        self.create_subscription(JointState, "joint_states", self._cb_joint_states, 10)

        self.get_logger().info("Brain node ready.")

    def _cb_joint_states(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            if self._model.existJointName(name):
                joint_id = self._model.getJointId(name)
                self._q[self._model.joints[joint_id].idx_q] = position
        self._publish_end_effector_pose(msg.header.stamp)

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
        raise NotImplementedError("TO DO")

    def _send_joint_command(self, q, gripper_pct=None):
        raise NotImplementedError("TO DO")


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

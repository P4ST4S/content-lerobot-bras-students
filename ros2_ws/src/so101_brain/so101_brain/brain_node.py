#!/usr/bin/env python3
"""brain_node.py - intelligence of the arm: kinematics, then pick & place.

Never talks to the backend: reads /joint_states, commands through /joint_command.

  - Forward kinematics: /end_effector_pose and /end_effector_point from /joint_states
  - Inverse kinematics: /target_position (topic) or /go_to_target (service)
  - Pick & place state machine: reads /object_position and /drop_box_position
    (last known value, never blocks on perception), restart with /start_pick
"""
import time
from enum import Enum
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
import rclpy.node
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from so101_interfaces.srv import GoToTarget
from std_msgs.msg import Empty

URDF_PATH = Path("/opt/so101/sim/so101_sim/assets/so101/so101_new_calib.urdf")
EE_FRAME = "gripper_frame_link"
ROBOT_FRAMES = ("", "world", "base_link")
DOWN = np.array([0.0, 0.0, -1.0])

IK_MAX_ITER = 200
IK_TOLERANCE = 1e-3
IK_DAMPING = 1e-4
IK_MAX_STEP = 0.2
IK_RESTARTS = 10

STATE_HZ = 10.0
HOVER_HEIGHT = 0.08
DROP_HEIGHT = 0.08
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 0.0
REACHED_TOLERANCE = 0.015
MOTION_TIMEOUT = 8.0
GRIPPER_WAIT = 1.0

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]


class State(Enum):
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    RELEASE = "RELEASE"
    RETURN = "RETURN"
    DONE = "DONE"


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
        self._ee_position = None

        self._state = State.IDLE
        self._state_start = time.monotonic()
        self._armed = self.declare_parameter("autostart", True).value
        self._object = None
        self._box = None
        self._pick = None
        self._home_q = None
        self._target = None

        self._ee_pose_pub = self.create_publisher(PoseStamped, "end_effector_pose", 10)
        self._ee_point_pub = self.create_publisher(PointStamped, "end_effector_point", 10)
        self._joint_command_pub = self.create_publisher(JointState, "joint_command", 10)
        self.create_subscription(JointState, "joint_states", self._cb_joint_states, 10)
        self.create_service(GoToTarget, "go_to_target", self._cb_go_to_target)
        self.create_subscription(PointStamped, "target_position", self._cb_target_position, 10)

        self.create_subscription(PoseStamped, "object_position", self._cb_object, 10)
        self.create_subscription(PoseStamped, "drop_box_position", self._cb_box, 10)
        self.create_subscription(Empty, "start_pick", self._cb_start_pick, 10)
        self.create_timer(1.0 / STATE_HZ, self._step)

        self.get_logger().info("Brain node ready.")

    # ------------------------------------------------------------ callbacks

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

    def _cb_object(self, msg: PoseStamped):
        p = msg.pose.position
        self._object = np.array([p.x, p.y, p.z])

    def _cb_box(self, msg: PoseStamped):
        p = msg.pose.position
        self._box = np.array([p.x, p.y, p.z])

    def _cb_start_pick(self, _msg: Empty):
        if self._state in (State.IDLE, State.DONE):
            self._armed = True
            self._set_state(State.IDLE)

    def _go_to(self, point, frame):
        if self._state not in (State.IDLE, State.DONE):
            return False, f"Busy: pick & place running ({self._state.value})"
        if frame not in ROBOT_FRAMES:
            return False, f"Unsupported frame '{frame}', expected world or base_link"

        q, error = self._solve_ik(np.array([point.x, point.y, point.z]))
        if error > IK_TOLERANCE:
            return False, f"Unreachable: closest solution is {error * 1000:.1f} mm away"

        self._send_joint_command(q)
        return True, f"IK solved, predicted error {error * 1000:.2f} mm"

    # ------------------------------------------------------- state machine

    def _step(self):
        elapsed = time.monotonic() - self._state_start

        if self._state == State.IDLE:
            if self._armed and self._object is not None and self._ee_position is not None:
                self._armed = False
                self._home_q = self._q.copy()
                self._pick = self._object.copy()
                self._move(State.APPROACH, self._pick + [0.0, 0.0, HOVER_HEIGHT], GRIPPER_OPEN)

        elif self._state == State.APPROACH:
            if self._arrived(elapsed):
                self._move(State.DESCEND, self._pick, GRIPPER_OPEN)

        elif self._state == State.DESCEND:
            if self._arrived(elapsed):
                self._send_joint_command(self._q, GRIPPER_CLOSED)
                self._set_state(State.GRASP)

        elif self._state == State.GRASP:
            if elapsed > GRIPPER_WAIT:
                self._move(State.LIFT, self._pick + [0.0, 0.0, HOVER_HEIGHT], GRIPPER_CLOSED)

        elif self._state == State.LIFT:
            if self._arrived(elapsed):
                if self._box is None:
                    if elapsed > MOTION_TIMEOUT:
                        self.get_logger().error("Drop box never seen, aborting")
                        self._go_home()
                    return
                self._move(State.TRANSPORT, self._box + [0.0, 0.0, DROP_HEIGHT], GRIPPER_CLOSED)

        elif self._state == State.TRANSPORT:
            if self._arrived(elapsed):
                self._send_joint_command(self._q, GRIPPER_OPEN)
                self._set_state(State.RELEASE)

        elif self._state == State.RELEASE:
            if elapsed > GRIPPER_WAIT:
                self._go_home()

        elif self._state == State.RETURN:
            if self._arrived(elapsed):
                self._set_state(State.DONE)

    def _set_state(self, state):
        self.get_logger().info(f"State: {self._state.value} -> {state.value}")
        self._state = state
        self._state_start = time.monotonic()

    def _move(self, state, target, gripper_pct):
        q, error = self._solve_ik(target, down=True)
        if error > IK_TOLERANCE:
            self.get_logger().error(
                f"{state.value}: target {np.round(target, 3)} unreachable "
                f"({error * 1000:.1f} mm), going home"
            )
            self._go_home()
            return
        tilt = np.degrees(np.arccos(np.clip(self._forward_kinematics(q).rotation[:, 2] @ DOWN, -1, 1)))
        self.get_logger().info(f"{state.value}: target {np.round(target, 3)}, gripper tilt {tilt:.0f} deg")
        self._target = target
        self._send_joint_command(q, gripper_pct)
        self._set_state(state)

    def _go_home(self):
        self._target = self._forward_kinematics(self._home_q).translation.copy()
        self._send_joint_command(self._home_q, GRIPPER_OPEN)
        self._set_state(State.RETURN)

    def _arrived(self, elapsed):
        if np.linalg.norm(self._ee_position - self._target) < REACHED_TOLERANCE:
            return True
        if elapsed > MOTION_TIMEOUT:
            self.get_logger().warn(
                f"{self._state.value}: timeout, "
                f"{np.linalg.norm(self._ee_position - self._target) * 1000:.0f} mm from target"
            )
            return True
        return False

    # ------------------------------------------------------------ kinematics

    def _forward_kinematics(self, q):
        pin.forwardKinematics(self._model, self._data, q)
        return pin.updateFramePlacement(self._model, self._data, self._ee_frame_id)

    def _publish_end_effector_pose(self, stamp):
        ee = self._forward_kinematics(self._q)
        self._ee_position = ee.translation.copy()
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
        self._ee_point_pub.publish(PointStamped(header=msg.header, point=msg.pose.position))

    def _solve_ik(self, target_position, down=False):
        lower, upper = self._model.lowerPositionLimit, self._model.upperPositionLimit
        rng = np.random.default_rng(0)
        seeds = [self._q.copy(), pin.neutral(self._model)]
        seeds += [rng.uniform(lower, upper) for _ in range(IK_RESTARTS)]

        best_q, best_error, best_tilt = None, np.inf, np.inf
        for seed in seeds:
            q, error, tilt = self._solve_ik_from(seed, target_position, down)
            if error < IK_TOLERANCE and tilt < best_tilt or best_error > IK_TOLERANCE and error < best_error:
                best_q, best_error, best_tilt = q, error, tilt
            if error < IK_TOLERANCE and (not down or tilt < 0.05):
                break
        return best_q, best_error

    def _solve_ik_from(self, q, target_position, down):
        if down:
            q = self._ik_iterations(q, target_position, down=True)
        q = self._ik_iterations(q, target_position, down=False)
        ee = self._forward_kinematics(q)
        error = float(np.linalg.norm(target_position - ee.translation))
        tilt = float(np.arccos(np.clip(ee.rotation[:, 2] @ DOWN, -1.0, 1.0)))
        return q, error, tilt

    def _ik_iterations(self, q, target_position, down):
        for _ in range(IK_MAX_ITER):
            ee = self._forward_kinematics(q)
            error = target_position - ee.translation
            tilt_error = np.cross(ee.rotation[:, 2], DOWN)
            if np.linalg.norm(error) < IK_TOLERANCE and (not down or np.linalg.norm(tilt_error) < 1e-3):
                break
            jacobian = pin.computeFrameJacobian(
                self._model, self._data, q, self._ee_frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            j_pos, j_rot = jacobian[:3], jacobian[3:]
            j_pos_pinv = j_pos.T @ np.linalg.inv(j_pos @ j_pos.T + IK_DAMPING * np.eye(3))
            dq = j_pos_pinv @ error
            if down:
                null_space = np.eye(self._model.nv) - j_pos_pinv @ j_pos
                j_rot_null = j_rot @ null_space
                dq += j_rot_null.T @ np.linalg.solve(
                    j_rot_null @ j_rot_null.T + IK_DAMPING * np.eye(3), tilt_error - j_rot @ dq
                )
            step = np.linalg.norm(dq)
            if step > IK_MAX_STEP:
                dq *= IK_MAX_STEP / step
            q = np.clip(q + dq, self._model.lowerPositionLimit, self._model.upperPositionLimit)
        return q

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

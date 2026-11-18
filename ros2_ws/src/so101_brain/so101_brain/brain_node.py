#!/usr/bin/env python3
"""brain_node.py - intelligence of the arm: kinematics, then drawing.

Never talks to the backend: reads /joint_states, commands through /joint_command.

  - Forward kinematics: /end_effector_pose and /end_effector_point from /joint_states
  - Inverse kinematics: /target_position (topic) or /go_to_target (service)
  - Drawing: /draw_image (std_msgs/String, image path) -> takes the felt pen out of its
    holder, draws the outlines, puts the pen back. Plan, ink and pen shown in RViz on
    /drawing/plan, /drawing/ink, /drawing/pen. (Pick & place lives on the main branch.)
"""
import time
from enum import Enum
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
import rclpy.node
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from so101_interfaces.srv import GoToTarget
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker

from so101_brain.image_to_strokes import image_to_strokes

URDF_PATH = Path("/opt/so101/sim/so101_sim/assets/so101/so101_new_calib.urdf")
EE_FRAME = "gripper_frame_link"
# gripper_frame_link sits on the fixed jaw face; its -x points to the moving jaw. The IK
# targets the axis of an 11 mm pen squeezed between the jaws, as measured in the sim
# (squeezed 1.3 mm into the pad). shortcut: sim value, re-measure on the real arm.
GRASP_OFFSET = (-0.0042, 0.0013, 0.0)
ROBOT_FRAMES = ("", "world", "base_link")
DOWN = np.array([0.0, 0.0, -1.0])

IK_MAX_ITER = 200
IK_TOLERANCE = 1e-3
IK_DAMPING = 1e-4
IK_MAX_STEP = 0.2
IK_RESTARTS = 10

STATE_HZ = 10.0
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 0.0
REACHED_TOLERANCE = 0.015
GRASP_TOLERANCE = 0.005
MOTION_TIMEOUT = 8.0
GRIPPER_WAIT = 1.0
PEN_ABOVE_MARGIN = 0.012  # jaw tips above the pen top when coming down onto it
HOLDER_CLEARANCE = 0.01   # pen tip above the holder top once pulled out
INSERT_GAP = 0.002        # pen tip above the table when dropped back in the holder

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]


INK_HEIGHT = 0.003  # pen tip closer than this to the paper = leaving ink


class State(Enum):
    IDLE = "IDLE"
    PEN_TRAVEL = "PEN_TRAVEL"
    PEN_DOWN = "PEN_DOWN"
    TRACE = "TRACE"
    PEN_UP = "PEN_UP"
    PEN_FETCH = "PEN_FETCH"
    PEN_REACH = "PEN_REACH"
    PEN_GRIP = "PEN_GRIP"
    PEN_EXTRACT = "PEN_EXTRACT"
    PEN_STOW = "PEN_STOW"
    PEN_INSERT = "PEN_INSERT"
    PEN_RELEASE = "PEN_RELEASE"
    PEN_LEAVE = "PEN_LEAVE"
    RETURN = "RETURN"
    DONE = "DONE"


class BrainNode(rclpy.node.Node):
    """Brain of the arm: kinematics, then drawing."""

    def __init__(self):
        super().__init__("so101_brain")

        self._model = pin.buildModelFromUrdf(str(URDF_PATH))
        tcp_id = self._model.getFrameId(EE_FRAME)
        tcp = self._model.frames[tcp_id]
        self._ee_frame_id = self._model.addFrame(
            pin.Frame(
                "grasp_center",
                tcp.parentJoint,
                tcp_id,
                tcp.placement * pin.SE3(np.eye(3), np.array(GRASP_OFFSET)),
                pin.FrameType.OP_FRAME,
            )
        )
        self._data = self._model.createData()
        self._q = pin.neutral(self._model)
        self._idx_q = {
            name: self._model.joints[self._model.getJointId(name)].idx_q for name in JOINT_NAMES
        }
        self._ee_position = None

        self._state = State.IDLE
        self._state_start = time.monotonic()
        self._home_q = None
        self._target = None

        # Drawing. Calibration knobs for the real arm: paper_z (table height under the
        # paper) and pen_holder_xy (holder position, measured in the robot frame).
        # The pen stands in the holder with its tip on the table, so grasping it with the
        # gripper frame at height pen_length leaves the tip pen_length below the gripper.
        self._paper_z = self.declare_parameter("paper_z", 0.0).value
        self._pen_length = self.declare_parameter("pen_length", 0.05).value
        self._pen_total = self.declare_parameter("pen_total", 0.088).value
        self._pen_holder_xy = tuple(self.declare_parameter("pen_holder_xy", [0.18, 0.15]).value)
        self._holder_height = self.declare_parameter("holder_height", 0.02).value
        self._pen_held = False
        self._pen_lift = self.declare_parameter("pen_lift", 0.02).value
        self._draw_center = tuple(self.declare_parameter("draw_center", [0.22, 0.0]).value)
        self._draw_size = self.declare_parameter("draw_size", 0.10).value
        self._trace_tolerance = self.declare_parameter("trace_tolerance", 0.003).value
        self._strokes = []
        self._stroke_i = self._point_i = 0
        self._ink = []
        self._last_ink = None
        self._plan_pub = self.create_publisher(Marker, "drawing/plan", 1)
        self._ink_pub = self.create_publisher(Marker, "drawing/ink", 1)
        self._pen_pub = self.create_publisher(Marker, "drawing/pen", 1)
        self.create_subscription(String, "draw_image", self._cb_draw_image, 10)

        self._ee_pose_pub = self.create_publisher(PoseStamped, "end_effector_pose", 10)
        self._ee_point_pub = self.create_publisher(PointStamped, "end_effector_point", 10)
        self._joint_command_pub = self.create_publisher(JointState, "joint_command", 10)
        self.create_subscription(JointState, "joint_states", self._cb_joint_states, 10)
        self.create_service(GoToTarget, "go_to_target", self._cb_go_to_target)
        self.create_subscription(PointStamped, "target_position", self._cb_target_position, 10)

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

    def _cb_draw_image(self, msg: String):
        if self._state not in (State.IDLE, State.DONE) or self._ee_position is None:
            self.get_logger().warn(f"Cannot draw now (state {self._state.value})")
            return
        try:
            strokes = image_to_strokes(msg.data, self._draw_center, self._draw_size)
        except FileNotFoundError as e:
            self.get_logger().error(str(e))
            return
        cx, cy, half = *self._draw_center, self._draw_size / 2
        checks = [(f"Paper corner {np.round(c, 3)}", self._pen_target(c))
                  for c in ((cx - half, cy - half), (cx - half, cy + half),
                            (cx + half, cy - half), (cx + half, cy + half))]
        checks += [("Pen holder (grasp)", self._holder_target(self._pen_length)),
                   ("Pen holder (above)", self._holder_target(self._above_pen_z()))]
        for name, target in checks:
            _, error = self._solve_ik(target, down=True)
            if error > IK_TOLERANCE:
                self.get_logger().error(f"{name} unreachable, aborting")
                return
        self.get_logger().info(
            f"Drawing {msg.data}: {len(strokes)} strokes, {sum(map(len, strokes))} points"
        )
        self._home_q = self._q.copy()
        self._strokes, self._stroke_i = strokes, 0
        self._ink, self._last_ink = [], None
        self._publish_lines(self._plan_pub, [p for s in strokes for p in zip(s[:-1], s[1:])],
                            ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.6))
        if self._pen_held:
            self._start_stroke()
        else:
            self._move(State.PEN_FETCH, self._holder_target(self._above_pen_z()), GRIPPER_OPEN)

    def _go_to(self, point, frame):
        if self._state not in (State.IDLE, State.DONE):
            return False, f"Busy: drawing ({self._state.value})"
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
        if self._ee_position is not None:
            self._publish_pen()

        if self._state in (State.PEN_TRAVEL, State.PEN_DOWN, State.TRACE, State.PEN_UP):
            self._step_drawing(elapsed)

        elif self._state.value.startswith("PEN_"):
            self._step_pen_handling(elapsed)

        elif self._state == State.RETURN:
            if self._arrived(elapsed):
                self._set_state(State.DONE)

    # ------------------------------------------------------- pen handling

    def _holder_target(self, z):
        return np.array([*self._pen_holder_xy, z])

    def _above_pen_z(self):
        """Gripper frame height with the jaw tips just above the pen standing in the holder."""
        return self._pen_total + PEN_ABOVE_MARGIN

    def _clear_z(self):
        """Gripper frame height with the held pen tip just above the holder."""
        return self._pen_length + self._holder_height + HOLDER_CLEARANCE

    def _step_pen_handling(self, elapsed):
        """Take the pen out of the holder (FETCH..EXTRACT) or put it back (STOW..LEAVE).
        Vertical moves only, right above the holder."""
        state = self._state
        if state == State.PEN_FETCH and self._arrived(elapsed):
            self._move(State.PEN_REACH, self._holder_target(self._pen_length), GRIPPER_OPEN)
        elif state == State.PEN_REACH and self._arrived(elapsed, GRASP_TOLERANCE):
            self._send_joint_command(self._q, GRIPPER_CLOSED)
            self._pen_held = True
            self._set_state(State.PEN_GRIP)
        elif state == State.PEN_GRIP and elapsed > GRIPPER_WAIT:
            self._move(State.PEN_EXTRACT, self._holder_target(self._clear_z()), GRIPPER_CLOSED)
        elif state == State.PEN_EXTRACT and self._arrived(elapsed):
            self._start_stroke()
        elif state == State.PEN_STOW and self._arrived(elapsed):
            self._move(State.PEN_INSERT, self._holder_target(self._pen_length + INSERT_GAP),
                       GRIPPER_CLOSED)
        elif state == State.PEN_INSERT and self._arrived(elapsed, GRASP_TOLERANCE):
            self._send_joint_command(self._q, GRIPPER_OPEN)
            self._pen_held = False
            self._set_state(State.PEN_RELEASE)
        elif state == State.PEN_RELEASE and elapsed > GRIPPER_WAIT:
            self._move(State.PEN_LEAVE, self._holder_target(self._above_pen_z()), GRIPPER_OPEN)
        elif state == State.PEN_LEAVE and self._arrived(elapsed):
            self._go_home(GRIPPER_OPEN)

    def _publish_pen(self):
        """Pen in RViz: in the gripper when held (FK), standing in the holder otherwise."""
        msg = Marker()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.type = Marker.CYLINDER
        msg.scale.x = msg.scale.y = 0.011
        msg.scale.z = self._pen_total
        msg.color = ColorRGBA(r=1.0, g=0.55, b=0.0, a=1.0)
        if self._pen_held:
            ee = self._forward_kinematics(self._q)
            center = ee.translation + ee.rotation[:, 2] * (self._pen_length - self._pen_total / 2)
            quat = pin.Quaternion(ee.rotation)
            msg.pose.orientation.x, msg.pose.orientation.y = quat.x, quat.y
            msg.pose.orientation.z, msg.pose.orientation.w = quat.z, quat.w
        else:
            center = self._holder_target(self._pen_total / 2)
            msg.pose.orientation.w = 1.0
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, center)
        self._pen_pub.publish(msg)

    # -------------------------------------------------------------- drawing

    def _pen_target(self, xy, lift=0.0):
        """Gripper frame target putting the pen tip at xy on the paper (pen vertical)."""
        return np.array([xy[0], xy[1], self._paper_z + self._pen_length + lift])

    def _start_stroke(self):
        if self._stroke_i >= len(self._strokes):
            self.get_logger().info("Drawing done")
            self._move(State.PEN_STOW, self._holder_target(self._clear_z()), GRIPPER_CLOSED)
            return
        self._point_i = 0
        start = self._strokes[self._stroke_i][0]
        self._move(State.PEN_TRAVEL, self._pen_target(start, self._pen_lift), GRIPPER_CLOSED)

    def _step_drawing(self, elapsed):
        self._record_ink()
        stroke = self._strokes[self._stroke_i]
        if self._state == State.PEN_TRAVEL:
            if self._arrived(elapsed):
                self._move(State.PEN_DOWN, self._pen_target(stroke[0]), GRIPPER_CLOSED)
        elif self._state == State.PEN_DOWN:
            if self._arrived(elapsed, self._trace_tolerance):
                self._set_state(State.TRACE)
        elif self._state == State.TRACE:
            if not self._arrived(elapsed, self._trace_tolerance):
                return
            self._point_i += 1
            if self._point_i >= len(stroke):
                self._move(State.PEN_UP, self._pen_target(stroke[-1], self._pen_lift), GRIPPER_CLOSED)
                return
            target = self._pen_target(stroke[self._point_i])
            q, error = self._solve_ik(target, down=True)
            if error > IK_TOLERANCE:
                self.get_logger().warn(f"Skipping unreachable point {np.round(target, 3)}")
                return
            self._target = target
            self._send_joint_command(q, GRIPPER_CLOSED)
            self._state_start = time.monotonic()  # timeout per point, not per stroke
        elif self._state == State.PEN_UP:
            if self._arrived(elapsed):
                self._stroke_i += 1
                self._start_stroke()

    def _record_ink(self):
        ee = self._forward_kinematics(self._q)
        tip = ee.translation + ee.rotation[:, 2] * self._pen_length
        if tip[2] > self._paper_z + INK_HEIGHT:
            self._last_ink = None
            return
        if self._last_ink is not None and np.linalg.norm(tip - self._last_ink) > 1e-3:
            self._ink.append((self._last_ink, tip))
            self._publish_lines(self._ink_pub, self._ink, ColorRGBA(r=0.05, g=0.05, b=0.05, a=1.0))
        if self._last_ink is None or np.linalg.norm(tip - self._last_ink) > 1e-3:
            self._last_ink = tip

    def _publish_lines(self, pub, segments, color):
        msg = Marker()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.type = Marker.LINE_LIST
        msg.scale.x = 0.002
        msg.color = color
        msg.pose.orientation.w = 1.0
        z = self._paper_z + 0.0005
        for a, b in segments:
            for p in (a, b):
                msg.points.append(Point(x=float(p[0]), y=float(p[1]), z=z))
        pub.publish(msg)

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
            self._go_home(gripper_pct)
            return
        tilt = np.degrees(np.arccos(np.clip(self._forward_kinematics(q).rotation[:, 2] @ DOWN, -1, 1)))
        self.get_logger().info(f"{state.value}: target {np.round(target, 3)}, gripper tilt {tilt:.0f} deg")
        self._target = target
        self._send_joint_command(q, gripper_pct)
        self._set_state(state)

    def _go_home(self, gripper_pct=GRIPPER_OPEN):
        self._target = self._forward_kinematics(self._home_q).translation.copy()
        self._send_joint_command(self._home_q, gripper_pct)
        self._set_state(State.RETURN)

    def _arrived(self, elapsed, tolerance=REACHED_TOLERANCE):
        if np.linalg.norm(self._ee_position - self._target) < tolerance:
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

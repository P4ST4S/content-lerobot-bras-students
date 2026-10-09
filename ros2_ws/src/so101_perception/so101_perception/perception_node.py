#!/usr/bin/env python3
"""perception_node.py — red cube and drop box detection, 3D in the robot frame.

Subscribes to /external_cam/image_raw and /external_cam/camera_info, reads the
camera pose from TF (world -> external_cam, OpenCV optical convention) and
publishes /object_position and /drop_box_position (PoseStamped, world frame).
"""
import cv2
import numpy as np
import rclpy
import rclpy.node
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

WORLD_FRAME = "world"

RED_RANGES = [((0, 120, 80), (8, 255, 255)), ((160, 120, 80), (180, 255, 255))]
BOX_RANGES = [((5, 80, 30), (20, 220, 160))]

OBJECT_HEIGHT = 0.025
OBJECT_OFFSET = (0.0, 0.0)
OBJECT_TOP = 0.05
OBJECT_FAR_EDGE_OFFSET = 0.03
BOX_HEIGHT = 0.025
OBJECT_MIN_PIXELS = 20
BOX_MIN_PIXELS = 400


def _quat_to_matrix(x, y, z, w):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class PerceptionNode(rclpy.node.Node):
    """Detects the red cube and the drop box and publishes their 3D positions."""

    def __init__(self):
        super().__init__("so101_perception")
        self._bridge = CvBridge()
        self._k_inv = None
        self._camera_frame = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._object_pub = self.create_publisher(PoseStamped, "object_position", 10)
        self._box_pub = self.create_publisher(PoseStamped, "drop_box_position", 10)
        self.create_subscription(CameraInfo, "external_cam/camera_info", self._cb_camera_info, 10)
        self.create_subscription(Image, "external_cam/image_raw", self._cb_image, 10)

        self.get_logger().info("Perception node ready.")

    def _cb_camera_info(self, msg: CameraInfo):
        if msg.k[0] == 0.0:
            self.get_logger().warn("Camera not calibrated yet, waiting", throttle_duration_sec=10.0)
            return
        self._k_inv = np.linalg.inv(np.array(msg.k).reshape(3, 3))
        self._camera_frame = msg.header.frame_id

    def _cb_image(self, msg: Image):
        camera_pose = self._fetch_cam_params()
        if camera_pose is None:
            return
        hsv = cv2.cvtColor(self._bridge.imgmsg_to_cv2(msg, "rgb8"), cv2.COLOR_RGB2HSV)

        blob = self._detect_object(hsv, RED_RANGES, OBJECT_MIN_PIXELS)
        if blob is not None:
            centroid, far_edge, cut = blob
            if cut:
                point = self._project_to_3d(far_edge, camera_pose, OBJECT_TOP)
                radial = point[:2] - camera_pose[1][:2]
                point[:2] -= OBJECT_FAR_EDGE_OFFSET * radial / np.linalg.norm(radial)
                point[2] = OBJECT_HEIGHT
            else:
                point = self._project_to_3d(centroid, camera_pose, OBJECT_HEIGHT)
            point[:2] += OBJECT_OFFSET
            self._publish_result(self._object_pub, point, msg.header.stamp)

        blob = self._detect_object(hsv, BOX_RANGES, BOX_MIN_PIXELS)
        if blob is not None:
            point = self._project_to_3d(blob[0], camera_pose, BOX_HEIGHT)
            self._publish_result(self._box_pub, point, msg.header.stamp)

    def _detect_object(self, hsv, ranges, min_pixels):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv, low, high)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if count < 2:
            return None
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h, area = stats[largest]
        if area < min_pixels:
            return None
        height, width = mask.shape
        cut = x <= 0 or y <= 0 or x + w >= width - 1 or y + h >= height - 1
        rows, cols = np.nonzero(labels == largest)
        far_edge = (cols[rows <= rows.min() + 1].mean(), rows.min())
        return centroids[largest], far_edge, cut

    def _project_to_3d(self, pixel, camera_pose, height):
        rotation, origin = camera_pose
        direction = rotation @ (self._k_inv @ np.array([pixel[0], pixel[1], 1.0]))
        return origin + (height - origin[2]) / direction[2] * direction

    def _publish_result(self, publisher, point, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = WORLD_FRAME
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = point.tolist()
        msg.pose.orientation.w = 1.0
        publisher.publish(msg)

    def _fetch_cam_params(self):
        if self._k_inv is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(WORLD_FRAME, self._camera_frame, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f"No camera pose yet: {e}", throttle_duration_sec=5.0)
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return _quat_to_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
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

"""Host-side webcam tools (macOS + Colima: the containers cannot see the USB webcam).

    python webcam.py preview --camera N       # find the webcam index (Q to quit)
    python webcam.py bridge --camera N        # stream frames to the control container (TCP)
    python webcam.py from-ros ost.yaml        # import the ROS camera_calibration result (K, distortion)
    python webcam.py extrinsics --camera N    # ArUco on the table -> camera pose in the robot frame

Calibration is saved to ros2_ws/calibration/camera.json (mounted in the containers).
"""
import argparse
import json
import socket
import struct
from pathlib import Path

import cv2
import numpy as np
import yaml

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "ros2_ws" / "calibration" / "camera.json"
WIDTH, HEIGHT = 640, 480

MARKER_SIZE = 0.10
MARKER_POSITION = (0.30, 0.0, 0.0)
ARUCO_DICTS = ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_ARUCO_ORIGINAL"]


def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {index} (macOS: allow camera access for the terminal)")
    return cap


def read(cap):
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera read failed")
    return cv2.resize(frame, (WIDTH, HEIGHT)) if frame.shape[1::-1] != (WIDTH, HEIGHT) else frame


def load_calibration():
    return json.loads(CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.is_file() else {}


def save_calibration(data):
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2))
    print(f"Saved {CALIBRATION_FILE}")


def cmd_preview(args):
    cap = open_camera(args.camera)
    print(f"Camera {args.camera}: click the window, then Q/ESC to quit")
    while True:
        cv2.imshow(f"camera {args.camera}", read(cap))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return


def cmd_bridge(args):
    cap = open_camera(args.camera)
    server = socket.create_server(("0.0.0.0", args.port), reuse_port=True)
    print(f"Streaming camera {args.camera} on tcp :{args.port}")
    while True:
        conn, addr = server.accept()
        print(f"Client connected: {addr}")
        try:
            while True:
                _, jpeg = cv2.imencode(".jpg", read(cap), [cv2.IMWRITE_JPEG_QUALITY, 85])
                conn.sendall(struct.pack(">I", len(jpeg)) + jpeg.tobytes())
        except OSError:
            print("Client disconnected")
            conn.close()


def cmd_from_ros(args):
    ost = yaml.safe_load(Path(args.file).read_text())
    data = load_calibration()
    data.update(
        width=ost["image_width"],
        height=ost["image_height"],
        K=np.array(ost["camera_matrix"]["data"]).reshape(3, 3).tolist(),
        dist=ost["distortion_coefficients"]["data"],
    )
    save_calibration(data)


def detect_marker(gray):
    for name in ARUCO_DICTS:
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)))
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            return name, corners[0].reshape(4, 2), int(ids[0][0])
    return None


def cmd_extrinsics(args):
    calib = load_calibration()
    if "K" not in calib:
        raise SystemExit("No intrinsics: run the ROS calibration then 'from-ros' first")
    k, dist = np.array(calib["K"]), np.array(calib["dist"])
    half = MARKER_SIZE / 2
    marker_corners = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float32)
    yaw = np.radians(args.yaw)
    world_marker = np.eye(4)
    world_marker[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    world_marker[:3, 3] = MARKER_POSITION
    cap = open_camera(args.camera)
    print("Robot axes drawn at the robot base: red = x (arm forward), green = y, blue = z (up).")
    print("If red does not point along the arm, rerun with --yaw 90 / 180 / -90.")
    print("Click the window, then ENTER: save, Q/ESC: quit")
    world_camera = None
    while True:
        frame = read(cap)
        found = detect_marker(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if found:
            name, corners, marker_id = found
            _, rvec, tvec = cv2.solvePnP(marker_corners, corners, k, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            camera_marker = np.eye(4)
            camera_marker[:3, :3] = cv2.Rodrigues(rvec)[0]
            camera_marker[:3, 3] = tvec.ravel()
            world_camera = world_marker @ np.linalg.inv(camera_marker)
            camera_world = np.linalg.inv(world_camera)
            axes = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], np.float32)
            pts, _ = cv2.projectPoints(axes, cv2.Rodrigues(camera_world[:3, :3])[0], camera_world[:3, 3], k, dist)
            pts = pts.reshape(-1, 2).astype(int)
            for end, color in zip(pts[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                cv2.line(frame, tuple(pts[0]), tuple(end), color, 3)
            cam = world_camera[:3, 3]
            cv2.putText(frame, f"{name} id={marker_id}  camera at ({cam[0]:.2f}, {cam[1]:.2f}, {cam[2]:.2f}) m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "marker NOT found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("extrinsics", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (10, 13) and world_camera is not None:
            break
        if key in (ord("q"), 27):
            return
    calib["T_world_camera"] = world_camera.tolist()
    save_calibration(calib)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, func in (("preview", cmd_preview), ("bridge", cmd_bridge), ("extrinsics", cmd_extrinsics)):
        p = sub.add_parser(name)
        p.add_argument("--camera", type=int, default=0)
        p.set_defaults(func=func)
        if name == "bridge":
            p.add_argument("--port", type=int, default=5835)
        if name == "extrinsics":
            p.add_argument("--yaw", type=float, default=0.0, help="marker rotation around z, degrees")
    p = sub.add_parser("from-ros")
    p.add_argument("file")
    p.set_defaults(func=cmd_from_ros)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

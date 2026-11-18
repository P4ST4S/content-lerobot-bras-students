#!/usr/bin/env python3
"""Demo web page: drop an image, the arm draws it, the RViz 3D view is shown in the page.

Runs in the rviz container (rclpy, ros2_ws mounted, noVNC files in /usr/share/novnc):
    python3 /ros2_ws/web/server.py             # http://localhost:8080
    python3 /ros2_ws/web/server.py --selftest

POST /draw               body = raw PNG/JPEG -> saved as images/web_upload.<ext>, published
POST /draw?sample=X.png  X must already be in images/ -> published
POST /plan?show=1|0      show/hide the grey planned strokes (/drawing/show_plan)
Publishes the image path on /draw_image, like docker/draw.sh. No auth: local demo only.
"""
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parent / "images"
NOVNC = Path("/usr/share/novnc")
PORT = 8080
MAX_BYTES = 10 * 1024 * 1024
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
         ".png": "image/png", ".svg": "image/svg+xml"}


def image_ext(data: bytes):
    """Extension from the magic bytes, None if not a PNG/JPEG."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return None


def sample_path(name: str, images: Path = IMAGES):
    """Path of an image already in images/, None for anything else (no free paths)."""
    files = {p.name for p in images.iterdir() if p.is_file()} if images.is_dir() else set()
    return images / name if name in files else None


def static_path(url_path: str, root: Path = NOVNC):
    """File under root for /novnc/<rel>, None if it escapes root or does not exist."""
    path = (root / url_path.removeprefix("/novnc/")).resolve()
    return path if path.is_file() and path.is_relative_to(root.resolve()) else None


class Handler(BaseHTTPRequestHandler):
    publish = None    # set in main()
    show_plan = None  # set in main()

    def _reply(self, code, body, ctype="text/plain; charset=utf-8"):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path).path
        if url == "/":
            return self._reply(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        path = static_path(url) if url.startswith("/novnc/") else None
        if path is None:
            return self._reply(404, "Not found")
        self._reply(200, path.read_bytes(), TYPES.get(path.suffix, "application/octet-stream"))

    def do_POST(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/plan":
            show = query.get("show", ["0"])[0] == "1"
            Handler.show_plan(show)
            return self._reply(200, "Plan shown" if show else "Plan hidden")
        if url.path != "/draw":
            return self._reply(404, "Not found")
        sample = query.get("sample", [None])[0]
        if sample is not None:
            path = sample_path(sample)
            if path is None:
                return self._reply(400, f"Unknown sample image: {sample}")
        else:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BYTES:
                return self._reply(400, "Send one PNG or JPEG image, 10 MB max")
            data = self.rfile.read(length)
            ext = image_ext(data)
            if ext is None:
                return self._reply(400, "Not a PNG or JPEG image")
            path = IMAGES / f"web_upload{ext}"
            path.write_bytes(data)
        Handler.publish(f"/ros2_ws/images/{path.name}")
        self._reply(200, f"Sent {path.name} to the arm")

    def log_message(self, *_):
        pass


def selftest():
    import tempfile
    assert image_ext(b"\x89PNG\r\n\x1a\n...") == ".png"
    assert image_ext(b"\xff\xd8\xff\xe0...") == ".jpg"
    assert image_ext(b"<html>") is None and image_ext(b"") is None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "star.png").write_bytes(b"x")
        (tmp / "sub").mkdir()
        (tmp / "sub" / "a.js").write_bytes(b"x")
        assert sample_path("star.png", tmp) == tmp / "star.png"
        assert sample_path("../star.png", tmp) is None and sample_path("sub", tmp) is None
        assert static_path("/novnc/sub/a.js", tmp) == (tmp / "sub" / "a.js").resolve()
        assert static_path("/novnc/../../etc/passwd", tmp) is None
        assert static_path("/novnc/sub", tmp) is None
    print("selftest OK")


def main():
    import rclpy
    from std_msgs.msg import Bool, String

    rclpy.init()
    node = rclpy.create_node("draw_web")
    pub = node.create_publisher(String, "draw_image", 10)
    plan_pub = node.create_publisher(Bool, "drawing/show_plan", 10)
    Handler.publish = lambda path: pub.publish(String(data=path))
    Handler.show_plan = lambda show: plan_pub.publish(Bool(data=show))
    print(f"Draw page on http://localhost:{PORT}", flush=True)
    try:
        ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()

"""image_to_strokes.py - turn an image file into pen strokes on the paper (robot frame).

Pure numpy + OpenCV, no ROS: dark shapes on light background -> their outlines,
as a list of (N, 2) arrays of XY points in meters, ordered to limit pen travel.

    python3 image_to_strokes.py              # self-check
    python3 image_to_strokes.py drawing.png  # print stroke stats
"""
import sys

import cv2
import numpy as np

WORK_PX = 400          # image resized so its longest side is this many pixels
SIMPLIFY_PX = 1.5      # approxPolyDP tolerance, in pixels
MIN_LENGTH_PX = 20     # outlines shorter than this are noise
STEP_M = 0.004         # spacing between consecutive points sent to the arm


def load_contours(image):
    """Grayscale image -> list of (N, 2) pixel outlines (closed loops)."""
    scale = WORK_PX / max(image.shape)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # ponytail: Otsu on dark-on-light drawings; swap for Canny if photos are needed
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    outlines = []
    for c in contours:
        if cv2.arcLength(c, True) < MIN_LENGTH_PX:
            continue
        c = cv2.approxPolyDP(c, SIMPLIFY_PX, True)[:, 0, :].astype(float)
        outlines.append(np.vstack([c, c[:1]]))  # close the loop
    return outlines, image.shape


def to_paper(outlines, shape, center, size):
    """Pixels -> meters in the robot frame, fitted in a size x size square around center.

    Image up (row 0) = +X (away from the robot), image right = -Y (robot's right).
    """
    h, w = shape
    scale = size / max(h, w)
    cx, cy = center
    return [np.column_stack([cx + (h / 2 - o[:, 1]) * scale, cy - (o[:, 0] - w / 2) * scale])
            for o in outlines]


def resample(stroke, step=STEP_M):
    """Evenly spaced points along a polyline, keeping its corners."""
    out = [stroke[0]]
    for a, b in zip(stroke[:-1], stroke[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.array(out)


def order(strokes, start=(0.0, 0.0)):
    """Greedy nearest neighbour on stroke start points; closed loops are rotated to begin
    at their point nearest to the pen."""
    # ponytail: greedy O(n^2), fine for a few hundred strokes
    remaining, ordered, pen = list(strokes), [], np.asarray(start)
    while remaining:
        best = min(range(len(remaining)),
                   key=lambda i: np.min(np.linalg.norm(remaining[i] - pen, axis=1)))
        s = remaining.pop(best)
        k = int(np.argmin(np.linalg.norm(s - pen, axis=1)))
        s = np.vstack([s[k:-1], s[:k + 1]])  # rotate the closed loop to start at k
        ordered.append(s)
        pen = s[-1]
    return ordered


def read_gray(path):
    """Image file -> grayscale. A transparent background becomes white: icon PNGs store it
    as transparent black, which would otherwise turn the whole image into one dark blob."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image '{path}'")
    if image.ndim == 2:
        return image
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        alpha = image[:, :, 3].astype(float) / np.iinfo(image.dtype).max
        white = np.iinfo(image.dtype).max
        gray = (gray * alpha + white * (1 - alpha)).astype(image.dtype)
    return gray


def image_to_strokes(path_or_image, center=(0.22, 0.0), size=0.10):
    image = read_gray(path_or_image) if isinstance(path_or_image, str) else path_or_image
    outlines, shape = load_contours(image)
    strokes = [resample(s) for s in to_paper(outlines, shape, center, size)]
    return order(strokes, start=center)


def _self_check():
    img = np.full((300, 300), 255, np.uint8)
    cv2.circle(img, (80, 80), 40, 0, -1)
    cv2.rectangle(img, (160, 160), (260, 260), 0, -1)
    center, size = (0.22, 0.0), 0.10
    strokes = image_to_strokes(img, center, size)
    assert len(strokes) == 2, len(strokes)
    pts = np.vstack(strokes)
    assert np.all(np.abs(pts - center) <= size / 2 + 1e-9), "stroke outside paper"
    for s in strokes:
        assert np.allclose(s[0], s[-1], atol=STEP_M), "loop not closed"
        assert np.max(np.linalg.norm(np.diff(s, axis=0), axis=1)) <= STEP_M + 1e-9
    # circle is top-left in the image -> +X (far) and +Y (robot's left)
    circle = min(strokes, key=lambda s: np.ptp(s[:, 0]))  # 80 px wide vs 100 px square
    assert circle[:, 0].mean() > center[0] and circle[:, 1].mean() > center[1]
    # transparent PNG (icon sites): background is transparent black, the disc must still win
    import tempfile, os
    rgba = np.zeros((200, 200, 4), np.uint8)
    cv2.circle(rgba, (100, 100), 60, (0, 0, 0, 255), -1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "icon.png")
        cv2.imwrite(path, rgba)
        disc = np.vstack(image_to_strokes(path, center, size))
    assert abs(np.ptp(disc[:, 0]) - size * 120 / 200) < 0.003, "transparent background drawn"
    print(f"OK: {len(strokes)} strokes, {len(pts)} points, transparent PNG OK")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        strokes = image_to_strokes(sys.argv[1])
        print(f"{len(strokes)} strokes, {sum(map(len, strokes))} points")
    else:
        _self_check()

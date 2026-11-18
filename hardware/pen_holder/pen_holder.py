"""Pen holder (phase 2, variant A): square base, central hole, 12 flexible fingers.

Generates the STL used both for 3D printing and as the RViz mesh:
    uv run --with manifold3d --with numpy hardware/pen_holder/pen_holder.py
Writes pen_holder.stl here and into ros2_ws/src/so101_description/meshes/, plus
pen_holder_print.stl: same part flipped for printing (send this one to the slicer).

Geometry (mm, origin = center of the bottom face, z up as it sits on the table):
  - through bore GUIDE_D keeps the pen vertical, the pen tip rests on the table;
  - top plate PLATE thick, cut into N_FINGERS tongues around a HOLE_D hole: a pen wider
    than HOLE_D bends the tongues down and they grip it;
  - under the plate a 45 deg cone leaves the tongues room to bend and makes the part
    printable without supports, upside down (plate on the bed).
"""
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold

BASE = 60.0        # base side
HEIGHT = 30.0      # total height
CORNER_R = 4.0
GUIDE_D = 12.0     # sized for Sharpie-type markers (~10-11.5 mm): little play, pen stays vertical
HOLE_D = 8.0       # finger tips: pens from ~8.5 mm up get gripped
CAVITY_D = 36.0    # finger length = (CAVITY_D - HOLE_D) / 2
PLATE = 0.8        # finger thickness: 2 layers of 0.4. Thicker = stiffer, but PLA cracks
N_FINGERS = 12
SLOT = 1.5         # gap between fingers
SEGMENTS = 96

HERE = Path(__file__).resolve().parent
OUTPUTS = [HERE / "pen_holder.stl",
           HERE.parents[1] / "ros2_ws/src/so101_description/meshes/pen_holder.stl"]


def build():
    body = (CrossSection.square((BASE - 2 * CORNER_R,) * 2, center=True)
            .offset(CORNER_R, JoinType.Round, circular_segments=SEGMENTS)
            .extrude(HEIGHT))
    guide = Manifold.cylinder(HEIGHT + 2, GUIDE_D / 2, circular_segments=SEGMENTS).translate((0, 0, -1))
    cone_h = (CAVITY_D - GUIDE_D) / 2  # 45 deg
    cone = Manifold.cylinder(cone_h, GUIDE_D / 2, CAVITY_D / 2,
                             circular_segments=SEGMENTS).translate((0, 0, HEIGHT - PLATE - cone_h))
    plate_cut = Manifold.cylinder(PLATE + 2, HOLE_D / 2, circular_segments=SEGMENTS)
    slot = Manifold.cube((CAVITY_D / 2 + 0.5, SLOT, PLATE + 2)).translate((0, -SLOT / 2, 0))
    for i in range(N_FINGERS):
        plate_cut += slot.rotate((0, 0, i * 360 / N_FINGERS + 180 / N_FINGERS))
    plate_cut = plate_cut.translate((0, 0, HEIGHT - PLATE - 1))
    return body - guide - cone - plate_cut


def write_stl(manifold, path):
    mesh = manifold.to_mesh()
    tri = np.asarray(mesh.vert_properties)[:, :3][np.asarray(mesh.tri_verts)].astype(np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    records = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    records["n"], records["v"] = normals, tri
    with open(path, "wb") as f:
        f.write(b"pen_holder".ljust(80, b" "))
        f.write(np.uint32(len(tri)).tobytes())
        f.write(records.tobytes())


def check(part):
    lo, hi = part.bounding_box()[:3], part.bounding_box()[3:]
    assert np.allclose(np.subtract(hi, lo), (BASE, BASE, HEIGHT), atol=0.01), (lo, hi)
    assert len(part.decompose()) == 1, "part must be one piece"
    probe = lambda x, y, z: (part ^ Manifold.cube((0.2, 0.2, 0.2), center=True).translate((x, y, z))).volume() > 0
    top = HEIGHT - PLATE / 2
    assert not probe(0, 0, top) and not probe(0, 0, HEIGHT / 2), "pen hole must be open"
    finger_r = (HOLE_D / 2 + CAVITY_D / 2) / 2
    assert probe(finger_r, 0, top), "finger missing"
    a = np.radians(180 / N_FINGERS)
    assert not probe(finger_r * np.cos(a), finger_r * np.sin(a), top), "slot missing"
    assert not probe(finger_r, 0, top - PLATE - 0.5), "no room under the fingers"


if __name__ == "__main__":
    part = build()
    check(part)
    for out in OUTPUTS:
        write_stl(part, out)
        print(f"{out}  ({part.num_tri()} triangles, {part.volume() / 1000:.1f} cm3)")
    # Print orientation: upside down, finger plate on the bed, no supports needed.
    printable = part.rotate((180, 0, 0)).translate((0, 0, HEIGHT))
    assert abs(printable.bounding_box()[2]) < 1e-6
    write_stl(printable, HERE / "pen_holder_print.stl")
    print(HERE / "pen_holder_print.stl")

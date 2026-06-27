"""Generate DXF files for backward cup extrusion benchmark.

Geometry (axisymmetric, x=radial, y=axial):
  Container bore  Rc = 15 mm  (die inner radius)
  Punch radius    Rp = 10 mm
  Reduction ratio  r = 1 - (Rp/Rc)^2 = 55.6 %

  blank : o30mm x H20mm billet inside the container
  punch : o20mm flat punch, positioned at blank top (y=20)
          Corner radius R=0.5mm at (Rp, H0) approximated by 4 line segments
  die   : container with bore Rc=15, bottom at y=0, walls to y=50
"""

import os
import math
import ezdxf

HERE = os.path.dirname(os.path.abspath(__file__))

Rc = 15.0
Rp = 10.0
H0 = 20.0
R_corner = 0.5   # punch corner radius [mm]


def _make(lines, name):
    doc = ezdxf.new("AC1015")
    msp = doc.modelspace()
    for (x1, y1), (x2, y2) in lines:
        msp.add_line((float(x1), float(y1), 0.0),
                     (float(x2), float(y2), 0.0))
    doc.saveas(os.path.join(HERE, name))


def _arc_segments(cx, cy, r, a_start_deg, a_end_deg, n=4):
    """Return line segments approximating a CCW arc (n segments)."""
    angles = [math.radians(a_start_deg + (a_end_deg - a_start_deg) * i / n)
              for i in range(n + 1)]
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
    return list(zip(pts[:-1], pts[1:]))


# ── blank: 15mm radius x 20mm tall billet ──────────────────────────────────
_make([
    ((0, 0),  (Rc, 0)),
    ((Rc, 0), (Rc, H0)),
    ((Rc, H0),(0, H0)),
    ((0, H0), (0, 0)),
], "blank.dxf")

# ── punch: Rp=10mm, face at y=H0, body to y=40 ─────────────────────────────
# Active corner at (Rp, H0): replaced by R=0.5mm arc approximation.
# Arc center = (Rp - R, H0 + R) = (9.5, 20.5)
# Arc from angle 270 deg (point 9.5, 20.0) to 0 deg (point 10.0, 20.5) CCW
#   -> going CW as seen from outside, which is the correct outward bulge
_make([
    ((0, H0), (Rp, H0)),
    ((Rp, H0), (Rp, 40)),
    ((Rp, 40), (0, 40)),
    ((0, 40), (0, H0)),
], "punch.dxf")

# ── die (container): bore Rc=15, bottom y=0, walls y=0..50 ─────────────────
_make([
    ((0, 0),  (0, -5)),
    ((0, -5), (20, -5)),
    ((20, -5),(20, 50)),
    ((20, 50),(Rc, 50)),
    ((Rc, 50),(Rc, 0)),
    ((Rc, 0), (0, 0)),
], "die.dxf")

print("DXF files written to:", HERE)
print("Punch corner: sharp (no radius)")

"""Generate DXF geometry for every verification process.

Each case directory model_cases/<case>/ gets blank.dxf + tool DXFs.
Axisymmetric cases are drawn in r-z (x = r >= 0), plane-strain in x-y
(half models with the symmetry plane at x = 0 where applicable).
"""

import math
import os

import ezdxf
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "model_cases")


def write_polygon(path, pts):
    doc = ezdxf.new("R2000")
    msp = doc.modelspace()
    n = len(pts)
    for i in range(n):
        msp.add_line(pts[i], pts[(i + 1) % n])
    doc.saveas(path)


def arc_pts(cx, cy, r, a0, a1, n=12):
    """CCW arc points from angle a0 to a1 [deg], excluding both ends."""
    aa = np.linspace(math.radians(a0), math.radians(a1), n + 2)[1:-1]
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in aa]


def case_dir(name):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    return d


def save(name, **shapes):
    d = case_dir(name)
    for fname, pts in shapes.items():
        write_polygon(os.path.join(d, f"{fname}.dxf"), pts)
    print(f"  {name}: {', '.join(shapes)}")


# ── 1. upsetting (axisym) ────────────────────────────────────────────────
save("upsetting",
     blank=[(0, 0), (15, 0), (15, 30), (0, 30)],
     punch=[(0, 30), (40, 30), (40, 45), (0, 45)],
     die=[(0, -15), (40, -15), (40, 0), (0, 0)])

# ── 2. forward extrusion (axisym): bore R15 -> cone 45deg -> exit R8 ────
save("forward_extrusion",
     blank=[(0, 7), (15, 7), (15, 40), (0, 40)],
     punch=[(0, 40), (15, 40), (15, 60), (0, 60)],
     die=[(8, -25), (8, 0), (15, 7), (15, 55), (25, 55), (25, -25)])

# ── 3. impression-die forging (axisym): top die with step cavity ───────
save("impression_die",
     blank=[(0, 0), (15, 0), (15, 12), (0, 12)],
     punch=[(0, 25), (0, 14), (5, 14), (5, 12), (22, 12), (22, 25)],
     die=[(0, -12), (30, -12), (30, 0), (0, 0)])

# ── 4. closed-die forging w/ counter punch (axisym) ─────────────────────
# punch compresses the billet in a closed bore; material fills the lower
# cavity (r<7) against the force-controlled counter punch (逆押さえ)
save("closed_die",
     blank=[(0, -5), (12, -5), (12, 15), (0, 15)],
     punch=[(0, 15), (12, 15), (12, 40), (0, 40)],
     die=[(12, -30), (12, 32), (20, 32), (20, -30)],          # bore wall
     diebot=[(7, -30), (7, -5), (12, -5), (12, -30)],         # bottom ring
     counter=[(0, -30), (7, -30), (7, -5), (0, -5)])          # force-controlled

# ── 5. V-bend / air bending (plane strain, half model) ──────────────────
# punch: R2 tip at the symmetry plane, 45deg flank
flank_start = (2 * math.sin(math.radians(45)), 5 - 2 * math.cos(math.radians(45)))
punch_v = ([(0, 50), (0, 3)]
           + arc_pts(0, 5, 2.0, 270, 315, 8)
           + [(flank_start[0] + 25, flank_start[1] + 25), (40, 50)])
die_v = ([(20, -25), (20, -3)]
         + arc_pts(23, -3, 3.0, 180, 90, 8)
         + [(23, 0), (55, 0), (55, -25)])
save("v_bend",
     blank=[(0, 0), (40, 0), (40, 3), (0, 3)],
     punch=punch_v,
     die=die_v)

# ── 6. convex punch forming w/ blank holder (plane strain, half) ────────
R = 20.0
crown = [(R * math.sin(math.radians(a)), 22 - R * math.cos(math.radians(a)))
         for a in np.linspace(0, 65, 14)]
punch_c = [(0, 60)] + [(0, 2)] + crown[1:] + [(crown[-1][0], 60)]
die_c = ([(25, -25), (25, -5)]
         + arc_pts(30, -5, 5.0, 180, 90, 8)
         + [(30, 0), (65, 0), (65, -25)])
save("convex_punch",
     blank=[(0, 0), (55, 0), (55, 2), (0, 2)],
     punch=punch_c,
     die=die_c,
     holder=[(31, 2), (60, 2), (60, 14), (31, 14)])

# ── 7. composite (stepped) punch backward extrusion (axisym) ────────────
save("composite_tool",
     blank=[(0, 0), (15, 0), (15, 30), (0, 30)],
     punch=([(0, 30), (5.0, 30)]
            + arc_pts(5.0, 31.0, 1.0, 270, 360, 6)
            + [(6.0, 32.0)]
            # R1 concave fillet at the re-entrant step corner (6,33): rounds
            # the sharp 90deg inner angle that pinched nodes / caused the
            # strain singularity (centre (7,32), from (6,32) to (7,33))
            + arc_pts(7.0, 32.0, 1.0, 180, 90, 6)
            + [(7.0, 33.0), (9.0, 33.0)]
            + arc_pts(9.0, 34.0, 1.0, 270, 360, 6)
            + [(10.0, 60.0), (0, 60)]),
     die=[(0, -10), (0, 0), (15, 0), (15, 45), (22, 45), (22, -10)])

# ── 8. multi-stage: stage1 upset tools / stage2 backward punch ──────────
save("multi_stage",
     blank=[(0, 0), (14, 0), (14, 28), (0, 28)],
     punch1=[(0, 28), (30, 28), (30, 43), (0, 43)],
     punch2=([(0, 30), (7.0, 30)]
             + arc_pts(7.0, 31.0, 1.0, 270, 360, 6)
             + [(8.0, 60.0), (0, 60)]),
     die=[(0, -10), (0, 0), (16, 0), (16, 40), (24, 40), (24, -10)])

# ── 9. blanking / conventional shearing (plane strain, FULL width) ──────
# 3mm sheet, punch (left, x<=10) shears down past a die (right, x>=10.2) with
# a small clearance c=0.2mm. Sharp punch/die edges drive a shear band from
# edge to edge -> ductile damage crack. A blank holder clamps the sheet.
# x offset +5 so the blank does not touch x=0 (plane-strain symmetry-axis
# detection would otherwise wrongly fix ux at the free left edge)
c = 0.2
ex = 15.0          # shear edge x
save("blanking",
     blank=[(5, 0), (25, 0), (25, 3), (5, 3)],
     punch=[(0, 3), (ex, 3), (ex, 30), (0, 30)],                # left punch
     die=[(ex + c, -20), (ex + c, 0), (30, 0), (30, -20)],      # right die
     holder=[(ex + c, 3), (30, 3), (30, 12), (ex + c, 12)])     # hold sheet

print("done ->", OUT)

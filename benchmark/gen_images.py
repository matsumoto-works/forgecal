"""Generate all images for the Medium article benchmark.

Images produced:
  01_geometry_setup.png        -- schematic of DXF geometry
  02_mises_s1.png              -- von Mises at stroke 1 mm
  03_mises_s3.png              -- von Mises at stroke 3 mm
  04_mises_s5.png              -- von Mises at stroke 5 mm
  05_mises_s7.png              -- von Mises at stroke 7 mm
  06_mises_s10.png             -- von Mises at stroke 10 mm
  07_peeq_s5.png               -- PEEQ at stroke 5 mm
  08_peeq_s10.png              -- PEEQ at stroke 10 mm
  09_load_stroke.png           -- load-stroke curves
  10_friction_sensitivity.png  -- p/sigma0 vs m (main validation)

Run from PlasticFEM_v4 root:
  python benchmark/gen_images.py
"""
from __future__ import annotations
import os, sys
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as mtri
import numpy as np
from scipy.ndimage import uniform_filter1d

HERE   = os.path.dirname(os.path.abspath(__file__))
CASES  = os.path.join(os.path.dirname(HERE), "results", "cases")
OUT    = os.path.join(HERE, "img")
os.makedirs(OUT, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
Rc, Rp, H0 = 15.0, 10.0, 20.0
sig0_ref    = 950.0   # MPa
A_punch     = np.pi * Rp**2
M_VALS      = [0.05, 0.10, 0.15]
COLORS      = ["#2196F3", "#FF9800", "#F44336"]
TAGS        = ["bce_m05", "bce_m10", "bce_m15"]
LABELS      = ["m = 0.05", "m = 0.10", "m = 0.15"]

# ── helper: find step closest to target stroke ───────────────────────────────
def find_step(f, target_stroke):
    best, best_err = None, 1e9
    for key in f.keys():
        if not key.startswith("step_"):
            continue
        g = f[key]
        if "tools/0" not in g:
            continue
        pts = g["tools/0"][()]
        s = H0 - float(pts[0, 0, 1])
        err = abs(s - target_stroke)
        if err < best_err:
            best, best_err = key, err
    return best


# ── helper: mask elements inside punch (penalty contact micro-penetration) ───
def _punch_mask(coords, elems, stroke):
    """True for elements whose centroid is inside the punch region.

    Penalty contact allows a small penetration at the punch corner.
    These 1-2 boundary elements appear as a visual 'island' separated from
    the main mesh by the white tool outline — masking them removes the artifact.
    """
    punch_face_z = H0 - stroke
    centroids = coords[elems].mean(axis=1)
    inside = (centroids[:, 0] < Rp - 0.05) & (centroids[:, 1] > punch_face_z + 0.05)
    return inside


# ── helper: draw one contour frame ──────────────────────────────────────────
def draw_contour(h5path, target_stroke, field, vmin, vmax, ax,
                 title_suffix="", cmap="jet"):
    with h5py.File(h5path) as f:
        key = find_step(f, target_stroke)
        g   = f[key]
        coords = g["coords"][()]
        elems  = g["elems"][()]
        ef     = g[field][()]
        pts    = g["tools/0"][()]
        punch_segs = pts          # (M, 2, 2)
        # die segments
        die_segs = g["tools/1"][()] if "tools/1" in g else np.zeros((0,2,2))
        stroke = H0 - float(pts[0, 0, 1])

    # mask micro-penetrating elements at punch corner
    bad = _punch_mask(coords, elems, stroke)
    elems_clean = elems[~bad]
    ef_clean    = ef[~bad]

    # element → node average (using cleaned elements)
    nf  = np.zeros(len(coords))
    cnt = np.zeros(len(coords))
    for e, el in enumerate(elems_clean):
        nf[el] += ef_clean[e]; cnt[el] += 1
    nf /= np.maximum(cnt, 1)
    nf = np.clip(nf, vmin, vmax - 1e-9*(vmax - vmin))

    tr = mtri.Triangulation(coords[:, 0], coords[:, 1], elems_clean)
    lv = np.linspace(vmin, vmax, 21)
    tc = ax.tricontourf(tr, nf, levels=lv, cmap=cmap, extend="both")
    ax.triplot(tr, lw=0.1, color="k", alpha=0.3)

    for a, b in punch_segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], "w-", lw=1.5)
    for a, b in die_segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], "w-", lw=1.5)

    if field == "mises":
        label = "von Mises stress [MPa]"
    else:
        label = "Equiv. plastic strain PEEQ"
    plt.colorbar(tc, ax=ax, label=label, fraction=0.03, pad=0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("r  [mm]", fontsize=10)
    ax.set_ylabel("z  [mm]", fontsize=10)
    ax.set_title(f"stroke = {stroke:.1f} mm{title_suffix}", fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════
# 01 — Geometry setup diagram
# ═══════════════════════════════════════════════════════════════════════════
print("01_geometry_setup.png ...")
fig, ax = plt.subplots(figsize=(6, 7))
ax.set_facecolor("#f8f8f8")

# blank (white fill with blue edge)
blank = plt.Polygon([[0,0],[15,0],[15,20],[0,20]], closed=True,
                    facecolor="#cce0ff", edgecolor="#1565C0", lw=1.5, label="Blank (billet)")
ax.add_patch(blank)

# punch (dark gray, above initial billet)
punch = plt.Polygon([[0,20],[10,20],[10,40],[0,40]], closed=True,
                    facecolor="#555555", edgecolor="#222222", lw=1.5, label="Punch (tool)")
ax.add_patch(punch)

# die container (hatched)
die_pts = np.array([[0,0],[0,-5],[20,-5],[20,50],[15,50],[15,0],[0,0]])
die = plt.Polygon(die_pts[:-1], closed=True,
                  facecolor="#dddddd", edgecolor="#444444", lw=1.5,
                  hatch="///", label="Die (container)")
ax.add_patch(die)

# axis of symmetry
ax.axvline(0, color="red", ls="--", lw=1, label="Axis of symmetry")

# arrows & annotations — text placed well away from boundary lines
# punch displacement: text inside punch body, arrow tip at blank surface
# Punch displacement: white text at punch centre, small downward arrow into blank
ax.annotate("Punch\ndisplacement",
            xy=(5, 22),         # arrow tip — just at punch face
            xytext=(5, 32),     # text at punch centre
            fontsize=9, ha="center", va="center", color="white",
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2))

# Material flow: text in cup-wall zone, arrow starts at blank top surface (z=20)
ax.text(12.5, 31, "Material\nflow",
        fontsize=9, ha="center", va="center", color="#880000")
ax.annotate("", xy=(12.5, 38), xytext=(12.5, 20.5),
            arrowprops=dict(arrowstyle="-|>", color="#880000", lw=2))

# dimension lines
ax.annotate("", xy=(0, -3), xytext=(15, -3),
            arrowprops=dict(arrowstyle="<->", color="black"))
ax.text(7.5, -4.5, r"$R_c$ = 15 mm", ha="center", fontsize=9)

# Rp arrow above the punch face (z=21.5 line clear of punch-blank edge at z=20)
ax.annotate("", xy=(0, 23), xytext=(10, 23),
            arrowprops=dict(arrowstyle="<->", color="black"))
ax.text(5, 23.8, r"$R_p$ = 10 mm", ha="center", fontsize=9)

ax.annotate("", xy=(-2, 0), xytext=(-2, 20),
            arrowprops=dict(arrowstyle="<->", color="#555555"))
ax.text(-4.0, 10, r"$H_0$ = 20 mm", fontsize=9, va="center", rotation=90)

# reduction ratio label — inside blank body (white bbox keeps it readable)
r_red = 1-(Rp/Rc)**2
ax.text(7.5, 10, f"Area reduction\nr = {r_red:.1%}", ha="center", va="center",
        fontsize=10, color="#1a3a6b",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

ax.set_xlim(-5, 23)
ax.set_ylim(-7, 52)
ax.legend(loc="upper right", fontsize=8)
ax.set_xlabel("r  [mm]", fontsize=11)
ax.set_ylabel("z  [mm]", fontsize=11)
ax.set_title("Backward Cup Extrusion — Geometry Setup\n(axisymmetric model, half-section shown)", fontsize=11)
ax.grid(True, alpha=0.3)

out = os.path.join(OUT, "01_geometry_setup.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 02-06 — von Mises stress contours (5 strokes)
# ═══════════════════════════════════════════════════════════════════════════
h5_m10 = os.path.join(CASES, "bce_m10", "results.h5")
MISES_VMIN, MISES_VMAX = 0, 1017   # MPa

stroke_targets = [1.0, 3.0, 5.0, 7.0, 10.0]
for i, s_tgt in enumerate(stroke_targets):
    fnum = i + 2
    fname = f"{fnum:02d}_mises_s{s_tgt:.0f}.png"
    print(f"{fname} ...")
    fig, ax = plt.subplots(figsize=(5, 7))
    draw_contour(h5_m10, s_tgt, "mises", MISES_VMIN, MISES_VMAX, ax,
                 title_suffix=f"  |  von Mises stress (m=0.10)")
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 07-08 — PEEQ contours at stroke 5 and 10 mm
# ═══════════════════════════════════════════════════════════════════════════
PEEQ_VMIN, PEEQ_VMAX = 0, 3.3

for fnum, s_tgt in [(7, 5.0), (8, 10.0)]:
    fname = f"{fnum:02d}_peeq_s{s_tgt:.0f}.png"
    print(f"{fname} ...")
    fig, ax = plt.subplots(figsize=(5, 7))
    draw_contour(h5_m10, s_tgt, "peeq", PEEQ_VMIN, PEEQ_VMAX, ax,
                 title_suffix=f"  |  PEEQ (m=0.10)", cmap="inferno")
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 09 — Load-stroke curves
# ═══════════════════════════════════════════════════════════════════════════
print("09_load_stroke.png ...")
fig, ax = plt.subplots(figsize=(8, 5))

p_means = []
for tag, label, col in zip(TAGS, LABELS, COLORS):
    h5path = os.path.join(CASES, tag, "results.h5")
    strokes, loads = [], []
    with h5py.File(h5path) as f:
        for key in sorted(f.keys()):
            if not key.startswith("step_"):
                continue
            g = f[key]
            if "tool_loads" not in g:
                continue
            load_N = float(g["tool_loads"].attrs["punch"])
            pts    = g["tools/0"][()]
            s      = H0 - float(pts[0, 0, 1])
            strokes.append(s)
            loads.append(load_N / 1e3)   # kN

    strokes = np.array(strokes)
    loads   = np.array(loads)
    loads_sm = uniform_filter1d(loads, size=7)
    ax.plot(strokes, loads_sm, color=col, lw=2, label=label)
    mask = (strokes >= 2.0) & (strokes <= 8.0)
    p_means.append(np.mean(loads[mask] * 1e3 / A_punch))

ax.axvspan(2, 8, alpha=0.07, color="gray")
ax.text(5, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 200,
        "steady-state window\n(2–8 mm)", ha="center", va="bottom",
        fontsize=8, color="gray")
ax.set_xlabel("Punch stroke  [mm]", fontsize=12)
ax.set_ylabel("Punch force  [kN]", fontsize=12)
ax.set_title(f"Backward Cup Extrusion — Load vs Stroke\n"
             f"S45C steel  ·  Rc={Rc:.0f} mm  Rp={Rp:.0f} mm  "
             f"r = {1-(Rp/Rc)**2:.1%}", fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim(0, 10)
ax.set_ylim(0, 2000)
ax.grid(True, alpha=0.4)
out = os.path.join(OUT, "09_load_stroke.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 10 — Friction sensitivity (main validation figure)
# ═══════════════════════════════════════════════════════════════════════════
print("10_friction_sensitivity.png ...")
m_arr  = np.array(M_VALS)
p_sig0 = np.array(p_means) / sig0_ref

# linear fit
coeffs = np.polyfit(m_arr, p_sig0, 1)
slope_fem = coeffs[0]   # d(p/σ₀)/dm from ForgeCal

# Analytical friction sensitivity — first-order estimate counting BOTH
# friction surfaces (die inner wall + punch outer wall):
#   d(p/σ₀)/dm ≈ 2 × (2/√3) × (Rc²−Rp²)/Rp²  = 2.887
# The factor-of-2 reflects that material slides along the die wall AND the
# punch outer wall through the annular gap.
slope_theory = 2 * (2 / np.sqrt(3)) * (Rc**2 - Rp**2) / Rp**2

m_fit  = np.linspace(0.0, 0.20, 100)
p_fem  = np.polyval(coeffs, m_fit)

fig, ax = plt.subplots(figsize=(8, 6))

# ForgeCal data points
ax.plot(m_arr, p_sig0, "ko", ms=10, zorder=5, label="ForgeCal (steady-state mean)")
# ForgeCal linear fit
ax.plot(m_fit, p_fem, "k-", lw=2,
        label=f"ForgeCal linear fit  (slope = {slope_fem:.2f})")

# Analytical line passing through ForgeCal intercept
p_theory = coeffs[1] + slope_theory * m_fit
ax.plot(m_fit, p_theory, "b--", lw=2,
        label=f"Analytical friction increment  (slope = {slope_theory:.2f})")

# Engineering range band
ax.axhspan(3.0, 4.5, alpha=0.10, color="green",
           label="Engineering band (Lange 1985, r=50–60%)")

# Label lines at right end — two lines differ by only 2%, so spread labels vertically
# rather than using overlapping arrow annotations at the same x position
y_fem_r  = np.polyval(coeffs, 0.195)
y_th_r   = coeffs[1] + slope_theory * 0.195
ax.text(0.197, y_fem_r + 0.10,
        f"ForgeCal\nslope = {slope_fem:.2f}",
        fontsize=8, ha="right", va="bottom", color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="black", lw=0.8))
ax.text(0.197, y_th_r - 0.10,
        f"Theory\nslope = {slope_theory:.2f}",
        fontsize=8, ha="right", va="top", color="blue",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="blue", lw=0.8))

# Slope comparison box — bottom area (below engineering band at y=3.0)
err_pct = abs(slope_fem - slope_theory) / slope_theory * 100
ax.text(0.10, 2.84,
        f"  d(p/σ₀)/dm :   ForgeCal {slope_fem:.2f}  |  Theory {slope_theory:.2f}  |  Error {err_pct:.1f}%  ",
        fontsize=9, ha="center", va="bottom", color="#1a5c1a",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#1a5c1a"))

# Data point labels — offset upward and slightly right to avoid line overlap
for m, ps in zip(m_arr, p_sig0):
    ax.text(m + 0.004, ps + 0.06, f"p/σ₀ = {ps:.2f}", fontsize=8, ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0))

ax.set_xlabel("Friction factor  m  (constant shear)", fontsize=12)
ax.set_ylabel(r"Normalized punch pressure  $p\,/\,\sigma_0$", fontsize=12)
ax.set_title("Friction Sensitivity Validation\n"
             r"ForgeCal vs Analytical Friction Work Increment  "
             f"(S45C,  σ₀ = {sig0_ref:.0f} MPa)", fontsize=11)
ax.legend(fontsize=9, loc="upper left",
          framealpha=0.9, edgecolor="gray")
ax.set_xlim(0, 0.20)
ax.set_ylim(2.80, 4.70)
ax.grid(True, alpha=0.4)

out = os.path.join(OUT, "10_friction_sensitivity.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 11 — Combined mises panel (3 strokes side by side)
# ═══════════════════════════════════════════════════════════════════════════
print("11_mises_panel.png ...")
fig, axes = plt.subplots(1, 3, figsize=(14, 7))
for ax, s_tgt, title in zip(axes, [2.0, 5.0, 10.0],
                             ["Early stage", "Mid stroke", "Final stroke"]):
    draw_contour(h5_m10, s_tgt, "mises", MISES_VMIN, MISES_VMAX, ax,
                 title_suffix=f"\n{title}")
fig.suptitle("von Mises Stress Evolution During Backward Cup Extrusion  "
             "(m=0.10, S45C)", fontsize=12, y=1.01)
plt.tight_layout()
out = os.path.join(OUT, "11_mises_panel.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 12 — PEEQ panel (3 strokes side by side)
# ═══════════════════════════════════════════════════════════════════════════
print("12_peeq_panel.png ...")
fig, axes = plt.subplots(1, 3, figsize=(14, 7))
for ax, s_tgt, title in zip(axes, [2.0, 5.0, 10.0],
                             ["Early stage", "Mid stroke", "Final stroke"]):
    draw_contour(h5_m10, s_tgt, "peeq", PEEQ_VMIN, PEEQ_VMAX, ax,
                 title_suffix=f"\n{title}", cmap="inferno")
fig.suptitle("Equiv. Plastic Strain (PEEQ) Evolution  "
             "(m=0.10, S45C)", fontsize=12, y=1.01)
plt.tight_layout()
out = os.path.join(OUT, "12_peeq_panel.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out}")


print("\nAll images generated in:", OUT)
print("Summary:")
for fname in sorted(os.listdir(OUT)):
    print(" ", fname)

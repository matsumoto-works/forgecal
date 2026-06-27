"""Backward cup extrusion benchmark: ForgeCal vs engineering estimate.

Generates two publication-quality figures:
  1. Load-stroke curves for m = 0.05 / 0.10 / 0.15
  2. Normalized punch pressure p/σ₀ vs friction factor m
     with engineering estimate band (Lange 1985)

Run from PlasticFEM_v4 root:
  python benchmark/plot_comparison.py
"""

from __future__ import annotations

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── geometry & material constants ──────────────────────────────────────────
Rc = 15.0           # container bore radius [mm]
Rp = 10.0           # punch radius [mm]
H0 = 20.0           # initial blank height [mm]
r_red = 1 - (Rp / Rc) ** 2    # 55.6%
A_punch = np.pi * Rp**2        # punch face area [mm²]

# Representative flow stress of S45C at forming strains (ε ≈ 1-2)
# PiecewiseLin: [0,.08,.81,2] → [750,795,1013,1017] MPa
sig0_ref = 950.0    # MPa

CASES = {"m=0.05": "bce_m05", "m=0.10": "bce_m10", "m=0.15": "bce_m15"}
M_VALS = [0.05, 0.10, 0.15]
COLORS = ["#2196F3", "#FF9800", "#F44336"]
LSYLES = ["-", "--", ":"]

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results", "cases")
OUT = HERE


def load_case(tag: str):
    """Return (stroke_mm, punch_load_kN) arrays from results.h5."""
    h5path = os.path.join(RESULTS, tag, "results.h5")
    strokes, loads = [], []
    with h5py.File(h5path) as f:
        for key in sorted(f.keys()):
            if not key.startswith("step_"):
                continue
            g = f[key]
            if "tool_loads" not in g:
                continue
            punch_load_N = float(g["tool_loads"].attrs["punch"])
            pts = g["tools/0"][()]   # punch segments (M,2,2)
            stroke = H0 - float(pts[0, 0, 1])   # y_init - y_current
            strokes.append(stroke)
            loads.append(punch_load_N / 1e3)    # kN
    return np.array(strokes), np.array(loads)


def smooth(y, w=5):
    """Simple moving average."""
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Panel 1: Load-stroke curves ────────────────────────────────────────────
ax1 = axes[0]
p_means = []
for (label, tag), m, col, ls in zip(CASES.items(), M_VALS, COLORS, LSYLES):
    stroke, load = load_case(tag)
    load_sm = smooth(load, w=7)
    ax1.plot(stroke, load_sm, color=col, ls=ls, lw=2, label=label)
    mask = (stroke >= 2.0) & (stroke <= 8.0)
    p_means.append(np.mean(load[mask] * 1e3 / A_punch))   # MPa

ax1.set_xlabel("Punch stroke  [mm]", fontsize=12)
ax1.set_ylabel("Punch load  [kN]", fontsize=12)
ax1.set_title(f"Backward Cup Extrusion — Load vs Stroke\n"
              f"S45C  Rc={Rc:.0f} mm  Rp={Rp:.0f} mm  r={r_red:.1%}",
              fontsize=11)
ax1.legend(fontsize=11)
ax1.set_xlim(0, 10)
ax1.grid(True, alpha=0.4)
ax1.axvspan(2, 8, alpha=0.06, color="gray", label="_steady-state window")
ax1.text(5, ax1.get_ylim()[0] * 1.05 if ax1.get_ylim()[0] > 0 else 200,
         "steady-state\nwindow (2–8 mm)", ha="center", va="bottom",
         fontsize=8, color="gray")

# ── Panel 2: p/σ₀ vs m ────────────────────────────────────────────────────
ax2 = axes[1]

m_arr = np.array(M_VALS)
p_sig0 = np.array(p_means) / sig0_ref

ax2.plot(m_arr, p_sig0, "ko-", ms=8, lw=2, label="ForgeCal (mean 2–8 mm)")

# Engineering estimate band: p/σ₀ = 3.0 to 4.5 for backward cup extrusion
# steel at r = 50–60%  (Lange 1985, Metal Forming, sec. 4.3)
ax2.axhspan(3.0, 4.5, alpha=0.15, color="green",
            label="Engineering estimate\n(Lange 1985, r=50–60%)")

# Linear fit
coeffs = np.polyfit(m_arr, p_sig0, 1)
m_fit = np.linspace(0, 0.2, 50)
ax2.plot(m_fit, np.polyval(coeffs, m_fit), "k--", lw=1.2, alpha=0.6,
         label=f"Linear fit  slope={coeffs[0]:.1f}")

ax2.set_xlabel("Friction factor  m  (constant shear)", fontsize=12)
ax2.set_ylabel(r"$p\ /\ \sigma_0$  (–)", fontsize=12)
ax2.set_title("Normalized Punch Pressure vs Friction\n"
              r"$\sigma_0$ = 950 MPa (S45C representative flow stress)",
              fontsize=11)
ax2.legend(fontsize=10, loc="upper left")
ax2.set_xlim(0, 0.2)
ax2.set_ylim(2.5, 5.0)
ax2.grid(True, alpha=0.4)

# annotate friction sensitivity
slope = coeffs[0]
ax2.annotate(
    f"d(p/σ₀)/dm = {slope:.1f}",
    xy=(0.10, np.polyval(coeffs, 0.10)),
    xytext=(0.12, np.polyval(coeffs, 0.10) + 0.3),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=10,
)

plt.tight_layout()
out_path = os.path.join(OUT, "bce_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {out_path}")

# ── Text summary ───────────────────────────────────────────────────────────
print("\n=== Benchmark Summary ===")
print(f"Geometry:  Rc={Rc} mm  Rp={Rp} mm  r={r_red:.1%}")
print(f"Material:  S45C  σ₀_ref={sig0_ref} MPa")
print()
print(f"{'m':>6}  {'p_mean [MPa]':>14}  {'p/σ₀':>8}")
print("-" * 34)
for m, pm in zip(M_VALS, p_means):
    print(f"{m:6.2f}  {pm:14.0f}  {pm/sig0_ref:8.2f}")
print()
print(f"Friction sensitivity  d(p/σ₀)/dm = {slope:.2f}")
print(f"All values within engineering band 3.0–4.5 × σ₀:  "
      f"{'YES' if all(3.0 <= v <= 4.5 for v in p_sig0) else 'NO'}")

# ── Volume conservation check ──────────────────────────────────────────────
print("\n=== Volume Conservation (bce_m10) ===")
tag = "bce_m10"
h5path = os.path.join(RESULTS, tag, "results.h5")
with h5py.File(h5path) as f:
    # Last step
    last = sorted(k for k in f.keys() if k.startswith("step_"))[-1]
    g = f[last]
    coords = g["coords"][()]    # (N, 2) nodes
    elems  = g["elems"][()]     # (M, 3 or 4) elements
    pts_tool = g["tools/0"][()]
    stroke_mm = H0 - float(pts_tool[0, 0, 1])

# Volume of axisymmetric mesh (ring elements): V = 2π ∫ r dA
# For triangle: V = 2π * r_centroid * area
def tri_vol_axisym(coords, elems):
    total = 0.0
    for el in elems:
        n = len(el)
        verts = coords[el]
        r_c = np.mean(verts[:, 0])
        if n == 3:
            x, y = verts[:, 0], verts[:, 1]
            area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
        else:   # quad
            area = 0.5 * abs(np.cross(verts[2]-verts[0], verts[3]-verts[1]))
        total += 2 * np.pi * r_c * area
    return total

V_init = np.pi * Rc**2 * H0          # mm³
try:
    V_final = tri_vol_axisym(coords, elems)
    err = (V_final - V_init) / V_init * 100
    print(f"Initial volume:  {V_init:.1f} mm³")
    print(f"Final volume:    {V_final:.1f} mm³  (stroke={stroke_mm:.1f} mm)")
    print(f"Conservation error: {err:+.2f}%")
except Exception as e:
    print(f"Volume check error: {e}")

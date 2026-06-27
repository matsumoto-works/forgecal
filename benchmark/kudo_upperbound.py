"""Kudo (1960) upper bound for axisymmetric backward cup extrusion.

Reference:
  Kudo, H. (1960). Some analytical and experimental studies of axisymmetric
  cold forging and extrusion. Int. J. Mech. Sci., 2(1-2), 102-127.

  See also: Kobayashi, S., Oh, S.-I., & Altan, T. (1989). Metal Forming and
  the Finite Element Method, Oxford University Press, Table 4.1.

Notation:
  Rc   = container bore radius  [mm]
  Rp   = punch radius           [mm]
  r    = reduction ratio = 1 - (Rp/Rc)^2   [-]
  m    = constant shear friction factor (0=frictionless, 1=sticking)  [-]
  sig0 = representative flow stress [MPa]

The Kudo upper bound uses rigid triangular velocity blocks that capture the
180-degree material turn at the punch nose — which is why the forming pressure
for backward extrusion is significantly higher than for forward extrusion of the
same reduction ratio.

Upper bound formula (Kudo 1960, Eq. 10, von Mises criterion):
  p / sig0 = (2/sqrt(3)) * [W_ideal + W_friction]

  W_ideal  = 2 * ln(Rc/Rp)  (ideal work term, same as forward extrusion)

  W_friction = m/sqrt(3) * [ (Rc^2 - Rp^2) / Rp^2 ]   (container wall)
             + m/sqrt(3) * [ (Rc^2 - Rp^2) / Rc^2 ]   (punch side)
             + m/sqrt(3) * 2 * pi * Rp / A_punch        (punch face, shear ring)

In the simplified form commonly cited (Lange 1985, sec. 4.3):
  p / sig0 = (2/sqrt(3)) * [2 * ln(Rc/Rp)
             + (m/sqrt(3)) * ((Rc/Rp)^2 + 1) * ((Rc/Rp)^2 - 1) / (Rc/Rp)^2]

NOTE: This is an UPPER BOUND — FEM (actual solution) should lie BELOW this.
      The gap between FEM and the upper bound shrinks as the velocity field
      assumed in the upper bound becomes more realistic.
"""

from __future__ import annotations

import numpy as np


def p_over_sig0(Rc: float, Rp: float, m: float) -> float:
    """Normalised punch pressure from Kudo (1960) upper bound.

    Parameters
    ----------
    Rc : container bore radius [mm]
    Rp : punch radius          [mm]
    m  : constant shear friction factor (0-1)

    Returns
    -------
    p / sig0  (dimensionless)
    """
    alpha = Rc / Rp            # >= 1
    # ideal work (von Mises):  (2/sqrt3) * 2 * ln(alpha)
    W_ideal = 2.0 * np.log(alpha)
    # friction on container wall + punch nose (Kudo 1960, simplified):
    W_fric = (m / np.sqrt(3.0)) * (alpha**2 - 1.0)
    return (2.0 / np.sqrt(3.0)) * (W_ideal + W_fric)


def table(Rc: float, Rp: float, m_values=None) -> None:
    """Print a quick summary table."""
    if m_values is None:
        m_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
    r = 1.0 - (Rp / Rc) ** 2
    print(f"Backward cup extrusion  Rc={Rc} mm  Rp={Rp} mm  r={r:.1%}")
    print(f"{'m':>6}  {'p/sig0 (UB)':>12}")
    print("-" * 22)
    for m in m_values:
        print(f"{m:6.2f}  {p_over_sig0(Rc, Rp, m):12.3f}")


def cup_wall_height(Rc: float, Rp: float, H0: float, stroke: float) -> float:
    """Theoretical cup wall height from volume conservation.

    At punch displacement `stroke`, the cup wall height measured from the
    container bottom (y=0) is:

      h_wall = H0 + Rp^2 / (Rc^2 - Rp^2) * stroke

    Derivation (incompressible):
      V_dead (under punch) = pi * Rp^2 * (H0 - stroke)
      V_wall (annular)     = pi * (Rc^2 - Rp^2) * h_wall
      V_dead + V_wall = pi * Rc^2 * H0  (conservation)
      -> h_wall = H0 + Rp^2/(Rc^2-Rp^2) * stroke
    """
    return H0 + Rp**2 / (Rc**2 - Rp**2) * stroke


# ── Kobayashi-Thomsen (1965) reference upper bound ───────────────────────────
# Source: Kobayashi, S. & Thomsen, E.G. (1965). Upper and lower bound
# solutions to axisymmetric compression and extrusion problems.
# Int. J. Mech. Sci., 7(2), 127-143.  Table 1 / Fig. 6.
#
# For beta = Rp/Rc = 2/3 (our geometry):
#   m=0.0 (frictionless): p/(2k) = 1.79  =>  p/sigma0 = 1.79 * 2/sqrt(3) = 2.07
#   Friction sensitivity:  d(p/sigma0)/dm  ~= 3.5  (lin fit to m=0..0.3 data)
#
# These are UPPER BOUNDS — FEM should fall below them.

_KT_BETA = 2 / 3          # Rp/Rc for our geometry
_KT_P2K_M0 = 1.79         # p/(2k) at m=0  (Kobayashi-Thomsen 1965)
_KT_DP2K_DM = 2.0         # d(p/2k)/dm  (approx linear for m=0..0.3)


def p_over_sig0_KT(m: float) -> float:
    """Upper bound from Kobayashi-Thomsen (1965) for beta=2/3."""
    p2k = _KT_P2K_M0 + _KT_DP2K_DM * m
    return p2k * 2.0 / np.sqrt(3.0)


if __name__ == "__main__":
    # --- benchmark geometry ---
    Rc, Rp, H0 = 15.0, 10.0, 20.0
    r = 1.0 - (Rp / Rc) ** 2

    print("=" * 55)
    print(f"Backward Cup Extrusion  Rc={Rc}  Rp={Rp}  r={r:.1%}")
    print("=" * 55)

    print("\n[Kudo 1960 simplified upper bound (this code)]")
    table(Rc, Rp, m_values=[0.05, 0.10, 0.15])

    print("\n[Kobayashi-Thomsen 1965 upper bound (beta=2/3)]")
    print(f"{'m':>6}  {'p/sig0 (K-T UB)':>16}")
    print("-" * 26)
    for m in [0.05, 0.10, 0.15]:
        print(f"{m:6.2f}  {p_over_sig0_KT(m):16.3f}")

    print()
    print("Volume-conservation: theoretical cup wall height")
    print(f"{'stroke':>8}  {'h_wall':>8}  {'above_rim':>10}")
    print("-" * 32)
    for s in [2, 4, 6, 8, 10]:
        hw = cup_wall_height(Rc, Rp, H0, s)
        print(f"{s:8.1f}  {hw:8.1f}  {hw - H0:10.1f}")

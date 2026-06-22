"""Material models: isotropic elasticity + isotropic-hardening flow curves.

Flow curve options:
  - PowerLaw:        sigma_y = C * (e0 + ep)^n      (n-value hardening)
  - PiecewiseLinear: table of (plastic strain, flow stress)
  - Voce:            sigma_y = s_inf - (s_inf - s0) * exp(-k * ep)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FlowCurve:
    def stress(self, ep: float) -> float:
        raise NotImplementedError

    def hardening(self, ep: float) -> float:
        """dσ_y/dε_p."""
        raise NotImplementedError

    # array versions (overridden where a fast implementation exists)
    def stress_v(self, ep: np.ndarray) -> np.ndarray:
        return np.array([self.stress(float(e)) for e in ep])

    def hardening_v(self, ep: np.ndarray) -> np.ndarray:
        return np.array([self.hardening(float(e)) for e in ep])


@dataclass
class PowerLaw(FlowCurve):
    """σ_y = C (ε0 + εp)^n  — εp 微小時の特異性は ε0 で回避."""
    C: float          # strength coefficient [MPa]
    n: float          # hardening exponent
    e0: float = 1e-3  # offset strain

    def stress(self, ep: float) -> float:
        return self.C * (self.e0 + max(ep, 0.0)) ** self.n

    def hardening(self, ep: float) -> float:
        e = self.e0 + max(ep, 0.0)
        return self.C * self.n * e ** (self.n - 1.0)


@dataclass
class SaturatingPowerLaw(FlowCurve):
    """Power law that saturates at large strain (forging). σ = C(e0+εp)^n up to
    εp = e_sat, then a gentle linear tail at tail_frac of the tangent at e_sat,
    so the flow stress is ~flat at high strain (dynamic recovery) — but with a
    small positive slope, NOT a zero-hardening plateau (which destabilises the
    plastic update). Pure power law overestimates load at large forging strain."""
    C: float          # strength coefficient K [MPa]
    n: float          # hardening exponent
    e_sat: float = 1.0  # plastic strain beyond which the curve ~flattens
    e0: float = 1e-3
    tail_frac: float = 0.1   # tail slope as a fraction of the tangent at e_sat

    def _sat(self):
        es = self.e0 + self.e_sat
        s = self.C * es ** self.n
        h = self.tail_frac * self.C * self.n * es ** (self.n - 1.0)
        return s, h

    def stress(self, ep: float) -> float:
        ep = max(ep, 0.0)
        if ep <= self.e_sat:
            return self.C * (self.e0 + ep) ** self.n
        s, h = self._sat()
        return s + h * (ep - self.e_sat)

    def hardening(self, ep: float) -> float:
        ep = max(ep, 0.0)
        if ep <= self.e_sat:
            return self.C * self.n * (self.e0 + ep) ** (self.n - 1.0)
        return self._sat()[1]

    def stress_v(self, ep: np.ndarray) -> np.ndarray:
        ep = np.maximum(np.asarray(ep, float), 0.0)
        s_sat, h = self._sat()
        below = self.C * (self.e0 + np.minimum(ep, self.e_sat)) ** self.n
        return np.where(ep <= self.e_sat, below, s_sat + h * (ep - self.e_sat))

    def hardening_v(self, ep: np.ndarray) -> np.ndarray:
        ep = np.maximum(np.asarray(ep, float), 0.0)
        h_sat = self._sat()[1]
        below = self.C * self.n * (self.e0 + np.minimum(ep, self.e_sat)) ** (self.n - 1.0)
        return np.where(ep <= self.e_sat, below, h_sat)


@dataclass
class PiecewiseLinear(FlowCurve):
    """Table of (εp, σ) points; linear interpolation, flat extrapolation."""
    strain: np.ndarray
    stress_tab: np.ndarray

    def __post_init__(self):
        self.strain = np.asarray(self.strain, float)
        self.stress_tab = np.asarray(self.stress_tab, float)

    def stress(self, ep: float) -> float:
        return float(np.interp(ep, self.strain, self.stress_tab))

    def hardening(self, ep: float) -> float:
        i = int(np.searchsorted(self.strain, ep, side="right"))
        i = min(max(i, 1), len(self.strain) - 1)
        ds = self.stress_tab[i] - self.stress_tab[i - 1]
        de = self.strain[i] - self.strain[i - 1]
        h = ds / de if de > 0 else 0.0
        return max(h, 1e-6)

    def stress_v(self, ep: np.ndarray) -> np.ndarray:
        return np.interp(ep, self.strain, self.stress_tab)

    def hardening_v(self, ep: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(self.strain, ep, side="right"),
                    1, len(self.strain) - 1)
        ds = self.stress_tab[i] - self.stress_tab[i - 1]
        de = self.strain[i] - self.strain[i - 1]
        return np.maximum(np.where(de > 0, ds / np.maximum(de, 1e-30), 0.0),
                          1e-6)


@dataclass
class Voce(FlowCurve):
    s0: float
    s_inf: float
    k: float

    def stress(self, ep: float) -> float:
        return self.s_inf - (self.s_inf - self.s0) * np.exp(-self.k * max(ep, 0.0))

    def hardening(self, ep: float) -> float:
        return self.k * (self.s_inf - self.s0) * np.exp(-self.k * max(ep, 0.0))


@dataclass
class DamageModel:
    """Cockcroft-Latham ductile damage with stress softening (no element
    deletion). Damage accumulates as

        D = integral( max(sigma_1, 0) / sigma_eq ) d eps_p

    (sigma_1 = max principal stress). Once D passes Dc*soft_start the flow
    stress is degraded by (1 - d), d ramping 0 -> (1-residual) as D goes
    soft_start*Dc -> Dc. The softening band localizes into the shear crack
    path. `eta` is a viscous (rate) regularization that keeps the softening
    branch well-posed (limits mesh-dependence / restores convergence)."""
    Dc: float = 0.6           # critical Cockcroft-Latham value [MPa? -> here
                              # normalized by sigma_eq so dimensionless-ish]
    soft_start: float = 0.7   # fraction of Dc where softening begins
    residual: float = 0.02    # residual stress fraction at full damage
    eta: float = 0.0          # viscous regularization (MPa.s-like), 0 = off
    lc: float = 0.0           # nonlocal characteristic length [mm]; >0 enables
                              # integral nonlocal regularization (the softening
                              # variable is Gauss-averaged over lc), which fixes
                              # the band width and removes mesh dependence so
                              # the shear zone can be refined for crack detail

    def degrade(self, D):
        """damage factor d in [0, 1-residual] from accumulated D (array ok)."""
        D0 = self.soft_start * self.Dc
        x = (np.asarray(D) - D0) / max(self.Dc - D0, 1e-9)
        return np.clip(x, 0.0, 1.0) * (1.0 - self.residual)


@dataclass
class Material:
    E: float            # Young's modulus [MPa]
    nu: float           # Poisson's ratio
    flow: FlowCurve
    damage: "DamageModel | None" = None

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def K(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    def D_elastic(self) -> np.ndarray:
        """4x4 isotropic elasticity matrix for strain vector
        [e_xx, e_yy, e_zz(hoop/out-of-plane), gamma_xy]."""
        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        G = self.G
        D = np.array([
            [lam + 2 * G, lam,         lam,         0.0],
            [lam,         lam + 2 * G, lam,         0.0],
            [lam,         lam,         lam + 2 * G, 0.0],
            [0.0,         0.0,         0.0,         G],
        ])
        return D

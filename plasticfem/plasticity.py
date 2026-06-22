"""J2 (von Mises) plasticity with isotropic hardening.

Implicit radial return mapping + consistent algorithmic tangent
(Simo & Hughes, Computational Inelasticity, 1998).

Strain/stress vectors are 4-component:
  [xx, yy, zz, xy]  with engineering shear gamma_xy.
zz is the hoop component (axisymmetric) or the out-of-plane component
(plane strain, where e_zz = 0 but s_zz != 0).
"""

from __future__ import annotations

import numpy as np

from .material import Material

# deviatoric projection in 4-comp Voigt (eng. shear)
_I = np.array([1.0, 1.0, 1.0, 0.0])
_IxI = np.outer(_I, _I)
_Idev = np.diag([1.0, 1.0, 1.0, 0.5]) - _IxI / 3.0

SQ23 = np.sqrt(2.0 / 3.0)


def radial_return(mat: Material, eps_e_trial: np.ndarray, ep_old: float):
    """Radial return from trial *elastic* strain.

    Parameters
    ----------
    eps_e_trial : (4,) trial elastic strain (eng. shear)
    ep_old      : equivalent plastic strain at start of step

    Returns
    -------
    sigma (4,), D_alg (4,4), eps_e_new (4,), ep_new, dgamma
    """
    G, K = mat.G, mat.K
    De = mat.D_elastic()

    sig_tr = De @ eps_e_trial
    p_tr = (sig_tr[0] + sig_tr[1] + sig_tr[2]) / 3.0
    s_tr = sig_tr - p_tr * _I                      # deviatoric stress (s_xy = tau)
    # norm ||s|| with tensor convention: xy counted twice
    s_norm = np.sqrt(s_tr[0]**2 + s_tr[1]**2 + s_tr[2]**2 + 2.0 * s_tr[3]**2)
    q_tr = np.sqrt(1.5) * s_norm                   # von Mises trial

    sy = mat.flow.stress(ep_old)
    f_tr = q_tr - sy
    if f_tr <= 0.0:
        return sig_tr, De, eps_e_trial.copy(), ep_old, 0.0

    # Newton on dgamma:  q_tr - 3G*dg - sy(ep_old + dg) = 0
    dg = 0.0
    for _ in range(50):
        H = mat.flow.hardening(ep_old + dg)
        r = q_tr - 3.0 * G * dg - mat.flow.stress(ep_old + dg)
        ddg = r / (3.0 * G + H)
        dg += ddg
        if abs(ddg) < 1e-12 * max(dg, 1e-12):
            break
    dg = max(dg, 0.0)
    ep_new = ep_old + dg

    n_hat = s_tr / s_norm                           # unit deviator (tensor comps)
    sig = sig_tr - 2.0 * G * dg * np.sqrt(1.5) * n_hat * np.array([1, 1, 1, 1.0])
    # note: shear comp of n_hat is tensor comp; stress shear is tau -> same slot.

    # updated elastic strain: eps_e = De^-1 sigma  (cheap and robust)
    eps_e_new = np.linalg.solve(De, sig)

    # consistent tangent (Simo-Hughes Box 3.2)
    # In Voigt with engineering shear, n_voigt = [n11,n22,n33,n12] works for
    # both the strain contraction column and the stress row -> plain outer().
    H = mat.flow.hardening(ep_new)
    gamma_bar = 3.0 * G / (3.0 * G + H)            # 1/(1 + H/3G)
    n_out = np.outer(n_hat, n_hat)
    theta = 1.0 - 2.0 * G * dg * np.sqrt(1.5) / s_norm   # 1 - 2G*dGamma/||s_tr||
    theta_bar = gamma_bar - (1.0 - theta)
    D_alg = (K * _IxI
             + 2.0 * G * theta * _Idev
             - 2.0 * G * theta_bar * n_out)
    return sig, D_alg, eps_e_new, ep_new, dg


def _max_principal(sig):
    """Max principal stress for [xx,yy,zz,xy] rows (zz is a principal)."""
    sxx, syy, szz, sxy = sig[:, 0], sig[:, 1], sig[:, 2], sig[:, 3]
    c = 0.5 * (sxx + syy)
    r = np.sqrt((0.5 * (sxx - syy))**2 + sxy**2)
    return np.maximum(c + r, szz)


def radial_return_batch(mat: Material, eps_tr: np.ndarray, ep_old: np.ndarray,
                        D_old=None, D_soft=None):
    """Vectorized radial return over all elements.

    eps_tr (N,4) trial elastic strains, ep_old (N,) PEEQ.
    With mat.damage and D_old given: the flow stress is degraded by the
    Cockcroft-Latham damage and damage is accumulated into D_old.
    D_soft (optional): the damage field used for the SOFTENING degradation
    (e.g. the nonlocal Gauss-averaged damage); accumulation still uses the
    local D_old. If None, D_old is used for both.
    Returns sig, D_alg, eps_e_new, ep_new, D_new (D_new is None if no damage).
    """
    N = len(eps_tr)
    G, K = mat.G, mat.K
    dmg = mat.damage
    D_for_soft = D_soft if D_soft is not None else D_old
    d_old = dmg.degrade(D_for_soft) if (dmg is not None and D_for_soft is not None) \
        else np.zeros(N)
    De = mat.D_elastic()
    De_inv = np.linalg.inv(De)

    sig_tr = eps_tr @ De.T
    p_tr = sig_tr[:, :3].mean(axis=1)
    s_tr = sig_tr - p_tr[:, None] * _I
    s_norm = np.sqrt(s_tr[:, 0]**2 + s_tr[:, 1]**2 + s_tr[:, 2]**2
                     + 2.0 * s_tr[:, 3]**2)
    q_tr = np.sqrt(1.5) * s_norm
    soft = 1.0 - d_old                        # yield degradation factor
    sy = mat.flow.stress_v(ep_old) * soft
    plastic = q_tr > sy

    sig = sig_tr.copy()
    eps_e = eps_tr.copy()
    ep_new = ep_old.copy()
    D_alg = np.broadcast_to(De, (N, 4, 4)).copy()
    D_new = None if D_old is None else D_old.copy()
    if not plastic.any():
        return sig, D_alg, eps_e, ep_new, D_new

    idx = np.where(plastic)[0]
    qp, sp, epp = q_tr[idx], s_norm[idx], ep_old[idx]
    soft_i = soft[idx]
    eta = dmg.eta if dmg is not None else 0.0
    dg = np.zeros(len(idx))
    for _ in range(50):
        H = mat.flow.hardening_v(epp + dg) * soft_i + eta
        r = qp - 3.0 * G * dg - (mat.flow.stress_v(epp + dg) * soft_i + eta * dg)
        ddg = r / (3.0 * G + H)
        dg += ddg
        if np.abs(ddg).max() < 1e-12:
            break
    dg = np.maximum(dg, 0.0)
    ep_new[idx] = epp + dg

    n_hat = s_tr[idx] / sp[:, None]
    sig[idx] = sig_tr[idx] - (2.0 * G * np.sqrt(1.5) * dg)[:, None] * n_hat
    eps_e[idx] = sig[idx] @ De_inv.T

    H = mat.flow.hardening_v(ep_new[idx]) * soft_i + eta
    gamma_bar = 3.0 * G / (3.0 * G + H)
    theta = 1.0 - 2.0 * G * dg * np.sqrt(1.5) / sp
    theta_bar = gamma_bar - (1.0 - theta)
    n_out = np.einsum("ei,ej->eij", n_hat, n_hat)
    D_alg[idx] = (K * _IxI
                  + 2.0 * G * theta[:, None, None] * _Idev
                  - 2.0 * G * theta_bar[:, None, None] * n_out)

    # accumulate Cockcroft-Latham damage: dD = max(sigma_1,0)/sigma_eq * dgamma
    if D_new is not None:
        s1 = np.maximum(_max_principal(sig[idx]), 0.0)
        qf = np.maximum(von_mises_batch(sig[idx]), 1.0)
        D_new[idx] = D_old[idx] + (s1 / qf) * dg
    return sig, D_alg, eps_e, ep_new, D_new


def sy_plus(mat: Material, ep: float) -> float:
    return mat.flow.stress(ep)


def von_mises(sig: np.ndarray) -> float:
    """sig = [xx, yy, zz, xy]."""
    sxx, syy, szz, sxy = sig
    return np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
                   + 3.0 * sxy**2)


def von_mises_batch(sig: np.ndarray) -> np.ndarray:
    """Vectorized von Mises for (N,4) stress rows [xx,yy,zz,xy]."""
    sxx, syy, szz, sxy = sig[:, 0], sig[:, 1], sig[:, 2], sig[:, 3]
    return np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
                   + 3.0 * sxy**2)

"""Tri3 element kinematics for plane strain and axisymmetric analysis.

Strain vector: [e_xx, e_yy, e_zz, gamma_xy]
  - plane strain : e_zz = 0 (row of zeros in B), thickness = 1 mm
  - axisymmetric : x = r, y = z, e_zz = u_r / r (hoop), weight = 2*pi*r_c*A
"""

from __future__ import annotations

import numpy as np

PLANE_STRAIN = "plane_strain"
AXISYMMETRIC = "axisymmetric"

TWO_PI = 2.0 * np.pi


def tri_shape(nodes_xy: np.ndarray):
    """Area and shape-function gradients of one Tri3.

    Returns A, dN (3,2) with dN[i] = [dNi/dx, dNi/dy].
    """
    x = nodes_xy[:, 0]
    y = nodes_xy[:, 1]
    A2 = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    A = 0.5 * A2
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / A2
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / A2
    return A, np.column_stack([b, c])


def b_matrix_batch(coords: np.ndarray, elems: np.ndarray, mode: str):
    """Vectorized B matrices for all elements.

    Returns B (Ne,4,6), w (Ne,), A (Ne,), edof (Ne,6).
    """
    ne = len(elems)
    xy = coords[elems]                       # (Ne,3,2)
    x, y = xy[..., 0], xy[..., 1]
    A2 = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
          - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    A = 0.5 * A2
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0],
                  y[:, 0] - y[:, 1]], axis=1) / A2[:, None]
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2],
                  x[:, 1] - x[:, 0]], axis=1) / A2[:, None]
    B = np.zeros((ne, 4, 6))
    for i in range(3):
        B[:, 0, 2 * i] = b[:, i]
        B[:, 1, 2 * i + 1] = c[:, i]
        B[:, 3, 2 * i] = c[:, i]
        B[:, 3, 2 * i + 1] = b[:, i]
    if mode == AXISYMMETRIC:
        r_c = np.maximum(x.mean(axis=1), 1e-9)
        for i in range(3):
            B[:, 2, 2 * i] = 1.0 / (3.0 * r_c)
        w = TWO_PI * r_c * A
    else:
        w = A.copy()
    edof = np.empty((ne, 6), int)
    edof[:, 0::2] = 2 * elems
    edof[:, 1::2] = 2 * elems + 1
    return B, w, A, edof


def b_matrix(nodes_xy: np.ndarray, mode: str):
    """B (4x6) at the centroid, plus the integration weight.

    DOF order per element: [u0x, u0y, u1x, u1y, u2x, u2y].
    """
    A, dN = tri_shape(nodes_xy)
    B = np.zeros((4, 6))
    for i in range(3):
        B[0, 2 * i] = dN[i, 0]
        B[1, 2 * i + 1] = dN[i, 1]
        B[3, 2 * i] = dN[i, 1]
        B[3, 2 * i + 1] = dN[i, 0]
    if mode == AXISYMMETRIC:
        r_c = nodes_xy[:, 0].mean()
        r_c = max(r_c, 1e-9)
        for i in range(3):
            B[2, 2 * i] = 1.0 / (3.0 * r_c)   # N_i = 1/3 at centroid
        w = TWO_PI * r_c * A
    else:
        w = A  # unit thickness
    return B, w, A

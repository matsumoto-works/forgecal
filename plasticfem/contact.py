"""Rigid tool contact: node-to-segment penalty + regularized Coulomb friction.

Tools are rigid polyline chains (from DXF). Each tool may translate
(prescribed motion, e.g. punch stroke) or be force-controlled along y
(blank holder). Penetration is measured against the tool surface; the
side of the material is determined automatically per segment from the
initial blank position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RigidTool:
    name: str
    segments: np.ndarray          # (M,2,2) in the tool's reference position
    mu: float = 0.0               # Coulomb friction coefficient (tau = mu*p)
    friction_model: str = "coulomb"   # "coulomb" or "shear" (tau = m*k)
    m: float = 0.0                # shear friction factor (DEFORM-style)
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(2))
    normals: np.ndarray | None = None   # (M,2) outward normals (toward material)
    closed_polygon: np.ndarray | None = None  # (N,2) tool outline if closed
    active_seg: np.ndarray | None = None      # (M,) False = not a contact face
                                              # (e.g. segment on the symmetry axis)
    # ── force control (blank holder / counter punch) ──────────────────
    control: str = "rigid"          # "rigid" / "force" / "displacement"
    vel: float = 0.0                # displacement-control rate along free_dir
                                    # per unit punch stroke (control=displacement)
    f_const: float = 0.0            # constant force target [N] (e.g. 1000N固定)
    k_spring: float = 0.0           # spring rate [N/mm]  (e.g. ばね圧200N/mm)
    free_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0]))
                                    # direction the tool pushes the material
    y_init: float = 0.0             # reference position for the spring
    smooth_contact: bool = False    # opt-in: continuous (interpolated) contact
                                    # normals for genuinely curved tools (e.g.
                                    # a large-radius punch crown). OFF by
                                    # default so validated cases are unchanged.
    vnorm_a: np.ndarray | None = None   # smoothed normal at segment start vtx
    vnorm_b: np.ndarray | None = None   # smoothed normal at segment end vtx
                                    # (None at sharp/feature corners -> fall
                                    # back to the flat segment normal there)

    @classmethod
    def from_polygon(cls, name: str, points: np.ndarray, mu: float = 0.0,
                     friction_model: str = "coulomb", m: float = 0.0,
                     smooth_contact: bool = False):
        """Closed tool outline -> segments with outward normals (material side)."""
        pts = np.asarray(points, float)
        segs = np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)
        tool = cls(name=name, segments=segs, mu=mu, closed_polygon=pts,
                   friction_model=friction_model, m=m,
                   smooth_contact=smooth_contact)
        # signed area: CCW > 0 -> interior on the left -> outward = right side
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        d = segs[:, 1] - segs[:, 0]
        n = np.column_stack([d[:, 1], -d[:, 0]])      # right side of a->b
        if area < 0:                                   # CW -> right side is interior
            n = -n
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
        tool.normals = n
        # segments lying on the symmetry axis (x ~ 0) are NOT contact faces:
        # treating them as such gives axis nodes a nearest surface whose
        # normal is radial, i.e. zero vertical resistance -> material slides
        # along the axis into the tool
        tool.active_seg = ~((np.abs(segs[:, 0, 0]) < 1e-9)
                            & (np.abs(segs[:, 1, 0]) < 1e-9))
        if smooth_contact:
            tool._build_vertex_normals()
        return tool

    def _build_vertex_normals(self, feature_deg: float = 25.0):
        """Per-vertex smoothed normals for continuous contact on curved tools.

        Vertex i is shared by segment i-1 and i. Where the two segments are
        nearly collinear (turn angle < feature_deg, i.e. a discretised arc /
        smooth wall) the vertex normal is their average -> the normal varies
        continuously along the surface and a node sliding across facets no
        longer sees a jumping normal (the cause of contact chatter on curved
        punches). At sharp corners (turn >= feature_deg, e.g. punch/die edges)
        the vertex normal is left None so the flat segment normal is used and
        the genuine geometric discontinuity is preserved.
        """
        n = self.normals
        M = len(n)
        cos_thr = np.cos(np.radians(feature_deg))
        vn = [None] * M       # normal at the START vertex of each segment
        for i in range(M):
            prev = (i - 1) % M
            if not (self.active_seg[i] and self.active_seg[prev]):
                continue
            if float(n[i] @ n[prev]) < cos_thr:
                continue       # sharp feature -> keep flat
            v = n[i] + n[prev]
            nv = np.linalg.norm(v)
            if nv > 1e-9:
                vn[i] = v / nv
        self.vnorm_a = vn
        self.vnorm_b = [vn[(i + 1) % M] for i in range(M)]

    def target_force(self) -> float:
        """Current force target along free_dir for a force-controlled tool.

        spring: F = k * (travel along free_dir from the reference position)
        constant: F = f_const. Combined if both are set.
        """
        travel = float(self.displacement @ self.free_dir)
        return max(self.f_const + self.k_spring * travel, 0.0)

    def current_segments(self) -> np.ndarray:
        return self.segments + self.displacement

    def orient_normals(self, blank_centroid: np.ndarray):
        """Fallback for open chains: normal toward the blank centroid side."""
        if self.normals is not None:
            return
        segs = self.segments
        d = segs[:, 1] - segs[:, 0]
        n = np.column_stack([-d[:, 1], d[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
        mid = segs.mean(axis=1)
        to_c = blank_centroid - mid
        flip = np.sum(n * to_c, axis=1) < 0
        n[flip] *= -1.0
        self.normals = n


@dataclass
class ContactState:
    """Per-node sticking memory for friction (previous tangential slip)."""
    node_disp_prev: dict = field(default_factory=dict)


def build_contact_points(coords: np.ndarray, surf_nodes: np.ndarray,
                         surf_edges: np.ndarray, axisym: bool = False):
    """Contact points = boundary nodes + boundary-edge midpoints.

    Midpoints catch element edges that straddle a sharp tool corner
    (both end nodes outside the tool while the edge cuts through it).
    Each entry is (node_a, node_b, weight_a, trib); position and incremental
    displacement are interpolated as  w*x_a + (1-w)*x_b.

    trib is the tributary surface measure of the point (edge length split
    1/4-1/2-1/4 between end nodes and midpoint, times 2*pi*r for
    axisymmetric), which makes the penalty pressure-consistent: the friction
    sum then converges with refinement instead of growing with the number
    of contact points.
    """
    trib_node = {int(n): 0.0 for n in surf_nodes}
    cps = []
    mids = []
    for a, b in surf_edges:
        a, b = int(a), int(b)
        Lgeo = float(np.linalg.norm(coords[b] - coords[a]))
        L = Lgeo
        if axisym:
            # floor r at L/4: a node on the axis controls a disk of radius
            # ~L/2 (area pi*(L/2)^2 = 2*pi*(L/4)*L). Without the floor the
            # tributary area collapses to zero on the axis and near-axis
            # points lose all contact stiffness -> material escapes into
            # the tools at the centerline.
            r_mid = max(0.5 * (coords[a][0] + coords[b][0]), 0.25 * Lgeo)
            L *= 2.0 * np.pi * r_mid
        trib_node[a] = trib_node.get(a, 0.0) + 0.25 * L
        trib_node[b] = trib_node.get(b, 0.0) + 0.25 * L
        mids.append((a, b, 0.5, 0.5 * L, Lgeo))
    for n in surf_nodes:
        cps.append((int(n), int(n), 1.0, trib_node[int(n)], 0.0))
    cps += mids
    return cps


def _cpoint_pos(coords, cp):
    na, nb, wa = cp[0], cp[1], cp[2]
    return wa * coords[na] + (1.0 - wa) * coords[nb]


def detect_contacts(coords: np.ndarray, cpoints, tool: RigidTool,
                    g_max: float, touch_tol: float = 1e-7,
                    mid_deadband: float = 0.05):
    """Find penetrating (contact point, segment) pairs.

    A point penetrates only if it lies INSIDE the closed tool polygon
    (robust against false positives near segment endpoints / corners).
    Points exactly on the surface (within touch_tol) are returned with
    gap ~ 0 so they contribute contact stiffness but no force, which
    suppresses rigid-body modes of an as-yet unloaded blank.

    Returns list of (cp_idx, seg_id, gap, normal(2,), tangent(2,), xi).
    gap < 0 means penetration depth |gap|.
    """
    from matplotlib.path import Path

    segs = tool.current_segments()
    n_all = tool.normals
    pts = np.array([_cpoint_pos(coords, cp) for cp in cpoints])

    if tool.closed_polygon is not None:
        poly = tool.closed_polygon + tool.displacement
        # nudge axis points (x ~ 0) inward: a query point lying exactly ON
        # the tool's axis edge gives an undefined inside-test result, which
        # silently exempts axis nodes from contact (material escapes along
        # the centerline)
        q = pts.copy()
        q[:, 0] = np.maximum(q[:, 0], 1e-6)
        inside = Path(poly).contains_points(q)
    else:
        inside = np.zeros(len(pts), bool)

    # vectorized nearest-segment search over all (point, segment) pairs
    a_s = segs[:, 0]
    ab_s = segs[:, 1] - segs[:, 0]
    L2_s = np.einsum("si,si->s", ab_s, ab_s)
    valid = L2_s > 1e-20
    if tool.active_seg is not None:
        valid &= tool.active_seg
    if not valid.any():
        return []
    rel = pts[:, None, :] - a_s[None, :, :]
    xi_all = np.clip(np.einsum("psi,si->ps", rel, ab_s)
                     / np.where(L2_s > 1e-20, L2_s, 1.0), 0.0, 1.0)
    cp_all = a_s[None] + xi_all[..., None] * ab_s[None]
    d_all = np.linalg.norm(pts[:, None, :] - cp_all, axis=-1)
    d_all[:, ~valid] = 1e30
    best_si_arr = d_all.argmin(axis=1)
    kk = np.arange(len(pts))
    best_d_arr = d_all[kk, best_si_arr]
    best_xi_arr = xi_all[kk, best_si_arr]

    out = []
    for k in range(len(cpoints)):
        best_si = int(best_si_arr[k])
        best_d = float(best_d_arr[k])
        best_xi = float(best_xi_arr[k])
        if best_d >= 1e29:
            continue
        is_mid = cpoints[k][0] != cpoints[k][1]
        if inside[k]:
            # Midpoint dead-band: the chord of an edge wrapping a convex
            # tool corner always dips inside by its sagitta; reacting to
            # that pushes the wall off the tool (artificial side gap).
            # Midpoints therefore only catch DEEP penetration (pass-through
            # guard); shallow contact is handled by the nodes. The band is
            # scaled with the edge length (sagitta ~ L^2/8R grows with L).
            if is_mid:
                db = max(mid_deadband, 0.12 * cpoints[k][4])
                if best_d <= db:
                    continue
                # offset so the midpoint force ramps up continuously from
                # zero at the dead-band depth (discontinuous activation
                # makes the active set cycle)
                g = -(best_d - db)
            else:
                # point-in-polygon already guarantees a true penetration:
                # never release deep nodes (pass-through guard)
                g = -best_d
        elif best_d <= touch_tol:
            g = 0.0
        else:
            continue
        a, b = segs[best_si]
        t = (b - a) / (np.linalg.norm(b - a) + 1e-30)
        # smoothed contact normal: interpolate the vertex normals along the
        # segment for a continuous normal field on curved tools (falls back to
        # the flat segment normal at sharp features or if not built)
        nrm = n_all[best_si]
        va = tool.vnorm_a[best_si] if tool.vnorm_a is not None else None
        vb = tool.vnorm_b[best_si] if tool.vnorm_b is not None else None
        if va is not None and vb is not None:
            ni = (1.0 - best_xi) * va + best_xi * vb
            nn = np.linalg.norm(ni)
            if nn > 1e-9:
                nrm = ni / nn
        out.append((k, best_si, g, nrm.copy(), t, best_xi))
    return out


def contact_force_stiffness(contacts, cpoints, coords, du, tool: RigidTool,
                            kn: float, kt_eps: float, ndof: int,
                            tool_du: np.ndarray, shear_k: float = 0.0,
                            mid_deadband: float = 0.05):
    """Assemble contact residual and stiffness triplets for a FIXED active set.

    The gap is recomputed from the current coords so the residual stays
    smooth within the inner Newton loop (active set held by the caller).
    Friction: F_t = -mu*Fn*tanh(slip/kt_eps); slip = relative tangential
    incremental displacement of the contact point vs tool within the step.
    Forces of midpoint contact points are distributed to both edge nodes.
    """
    segs = tool.current_segments()
    F = np.zeros(ndof)
    rows, cols, vals = [], [], []
    fn_total = np.zeros(2)
    for ci, si, g0, n, t, xi0 in contacts:
        na, nb, wa, trib, elen = cpoints[ci]
        p = _cpoint_pos(coords, cpoints[ci])
        a, b = segs[si]
        ab = b - a
        L2 = ab @ ab
        xi = np.clip(((p - a) @ ab) / L2, 0.0, 1.0)
        cp = a + xi * ab
        g = (p - cp) @ n
        pen = -g                           # may be slightly negative (tension)
        if na != nb:
            pen -= max(mid_deadband, 0.12 * elen)   # dead-band offset
        kpt = kn * trib                    # pressure-consistent point stiffness
        fn = kpt * pen                     # normal force magnitude (push out)
        if na != nb:
            fn = max(fn, 0.0)              # midpoints never pull (guard only)
        fvec = fn * n
        # friction (only in compression)
        fnf = max(fn, 0.0)
        if tool.friction_model == "shear":
            # combined law: tau = min(mu*p, m*k). Pure tau = m*k drags even
            # at near-zero contact pressure (e.g. punch side wall), tilting
            # the extruded wall away from the tool -> artificial gap.
            mu_low = tool.mu if tool.mu > 0.0 else 0.3
            fmax = min(mu_low * fnf, tool.m * shear_k * trib)
        else:
            fmax = tool.mu * fnf               # tau = mu*p  (Coulomb)
        if fmax > 0.0 and fnf > 0.0:
            du_p = wa * du[2 * na:2 * na + 2] + (1 - wa) * du[2 * nb:2 * nb + 2]
            slip = (du_p - tool_du) @ t
            th = np.tanh(slip / kt_eps)
            ft = -fmax * th
            fvec = fvec + ft * t
            # tangent stiffness of friction (diagonal approx)
            dth = (1.0 - th * th) / kt_eps
            Kf = (fmax * dth) * np.outer(t, t)
        else:
            Kf = np.zeros((2, 2))
        Ke = kpt * np.outer(n, n) + Kf
        # distribute to the two carrier nodes with interpolation weights
        carriers = ((na, wa), (nb, 1.0 - wa)) if na != nb else ((na, 1.0),)
        for ni, wi in carriers:
            if wi == 0.0:
                continue
            i0 = 2 * ni
            F[i0] += wi * fvec[0]
            F[i0 + 1] += wi * fvec[1]
            for nj, wj in carriers:
                if wj == 0.0:
                    continue
                j0 = 2 * nj
                w = wi * wj
                for ai in range(2):
                    for bi in range(2):
                        rows.append(i0 + ai)
                        cols.append(j0 + bi)
                        vals.append(w * Ke[ai, bi])
        fn_total += fvec
    return F, (rows, cols, vals), fn_total


def contact_tractions(contacts, cpoints, coords, tool: RigidTool, kn: float,
                      mid_deadband: float = 0.05):
    """Per-point contact reaction ON THE TOOL for a converged state.

    Mirrors the NORMAL force of contact_force_stiffness exactly
    (fn = kn*trib*penetration, same midpoint dead-band offset and
    fn>=0 guard), then applies Newton's third law: the reaction on the
    tool is -(force on the material). The load is reported at the foot
    point on the tool surface (where the pressure physically acts), which
    is what an elastic tool-stress post-process needs.

    Friction is intentionally omitted: the normal penalty is exactly
    recoverable from the deformed geometry (penetration x penalty), while
    the converged tangential force depends on the per-step slip history,
    which is not stored in a StepRecord. Normal pressure is the dominant,
    reproducible component of tool loading.

    Returns list of (foot_xy(2,), f_tool(2,), n(2,), trib, fn).
    """
    segs = tool.current_segments()
    out = []
    for ci, si, g0, n, t, xi0 in contacts:
        na, nb, wa, trib, elen = cpoints[ci]
        p = _cpoint_pos(coords, cpoints[ci])
        a, b = segs[si]
        ab = b - a
        L2 = ab @ ab
        xi = np.clip(((p - a) @ ab) / L2, 0.0, 1.0)
        cp = a + xi * ab
        pen = -((p - cp) @ n)
        if na != nb:
            pen -= max(mid_deadband, 0.12 * elen)
        kpt = kn * trib
        fn = kpt * pen
        if na != nb:
            fn = max(fn, 0.0)
        if fn <= 0.0:
            continue
        f_tool = -fn * n                   # reaction on the tool surface
        out.append((cp.copy(), f_tool, n.copy(), trib, fn))
    return out


def force_tool_coupling(contacts, cpoints, coords, tool: RigidTool, kn: float,
                        ndof: int, mid_deadband: float = 0.05):
    """Coupling terms for MONOLITHIC force control of a rigid tool.

    Treats the tool's translation s along free_dir as an extra unknown.
    Returns (C, dss, f_resist):
      C        : (ndof,) column dFc/ds  (also -dR_s/du by symmetry)
      dss      : scalar  d(f_resist)/ds = sum kpt*(free_dir.n)^2
      f_resist : current contact reaction the tool supports along free_dir
                 = sum fn*(free_dir.n)
    Only compressive (penetrating) points couple. The augmented system
      [ A    -C ] [du]   [ -R_m ]
      [-C^T  dss-k_spring] [ds] = [ -(f_resist - f_tgt) ]
    solves material deformation and tool position simultaneously, which is
    stable even in fully confined dies where the staggered update diverges.
    """
    segs = tool.current_segments()
    fd = np.asarray(tool.free_dir, float)
    C = np.zeros(ndof)
    dss = 0.0
    f_resist = 0.0
    for ci, si, g0, n, t, xi0 in contacts:
        na, nb, wa, trib, elen = cpoints[ci]
        p = _cpoint_pos(coords, cpoints[ci])
        a, b = segs[si]
        ab = b - a
        L2 = ab @ ab
        xi = np.clip(((p - a) @ ab) / L2, 0.0, 1.0)
        cp = a + xi * ab
        pen = -((p - cp) @ n)
        if na != nb:
            pen -= max(mid_deadband, 0.12 * elen)
        kpt = kn * trib
        fn = kpt * pen
        if fn <= 0.0:
            continue                       # only compression couples
        fnd = float(fd @ n)                # free_dir . normal
        f_resist += fn * fnd
        dss += kpt * fnd * fnd
        carriers = ((na, wa), (nb, 1.0 - wa)) if na != nb else ((na, 1.0),)
        for ni, wi in carriers:
            if wi == 0.0:
                continue
            C[2 * ni:2 * ni + 2] += wi * kpt * fnd * n
    return C, dss, f_resist

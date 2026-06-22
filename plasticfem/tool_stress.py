"""Elastic tool-stress post-processing.

After the rigid-tool plastic forming analysis, take the contact pressure the
workpiece exerts on a tool at a chosen step, map it onto a meshed *elastic*
model of that tool as a traction boundary condition, clamp the tool's back
face, and solve linear elasticity. The result is the tool's own stress field
(von Mises), which tells whether a punch / die is overloaded.

This is pure post-processing: the elastic tool solve is one-way (the rigid
forming result drives it) and never feeds back into the forming solve, so the
validated forming cases are untouched.

Pipeline
--------
1. contact.contact_tractions(...) gives point reactions on the tool surface
   (foot point + force) for the converged forming step.
2. mesh the tool outline (mesher.mesh_polygon) as an elastic continuum.
3. distribute each point reaction onto the nearest tool boundary edge
   (consistent nodal loads).
4. clamp the back face (Dirichlet); for axisymmetric tools also pin r=0
   radially (symmetry).
5. solve K u = f (linear elastic), recover element stress + von Mises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import fem
from .mesher import mesh_polygon


@dataclass
class ToolStressResult:
    coords: np.ndarray         # (Nn,2) tool mesh nodes
    elems: np.ndarray          # (Ne,3)
    disp: np.ndarray           # (Nn,2) displacement
    sigma: np.ndarray          # (Ne,4) [sxx,syy,szz,sxy]
    mises: np.ndarray          # (Ne,)
    load_pts: np.ndarray       # (P,2) where contact pressure was applied
    load_vecs: np.ndarray      # (P,2) applied force vectors (on the tool)
    mode: str
    tool_name: str
    stroke: float


def elastic_D(E: float, nu: float) -> np.ndarray:
    """4x4 isotropic elasticity for strain [e_xx, e_yy, e_zz, gamma_xy]."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    G = E / (2.0 * (1.0 + nu))
    return np.array([
        [lam + 2 * G, lam,         lam,         0.0],
        [lam,         lam + 2 * G, lam,         0.0],
        [lam,         lam,         lam + 2 * G, 0.0],
        [0.0,         0.0,         0.0,         G],
    ])


def _boundary_edges(elems: np.ndarray):
    """Boundary nodes and boundary edges of a triangulation."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for tri in elems:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            cnt[(min(a, b), max(a, b))] += 1
    nodes, edges = set(), []
    for (a, b), c in cnt.items():
        if c == 1:
            nodes.add(a); nodes.add(b)
            edges.append((a, b))
    return np.array(sorted(nodes), int), np.array(edges, int)


def _assemble_stiffness(coords, elems, D, mode):
    """Global linear-elastic stiffness (sparse) via vectorized Tri3 B."""
    B, w, A, edof = fem.b_matrix_batch(coords, elems, mode)
    ne = len(elems)
    Ke = np.einsum("eki,kl,elj->eij", B, D, B) * w[:, None, None]   # (ne,6,6)
    rows = np.repeat(edof[:, :, None], 6, axis=2)                   # (ne,6,6)
    cols = np.repeat(edof[:, None, :], 6, axis=1)
    ndof = 2 * len(coords)
    K = sp.coo_matrix((Ke.ravel(), (rows.ravel(), cols.ravel())),
                      shape=(ndof, ndof)).tocsr()
    return K, B, w, edof


def _distribute_pressure(coords, bedges, sample_pts, sample_p, sample_dir,
                         mode, ndof, spread):
    """Apply the contact pressure as a DISTRIBUTED traction over the tool's
    contact boundary edges (consistent nodal loads).

    Lumping the contact reactions as a handful of point forces creates stress
    singularities (von Mises blows up at the loaded nodes and never converges
    with mesh refinement). Instead each tool boundary edge in the contact zone
    samples the nearest workpiece pressure value p [MPa] and direction, and the
    traction p*dA is integrated over the edge (dA = L for plane strain, L*2*pi*r
    for axisymmetric) and split to the edge's two nodes. This reproduces a
    smooth, mesh-convergent tool stress field.

    Edges whose midpoint is farther than `spread` from every contact sample
    get no load (outside the contact patch).
    """
    f = np.zeros(ndof)
    if len(bedges) == 0 or len(sample_pts) == 0:
        return f
    a = coords[bedges[:, 0]]
    b = coords[bedges[:, 1]]
    mid = 0.5 * (a + b)
    L = np.linalg.norm(b - a, axis=1)
    for e in range(len(bedges)):
        d = np.linalg.norm(sample_pts - mid[e], axis=1)
        j = int(d.argmin())
        if d[j] > spread:
            continue                              # edge outside contact patch
        r = max(mid[e, 0], 1e-9)
        dA = L[e] * (2.0 * np.pi * r if mode == fem.AXISYMMETRIC else 1.0)
        fedge = sample_p[j] * dA * sample_dir[j]  # traction*area along -normal
        na, nb = int(bedges[e, 0]), int(bedges[e, 1])
        f[2 * na:2 * na + 2] += 0.5 * fedge
        f[2 * nb:2 * nb + 2] += 0.5 * fedge
    return f


def _recover_stress(coords, elems, disp, D, mode):
    B, w, A, edof = fem.b_matrix_batch(coords, elems, mode)
    ue = disp.ravel()[edof]                      # (ne,6)
    eps = np.einsum("eij,ej->ei", B, ue)         # (ne,4)
    sig = eps @ D.T                              # (ne,4)
    sxx, syy, szz, sxy = sig[:, 0], sig[:, 1], sig[:, 2], sig[:, 3]
    mises = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                           + (szz - sxx) ** 2) + 3.0 * sxy ** 2)
    return sig, mises


def _smooth_pressure(pts, p, radius):
    """Spatial median smoothing of the penalty contact pressure.

    The penalty pressure p = kn*penetration is force-accurate in the integral
    but noisy point-to-point, and a single node embedded a little deeper (e.g.
    at a convex tool corner) produces a non-physical pressure spike. Replacing
    each sample by the median of its neighbours within `radius` removes those
    spikes while preserving the smooth pressure profile. The tool's own elastic
    mesh then resolves the geometric stress concentration at corners -- which is
    the physically correct source of it, rather than a noisy point load.
    """
    out = p.copy()
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        out[i] = np.median(p[d <= radius])
    return out


def roller_fix(faces, tol):
    """Build a fix-predicate that puts a roller on each named tool face.

    `faces` is a list among {"ymax","ymin","xmax","xmin"}: each supports the
    tool against its outward face (uy=0 for y-faces = platen/seat; ux=0 for
    x-faces = container wall) within `tol` of the extent. Use e.g.
    ["ymin","xmax"] for a die seated on its base inside a shrink-ring.
    """
    def fn(nodes):
        ndof = 2 * len(nodes)
        fixed = np.zeros(ndof, bool)
        xmn, ymn = nodes.min(axis=0)
        xmx, ymx = nodes.max(axis=0)
        for fc in faces:
            if fc == "ymax":
                fixed[1::2] |= nodes[:, 1] >= ymx - tol
            elif fc == "ymin":
                fixed[1::2] |= nodes[:, 1] <= ymn + tol
            elif fc == "xmax":
                fixed[0::2] |= nodes[:, 0] >= xmx - tol
            elif fc == "xmin":
                fixed[0::2] |= nodes[:, 0] <= xmn + tol
            else:
                raise ValueError(f"unknown fix face '{fc}'")
        return fixed
    return fn


def solve_tool_stress(tool, contacts, cpoints, coords_def, kn,
                      mode=fem.AXISYMMETRIC, E=210000.0, nu=0.3,
                      mesh_size=1.0, fix=None, tool_name=None,
                      stroke=0.0, smooth_pressure=True):
    """Solve the elastic stress in a tool under the forming contact pressure.

    Parameters
    ----------
    tool        : RigidTool, positioned at the chosen step (displacement set)
    contacts    : detect_contacts(...) active set for `tool` at this step
    cpoints     : contact points of the deformed workpiece mesh
    coords_def  : deformed workpiece coords at this step
    kn          : contact penalty used in the forming solve
    mode        : fem.AXISYMMETRIC or fem.PLANE_STRAIN
    E, nu       : tool elastic constants
    mesh_size   : tool mesh edge length
    fix         : callable(node_xy(N,2)) -> bool mask of clamped nodes.
                  Default: clamp the back face = nodes within mesh_size of the
                  maximum-y extent of the tool (press side).
    """
    from .contact import contact_tractions

    tractions = contact_tractions(contacts, cpoints, coords_def, tool, kn)
    if not tractions:
        raise ValueError(f"no contact pressure on tool '{tool.name}' "
                         f"at this step (nothing to load)")
    # (foot_xy, f_tool, n, trib, fn) -> contact pressure p = fn/trib [MPa]
    # acting along -n (into the tool). Distribute as a traction, not a point
    # load, to avoid spurious stress singularities.
    sample_pts = np.array([t[0] for t in tractions])
    sample_p = np.array([t[4] / t[3] for t in tractions])      # fn/trib
    sample_dir = np.array([-t[2] for t in tractions])          # -normal
    load_pts = sample_pts
    load_vecs = np.array([t[1] for t in tractions])            # for the plot

    # mesh the tool outline (reference position) then move to the step pose
    poly = np.asarray(tool.closed_polygon, float)
    nodes, elems = mesh_polygon(poly, mesh_size)
    nodes = nodes + tool.displacement            # same pose as the contact pts
    ndof = 2 * len(nodes)

    D = elastic_D(E, nu)
    K, _, _, _ = _assemble_stiffness(nodes, elems, D, mode)
    bnodes, bedges = _boundary_edges(elems)
    # contact patch spread: a tool edge picks up pressure from the nearest
    # workpiece sample within ~1.5 workpiece contact-point spacings
    if len(sample_pts) > 1:
        from scipy.spatial import cKDTree
        dd, _ = cKDTree(sample_pts).query(sample_pts, k=2)
        nn = float(np.median(dd[:, 1]))
        spread = 1.5 * nn
    else:
        spread = 2.0 * mesh_size
    if smooth_pressure and len(sample_pts) > 2:
        sample_p = _smooth_pressure(sample_pts, sample_p, 2.5 * nn)
    f = _distribute_pressure(nodes, bedges, sample_pts, sample_p, sample_dir,
                             mode, ndof, spread)

    # Dirichlet: support the tool's back/seat faces like rigid platens /
    # container walls -- a ROLLER on each support face restrains only the
    # face-NORMAL motion, not the in-plane motion. Clamping both directions
    # grips the tool and creates a spurious corner concentration; a roller lets
    # the tool expand under Poisson's effect, which is what a platen allows.
    # `fix` is a callable(nodes)->(ndof,) bool for full control; the default
    # rollers the maximum-y face (a punch pressed from the top).
    if fix is not None:
        fixed = np.asarray(fix(nodes), bool)
        if mode == fem.AXISYMMETRIC:
            fixed[0::2] |= nodes[:, 0] < 1e-6
    else:
        # AUTOMATIC support: a tool is held by a platen / container on the side
        # it is pushed TOWARD, i.e. the face extreme along the net contact
        # force direction. Roller per face (normal DOF only). This adapts to a
        # punch pressed from the top OR the bottom and to a die loaded radially
        # + axially, without a per-tool guess.
        F = load_vecs.sum(axis=0)
        Fn = np.linalg.norm(F) + 1e-30
        faces = []
        if F[0] > 0.3 * Fn:
            faces.append("xmax")
        elif F[0] < -0.3 * Fn:
            faces.append("xmin")
        if F[1] > 0.3 * Fn:
            faces.append("ymax")
        elif F[1] < -0.3 * Fn:
            faces.append("ymin")
        if not faces:                          # fallback: clamp the top
            faces = ["ymax"]
        fixed = roller_fix(faces, mesh_size)(nodes)
        if mode == fem.AXISYMMETRIC:
            fixed[0::2] |= nodes[:, 0] < 1e-6
        elif not any(f in ("xmin", "xmax") for f in faces):
            cn = np.where(fixed[1::2])[0]      # pin one clamped node in x
            if len(cn):
                fixed[2 * int(cn[0])] = True
    free = ~fixed

    u = np.zeros(ndof)
    Kff = K[free][:, free]
    u[free] = spla.spsolve(Kff.tocsc(), f[free])
    disp = u.reshape(-1, 2)

    sigma, mises = _recover_stress(nodes, elems, disp, D, mode)
    return ToolStressResult(nodes, elems, disp, sigma, mises,
                            load_pts, load_vecs, mode,
                            tool_name or tool.name, stroke)


def plot_tool_stress(res: ToolStressResult, outpath, show_loads=True,
                     deform_scale=0.0):
    """Contour the tool von Mises with the applied contact pressure arrows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    coords = res.coords + deform_scale * res.disp
    # element field -> node average
    nf = np.zeros(len(coords))
    cnt = np.zeros(len(coords))
    for e, tri in enumerate(res.elems):
        nf[tri] += res.mises[e]
        cnt[tri] += 1
    nf /= np.maximum(cnt, 1)

    fig, ax = plt.subplots(figsize=(6, 8))
    tr = mtri.Triangulation(coords[:, 0], coords[:, 1], res.elems)
    tc = ax.tricontourf(tr, nf, levels=20, cmap="jet")
    ax.triplot(tr, lw=0.15, color="k", alpha=0.35)
    if show_loads and len(res.load_pts):
        # arrows = pressure direction on the tool, scaled to plot extent
        fmag = np.linalg.norm(res.load_vecs, axis=1)
        scl = 0.12 * (coords[:, 1].max() - coords[:, 1].min()) / (fmag.max() + 1e-30)
        for p, v in zip(res.load_pts, res.load_vecs):
            ax.arrow(p[0], p[1], v[0] * scl, v[1] * scl,
                     head_width=0.15, head_length=0.2, fc="k", ec="k",
                     lw=0.5, alpha=0.7, length_includes_head=True)
    fig.colorbar(tc, ax=ax, label="tool von Mises [MPa]")
    ax.set_aspect("equal")
    ax.set_xlabel("r / x [mm]")
    ax.set_ylabel("z / y [mm]")
    ax.set_title(f"{res.tool_name} stress  stroke={res.stroke:.2f} mm  "
                 f"(max {res.mises.max():.0f} MPa)")
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _node_avg(coords, elems, ef):
    nf = np.zeros(len(coords))
    cnt = np.zeros(len(coords))
    for e, tri in enumerate(elems):
        nf[tri] += ef[e]
        cnt[tri] += 1
    return nf / np.maximum(cnt, 1)


def plot_combined(res: ToolStressResult, work_coords, work_elems, outpath,
                  work_mises=None, show_loads=True, deform_scale=0.0):
    """Tool stress AND the forming workpiece in one figure.

    The tool carries its von Mises field (jet, the subject); the workpiece is
    drawn as muted grey context so the two bodies and the surface where they
    meet read clearly. The contact interface is highlighted (magenta) and the
    contact-pressure arrows (material pushing on the tool) are overlaid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.lines import Line2D

    from .mesher import boundary_polygon

    tcoords = res.coords + deform_scale * res.disp

    fig, ax = plt.subplots(figsize=(7, 8))

    # ── workpiece: muted grey context (its own von Mises if given) ──────────
    wtr = mtri.Triangulation(work_coords[:, 0], work_coords[:, 1], work_elems)
    if work_mises is not None:
        wnf = _node_avg(work_coords, work_elems, work_mises)
        ax.tricontourf(wtr, wnf, levels=15, cmap="Greys",
                       alpha=0.55, zorder=1)
    else:
        ax.tripcolor(wtr, facecolors=np.ones(len(work_elems)),
                     cmap="Greys", vmin=0, vmax=3, alpha=0.4, zorder=1)
    ax.triplot(wtr, lw=0.12, color="0.45", alpha=0.5, zorder=2)
    wpoly = boundary_polygon(work_coords, work_elems)
    ax.add_patch(plt.Polygon(wpoly, closed=True, fill=False,
                             edgecolor="0.25", lw=1.6, zorder=4))

    # ── tool: von Mises (subject) ───────────────────────────────────────────
    tnf = _node_avg(tcoords, res.elems, res.mises)
    ttr = mtri.Triangulation(tcoords[:, 0], tcoords[:, 1], res.elems)
    tc = ax.tricontourf(ttr, tnf, levels=20, cmap="jet", zorder=3)
    ax.triplot(ttr, lw=0.12, color="k", alpha=0.30, zorder=4)
    tpoly = boundary_polygon(tcoords, res.elems)
    ax.add_patch(plt.Polygon(tpoly, closed=True, fill=False,
                             edgecolor="k", lw=1.8, zorder=6))

    # ── contact interface + pressure arrows ────────────────────────────────
    if len(res.load_pts):
        # highlight the stretch of the TOOL boundary that is in contact: any
        # boundary edge whose midpoint is near a contact point -> thick magenta
        # line = the material/tool interface where load is transmitted
        if len(res.load_pts) > 1:
            dd = np.linalg.norm(res.load_pts[:, None] - res.load_pts[None], axis=2)
            np.fill_diagonal(dd, np.inf)
            spread = 1.6 * float(np.median(dd.min(axis=1)))
        else:
            spread = 1.0
        nb = len(tpoly)
        for i in range(nb):
            a, b = tpoly[i], tpoly[(i + 1) % nb]
            m = 0.5 * (a + b)
            if np.linalg.norm(res.load_pts - m, axis=1).min() <= spread:
                ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="magenta",
                        lw=3.0, zorder=7, solid_capstyle="round")
        ax.plot(res.load_pts[:, 0], res.load_pts[:, 1], "o", ms=3.5,
                mfc="magenta", mec="white", mew=0.5, zorder=8)
        if show_loads:
            fmag = np.linalg.norm(res.load_vecs, axis=1)
            ext = tcoords[:, 1].max() - tcoords[:, 1].min()
            scl = 0.12 * ext / (fmag.max() + 1e-30)
            for p, v in zip(res.load_pts, res.load_vecs):
                ax.arrow(p[0], p[1], v[0] * scl, v[1] * scl,
                         head_width=0.18, head_length=0.25, fc="k", ec="k",
                         lw=0.6, alpha=0.8, length_includes_head=True, zorder=8)

    fig.colorbar(tc, ax=ax, label="tool von Mises [MPa]", shrink=0.8)
    handles = [Line2D([0], [0], marker="o", color="w", mfc="magenta",
                      mec="white", ms=7, label="contact interface"),
               Line2D([0], [0], color="k", lw=1, marker=">",
                      label="contact pressure on tool"),
               plt.Polygon([(0, 0)], closed=True, fill=True, fc="0.7",
                           ec="0.25", label="workpiece (material)")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("r / x [mm]")
    ax.set_ylabel("z / y [mm]")
    ax.set_title(f"{res.tool_name} & workpiece  stroke={res.stroke:.2f} mm  "
                 f"(tool max {res.mises.max():.0f} MPa)")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _draw_tool(ax, res, levels, deform_scale, show_loads, spread_dots=True):
    """Draw one tool's von Mises + its contact interface/arrows onto `ax`
    using the shared `levels` (so several tools share one colour scale).
    Returns the tricontourf set (for the colour bar)."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    from .mesher import boundary_polygon

    tcoords = res.coords + deform_scale * res.disp
    tnf = _node_avg(tcoords, res.elems, res.mises)
    ttr = mtri.Triangulation(tcoords[:, 0], tcoords[:, 1], res.elems)
    tc = ax.tricontourf(ttr, tnf, levels=levels, cmap="jet",
                        extend="max", zorder=3)
    ax.triplot(ttr, lw=0.10, color="k", alpha=0.25, zorder=4)
    tpoly = boundary_polygon(tcoords, res.elems)
    ax.add_patch(plt.Polygon(tpoly, closed=True, fill=False,
                             edgecolor="k", lw=1.6, zorder=6))
    if len(res.load_pts):
        if len(res.load_pts) > 1:
            dd = np.linalg.norm(res.load_pts[:, None] - res.load_pts[None], axis=2)
            np.fill_diagonal(dd, np.inf)
            spread = 1.6 * float(np.median(dd.min(axis=1)))
        else:
            spread = 1.0
        nb = len(tpoly)
        for i in range(nb):
            a, b = tpoly[i], tpoly[(i + 1) % nb]
            m = 0.5 * (a + b)
            if np.linalg.norm(res.load_pts - m, axis=1).min() <= spread:
                ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="magenta",
                        lw=2.6, zorder=7, solid_capstyle="round")
        if spread_dots:
            ax.plot(res.load_pts[:, 0], res.load_pts[:, 1], "o", ms=3.0,
                    mfc="magenta", mec="white", mew=0.5, zorder=8)
        if show_loads:
            fmag = np.linalg.norm(res.load_vecs, axis=1)
            ext = tcoords[:, 1].max() - tcoords[:, 1].min()
            scl = 0.10 * ext / (fmag.max() + 1e-30)
            for p, v in zip(res.load_pts, res.load_vecs):
                ax.arrow(p[0], p[1], v[0] * scl, v[1] * scl,
                         head_width=0.16, head_length=0.22, fc="k", ec="k",
                         lw=0.5, alpha=0.8, length_includes_head=True, zorder=8)
    return tc


def plot_all_tools(results, work_coords, work_elems, outpath,
                   work_mises=None, show_loads=True, deform_scale=0.0,
                   material_cmap="jet"):
    """ALL tools + the workpiece in one figure with a SHARED colour scale.

    Every tool is drawn with the same von Mises scale (so the most highly
    stressed tool is directly comparable) and each tool's contact interface is
    highlighted in magenta. `results` is a list of ToolStressResult.

    material_cmap : "jet"      -> the workpiece is shown with the SAME jet von
                                 Mises scale as the tools (one unified field; the
                                 material reads low because it flows at ~yield
                                 while the tools reach several times that).
                    "grey"     -> muted grey context (subject = tools).
                    "separate" -> the workpiece gets its OWN colour map (viridis)
                                 and OWN scale with a second colour bar, so both
                                 the material's and the tools' internal stress
                                 distributions are fully resolved despite their
                                 very different magnitudes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.lines import Line2D

    from .mesher import boundary_polygon

    vmax = max(float(r.mises.max()) for r in results)
    levels = np.linspace(0.0, vmax, 21)

    fig, ax = plt.subplots(figsize=(8, 9))

    # workpiece
    wc = None          # material contour set (for its own colour bar)
    wtr = mtri.Triangulation(work_coords[:, 0], work_coords[:, 1], work_elems)
    if work_mises is not None and material_cmap == "separate":
        # material on its OWN viridis scale -> resolves the material's internal
        # field, which a tool-dominated shared scale would flatten to blue
        wnf = _node_avg(work_coords, work_elems, work_mises)
        wlevels = np.linspace(0.0, float(work_mises.max()) or 1.0, 21)
        wc = ax.tricontourf(wtr, wnf, levels=wlevels, cmap="viridis", zorder=1)
        ax.triplot(wtr, lw=0.10, color="k", alpha=0.20, zorder=2)
        wlw = 2.0
    elif work_mises is not None and material_cmap == "jet":
        # same jet scale as the tools -> one unified stress field
        wnf = _node_avg(work_coords, work_elems, work_mises)
        ax.tricontourf(wtr, wnf, levels=levels, cmap="jet",
                       extend="max", zorder=1)
        ax.triplot(wtr, lw=0.10, color="k", alpha=0.20, zorder=2)
        wlw = 2.0       # thicker outline so the material body stays readable
    elif work_mises is not None:
        wnf = _node_avg(work_coords, work_elems, work_mises)
        ax.tricontourf(wtr, wnf, levels=15, cmap="Greys", alpha=0.55, zorder=1)
        ax.triplot(wtr, lw=0.10, color="0.45", alpha=0.5, zorder=2)
        wlw = 1.6
    else:
        ax.tripcolor(wtr, facecolors=np.ones(len(work_elems)), cmap="Greys",
                     vmin=0, vmax=3, alpha=0.4, zorder=1)
        ax.triplot(wtr, lw=0.10, color="0.45", alpha=0.5, zorder=2)
        wlw = 1.6
    wpoly = boundary_polygon(work_coords, work_elems)
    ax.add_patch(plt.Polygon(wpoly, closed=True, fill=False,
                             edgecolor="k", lw=wlw, zorder=5))

    tc = None
    for r in results:
        tc = _draw_tool(ax, r, levels, deform_scale, show_loads)
        # label each tool near the top of its body
        tcoords = r.coords + deform_scale * r.disp
        top = tcoords[tcoords[:, 1].argmax()]
        ax.annotate(f"{r.tool_name}\nmax {r.mises.max():.0f} MPa",
                    xy=top, fontsize=8, ha="center", va="bottom",
                    zorder=9, color="k",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    unified = (work_mises is not None and material_cmap == "jet")
    separate = wc is not None
    cbar_label = ("von Mises [MPa]  (tools + material)" if unified
                  else "tool von Mises [MPa]")
    fig.colorbar(tc, ax=ax, label=cbar_label, shrink=0.8, pad=0.02)
    if separate:
        # second colour bar for the material's own scale
        fig.colorbar(wc, ax=ax, label="material von Mises [MPa]",
                     shrink=0.8, pad=0.10)
    handles = [Line2D([0], [0], color="magenta", lw=2.6,
                      label="contact interface"),
               Line2D([0], [0], color="k", lw=1, marker=">",
                      label="contact pressure on tool")]
    if not unified and not separate:
        handles.append(plt.Polygon([(0, 0)], closed=True, fill=True, fc="0.7",
                                   ec="0.25", label="workpiece (material)"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("r / x [mm]")
    ax.set_ylabel("z / y [mm]")
    names = " + ".join(r.tool_name for r in results)
    if unified:
        extra = "  [material on same scale]"
    elif separate:
        extra = "  [material on its own scale]"
    else:
        extra = ""
    ax.set_title(f"all tools ({names}) & workpiece  "
                 f"stroke={results[0].stroke:.2f} mm{extra}")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return outpath

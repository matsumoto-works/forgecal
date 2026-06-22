"""Incremental quasi-static solver.

Updated Lagrange: each converged step adds the incremental displacement to
the node coordinates; strain increments are evaluated with B computed on the
configuration at the start of the step (hypoelastic formulation).

Newton-Raphson with consistent algorithmic tangent, penalty contact against
rigid tools, adaptive sub-stepping and Gmsh-based adaptive remeshing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import fem
from .contact import (RigidTool, build_contact_points, detect_contacts,
                      contact_force_stiffness, force_tool_coupling)
from .diagnostics import detect_fold, diagnose
from .material import Material
from .mesher import boundary_polygon, element_quality, mesh_polygon
from .plasticity import radial_return_batch, von_mises


@dataclass
class SimConfig:
    mode: str = fem.AXISYMMETRIC          # or fem.PLANE_STRAIN
    stroke: float = 20.0                  # total punch travel [mm]
    punch_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0]))
    d_stroke: float = 0.05                # stroke increment [mm]
    mesh_size: float = 1.5                # [mm]
    kn: float | None = None               # penalty stiffness; default 20*E*size
    kt_eps: float = 0.01                  # friction slip regularization [mm]
    contact_gmax: float | None = None     # capture distance; default mesh_size
    newton_tol: float = 1e-4      # relative to current force scale
    newton_maxit: int = 30
    activeset_maxit: int = 12     # outer contact active-set updates
    du_cap_factor: float = 4.0    # max |ddu| per iteration = factor * d_stroke
    force_ramp_stroke: float = 1.5  # ramp force-tool targets over this punch
                                    # travel [mm] (avoids step-0 runaway of a
                                    # constant-force counter punch)
    max_stall_remesh: int = 6     # give up a run after this many consecutive
                                  # failed-step remeshes without progress
                                  # (prevents hour-long livelock under locking)
    remesh_aspect: float = 3.5            # absolute aspect floor for trigger
    remesh_aspect_factor: float = 2.5     # relative trigger: remesh when max
                                          # aspect exceeds factor x the fresh-
                                          # mesh baseline. Makes the trigger
                                          # adapt to intrinsically anisotropic
                                          # meshes (thin sheet) without a
                                          # per-case flag: thin elements have a
                                          # high baseline so they don't trip it,
                                          # while distorted bulk elements do.
    remesh_check_every: int = 3
    remesh_stroke: float = 1.0            # force remesh every X mm of stroke
    relax_after_remesh: bool = True       # zero-stroke equilibrium pass
    # Conservative remap (experimental, default OFF). Rescales the monotone
    # history vars (PEEQ/damage) at each remesh so their volume integral is
    # preserved (counters the diffusion of plain interpolation). Regression
    # testing (2026-06-14) showed it does NOT give a clean win: the only
    # variant that improved the backward-extrusion load tracking also switched
    # the base interpolation to an SPR/real-mesh scheme that DESTABILISED the
    # fully confined closed-die case; the conservative rescale on top of the
    # validated centroid interpolation gave no net benefit. Kept OFF (behaviour
    # identical to the original remap) and available for future work.
    remap_conserve: bool = False
    remap_conserve_clamp: float = 0.50    # max per-remesh rescale = +/- fraction
    edge_midpoint_contact: bool = False   # DEFORM-style: node-only contact
    pen_tol: float = 0.10                 # allowed penetration depth [mm];
                                          # deeper -> immediate remesh + projection
    pen_check_min_stroke: float = 0.10    # min stroke between pen-triggered remeshes
    refine_factor: float = 0.3            # local size = factor * mesh_size
    refine_radius: float = 3.0            # [mm] around punch working corners
    refine_strain: float = 0.8            # also refine where PEEQ exceeds this
    refine_box: tuple | None = None       # (xmin,xmax,ymin,ymax): force fine
                                          # mesh in this zone (e.g. the shear
                                          # band of a blanking die) to resolve
                                          # rollover / burr / fracture surface
    refine_box_factor: float = 0.25       # size = factor * mesh_size in the box
    refine_points: bool = False           # opt-in punch-corner/high-strain
                                          # point refinement (off: was a no-op
                                          # historically; enabling destabilises
                                          # validated uniform-mesh cases)
    axis_tol: float = 1e-6                # |x| < tol -> ux = 0 (axisym axis)
    holder_force: float = 0.0             # blank holder target force [N]
    min_substep: float = 1e-4
    fbar: bool = False                    # nodal-averaged volumetric strain
                                          # (F-bar) for T3 volumetric locking.
                                          # EXPERIMENTAL: internal force is the
                                          # exact non-local B-bar form but the
                                          # tangent is only approximate, so it
                                          # currently thrashes Newton; needs the
                                          # consistent non-local tangent before
                                          # it can be enabled by default.
    # ── failure diagnosis & material-fold (self-contact onset) detection ──
    # fold_check is OFF by default: the proactive in-loop check FALSE-POSITIVED
    # on confined dies (closed_die etc.) where free surfaces legitimately come
    # close, stopping a previously-completing run early. detect_fold is still
    # used by diagnose() ON ABORT to classify a genuine fold-driven failure.
    fold_check: bool = False      # proactively scan for a self-contact onset
    fold_stop: bool = True        # halt with a design-NG message when fold_check
                                  # finds one (only relevant if fold_check=True)
    fold_check_every: int = 2     # check cadence in steps
    fold_tol_factor: float = 0.35  # fold gap threshold = factor * mesh_size
    fold_cos: float = 0.5         # opposing-normal threshold (n_i.n_j < -cos)


@dataclass
class StepRecord:
    stroke: float
    punch_load: float          # |force along punch dir| [N]
    coords: np.ndarray
    elems: np.ndarray
    sigma: np.ndarray          # (Ne,4)
    ep: np.ndarray             # (Ne,)
    mises: np.ndarray          # (Ne,)
    tool_segs: list = None     # per-tool (M,2,2) segments at this step
                               # (snapshot so post-plots draw tools correctly)
    stage: int = 0             # process stage index (multi-stage)
    damage: np.ndarray = None  # Cockcroft-Latham damage (if enabled)
    eps_e: np.ndarray = None   # (Ne,4) elastic strain tensor [exx,eyy,ezz,gxy]
    vel: np.ndarray = None     # (Nn,2) nodal increment of this step (velocity
                               # field = displacement per stroke increment)
    tool_loads: dict = None    # {tool_name: reaction load [N]} this step
    tool_names: list = None    # tool names in the SAME order as tool_segs
                               # (so a viewer can map a name to its geometry)
    kn: float = 0.0            # contact penalty used in the run (so a post
                               # tool-stress recovers the exact contact force)
    mesh_size: float = 0.0     # target mesh size of the run


class Simulation:
    def __init__(self, cfg: SimConfig, mat: Material,
                 blank_outline: np.ndarray, tools: list[RigidTool],
                 punch_name: str = "punch", holder_name: str | None = None):
        self.cfg = cfg
        self.mat = mat
        self.tools = tools
        self.punch = next(t for t in tools if t.name == punch_name)
        self.holder = (next((t for t in tools if t.name == holder_name), None)
                       if holder_name else None)

        centroid0 = np.asarray(blank_outline, float).mean(axis=0)
        for t in tools:
            t.orient_normals(centroid0)
        self.coords, self.elems = mesh_polygon(blank_outline, cfg.mesh_size,
                                               **self._refine_params())
        self._update_aspect_baseline()
        # geometric thin-sheet detector: few element layers through the
        # thinnest dimension of the blank. Sheets must NOT be periodically
        # remeshed (it destroys the through-thickness layers); bulk parts
        # benefit from periodic remesh. (Aspect-based detection is unreliable
        # because gmsh fills even thin sections with near-equilateral cells.)
        bb = np.asarray(blank_outline, float)
        thk = float((bb.max(axis=0) - bb.min(axis=0)).min())
        self._is_sheet = (thk / cfg.mesh_size) < 6.0

        ne = len(self.elems)
        self.eps_e = np.zeros((ne, 4))
        self.ep = np.zeros(ne)
        self.sigma = np.zeros((ne, 4))
        self.damage = np.zeros(ne)        # Cockcroft-Latham damage (if enabled)

        # penalty per unit contact area [N/mm^2 per mm penetration];
        # fn_i = kn * tributary_area_i * penetration_i
        self.kn = cfg.kn if cfg.kn is not None else 2.0 * mat.E / cfg.mesh_size
        self.gmax = cfg.contact_gmax if cfg.contact_gmax is not None else cfg.mesh_size
        self.history: list[StepRecord] = []
        self.load_curve: list[tuple[float, float]] = []  # (stroke, load N)
        self.tool_load_history: list[dict] = []   # per-step {tool: load N}
        self._tool_loads: dict = {}
        self.remesh_count = 0
        self.stage = 0            # process stage index (set by multi-stage runner)
        self.tracers = None       # flow-net tracer points (set by init_tracers)
        self._adv = None
        self.diagnosis = None     # set on abort / fold-stop (see diagnostics.py)
        self._update_boundary()

    # ── helpers ───────────────────────────────────────────────────────────
    def _update_boundary(self):
        from collections import defaultdict
        cnt = defaultdict(int)
        for tri in self.elems:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                cnt[(min(a, b), max(a, b))] += 1
        s = set()
        edges = []
        for (a, b), c in cnt.items():
            if c == 1:
                s.add(a); s.add(b)
                edges.append((a, b))
        self.surf_nodes = np.array(sorted(s), int)
        self.surf_edges = np.array(edges, int)
        self._rebuild_cpoints()
        self.axis_nodes = np.where(np.abs(self.coords[:, 0]) < self.cfg.axis_tol)[0]

    def _shear_k(self) -> float:
        """Shear yield stress k = sigma_y / sqrt(3) at the mean PEEQ
        (used by the DEFORM-style shear friction model tau = m*k)."""
        return self.mat.flow.stress(float(self.ep.mean())) / np.sqrt(3.0)

    def _build_nl_weight(self):
        """Build the (sparse, row-normalized) nonlocal averaging operator W
        for the current mesh: D_nl = W @ D. Rebuilt only when the mesh
        changes (cached), so the per-increment nonlocal average is one fast
        sparse matvec instead of a neighbour loop."""
        dmg = self.mat.damage
        lc = dmg.lc if dmg is not None else 0.0
        if lc <= 0.0:
            self._nl_W = None
            return
        from scipy.spatial import cKDTree
        cent = self.coords[self.elems].mean(axis=1)
        x = self.coords[self.elems][..., 0]
        y = self.coords[self.elems][..., 1]
        area = np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                      - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])) * 0.5
        tree = cKDTree(cent)
        inv2l2 = 1.0 / (2.0 * lc * lc)
        rows, cols, vals = [], [], []
        for e, nb in enumerate(tree.query_ball_point(cent, 3.0 * lc)):
            nb = np.asarray(nb, int)
            d2 = ((cent[nb] - cent[e])**2).sum(axis=1)
            wgt = np.exp(-d2 * inv2l2) * area[nb]
            wgt /= max(wgt.sum(), 1e-30)
            rows.extend([e] * len(nb)); cols.extend(nb); vals.extend(wgt)
        self._nl_W = sp.csr_matrix((vals, (rows, cols)),
                                   shape=(len(cent), len(cent)))

    def _nonlocal_avg(self, field):
        """Nonlocal Gauss average of an element field over lc (= W @ field).
        Introduces a length scale so the softening band width is set by lc,
        not the mesh -> mesh-independent and refineable."""
        if getattr(self, "_nl_W", None) is None:
            self._build_nl_weight()
        if self._nl_W is None:
            return field
        return np.asarray(self._nl_W @ field).ravel()

    def _rebuild_cpoints(self):
        self.cpoints = build_contact_points(
            self.coords, self.surf_nodes, self.surf_edges,
            axisym=(self.cfg.mode == fem.AXISYMMETRIC))
        if not self.cfg.edge_midpoint_contact:
            # keep node points only (tributary weights already computed)
            self.cpoints = [c for c in self.cpoints if c[0] == c[1]]

    def _fixed_dofs(self):
        fixed = [2 * n for n in self.axis_nodes]  # ux = 0 on axis
        if len(self.axis_nodes) == 0:
            # no symmetry axis (e.g. full-model blanking): pin ux of one
            # far anchor node to remove the horizontal rigid-body mode
            # (choose the bottom-most, right-most node — clamped region)
            c = self.coords
            anchor = int(np.lexsort((-c[:, 0], c[:, 1]))[0])
            fixed.append(2 * anchor)
        return np.array(sorted(set(fixed)), int)

    def springback(self, nsub: int = 10, maxit: int = 30, tol: float = 1e-6):
        """Free elastic springback by incremental reaction release.

        The stored stress is balanced by tool reactions R0 = Fint (on the
        free DOFs). Removing the tools at once can diverge, so R0 is released
        gradually: residual = Fint(u) - lambda * R0 with lambda ramped 1 -> 0
        over nsub damped sub-increments. At lambda = 0 the body is traction
        free (sprung back). The plastic state evolves through radial return,
        so any reverse yielding during recovery is captured. Returns the
        recovered StepRecord.
        """
        coords0 = self.coords.copy()
        ndof = 2 * len(coords0)
        pre = self._precompute_B(coords0)
        if pre is None:
            return None
        fixed = set(int(x) for x in self._fixed_dofs())
        anchor = (self.axis_nodes[np.argmin(coords0[self.axis_nodes, 1])]
                  if len(self.axis_nodes) else int(np.argmin(coords0[:, 1])))
        fixed.add(2 * int(anchor) + 1)
        if len(self.axis_nodes) == 0:          # no symmetry plane: also pin ux
            fixed.add(2 * int(anchor))
        free = np.setdiff1d(np.arange(ndof), np.array(sorted(fixed), int))

        Fint0, _, _ = self._assemble(pre, np.zeros(ndof), ndof)
        R0 = Fint0.copy()
        R0[list(fixed)] = 0.0
        ref = max(np.linalg.norm(R0[free]), 1.0)

        du = np.zeros(ndof)
        state = None
        for k in range(1, nsub + 1):
            lam = 1.0 - k / nsub
            for _ in range(maxit):
                Fint, K, state = self._assemble(pre, du, ndof)
                r = (Fint - lam * R0)[free]
                if np.linalg.norm(r) < tol * ref:
                    break
                try:
                    ddu = spla.spsolve(K[free][:, free].tocsc(), -r)
                except Exception:
                    return None
                if not np.all(np.isfinite(ddu)):
                    return None
                cap = 0.5 * self.cfg.mesh_size
                mx = np.abs(ddu).max()
                if mx > cap:
                    ddu *= cap / mx
                du[free] += ddu
        if state is None:
            return None
        sig, eps_e, ep, D_new = state
        self.sigma, self.eps_e, self.ep = sig.copy(), eps_e.copy(), ep.copy()
        if D_new is not None:
            self.damage = D_new.copy()
        self.coords = coords0 + du.reshape(-1, 2)
        rec = self._make_record(self.history[-1].stroke if self.history else 0.0,
                                0.0)
        self.history.append(rec)
        return rec

    # ── assembly (vectorized; B cached per increment) ────────────────────
    def _precompute_B(self, coords0):
        """B matrices on the step-start configuration (constant during the
        Newton iterations of one increment - hypoelastic UL)."""
        B, w, A, edof = fem.b_matrix_batch(coords0, self.elems, self.cfg.mode)
        if A.min() <= 0:
            return None
        rows = np.repeat(edof, 6, axis=1).ravel()        # (Ne*36,)
        cols = np.tile(edof, (1, 6)).ravel()
        fbar = self._build_fbar(B, w, edof, ndof=2 * len(coords0)) \
            if self.cfg.fbar else None
        return B, w, edof, rows, cols, fbar

    def _build_fbar(self, B, w, edof, ndof):
        """Sparse F-bar strain-displacement operator B_bar (4Ne x ndof).

        B_bar = B_global + (1/3) Sm (M - I) T   where
          B_global : standard strain operator (4Ne x ndof)
          T        : trace-strain operator (theta_e = T u),  row e = b_theta_e
          M        : nodal volumetric averaging,
                     M = (1/3) C^T diag(1/W) C diag(w)
          Sm       : scatters a per-element scalar onto the xx,yy,zz rows
        Constant during one increment (depends only on coords0, w) -> built
        once, reused over the Newton iterations; gives a CONSISTENT tangent
        K = B_bar^T (w D) B_bar.
        """
        ne = len(self.elems)
        elems = self.elems
        # B_global (4Ne x ndof)
        r = (4 * np.arange(ne)[:, None, None] + np.arange(4)[None, :, None]
             + np.zeros((1, 1, 6), int)).ravel()
        c = np.broadcast_to(edof[:, None, :], (ne, 4, 6)).ravel()
        B_global = sp.csr_matrix((B.ravel(), (r, c)), shape=(4 * ne, ndof))
        # T (Ne x ndof): theta_e = (B[0]+B[1]+B[2]) . u_e
        b_th = B[:, 0, :] + B[:, 1, :] + B[:, 2, :]            # (Ne,6)
        rT = np.repeat(np.arange(ne), 6)
        T = sp.csr_matrix((b_th.ravel(), (rT, edof.ravel())), shape=(ne, ndof))
        # C (nn x Ne) incidence, W_n = C w
        nn = ndof // 2
        rC = elems.ravel()
        cC = np.repeat(np.arange(ne), 3)
        C = sp.csr_matrix((np.ones(3 * ne), (rC, cC)), shape=(nn, ne))
        W = np.asarray(C @ w).ravel()
        invW = sp.diags(1.0 / np.maximum(W, 1e-30))
        M = (1.0 / 3.0) * (C.T @ (invW @ (C @ sp.diags(w))))   # (Ne x Ne)
        # Sm (4Ne x Ne): scalar -> xx,yy,zz rows
        rows_s = np.concatenate([4 * np.arange(ne) + k for k in (0, 1, 2)])
        cols_s = np.tile(np.arange(ne), 3)
        Sm = sp.csr_matrix((np.ones(3 * ne), (rows_s, cols_s)),
                           shape=(4 * ne, ne))
        Bbar = (B_global + (1.0 / 3.0) * Sm @ ((M - sp.identity(ne)) @ T)).tocsr()
        return dict(Bbar=Bbar, w=w, ne=ne)

    def _assemble(self, pre, du, ndof):
        """Internal force + consistent tangent using precomputed B.

        With cfg.fbar, the strain is evaluated through the sparse F-bar
        operator B_bar (nodal-averaged volumetric part), and BOTH the
        internal force (F = B_bar^T (w sigma)) and the tangent
        (K = B_bar^T (w D) B_bar) use the SAME B_bar -> fully consistent,
        which is required for Newton to converge under volumetric locking.
        """
        B, w, edof, rows, cols, fbar = pre

        Dst = self.damage if self.mat.damage is not None else None
        Dsoft = getattr(self, "_D_soft", None)   # nonlocal softening field
        if fbar is None:
            deps = np.einsum("eij,ej->ei", B, du[edof])
            eps_tr = self.eps_e + deps
            sig, Dalg, eps_e_new, ep_new, D_new = radial_return_batch(
                self.mat, eps_tr, self.ep, Dst, Dsoft)
            fe = np.einsum("e,eai,ea->ei", w, B, sig)
            Fint = np.zeros(ndof)
            np.add.at(Fint, edof.ravel(), fe.ravel())
            Ke = np.einsum("e,eai,eab,ebj->eij", w, B, Dalg, B)
            K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(ndof, ndof))
            return Fint, K, (sig, eps_e_new, ep_new, D_new)

        # ── F-bar path (sparse, consistent) ──────────────────────────────
        Bbar = fbar["Bbar"]
        ne = fbar["ne"]
        eps_bar = (Bbar @ du).reshape(ne, 4)               # F-bar strain incr
        eps_tr = self.eps_e + eps_bar
        sig, Dalg, eps_e_new, ep_new, D_new = radial_return_batch(
            self.mat, eps_tr, self.ep, Dst, Dsoft)
        # F = B_bar^T (w sigma)
        sig_stack = (sig * w[:, None]).ravel()             # (4Ne,)
        Fint = np.asarray(Bbar.T @ sig_stack).ravel()
        # K = B_bar^T blkdiag(w D) B_bar
        wD = Dalg * w[:, None, None]                       # (Ne,4,4)
        bi = (4 * np.arange(ne)[:, None, None]
              + np.arange(4)[None, :, None] + np.zeros((1, 1, 4), int)).ravel()
        bj = (4 * np.arange(ne)[:, None, None]
              + np.zeros((1, 4, 1), int) + np.arange(4)[None, None, :]).ravel()
        Dblk = sp.csr_matrix((wD.ravel(), (bi, bj)), shape=(4 * ne, 4 * ne))
        K = (Bbar.T @ Dblk @ Bbar).tocsr()
        return Fint, K, (sig, eps_e_new, ep_new, D_new)

    # ── one stroke increment ─────────────────────────────────────────────
    def _solve_increment(self, dstroke: float):
        cfg = self.cfg
        coords0 = self.coords.copy()
        self._rebuild_cpoints()      # tributary lengths follow deformation
        # nonlocal softening field (constant during this increment's Newton)
        if self.mat.damage is not None and self.mat.damage.lc > 0.0:
            self._D_soft = self._nonlocal_avg(self.damage)
        else:
            self._D_soft = None
        nn = len(coords0)
        ndof = 2 * nn
        du = np.zeros(ndof)
        tool_du = {t.name: np.zeros(2) for t in self.tools}
        disp0 = {t.name: t.displacement.copy() for t in self.tools}
        tool_du[self.punch.name] = cfg.punch_dir * dstroke
        self.punch.displacement = self.punch.displacement + cfg.punch_dir * dstroke
        # other displacement-controlled tools (e.g. a counter punch driven at
        # a prescribed rate instead of force control): move free_dir * vel * d
        for t in self.tools:
            if t is self.punch or getattr(t, "control", "rigid") != "displacement":
                continue
            step_mv = t.free_dir * getattr(t, "vel", 0.0) * dstroke
            tool_du[t.name] = step_mv
            t.displacement = t.displacement + step_mv

        fixed = self._fixed_dofs()
        free = np.setdiff1d(np.arange(ndof), fixed)

        self.last_newton = []

        def fail():
            for t in self.tools:
                t.displacement = disp0[t.name]
            self.punch.displacement = (disp0[self.punch.name])
            return False, None

        pre = self._precompute_B(coords0)
        if pre is None:
            return fail()      # inverted element at step start

        # force-controlled tools get an extra scalar DOF (position along
        # free_dir) solved monolithically with the material
        force_tools = [t for t in self.tools if t.control == "force"]
        s_ft = {t.name: 0.0 for t in force_tools}   # this-step increment of pos
        base_s = {t.name: float(disp0[t.name] @ t.free_dir)
                  for t in force_tools}             # prior travel along free_dir

        # ── outer active-set loop ────────────────────────────────────
        prev_sets = None
        tool_forces: dict[str, np.ndarray] = {}
        for outer in range(cfg.activeset_maxit):
            cur = coords0 + du.reshape(-1, 2)
            all_contacts = {t.name: detect_contacts(cur, self.cpoints, t,
                                                    self.gmax)
                            for t in self.tools}
            # compare active sets by contact POINT only: a node sliding from
            # one tool segment to the adjacent one (e.g. across the punch
            # corner) is the same contact, not a set change - otherwise the
            # outer loop ping-pongs forever at corners
            sets = {nm: frozenset(c[0] for c in cs)
                    for nm, cs in all_contacts.items()}
            if prev_sets is not None and sets == prev_sets and converged:
                break          # set stable and equilibrium (incl. tool force)
            prev_sets = sets

            # ── inner Newton with FIXED active set ──────────────────
            converged = False
            damp = 1.0
            nrm_hist = []
            for it in range(cfg.newton_maxit):
                Fint, K, new_state = self._assemble(pre, du, ndof)

                Fc = np.zeros(ndof)
                crows, ccols, cvals = [], [], []
                punch_force = np.zeros(2)
                cur = coords0 + du.reshape(-1, 2)
                for t in self.tools:
                    contacts = all_contacts[t.name]
                    if not contacts:
                        continue
                    Fc_t, (r_, c_, v_), ftot = contact_force_stiffness(
                        contacts, self.cpoints, cur, du, t, self.kn,
                        cfg.kt_eps, ndof, tool_du[t.name],
                        shear_k=self._shear_k())
                    Fc += Fc_t
                    crows.extend(r_); ccols.extend(c_); cvals.extend(v_)
                    tool_forces[t.name] = -ftot   # reaction on the tool
                    if t is self.punch:
                        punch_force = -ftot

                Kc = sp.csr_matrix((cvals, (crows, ccols)), shape=(ndof, ndof))
                R = Fint - Fc
                Rf = R[free]

                # ── monolithic force-control augmentation ──────────────
                # add one scalar DOF per force tool: its position along
                # free_dir, coupled to the material and constrained by
                # f_resist = f_target
                nft = len(force_tools)
                R_s = np.zeros(nft)
                Ccols = []
                dss = np.zeros(nft)
                for j, t in enumerate(force_tools):
                    C, d_ss, f_res = force_tool_coupling(
                        all_contacts[t.name], self.cpoints, cur, t,
                        self.kn, ndof)
                    f_tgt = t.f_const + t.k_spring * (base_s[t.name]
                                                      + s_ft[t.name])
                    R_s[j] = f_res - f_tgt
                    Ccols.append(C[free])
                    # d(R_s)/ds = d(f_resist)/ds - k_spring. Floor the
                    # magnitude so the s-DOF stays non-singular when the tool
                    # is barely in contact (d_ss ~ 0); the floor is a
                    # fictitious stiffness that lets the tool advance to make
                    # contact and vanishes from the converged state (R_s->0).
                    dval = d_ss - t.k_spring
                    floor = self.kn * self.cfg.mesh_size
                    if abs(dval) < floor:
                        dval = floor
                    dss[j] = dval
                nrm = np.linalg.norm(np.concatenate([Rf, R_s]))
                self.last_newton.append(nrm)
                fscale = max(np.linalg.norm(Fint), np.linalg.norm(Fc),
                             np.linalg.norm([t.f_const for t in force_tools])
                             if nft else 0.0, 1.0)
                if nrm < cfg.newton_tol * fscale or nrm < 1e-8:
                    converged = True
                    break
                Asub = (K + Kc)[free][:, free]
                try:
                    if nft == 0:
                        ddu = spla.spsolve(Asub.tocsc(), -Rf)
                        dds = np.zeros(0)
                    else:
                        # build augmented [[A, -C],[-C^T, dss]]
                        Cmat = sp.csc_matrix(np.array(Ccols).T)  # (nfree, nft)
                        top = sp.hstack([Asub, -Cmat])
                        bot = sp.hstack([(-Cmat).T,
                                         sp.csc_matrix(np.diag(dss))])
                        Aug = sp.vstack([top, bot]).tocsc()
                        rhs = -np.concatenate([Rf, R_s])
                        sol = spla.spsolve(Aug, rhs)
                        ddu = sol[:len(free)]
                        dds = sol[len(free):]
                except Exception:
                    return fail()
                if not np.all(np.isfinite(ddu)):
                    return fail()
                cap = cfg.du_cap_factor * max(dstroke, 1e-2)
                mx = np.abs(ddu).max()
                if mx > cap:
                    ddu *= cap / mx
                nrm_hist.append(nrm)
                if len(nrm_hist) >= 3 and nrm > 0.7 * max(nrm_hist[-3:-1]):
                    damp = max(damp * 0.6, 0.05)
                else:
                    damp = min(damp * 1.3, 1.0)
                du[free] += damp * ddu
                # apply the force-tool position increments
                for j, t in enumerate(force_tools):
                    ds = float(damp * dds[j])
                    s_ft[t.name] += ds
                    t.displacement = t.displacement + t.free_dir * ds
                    tool_du[t.name] = tool_du[t.name] + t.free_dir * ds
            if not converged:
                return fail()

        if not converged:
            return fail()
        # record per-tool reaction loads (punch along punch_dir; force tools
        # along their free_dir = clamp / back-pressure they supply) for the
        # load-stroke graph
        self._tool_loads = {}
        for t in self.tools:
            f = tool_forces.get(t.name, np.zeros(2))
            d = cfg.punch_dir if t is self.punch else t.free_dir
            self._tool_loads[t.name] = abs(float(f @ d))
        # commit state
        sig, eps_e, ep, D_new = new_state
        self.sigma = sig.copy()
        self.eps_e = eps_e.copy()
        self.ep = ep.copy()
        if D_new is not None:
            self.damage = D_new.copy()
        self.coords = coords0 + du.reshape(-1, 2)
        # remember the increment (pre-remesh mesh + displacement) so flow-net
        # tracers can be advected on the valid configuration
        self._adv = (coords0, du.copy())
        return True, punch_force

    # ── local refinement size field ──────────────────────────────────────
    def _size_func(self, high_strain_pts: np.ndarray | None = None):
        """Target size callable: fine near punch working corners (and
        optionally near high-strain material points), coarse elsewhere."""
        cfg = self.cfg
        base = cfg.mesh_size
        fine = cfg.refine_factor * base
        r0, r1 = cfg.refine_radius, 2.0 * cfg.refine_radius

        if self.punch.closed_polygon is not None:
            pts = self.punch.closed_polygon + self.punch.displacement
            ymin = pts[:, 1].min()
            zones = [pts[pts[:, 1] < ymin + 2.5]]   # working face vertices
        else:
            zones = []
        if high_strain_pts is not None and len(high_strain_pts):
            zones.append(high_strain_pts)
        zone_pts = np.vstack(zones) if zones else None
        return zone_pts  # (kept name for callers; now returns refine points)

    def _refine_params(self, high_strain_pts=None):
        """gmsh size-field parameters for local refinement (Box + point
        Distance/Threshold). Returns a kwargs dict for mesh_polygon.

        Point-based (corner / high-strain) refinement is opt-in via
        cfg.refine_points — it was historically a no-op and enabling it can
        destabilise the validated uniform-mesh cases, so it stays OFF unless
        requested. The explicit refine_box (e.g. blanking shear zone) always
        applies."""
        cfg = self.cfg
        kw = {}
        if getattr(cfg, "refine_points", False):
            pts = self._size_func(high_strain_pts)
            if pts is not None and len(pts):
                kw.update(refine_pts=pts,
                          refine_size=cfg.refine_factor * cfg.mesh_size,
                          refine_dist=cfg.refine_radius)
        if cfg.refine_box is not None:
            kw.update(refine_box=cfg.refine_box,
                      box_size=cfg.refine_box_factor * cfg.mesh_size)
        return kw

    def _update_aspect_baseline(self):
        """Record the fresh-mesh aspect quality (90th pct) so the remesh
        trigger can fire on relative degradation rather than an absolute
        threshold that penalizes intrinsically thin (sheet) elements."""
        _, aspect = element_quality(self.coords, self.elems)
        self._aspect_base = float(np.percentile(aspect, 90)) if len(aspect) else 1.0
        self._nl_W = None        # invalidate nonlocal operator (mesh changed)

    # ── remeshing ─────────────────────────────────────────────────────────
    def _maybe_remesh(self, force=False):
        area, aspect = element_quality(self.coords, self.elems)
        # relative trigger: degraded beyond factor x baseline, OR above the
        # absolute floor, OR inverted element
        trig = max(self.cfg.remesh_aspect,
                   self.cfg.remesh_aspect_factor * getattr(self, "_aspect_base", 1.0))
        if not force and (aspect.max() < trig and area.min() > 0):
            return False
        poly = boundary_polygon(self.coords, self.elems)
        # refine near punch corners + zones already strained heavily;
        # adaptive threshold: at large deformation most of the billet exceeds
        # a fixed PEEQ value, which would refine everything and dilute the
        # corner refinement -> track the top of the strain distribution
        cents = self.coords[self.elems].mean(axis=1)
        thr = max(self.cfg.refine_strain, 0.6 * float(self.ep.max()))
        hs = cents[self.ep > thr]
        try:
            new_nodes, new_elems = mesh_polygon(poly, self.cfg.mesh_size,
                                                **self._refine_params(hs))
        except Exception:
            return False
        new_sigma, new_eps_e, new_ep, new_damage = self._remap_state(
            self.coords, self.elems, new_nodes, new_elems)
        self.coords = new_nodes
        self.elems = new_elems
        self.sigma = new_sigma
        self.eps_e = new_eps_e
        self.ep = new_ep
        self.damage = new_damage
        self._update_boundary()
        self._project_out_of_tools()
        self._update_aspect_baseline()
        self.remesh_count += 1
        return True

    def _remap_state(self, old_nodes, old_elems, new_nodes, new_elems):
        """Conservative state transfer old mesh -> new mesh.

        Base interpolation is the proven linear interpolation over the old
        element centroids (nearest-centroid fallback outside the centroid hull).
        On top of it the monotone history variables (PEEQ, damage) are rescaled
        so their VOLUME INTEGRAL is conserved across the remesh: plain
        interpolation diffuses the plastic-strain peaks, so without this the
        hardening (hence flow stress and load) drifts down a little at every
        remesh -> the cumulative late-stage load deficit (B4) and the asymmetric
        load-sawtooth troughs (B1). Conserving the integral removes that drift.

        The conservative rescale is the part that fixes the load; an earlier
        attempt to also switch the base interpolation to an SPR nodal-recovery /
        real-mesh scheme destabilised the convergence of the fully confined
        closed-die case at peak load, so the base scheme is kept as the
        validated centroid interpolation.
        """
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import cKDTree

        axisym = self.cfg.mode == fem.AXISYMMETRIC

        def evol(nodes, elems):
            p = nodes[elems]
            x, y = p[..., 0], p[..., 1]
            A = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                             - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
            if axisym:
                rc = np.maximum(x.mean(axis=1), 1e-9)
                return 2.0 * np.pi * rc * A
            return A

        Vold = evol(old_nodes, old_elems)
        Vnew = evol(new_nodes, new_elems)
        old_cent = old_nodes[old_elems].mean(axis=1)
        new_cent = new_nodes[new_elems].mean(axis=1)
        _, near = cKDTree(old_cent).query(new_cent)

        def interp(field):
            f_near = field[near]
            try:
                f_lin = LinearNDInterpolator(old_cent, field)(new_cent)
            except Exception:
                return f_near.copy()
            bad = ~np.isfinite(f_lin)
            if f_lin.ndim > 1:
                bad = bad.any(axis=1)
            f_lin[bad] = f_near[bad]
            return f_lin

        def conserve(fe, fv):
            """Rescale fv so its volume integral matches fe's (gentle clamp)."""
            if not self.cfg.remap_conserve:
                return fv
            old_int = float(np.dot(fe, Vold))
            new_int = float(np.dot(fv, Vnew))
            if old_int > 1e-30 and new_int > 1e-30:
                cl = self.cfg.remap_conserve_clamp
                fv = fv * min(max(old_int / new_int, 1.0 - cl), 1.0 + cl)
            return fv

        new_sigma = interp(self.sigma)
        new_eps_e = interp(self.eps_e)
        new_ep = np.maximum(conserve(self.ep, interp(self.ep)), 0.0)
        new_damage = np.maximum(conserve(self.damage, interp(self.damage)), 0.0)
        return new_sigma, new_eps_e, new_ep, new_damage

    def _project_out_of_tools(self, snap_tol: float = 0.05):
        """Boundary regularization after remesh.

        - nodes inside a tool are pushed back onto the surface;
        - nodes just OUTSIDE the surface (within snap_tol) are snapped onto
          it. Without the snap, interpolation noise plus the one-way push
          ratchets the boundary outward at every remesh, opening an
          artificial gap along the punch side wall.
        """
        from matplotlib.path import Path
        for tool in self.tools:
            if tool.closed_polygon is None:
                continue
            poly = tool.closed_polygon + tool.displacement
            path = Path(poly)
            segs = tool.current_segments()
            pts = self.coords[self.surf_nodes]
            q = pts.copy()
            q[:, 0] = np.maximum(q[:, 0], 1e-6)   # axis-edge nudge
            inside = path.contains_points(q)
            for k in range(len(self.surf_nodes)):
                nid = self.surf_nodes[k]
                p = self.coords[nid]
                best_d, best_cp = 1e30, None
                for si in range(len(segs)):
                    if tool.active_seg is not None and not tool.active_seg[si]:
                        continue
                    a, b = segs[si]
                    ab = b - a
                    L2 = ab @ ab
                    if L2 < 1e-20:
                        continue
                    xi = np.clip(((p - a) @ ab) / L2, 0.0, 1.0)
                    cp = a + xi * ab
                    d = np.linalg.norm(p - cp)
                    if d < best_d:
                        best_d, best_cp = d, cp
                if best_cp is None:
                    continue
                if inside[k] or best_d <= snap_tol:
                    self.coords[nid] = best_cp

    def _max_penetration(self) -> float:
        """Deepest penetration of any boundary node into any tool [mm]."""
        from matplotlib.path import Path
        worst = 0.0
        pts = self.coords[self.surf_nodes]
        for tool in self.tools:
            if tool.closed_polygon is None:
                continue
            poly = tool.closed_polygon + tool.displacement
            q = pts.copy()
            q[:, 0] = np.maximum(q[:, 0], 1e-6)   # axis-edge nudge
            inside = Path(poly).contains_points(q)
            if not inside.any():
                continue
            segs = tool.current_segments()
            qpts = pts[inside]                       # (P,2) penetrating nodes
            a_s = segs[:, 0]                         # (S,2)
            ab_s = segs[:, 1] - segs[:, 0]           # (S,2)
            L2_s = np.einsum("si,si->s", ab_s, ab_s)
            valid = L2_s > 1e-20
            if tool.active_seg is not None:
                valid &= tool.active_seg
            if not valid.any():
                continue
            rel = qpts[:, None, :] - a_s[None, :, :]          # (P,S,2)
            xi = np.clip(np.einsum("psi,si->ps", rel, ab_s)
                         / np.where(L2_s > 1e-20, L2_s, 1.0), 0.0, 1.0)
            foot = a_s[None] + xi[..., None] * ab_s[None]      # (P,S,2)
            d = np.linalg.norm(qpts[:, None, :] - foot, axis=-1)  # (P,S)
            d[:, ~valid] = 1e30
            # nearest segment per node, then deepest over all penetrating nodes
            worst = max(worst, float(d.min(axis=1).max()))
        return worst

    # ── flow-net tracers (DEFORM-style fiber flow) ───────────────────────
    def init_tracers(self, spacing=None, n_div=None):
        """Seed a grid of Lagrangian tracer points over the initial material.
        Advected with the material every increment (independent of the FE mesh,
        so they survive remeshing) -> their deformed grid is the flow net /
        fiber flow, revealing material flow, laps and defects.

        n_div: if given, the grid is n_div cells across the FULL material extent
        in each direction (so the lines reach the material edges, and the
        density is width/n_div x height/n_div). Otherwise a square `spacing` (or
        2*mesh_size) is used."""
        import matplotlib.tri as mtri
        lo = self.coords.min(axis=0)
        hi = self.coords.max(axis=0)
        if n_div is not None:
            xs = np.linspace(lo[0], hi[0], int(n_div) + 1)
            ys = np.linspace(lo[1], hi[1], int(n_div) + 1)
        else:
            sp_ = spacing or 2.0 * self.cfg.mesh_size
            xs = np.arange(lo[0], hi[0] + sp_ * 0.5, sp_)
            ys = np.arange(lo[1], hi[1] + sp_ * 0.5, sp_)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        tri = mtri.Triangulation(self.coords[:, 0], self.coords[:, 1], self.elems)
        inside = tri.get_trifinder()(pts[:, 0], pts[:, 1]) >= 0
        self._tr_grid_shape = gx.shape
        self._tr_inside = inside.reshape(gx.shape)
        # axisymmetric: tracers seeded on the symmetry axis (r=0) must STAY on
        # it. They sit on the mesh boundary, where trifinder returns -1 and the
        # nearest-element fallback would inject a spurious radial drift.
        self._tr_axis = (np.abs(pts[:, 0]) < max(self.cfg.axis_tol, 1e-9)
                         if self.cfg.mode == fem.AXISYMMETRIC
                         else np.zeros(len(pts), bool))
        self.tracers = pts
        self.tracer_history = [pts.copy()]

    def _advect_tracers(self):
        """Move tracers by the last increment's displacement, interpolated on
        the pre-increment mesh (barycentric). Tracers that fall just outside
        the mesh (the free surface moved past them) are advected by their
        NEAREST element instead of being left behind — otherwise surface
        tracers stall and the flow net overshoots the deformed material."""
        if getattr(self, "tracers", None) is None or self._adv is None:
            return
        import matplotlib.tri as mtri
        from scipy.spatial import cKDTree
        coords0, du = self._adv
        d2 = du.reshape(-1, 2)
        elems = self.elems
        tri = mtri.Triangulation(coords0[:, 0], coords0[:, 1], elems)
        finder = tri.get_trifinder()
        p = self.tracers
        ti = finder(p[:, 0], p[:, 1])
        cent = coords0[elems].mean(axis=1)
        ctree = cKDTree(cent)
        for k in range(len(p)):
            e = ti[k]
            clamp = False
            if e < 0:                          # outside: use nearest element
                e = int(ctree.query(p[k])[1])
                clamp = True
            n0, n1, n2 = elems[e]
            a, b, c = coords0[n0], coords0[n1], coords0[n2]
            det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(det) < 1e-30:
                continue
            l0 = ((b[1] - c[1]) * (p[k, 0] - c[0])
                  + (c[0] - b[0]) * (p[k, 1] - c[1])) / det
            l1 = ((c[1] - a[1]) * (p[k, 0] - c[0])
                  + (a[0] - c[0]) * (p[k, 1] - c[1])) / det
            if clamp:                          # clamp bary coords to the element
                l0 = min(max(l0, 0.0), 1.0)
                l1 = min(max(l1, 0.0), 1.0)
            l2 = 1.0 - l0 - l1
            p[k] += l0 * d2[n0] + l1 * d2[n1] + l2 * d2[n2]
        # axisymmetric: no tracer crosses the axis; axis tracers stay on it
        if self.cfg.mode == fem.AXISYMMETRIC:
            np.maximum(p[:, 0], 0.0, out=p[:, 0])
            ax = getattr(self, "_tr_axis", None)
            if ax is not None:
                p[ax, 0] = 0.0

    def _relax(self):
        """Zero-stroke equilibrium pass after remesh: re-balances the mapped
        (non-equilibrated) stress field and removes the load-drop artifact."""
        try:
            ok, _ = self._solve_increment(0.0)
        except Exception:
            ok = False
        return ok

    def _make_record(self, stroke, load):
        from .plasticity import von_mises_batch
        mises = von_mises_batch(self.sigma)
        segs = [t.current_segments().copy() for t in self.tools]
        # nodal velocity field = the last committed increment du (on the
        # current mesh); None at step 0 or right after a mesh change
        vel = None
        if self._adv is not None:
            du = self._adv[1].reshape(-1, 2)
            if len(du) == len(self.coords):
                vel = du.copy()
        return StepRecord(stroke, load, self.coords.copy(), self.elems.copy(),
                          self.sigma.copy(), self.ep.copy(), mises,
                          tool_segs=segs, stage=self.stage,
                          damage=self.damage.copy(),
                          eps_e=self.eps_e.copy(), vel=vel,
                          tool_loads=dict(self._tool_loads),
                          tool_names=[t.name for t in self.tools],
                          kn=float(self.kn), mesh_size=float(self.cfg.mesh_size))

    # ── main loop ─────────────────────────────────────────────────────────
    def run(self, callback=None, record_initial=True, base_stroke=0.0):
        cfg = self.cfg
        # step-0 snapshot: initial (carried) state before this stage loads
        if record_initial:
            rec0 = self._make_record(base_stroke, 0.0)
            self.history.append(rec0)
            if callback:
                callback(0, rec0)
        stroke_done = 0.0
        stroke_since_remesh = 0.0
        dstr = cfg.d_stroke
        step = 0
        stall_remesh = 0          # consecutive failed-step remeshes
        while stroke_done < cfg.stroke - 1e-9:
            d = min(dstr, cfg.stroke - stroke_done)
            ok, pf = self._solve_increment(d)
            if not ok:
                # remesh at most once per failing step (re-remeshing on every
                # substep halving is what livelocks for an hour under locking)
                if stall_remesh == 0 and self._maybe_remesh(force=True):
                    if cfg.relax_after_remesh:
                        self._relax()
                    ok, pf = self._solve_increment(d)
                    if ok:
                        stroke_since_remesh = 0.0
                if not ok and dstr < 32 * cfg.min_substep and cfg.edge_midpoint_contact:
                    # last resort: midpoint chattering can deadlock tiny
                    # increments -> pass this one with node-only contact
                    cfg.edge_midpoint_contact = False
                    ok, pf = self._solve_increment(d)
                    cfg.edge_midpoint_contact = True
                    if ok:
                        print(f"    [recover] node-only contact pass at d={d:.5f}")
                if not ok:
                    trace = ", ".join(f"{v:.2e}" for v in self.last_newton[-8:])
                    print(f"    [fail] d={d:.5f} residual trace: ...{trace}")
                    dstr *= 0.5
                    stall_remesh += 1
                    if dstr < cfg.min_substep or stall_remesh > cfg.max_stall_remesh:
                        print(f"    [abort] no convergence at stroke="
                              f"{stroke_done:.3f} mm after {stall_remesh} "
                              f"attempts -> returning partial result "
                              f"({len(self.history)} steps)")
                        self.diagnosis = diagnose(self)
                        print(self.diagnosis.format())
                        return self.history
                    continue
            stall_remesh = 0
            stroke_done += d
            stroke_since_remesh += d
            step += 1
            load = abs(float(pf @ cfg.punch_dir))
            self.load_curve.append((base_stroke + stroke_done, load))
            self.tool_load_history.append(
                dict(self._tool_loads, _stroke=base_stroke + stroke_done))
            rec = self._make_record(base_stroke + stroke_done, load)
            self.history.append(rec)
            if self.tracers is not None:
                self._advect_tracers()
                self.tracer_history.append(self.tracers.copy())
            if callback:
                if callback(step, rec):     # truthy return -> stop cleanly
                    return self.history
            # material-fold (self-contact onset) watch: a fold means the part
            # would lap/entrap -> report it as a design-NG and stop cleanly,
            # rather than grinding into the remesh livelock it inevitably causes
            if cfg.fold_check and step % max(cfg.fold_check_every, 1) == 0:
                fold = detect_fold(self.coords, self.elems, self.surf_edges,
                                   cfg.mesh_size, tol_factor=cfg.fold_tol_factor,
                                   cos_thresh=cfg.fold_cos)
                if fold is not None:
                    self.diagnosis = diagnose(self)
                    print(f"    [design-NG] material fold at stroke="
                          f"{stroke_done:.3f} mm -> stopping "
                          f"({len(self.history)} steps)")
                    print(self.diagnosis.format())
                    if cfg.fold_stop:
                        return self.history
            # grow increment back
            if dstr < cfg.d_stroke:
                dstr = min(dstr * 1.5, cfg.d_stroke)
            # DEFORM-style penetration tolerance: deep penetration is the
            # remesh trigger (remesh projects the boundary back onto the
            # tool surfaces); shallow penetration is simply tolerated
            # deep penetration forces a remesh (boundary re-projection);
            # the stroke interval only triggers a quality CHECK (the relative
            # aspect criterion decides) so a healthy mesh - e.g. a thin sheet -
            # is not needlessly re-triangulated, which would destroy its
            # through-thickness layers
            pen_trigger = (stroke_since_remesh >= cfg.pen_check_min_stroke
                           and self._max_penetration() > cfg.pen_tol)
            # periodic forced remesh refreshes a distorting BULK mesh (improves
            # accuracy), but destroys a thin SHEET's through-thickness layers,
            # so it is suppressed for sheets (geometric thickness detector).
            # The relative-degradation trigger still catches real degradation.
            periodic = (not self._is_sheet
                        and stroke_since_remesh >= cfg.remesh_stroke)
            force_rm = pen_trigger or periodic
            do_check = (force_rm or step % cfg.remesh_check_every == 0)
            if do_check:
                if self._maybe_remesh(force=force_rm):
                    stroke_since_remesh = 0.0
                    if cfg.relax_after_remesh:
                        self._relax()
        return self.history

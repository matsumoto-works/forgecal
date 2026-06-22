"""Failure diagnosis and material-fold (self-contact onset) detection.

The solver only models material<->rigid-tool contact, not material self-contact.
When a free surface folds onto itself the two faces would interpenetrate, which
shows up as a non-convergence -> remesh livelock with a cryptic "no convergence"
message. This module turns that into actionable feedback:

* detect_fold()  finds an incipient self-contact (two boundary faces with
  opposing outward normals coming together) and reports WHERE. Because a fold
  means the product itself is defective (a lap / entrapment), detecting it and
  halting with "this design needs improvement" is the useful outcome even
  without true self-contact resolution.
* diagnose()  classifies WHY a step failed (fold / collapsed-or-inverted mesh /
  excessive penetration / plain non-convergence) and suggests a remedy, instead
  of only printing the raw residual trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesher import element_quality


@dataclass
class Diagnosis:
    cause: str                       # fold / mesh_collapse / penetration / nonconvergence
    title: str                       # one-line human summary
    suggestions: list[str] = field(default_factory=list)
    location: tuple | None = None    # (r/x, z/y) [mm] of the worst spot
    metrics: dict = field(default_factory=dict)
    design_ng: bool = False          # True -> the geometry/process is the problem

    def format(self) -> str:
        tag = "design-NG" if self.design_ng else "diagnosis"
        lines = [f"    [{tag}] {self.title}"]
        if self.location is not None:
            lines.append(f"      場所 (r,z) = ({self.location[0]:.2f}, "
                         f"{self.location[1]:.2f}) mm")
        if self.metrics:
            m = "  ".join(f"{k}={v}" for k, v in self.metrics.items())
            lines.append(f"      指標: {m}")
        for s in self.suggestions:
            lines.append(f"      → {s}")
        return "\n".join(lines)


def _boundary_normals(coords, elems, surf_edges):
    """Outward unit normal + midpoint of every boundary edge.

    The normal is defined physically (pointing away from the edge's interior
    triangle vertex), so it is independent of how surf_edges happens to be
    ordered."""
    # map each undirected edge -> the opposite (interior) vertex of its triangle
    opp = {}
    for tri in elems:
        for a, b, c in ((tri[0], tri[1], tri[2]),
                        (tri[1], tri[2], tri[0]),
                        (tri[2], tri[0], tri[1])):
            opp[(min(a, b), max(a, b))] = c
    mids = np.empty((len(surf_edges), 2))
    nrm = np.empty((len(surf_edges), 2))
    for i, (a, b) in enumerate(surf_edges):
        pa, pb = coords[a], coords[b]
        mid = 0.5 * (pa + pb)
        e = pb - pa
        n = np.array([e[1], -e[0]], float)
        ln = np.hypot(*n)
        n = n / ln if ln > 1e-12 else n
        c = opp.get((min(a, b), max(a, b)))
        if c is not None and np.dot(mid - coords[c], n) < 0:
            n = -n                       # flip to point away from the interior
        mids[i] = mid
        nrm[i] = n
    return mids, nrm


def detect_fold(coords, elems, surf_edges, mesh_size,
                tol_factor=0.6, cos_thresh=0.5):
    """Detect incipient material self-contact (a fold).

    A fold is two boundary faces whose OUTWARD normals oppose each other
    (n_i . n_j < -cos_thresh) and whose midpoints have come within
    tol_factor * mesh_size, while not sharing a node. Returns
    (location_xy, gap) for the tightest such pair, or None."""
    if len(surf_edges) < 4:
        return None
    mids, nrm = _boundary_normals(coords, elems, surf_edges)
    tol = tol_factor * mesh_size
    ea, eb = surf_edges[:, 0], surf_edges[:, 1]
    best = None
    # pairwise midpoint distance + opposing-normal test (E ~ O(sqrt(N)), cheap)
    d = np.linalg.norm(mids[:, None, :] - mids[None, :, :], axis=-1)
    ndot = nrm @ nrm.T
    shares = ((ea[:, None] == ea[None, :]) | (ea[:, None] == eb[None, :]) |
              (eb[:, None] == ea[None, :]) | (eb[:, None] == eb[None, :]))
    iu = np.triu_indices(len(surf_edges), k=1)
    mask = (d[iu] < tol) & (ndot[iu] < -cos_thresh) & (~shares[iu])
    if not mask.any():
        return None
    cand = np.flatnonzero(mask)
    gaps = d[iu][cand]
    k = cand[np.argmin(gaps)]
    i, j = iu[0][k], iu[1][k]
    loc = 0.5 * (mids[i] + mids[j])
    return (float(loc[0]), float(loc[1])), float(d[i, j])


def detect_tool_gap_ingress(coords, surf_nodes, tools, g_tol, cos_thresh=0.3):
    """Material wedged in a gap between two DISTINCT tools whose surfaces face
    the node from OPPOSITE sides -> material has extruded into a parting line /
    tool-tool gap (no escape; usually over-confinement / back-pressure). Returns
    ((r,z), (toolA, toolB)) or None. Meant to be run ON ABORT (a failure cause);
    a completed run with intended flash never triggers it.

    Unlike a closing corner (both tool surfaces face the SAME side), a true gap
    sandwiches the material -> the two facing normals oppose (n_A . n_B < 0)."""
    if surf_nodes is None or len(surf_nodes) < 1:
        return None
    pts = coords[surf_nodes]
    dists, normals, names = [], [], []
    idx = np.arange(len(pts))
    for tool in tools:
        seg = getattr(tool, "current_segments", lambda: None)()
        if seg is None or len(seg) == 0:
            continue
        a, b = seg[:, 0], seg[:, 1]
        ab = b - a
        L2 = np.einsum("si,si->s", ab, ab)
        rel = pts[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum("psi,si->ps", rel, ab)
                    / np.where(L2 > 1e-20, L2, 1.0), 0.0, 1.0)
        foot = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(pts[:, None, :] - foot, axis=-1)
        si = d.argmin(1)
        e = ab[si]
        n = np.stack([e[:, 1], -e[:, 0]], 1)
        n = n / np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-12)[:, None]
        sgn = np.sign(np.einsum("pi,pi->p", pts - foot[idx, si], n))
        sgn[sgn == 0] = 1.0
        dists.append(d[idx, si])
        normals.append(n * sgn[:, None])
        names.append(tool.name)
    if len(dists) < 2:
        return None
    D = np.stack(dists, 1)
    Nn = np.stack(normals, 1)
    order = np.argsort(D, 1)
    best = None
    for i in idx:
        t1, t2 = order[i, 0], order[i, 1]
        if D[i, t1] < g_tol and D[i, t2] < g_tol \
                and float(Nn[i, t1] @ Nn[i, t2]) < -cos_thresh:
            score = D[i, t1] + D[i, t2]
            if best is None or score < best[0]:
                best = (score, (float(pts[i, 0]), float(pts[i, 1])),
                        (names[t1], names[t2]))
    return (best[1], best[2]) if best is not None else None


def diagnose(sim) -> Diagnosis:
    """Classify why the current increment failed and suggest a remedy."""
    cfg = sim.cfg
    coords, elems = sim.coords, sim.elems

    # 1) material fold / self-contact onset -> the design is the problem.
    # GUARDED by cfg.fold_check (default OFF): detect_fold can't yet tell a true
    # self-contact from two free surfaces each pressed against opposite die walls
    # in a (legitimately) nearly-full confined die, so it false-flags "design-NG"
    # there. Until it is contact-aware, only report folds when explicitly enabled.
    fold = (detect_fold(coords, elems, sim.surf_edges, cfg.mesh_size,
                        tol_factor=cfg.fold_tol_factor, cos_thresh=cfg.fold_cos)
            if cfg.fold_check else None)
    if fold is not None:
        (loc, gap) = fold
        return Diagnosis(
            cause="fold", design_ng=True,
            title="材料の折れ返り（自己接触）を検知 — この設計は巻き込み欠陥が発生します",
            location=loc, metrics={"gap[mm]": f"{gap:.3f}"},
            suggestions=[
                "工具の入隅/角の R を大きくして材料が回り込まないようにする",
                "逃がし・抜き勾配を追加、または素材形状（プリフォーム）を見直す",
                "1工程あたりの変形量を減らす（工程分割）",
                "本ソルバは自己接触を解かないため、折れ返り発生＝NG判定として停止",
            ])

    # 1.5) material extruded into a tool-tool gap (no escape) -> usually
    # over-confinement / too much back-pressure, or an unintended parting-line
    # gap in the tooling. Checked on abort only (a completed flash never trips).
    # The PUNCH is excluded: material squeezed between the moving punch and a die
    # is normal compression -> only fixed-tool gaps (e.g. die-diebot) are flagged.
    non_punch = [t for t in sim.tools if t is not getattr(sim, "punch", None)]
    gap_in = detect_tool_gap_ingress(
        coords, sim.surf_nodes, non_punch,
        g_tol=max(0.9 * cfg.mesh_size, 0.4), cos_thresh=0.3)
    if gap_in is not None:
        loc, (ta, tb) = gap_in
        return Diagnosis(
            cause="tool_gap", design_ng=False,
            title=f"材料が工具間ギャップ（{ta}–{tb}）に侵入 — 逃げ場がない",
            location=loc,
            suggestions=[
                "背圧（板押さえ/カウンター）を下げて材料に逃げ道を作る",
                "工具間の隙間（パーティングライン）を見直す／密着させる",
                "素材体積・プリフォーム形状が過大でないか確認",
                "（意図したバリの場合）バリ溝・逃がし設計が適切か確認",
            ])

    # 2) collapsed / inverted elements
    area, aspect = element_quality(coords, elems)
    sign = np.sign(np.median(area))
    inv = (area * sign) <= 0.0
    amag = np.abs(area)
    med = np.median(amag)
    collapsed = amag < 0.02 * med
    if inv.any() or collapsed.any():
        bad = np.flatnonzero(inv | collapsed)
        worst = bad[np.argmin(amag[bad])]
        c = coords[elems[worst]].mean(axis=0)
        return Diagnosis(
            cause="mesh_collapse",
            title="メッシュ要素の潰れ/反転を検知 — 局所のメッシュ破綻",
            location=(float(c[0]), float(c[1])),
            metrics={"反転": int(inv.sum()), "潰れ": int(collapsed.sum()),
                     "max_aspect": f"{float(aspect.max()):.0f}"},
            suggestions=[
                "メッシュ寸法を変える: 本ソルバはリメッシュ嵐が起きやすく、"
                "まず少し【粗く】すると安定することが多い（要素が大きく歪みに強い）。"
                "それでも粗すぎて崩れるなら逆に【細かく】",
                "解析ステップを少し大きく/小さく振って安定する値を探す（非単調）",
                "崩れる箇所だけ局所細分化（refine_box）で解像度を上げる",
            ])

    # 3) excessive contact penetration
    pen = sim._max_penetration()
    if pen > max(cfg.pen_tol, 0.3 * cfg.mesh_size):
        return Diagnosis(
            cause="penetration",
            title=f"接触貫入が過大（{pen:.2f} mm） — 接触/リメッシュ設定の問題",
            metrics={"max_pen[mm]": f"{pen:.3f}", "pen_tol[mm]": f"{cfg.pen_tol}"},
            suggestions=[
                "接触ペナルティ kn を上げる（cfg.kn）",
                "pen_tol を下げて深い貫入で早めにリメッシュ＆再投影",
                "d_stroke を小さくして 1 ステップの侵入量を減らす",
            ])

    # 4) plain non-convergence (mesh looks healthy)
    res = list(getattr(sim, "last_newton", []) or [])
    ratio = (res[-1] / max(res[0], 1e-30)) if len(res) >= 2 else float("nan")
    return Diagnosis(
        cause="nonconvergence",
        title="Newton 反復が収束せず（メッシュ・接触は健全）",
        metrics={"res_last": f"{res[-1]:.2e}" if res else "n/a",
                 "res_last/first": f"{ratio:.2f}" if ratio == ratio else "n/a"},
        suggestions=[
            "メッシュ寸法・解析ステップを両方向（粗く/細かく）に振って安定値を探す"
            "（本ソルバの安定帯は非単調）",
            "newton_maxit / activeset_maxit を増やす",
            "体積ロッキングが疑われる場合は F-bar を有効化（cfg.fbar）",
            "材料モデル・摩擦設定（m, mu）が極端でないか確認",
        ])

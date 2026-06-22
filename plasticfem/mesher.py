"""Gmsh-based triangulation of a closed polygon (blank outline)."""

from __future__ import annotations

import numpy as np


def mesh_polygon(points: np.ndarray, size: float, order: int = 1,
                 size_func=None, refine_box=None, box_size=None,
                 refine_pts=None, refine_size=None, refine_dist=None):
    """Triangulate a closed polygon.

    Parameters
    ----------
    points      : (N,2) polygon vertices (no repeated last point)
    size        : target element edge length [mm]
    refine_box  : (xmin,xmax,ymin,ymax) box of size `box_size` (gmsh Box field)
    refine_pts  : (M,2) points around which size -> `refine_size` within
                  `refine_dist` (gmsh Distance+Threshold field)
    size_func   : deprecated callable (kept for compatibility; ignored if the
                  field parameters are given)

    Local refinement uses gmsh size FIELDS (reliable), not the size callback
    (which the boundary point sizes override).

    Returns
    -------
    nodes (Nn,2), elems (Ne,3) int
    """
    import gmsh
    # interruptible=False: skip gmsh's signal handler, which otherwise raises
    # "signal only works in main thread" when meshing from a non-main thread
    # (e.g. the Streamlit script-runner thread in the post-processor app).
    # A previous call that errored out can leave gmsh initialized; start from a
    # clean slate so repeated meshing (e.g. dragging the tool-mesh slider) never
    # trips "Gmsh has already been initialized" / "has not been initialized".
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(interruptible=False)
    try:
        # option-setting kept INSIDE the try so the finally always finalizes
        # (a failure here previously leaked the initialized state to the next call)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        use_fields = (refine_box is not None) or (refine_pts is not None
                                                  and len(refine_pts) > 0)
        if use_fields:
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMin", size * 0.6)
        gmsh.model.add("blank")
        tags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, size)
                for x, y in points]
        lines = [gmsh.model.geo.addLine(tags[i], tags[(i + 1) % len(tags)])
                 for i in range(len(tags))]
        loop = gmsh.model.geo.addCurveLoop(lines)
        gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        if use_fields:
            fields = []
            mf = gmsh.model.mesh.field
            if refine_box is not None:
                bs = box_size if box_size is not None else 0.25 * size
                fb = mf.add("Box")
                mf.setNumber(fb, "VIn", bs)
                mf.setNumber(fb, "VOut", size)
                mf.setNumber(fb, "XMin", refine_box[0])
                mf.setNumber(fb, "XMax", refine_box[1])
                mf.setNumber(fb, "YMin", refine_box[2])
                mf.setNumber(fb, "YMax", refine_box[3])
                mf.setNumber(fb, "Thickness", 0.5 * size)
                fields.append(fb)
            if refine_pts is not None and len(refine_pts) > 0:
                rs = refine_size if refine_size is not None else 0.3 * size
                rd = refine_dist if refine_dist is not None else 3.0
                ptags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0)
                         for x, y in refine_pts]
                gmsh.model.geo.synchronize()
                fd = mf.add("Distance")
                mf.setNumbers(fd, "PointsList", [float(t) for t in ptags])
                ft = mf.add("Threshold")
                mf.setNumber(ft, "InField", fd)
                mf.setNumber(ft, "SizeMin", rs)
                mf.setNumber(ft, "SizeMax", size)
                mf.setNumber(ft, "DistMin", 0.0)
                mf.setNumber(ft, "DistMax", rd)
                fields.append(ft)
            fmin = mf.add("Min")
            mf.setNumbers(fmin, "FieldsList", [float(f) for f in fields])
            mf.setAsBackgroundMesh(fmin)

        gmsh.model.mesh.generate(2)

        ntags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.array(coords, float).reshape(-1, 3)[:, :2]
        remap = {t: i for i, t in enumerate(ntags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(dim=2)
        tri = None
        for et, en in zip(etypes, enodes):
            if et == 2:  # 3-node triangle
                tri = np.array([remap[t] for t in en], int).reshape(-1, 3)
        if tri is None:
            raise RuntimeError("gmsh produced no triangles")
        # ensure CCW orientation
        v1 = nodes[tri[:, 1]] - nodes[tri[:, 0]]
        v2 = nodes[tri[:, 2]] - nodes[tri[:, 0]]
        flip = (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        return nodes, tri
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def boundary_polygon(nodes: np.ndarray, elems: np.ndarray) -> np.ndarray:
    """Extract ordered boundary polygon of a triangulated mesh (largest loop)."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for tri in elems:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            cnt[(min(a, b), max(a, b))] += 1
    bedges = []
    for tri in elems:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if cnt[(min(a, b), max(a, b))] == 1:
                bedges.append((a, b))  # keeps CCW order of element
    nxt = {a: b for a, b in bedges}
    loops = []
    visited = set()
    for start in list(nxt):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        cur = nxt[start]
        while cur != start and cur in nxt and cur not in visited:
            loop.append(cur)
            visited.add(cur)
            cur = nxt[cur]
        loops.append(loop)
    best = max(loops, key=len)
    return nodes[np.array(best, int)]


def element_quality(nodes: np.ndarray, elems: np.ndarray):
    """Return (areas, aspect ratios) of all triangles."""
    p0, p1, p2 = nodes[elems[:, 0]], nodes[elems[:, 1]], nodes[elems[:, 2]]
    v1, v2, v3 = p1 - p0, p2 - p1, p0 - p2
    area = 0.5 * (v1[:, 0] * (-v3[:, 1]) - v1[:, 1] * (-v3[:, 0]))
    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)
    l3 = np.linalg.norm(v3, axis=1)
    lmax = np.maximum(np.maximum(l1, l2), l3)
    # aspect = longest edge / (2*inradius); equilateral -> ~1.15
    s = 0.5 * (l1 + l2 + l3)
    inr = np.where(s > 0, np.abs(area) / s, 0.0)
    aspect = np.where(inr > 1e-12, lmax / (2.0 * np.sqrt(3.0) * inr), 1e9)
    return area, aspect

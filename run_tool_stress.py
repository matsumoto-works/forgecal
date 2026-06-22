"""Elastic tool-stress post-processing driver.

Reads a finished forming run (results/.../results.h5), picks the step nearest
a requested stroke, reconstructs the chosen tool at that pose, recovers the
workpiece->tool contact pressure, and solves the tool's elastic stress field.

Example
-------
  anaconda3\\python.exe run_tool_stress.py results/verify_20260614 \\
        --stroke 11.85 --tool punch --tool-mesh 0.8

The DXF tool outlines and the workpiece mesh size must match the run that
produced the h5 (defaults are the backward-extrusion case).
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plasticfem import fem
from plasticfem.contact import RigidTool, build_contact_points, detect_contacts
from plasticfem.geometry import read_dxf
from plasticfem.tool_stress import (solve_tool_stress, plot_tool_stress,
                                     plot_combined, plot_all_tools, roller_fix)

HERE = os.path.dirname(os.path.abspath(__file__))


def _boundary(elems):
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


def load_step(h5path, target_stroke):
    """Return (coords, elems, stroke, {tool_idx: segs}) for the step whose
    stroke is closest to target_stroke."""
    with h5py.File(h5path, "r") as h:
        keys = sorted(k for k in h.keys() if k.startswith("step_"))
        strokes = np.array([h[k].attrs["stroke_mm"] for k in keys])
        i = int(np.abs(strokes - target_stroke).argmin())
        g = h[keys[i]]
        coords = g["coords"][:]
        elems = g["elems"][:]
        mises = g["mises"][:] if "mises" in g else None
        stroke = float(g.attrs["stroke_mm"])
        tool_segs = {}
        if "tools" in g:
            for tk in g["tools"].keys():
                tool_segs[int(tk)] = g["tools"][tk][:]
    return coords, elems, mises, stroke, tool_segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="run output dir containing results.h5")
    ap.add_argument("--stroke", type=float, default=None,
                    help="stroke [mm] to analyse (default: last/deepest step)")
    ap.add_argument("--tool", default="punch",
                    help="tool name to stress, or 'all' for every tool "
                         "(punch+die) plus the material in one figure")
    ap.add_argument("--tool-mesh", type=float, default=0.8,
                    help="tool mesh edge length [mm]")
    ap.add_argument("--tool-E", type=float, default=210000.0)
    ap.add_argument("--tool-nu", type=float, default=0.3)
    ap.add_argument("--work-mesh", type=float, default=1.5,
                    help="workpiece mesh size of the run (sets contact penalty)")
    ap.add_argument("--work-E", type=float, default=210000.0)
    ap.add_argument("--mode", default="axisym", choices=["axisym", "plane"])
    ap.add_argument("--midpoints", action="store_true",
                    help="run used edge-midpoint contact (default: node-only)")
    ap.add_argument("--deform-scale", type=float, default=0.0,
                    help="amplify tool displacement in the plot")
    ap.add_argument("--material-color", default="jet",
                    choices=["jet", "grey", "separate"],
                    help="in the all-tools figure, draw the material on the "
                         "same jet scale as the tools (jet), as grey context "
                         "(grey), or with its own colour map + colour bar so "
                         "both internal stress fields resolve (separate)")
    ap.add_argument("--fix-face", default=None,
                    help="comma list of support faces among ymax,ymin,xmax,"
                         "xmin (roller per face). Default ymax (punch top). "
                         "Use e.g. ymin,xmax for a die on its base in a ring")
    args = ap.parse_args()

    mode = fem.AXISYMMETRIC if args.mode == "axisym" else fem.PLANE_STRAIN
    h5path = os.path.join(args.results_dir, "results.h5")
    if not os.path.exists(h5path):
        sys.exit(f"no results.h5 in {args.results_dir}")

    # rebuild the same rigid tools the run used (DXF outlines, tool order)
    punch_shape = read_dxf(os.path.join(HERE, "model", "punch.dxf"))
    die_shape = read_dxf(os.path.join(HERE, "model", "die.dxf"))
    punch = RigidTool.from_polygon("punch", punch_shape.outline.points,
                                   friction_model="shear", m=0.12)
    die = RigidTool.from_polygon("die", die_shape.outline.points,
                                 friction_model="shear", m=0.12)
    tools = [punch, die]

    target = args.stroke if args.stroke is not None else 1e9
    coords, elems, work_mises, stroke, tool_segs = load_step(h5path, target)
    print(f"step @ stroke {stroke:.3f} mm  ({len(coords)} workpiece nodes)")

    # workpiece contact points (node-only by default, matching the run)
    surf_nodes, surf_edges = _boundary(elems)
    cpoints = build_contact_points(coords, surf_nodes, surf_edges,
                                   axisym=(mode == fem.AXISYMMETRIC))
    if not args.midpoints:
        cpoints = [c for c in cpoints if c[0] == c[1]]
    kn = 2.0 * args.work_E / args.work_mesh    # same penalty as the forming run

    # per-tool default support (overridden by --fix-face if given):
    #   punch -> top face pressed by the press platen (solver default)
    #   die   -> seated on its base inside a container ring
    default_fix = {"punch": None, "die": ["ymin", "xmax"]}
    if args.fix_face:
        override = [s.strip() for s in args.fix_face.split(",")]
    else:
        override = None

    selected = tools if args.tool == "all" else \
        [next(t for t in tools if t.name == args.tool)]

    results = []
    for tool in selected:
        ti = tools.index(tool)
        if ti in tool_segs:
            tool.displacement = tool_segs[ti][0, 0] - tool.segments[0, 0]
        contacts = detect_contacts(coords, cpoints, tool, g_max=args.work_mesh)
        print(f"\ntool '{tool.name}'  displacement="
              f"{np.round(tool.displacement, 3)}  "
              f"contacts={len(contacts)}")
        faces = override if override is not None \
            else default_fix.get(tool.name, None)
        fix = roller_fix(faces, tol=args.tool_mesh) if faces else None
        res = solve_tool_stress(tool, contacts, cpoints, coords, kn,
                                mode=mode, E=args.tool_E, nu=args.tool_nu,
                                mesh_size=args.tool_mesh, tool_name=tool.name,
                                stroke=stroke, fix=fix)
        results.append(res)
        Ftot = res.load_vecs.sum(axis=0)
        print(f"  tool mesh: {len(res.coords)} nodes, {len(res.elems)} elems"
              f"  |F|={np.linalg.norm(Ftot)/1000.0:.1f} kN"
              f"  max vM={res.mises.max():.0f} MPa")

        out = os.path.join(args.results_dir,
                           f"tool_stress_{res.tool_name}_s{stroke:05.2f}mm.png")
        plot_tool_stress(res, out, deform_scale=args.deform_scale)
        out2 = os.path.join(
            args.results_dir,
            f"tool_stress_{res.tool_name}_s{stroke:05.2f}mm_combined.png")
        plot_combined(res, coords, elems, out2, work_mises=work_mises,
                      deform_scale=args.deform_scale)

    if len(results) > 1:
        outall = os.path.join(args.results_dir,
                              f"tool_stress_ALL_s{stroke:05.2f}mm.png")
        plot_all_tools(results, coords, elems, outall, work_mises=work_mises,
                       deform_scale=args.deform_scale,
                       material_cmap=args.material_color)
        print(f"\nsaved {outall}")


if __name__ == "__main__":
    main()

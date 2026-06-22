"""Backward extrusion verification case (model/*.dxf vs DEFORM).

Blank: R20 x H30 (S45C), punch R12 (shoulder R1), die bore R20.
Axisymmetric, isothermal, Coulomb friction mu=0.12.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plasticfem import fem
from plasticfem.contact import RigidTool
from plasticfem.geometry import read_dxf
from plasticfem.material import Material, PiecewiseLinear
from plasticfem.post import plot_step, save_hdf5, save_load_curve
from plasticfem.solver import SimConfig, Simulation

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stroke", type=float, default=20.0)
    ap.add_argument("--mesh", type=float, default=1.5)
    ap.add_argument("--dstroke", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot-every", type=int, default=20)
    ap.add_argument("--no-relax", action="store_true")
    ap.add_argument("--midpoints", action="store_true",
                    help="enable edge-midpoint contact (default: node-only)")
    ap.add_argument("--pen-tol", type=float, default=0.10)
    ap.add_argument("--remesh-stroke", type=float, default=1.0)
    ap.add_argument("--flownet", action="store_true",
                    help="seed flow-net tracers and save them for the viewer")
    args = ap.parse_args()

    outdir = args.out or os.path.join(HERE, "results",
                                      time.strftime("%Y%m%d_%H%M%S") + "_backward_extrusion")
    os.makedirs(outdir, exist_ok=True)

    # ── geometry ──────────────────────────────────────────────
    blank = read_dxf(os.path.join(HERE, "model", "blank.dxf")).outline.points
    punch_shape = read_dxf(os.path.join(HERE, "model", "punch.dxf"))
    die_shape = read_dxf(os.path.join(HERE, "model", "die.dxf"))

    # ── material: S45C piecewise-linear flow curve ────────────
    mat = Material(E=210000.0, nu=0.3,
                   flow=PiecewiseLinear(strain=[0.0, 0.08, 0.81, 2.0],
                                        stress_tab=[750.0, 795.0, 1013.0, 1017.0]))

    # ── tools ─────────────────────────────────────────────────
    # DEFORM uses shear friction m=0.12 -> use the same model for comparison
    punch = RigidTool.from_polygon("punch", punch_shape.outline.points,
                                   friction_model="shear", m=0.12)
    die = RigidTool.from_polygon("die", die_shape.outline.points,
                                 friction_model="shear", m=0.12)

    cfg = SimConfig(mode=fem.AXISYMMETRIC, stroke=args.stroke,
                    d_stroke=args.dstroke, mesh_size=args.mesh,
                    remesh_stroke=args.remesh_stroke,
                    relax_after_remesh=not args.no_relax,
                    edge_midpoint_contact=args.midpoints,
                    pen_tol=args.pen_tol)

    sim = Simulation(cfg, mat, blank, [punch, die])
    if args.flownet:
        sim.init_tracers(n_div=20)
    print(f"mesh: {len(sim.coords)} nodes, {len(sim.elems)} elements")
    print(f"output: {outdir}")

    t0 = time.time()

    def cb(step, rec):
        if step % 10 == 0 or step == 1:
            print(f"  step {step:4d}  stroke={rec.stroke:6.3f} mm  "
                  f"load={rec.punch_load/1000.0:8.1f} kN  "
                  f"PEEQmax={rec.ep.max():.3f}  remesh={sim.remesh_count}  "
                  f"({time.time()-t0:.0f}s)")
        if step % args.plot_every == 0:
            tag = f"{step:05d}_s{rec.stroke:05.2f}mm"
            plot_step(rec, sim.tools, outdir, tag=tag, field="mises")
            plot_step(rec, sim.tools, outdir, tag=tag, field="peeq")

    history = sim.run(callback=cb)

    # DEFORM reference points (read from Images_BackwardExtrusion_basic /
    # comparison report; tons-SI converted to kN)
    deform_ref = os.path.join(outdir, "deform_ref.csv")
    with open(deform_ref, "w") as f:
        f.write("stroke_mm,load_kN\n")
        for s_, p_ in [(0.1, 635), (0.5, 1115), (1.0, 1227), (2.0, 1310),
                       (5.0, 1392), (10.0, 1470), (15.0, 1566), (20.0, 1483)]:
            f.write(f"{s_},{p_}\n")
    save_load_curve(sim.load_curve, outdir, deform_csv=deform_ref)
    save_hdf5(history, outdir, sim=sim if args.flownet else None)
    rec = history[-1]
    tag = f"final_s{rec.stroke:05.2f}mm"
    plot_step(rec, sim.tools, outdir, prefix="final", tag=tag, field="mises")
    plot_step(rec, sim.tools, outdir, prefix="final", tag=tag, field="peeq")

    print(f"\ndone: {len(history)} steps, {time.time()-t0:.0f}s, "
          f"remesh x{sim.remesh_count}")
    print(f"final load = {rec.punch_load/1000.0:.1f} kN @ {rec.stroke:.2f} mm "
          f"(DEFORM: ~1640 kN @ 20 mm)")


if __name__ == "__main__":
    main()

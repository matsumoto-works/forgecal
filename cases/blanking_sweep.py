"""Blanking parameter study: shear-surface quality vs clearance & holder force.

For each (clearance, holder force) the sheet is sheared with nonlocal ductile
damage; the run stops at cut-through (punch load drops to a small fraction of
its peak — the point where the shear surface is fully formed, since element
deletion is intentionally NOT used). Outputs per case:
  - max punch shear load
  - the damage (fracture) + PEEQ contour of the shear surface
  - rollover depth (vertical drop of the cut edge) as a surface-quality proxy
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plasticfem import fem
from plasticfem.contact import RigidTool
from plasticfem.material import DamageModel, Material, PowerLaw
from plasticfem.post import plot_step
from plasticfem.solver import SimConfig, Simulation

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "blanking_sweep")
EX = 15.0            # shear edge x [mm]
THK = 3.0            # sheet thickness [mm]


def make_tools(clearance):
    """Build punch / die / holder polygons for a given clearance [mm]."""
    blank = np.array([(5, 0), (25, 0), (25, THK), (5, THK)], float)
    punch = np.array([(0, THK), (EX, THK), (EX, 30), (0, 30)], float)
    die = np.array([(EX + clearance, -20), (EX + clearance, 0),
                    (30, 0), (30, -20)], float)
    holder = np.array([(EX + clearance, THK), (30, THK),
                       (30, 12), (EX + clearance, 12)], float)
    return blank, punch, die, holder


def run_case(clearance, holder_N, mat, mesh=0.4):
    blank, ppoly, dpoly, hpoly = make_tools(clearance)
    punch = RigidTool.from_polygon("punch", ppoly, friction_model="coulomb",
                                   mu=0.08)
    die = RigidTool.from_polygon("die", dpoly, friction_model="coulomb", mu=0.08)
    holder = RigidTool.from_polygon("holder", hpoly, friction_model="coulomb",
                                    mu=0.08)
    holder.control = "force"
    holder.f_const = holder_N
    holder.free_dir = np.array([0.0, -1.0])
    # uniform-ish mesh for the parameter trend study (fast); nonlocal damage
    # keeps it mesh-objective. A detailed fine-box run is done separately for
    # surface visualization.
    cfg = SimConfig(mode=fem.PLANE_STRAIN, stroke=2.6, d_stroke=0.025,
                    mesh_size=mesh, max_stall_remesh=8, min_substep=2e-5)
    sim = Simulation(cfg, mat, blank, [punch, die, holder], punch_name="punch")

    peak = [0.0]
    cut = [None]

    def cb(step, rec):
        peak[0] = max(peak[0], rec.punch_load)
        if step % 20 == 0:
            print(f"    .. step {step} s={rec.stroke:.2f} "
                  f"load={rec.punch_load/1000:.2f}kN peak={peak[0]/1000:.2f} "
                  f"rm={sim.remesh_count}", flush=True)
        # cut-through: load dropped below 8% of peak after a real peak ->
        # the shear surface is fully formed; stop (return True) to skip the
        # ill-posed grinding of the already-separated ligament
        if cut[0] is None and peak[0] > 50.0 and rec.punch_load < 0.08 * peak[0]:
            cut[0] = step
            return True
        return False
    sim.run(callback=cb, record_initial=False)
    return sim, peak[0], cut[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    mat = Material(E=210000.0, nu=0.3, flow=PowerLaw(C=550.0, n=0.22, e0=2e-3),
                   damage=DamageModel(Dc=0.5, soft_start=0.6, residual=0.08,
                                      eta=120.0, lc=0.9))
    # clearance 10/15/20% of thickness (all >= element size, resolvable);
    # holder 2 kN, plus one 10% case at 1 kN to see holder-force effect
    cases = [("c10", 0.10 * THK, 2000.0), ("c15", 0.15 * THK, 2000.0),
             ("c20", 0.20 * THK, 2000.0), ("c10_hold1k", 0.10 * THK, 1000.0)]
    summary = []
    for name, cl, hN in cases:
        t0 = time.time()
        sim, peak, cut = run_case(cl, hN, mat)
        rec = sim.history[-1]
        # rollover proxy: how far the cut edge (near x=EX) on the punch side
        # was dragged down below the original top surface
        c = rec.coords
        near = c[(c[:, 0] > EX - 1.2) & (c[:, 0] < EX) & (c[:, 1] > 0)]
        rollover = THK - near[:, 1].max() if len(near) else float("nan")
        plot_step(rec, [], OUT, prefix=name, tag="dmg", field="damage",
                  vmin=0, vmax=1.0)
        plot_step(rec, [], OUT, prefix=name, tag="peeq", field="peeq")
        summary.append((name, cl, hN, peak / 1000.0, rec.stroke, rollover,
                        time.time() - t0))
        print(f"{name}: clr={cl:.2f}mm hold={hN/1000:.0f}kN  "
              f"peak={peak/1000:.2f}kN  reached={rec.stroke:.2f}mm  "
              f"rollover={rollover:.2f}mm  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(OUT, "summary.csv"), "w") as f:
        f.write("case,clearance_mm,holder_kN,peak_load_kN,reached_mm,"
                "rollover_mm,seconds\n")
        for r in summary:
            f.write("%s,%.3f,%.1f,%.3f,%.3f,%.3f,%.0f\n"
                    % (r[0], r[1], r[2] / 1000.0, r[3], r[4], r[5], r[6]))
    print("done ->", OUT)


if __name__ == "__main__":
    main()

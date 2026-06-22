"""Generic process-case runner for the multi-process verification campaign.

usage:  python run_case.py <case-name> [--stroke S] [--mesh M]
        python run_case.py --list
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# diagnostic prints contain Japanese / em-dash; don't let a narrow console
# encoding (Windows cp932) crash a run -> replace unencodable chars.
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from plasticfem import fem
from plasticfem.contact import RigidTool
from plasticfem.geometry import read_dxf
from plasticfem.material import (DamageModel, Material, PiecewiseLinear,
                                 PowerLaw)
from plasticfem.post import (plot_flownet, plot_history, plot_step,
                             save_hdf5, save_load_curve)
from plasticfem.solver import SimConfig, Simulation

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.join(HERE, "model_cases")

S45C = dict(E=210000.0, nu=0.3,
            flow=PiecewiseLinear(strain=[0, 0.08, 0.81, 2.0],
                                 stress_tab=[750, 795, 1013, 1017]))
SHEET = dict(E=210000.0, nu=0.3, flow=PowerLaw(C=550.0, n=0.22, e0=2e-3))
# ductile sheet steel with Cockcroft-Latham damage softening (for blanking)
SHEET_DMG = dict(E=210000.0, nu=0.3, flow=PowerLaw(C=550.0, n=0.22, e0=2e-3),
                 damage=DamageModel(Dc=0.5, soft_start=0.6, residual=0.08,
                                    eta=120.0, lc=0.9))


def T(file, *, model="shear", mu=0.0, m=0.12, **kw):
    """Tool spec helper."""
    return dict(file=file, friction_model=model, mu=mu, m=m, **kw)


CASES = {
    "upsetting": dict(
        mode=fem.AXISYMMETRIC, mat=S45C, stroke=12.0, mesh=1.5,
        tools=[T("punch", m=0.4), T("die", m=0.4)]),
    "forward_extrusion": dict(
        mode=fem.AXISYMMETRIC, mat=S45C, stroke=15.0, mesh=1.2,
        tools=[T("punch", m=0.12), T("die", m=0.12)]),
    "impression_die": dict(
        mode=fem.AXISYMMETRIC, mat=S45C, stroke=5.0, mesh=1.0,
        tools=[T("punch", m=0.3), T("die", m=0.3)]),
    "closed_die": dict(
        mode=fem.AXISYMMETRIC, mat=S45C, stroke=5.0, mesh=1.0,
        tools=[T("punch", m=0.2), T("die", m=0.2), T("diebot", m=0.2),
               # counter punch FORCE-controlled (逆押さえ背圧 40kN). Solved
               # monolithically (tool position is an extra unknown coupled to
               # the material) -> stable even in this fully confined die where
               # the staggered position-force update diverges. The counter
               # retreats while maintaining the back-pressure as material
               # extrudes down the r0-7 throat.
               T("counter", m=0.2, control="force", f_const=40e3,
                 free_dir=(0.0, 1.0))]),
    "v_bend": dict(
        mode=fem.PLANE_STRAIN, mat=SHEET, stroke=12.0, mesh=0.8,
        dstroke=0.04, springback_end=True,
        # thin sheet: handled automatically by the relative-degradation remesh
        # trigger (no per-case remesh flag needed)
        tools=[T("punch", model="coulomb", mu=0.10),
               T("die", model="coulomb", mu=0.10)]),
    "convex_punch": dict(
        mode=fem.PLANE_STRAIN, mat=SHEET, stroke=13.0, mesh=0.7,
        dstroke=0.04,
        cfg=dict(max_stall_remesh=10, min_substep=2e-5),
        tools=[T("punch", model="coulomb", mu=0.10, smooth_contact=True),
               T("die", model="coulomb", mu=0.10, smooth_contact=True),
               # preloaded spring blank holder: f_const = spring preload so the
               # flange is actually clamped (light 200N preload left the sheet
               # un-held -> it just draped, near-rigid-body, chattering)
               T("holder", model="coulomb", mu=0.10, control="force",
                 k_spring=200.0, f_const=800.0, free_dir=(0.0, -1.0))]),
    "convex_punch_ف1000N": None,  # placeholder removed below
    "composite_tool": dict(
        mode=fem.AXISYMMETRIC, mat=S45C, stroke=10.0, mesh=1.2,
        cfg=dict(max_stall_remesh=12, min_substep=2e-5),
        tools=[T("punch", m=0.12, smooth_contact=True),
               T("die", m=0.12, smooth_contact=True)]),
    "multi_stage": dict(
        mode=fem.AXISYMMETRIC, mat=S45C, mesh=1.5,
        stages=[dict(punch="punch1", stroke=8.0, tools=[
                     T("punch1", m=0.4), T("die", m=0.12)]),
                dict(punch="punch2", stroke=10.0, tools=[
                     T("punch2", m=0.12), T("die", m=0.12)])]),
    "blanking": dict(
        # conventional shearing / blanking with Cockcroft-Latham damage
        # softening (no element deletion). Punch shears the 3mm sheet past the
        # die edge (0.2mm clearance); the shear band damages and softens to a
        # crack. The shear zone (x~15) is finely meshed to resolve the
        # rollover, burr and fracture surface; run to 98% of thickness.
        mode=fem.PLANE_STRAIN, mat=SHEET_DMG, stroke=2.94, mesh=0.4,
        dstroke=0.02,
        # nonlocal damage (DamageModel.lc) fixes the softening band width, so
        # the shear zone CAN now be refined to resolve the fracture surface
        # without the mesh-dependent runaway. Looser substep limits let the
        # solver grind through the final (near-separated) ligament.
        cfg=dict(refine_box=(13.5, 16.8, -1.0, 4.0), refine_box_factor=0.45,
                 max_stall_remesh=12, min_substep=1e-5),
        tools=[T("punch", model="coulomb", mu=0.08),
               T("die", model="coulomb", mu=0.08),
               T("holder", model="coulomb", mu=0.08, control="force",
                 f_const=2000.0, free_dir=(0.0, -1.0))]),
}
del CASES["convex_punch_ف1000N"]
# fixed-force holder variant (1000N固定)
CASES["convex_punch_fixed"] = dict(
    mode=fem.PLANE_STRAIN, mat=SHEET, stroke=13.0, mesh=0.7, dstroke=0.04,
    dxf_dir="convex_punch",
    tools=[T("punch", model="coulomb", mu=0.10),
           T("die", model="coulomb", mu=0.10),
           T("holder", model="coulomb", mu=0.10, control="force",
             f_const=1000.0, free_dir=(0.0, -1.0))])

# ── s_forging: axisymmetric coining/forging with a force-controlled stripper ──
# OnShape-authored DXFs (blank 35x10 disc, Punch on R0-10 with a nose radius,
# die wall at R=35, stripper holding R20-35 from above). Stage-1 only (punch2
# is the second operation, not verified here). Three stripper-force variants
# (20/200/400 kN) mirror the DEFORM runs. Stripper is force-controlled and free
# to retreat upward while maintaining the hold-down force (free_dir down, like
# the convex_punch blank holder). DEFORM: top-die ~136 t (=1334 kN) @6mm, 400kN.
# Stage-1 only (punch2 is the second operation, not verified here). DEFORM
# "Top Die" load monitor = upper-ram total = punch forming load + stripper
# spring force (the stripper spring is mounted on the top die), so compare
# FEM punch+stripper against DEFORM Top Die (or FEM punch against Top Die -
# Stripper). Verified within +-5% (400kN runs to 6mm; 200/20kN match where
# they reach). Known limitation: the low-hold-down runs (20/200 kN) abort at
# ~1.3-1.6mm from a punch-nose material fold (Newton residual will not drop
# even at tiny substeps -> remesh livelock); k_spring / smooth_contact /
# looser newton_tol were all tried and none fix it -> needs self-contact /
# fold handling, not a parameter. k_spring left env-tunable (default 0).
_SF_KSPRING = float(os.environ.get("SF_KSPRING_PER_MM", 0.0))  # 1/mm
for _sf in (20, 200, 400):
    CASES[f"s_forging_{_sf}"] = dict(
        mode=fem.AXISYMMETRIC, mat=S45C, stroke=6.0, mesh=0.6, dstroke=0.05,
        dxf_dir="s_forging", punch="Punch",
        tools=[T("Punch", m=0.12), T("die", m=0.12),
               T("stripper", m=0.12, control="force",
                 f_const=_sf * 1e3, k_spring=_SF_KSPRING * _sf * 1e3,
                 free_dir=(0.0, -1.0))])
del _sf


def build_tool(dxf_dir, spec):
    shape = read_dxf(os.path.join(dxf_dir, spec["file"] + ".dxf"))
    tool = RigidTool.from_polygon(
        spec["file"], shape.outline.points,
        mu=spec.get("mu", 0.0),
        friction_model=spec.get("friction_model", "coulomb"),
        m=spec.get("m", 0.0),
        smooth_contact=spec.get("smooth_contact", False))
    tool.control = spec.get("control", "rigid")
    tool.f_const = spec.get("f_const", 0.0)
    tool.k_spring = spec.get("k_spring", 0.0)
    tool.free_dir = np.asarray(spec.get("free_dir", (0.0, -1.0)), float)
    tool.vel = spec.get("vel", 0.0)
    return tool


def run_stage(sim, label, t0, base_stroke, record_initial):
    """Run one stage; plotting is deferred to a fixed-scale post pass."""
    def cb(step, rec):
        if step % 25 == 0 or step <= 1:
            print(f"  [{label}] step {step:4d} s={rec.stroke:6.2f} "
                  f"load={rec.punch_load/1000:8.1f}kN PEEQ={rec.ep.max():.2f} "
                  f"rm={sim.remesh_count} ({time.time()-t0:.0f}s)", flush=True)
    return sim.run(callback=cb, record_initial=record_initial,
                   base_stroke=base_stroke)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--stroke", type=float, default=None)
    ap.add_argument("--mesh", type=float, default=None)
    ap.add_argument("--plot-every", type=int, default=25)
    ap.add_argument("--springback", action="store_true",
                    help="apply a final free springback after the last stage "
                         "(works for single-stage cases too)")
    ap.add_argument("--flownet", action="store_true",
                    help="seed flow-net tracers and output the deformed grid")
    args = ap.parse_args()

    if args.case == "--list":
        print("\n".join(CASES))
        return
    spec = CASES[args.case]
    dxf_dir = os.path.join(MC, spec.get("dxf_dir", args.case))
    outdir = os.path.join(HERE, "results", "cases", args.case)
    os.makedirs(outdir, exist_ok=True)

    mat = Material(**spec["mat"])
    blank = read_dxf(os.path.join(dxf_dir, "blank.dxf")).outline.points
    mesh = args.mesh or spec["mesh"]
    t0 = time.time()

    stages = spec.get("stages")
    if stages is None:
        stages = [dict(punch=spec.get("punch", "punch"), stroke=spec["stroke"],
                       tools=spec["tools"])]
    history_all = []
    sim = None
    base_stroke = 0.0
    for si, st in enumerate(stages):
        tools = [build_tool(dxf_dir, ts) for ts in st["tools"]]
        stroke = args.stroke if (args.stroke and len(stages) == 1) else st["stroke"]
        cfg = SimConfig(mode=spec["mode"], stroke=stroke,
                        d_stroke=spec.get("dstroke", 0.05), mesh_size=mesh,
                        **spec.get("cfg", {}))
        prev = sim
        sim = Simulation(cfg, mat, blank, tools, punch_name=st["punch"])
        sim.stage = si
        if prev is not None:
            # carry deformed mesh + state into the next stage
            sim.coords = prev.coords.copy()
            sim.elems = prev.elems.copy()
            sim.sigma = prev.sigma.copy()
            sim.eps_e = prev.eps_e.copy()
            sim.ep = prev.ep.copy()
            sim._update_boundary()
            # inter-stage handling: unload under the new tool set. For BULK
            # forging elastic springback is negligible (huge plastic strain),
            # so _relax is adequate and robust. TRUE free springback
            # (springback()) matters for sheet bending but needs damped/
            # substepped solve to be stable on large residual stress -> future.
            if spec.get("springback_between"):
                sb = sim.springback()
                if sb is not None:
                    sb.stage = si
                    history_all.append(sb)
            # drop the new punch onto the deformed blank top
            pt = sim.punch
            ymin = (pt.closed_polygon[:, 1] + pt.displacement[1]).min()
            top = float(sim.coords[:, 1].max())
            if ymin > top:
                pt.displacement = pt.displacement + np.array([0.0, top - ymin])
            sim._relax()
        if args.flownet and prev is None:
            sim.init_tracers(n_div=20)
        label = f"st{si+1}" if len(stages) > 1 else "run"
        print(f"[{args.case}] stage {si+1}/{len(stages)}: "
              f"{len(sim.coords)} nodes, stroke {stroke}mm (base {base_stroke:.1f})")
        history_all += run_stage(sim, label, t0, base_stroke,
                                 record_initial=True)
        base_stroke += stroke

    # final free springback (key result for bending / sheet forming).
    # available for ANY case (incl. single-stage) via --springback or the
    # per-case springback_end flag
    if spec.get("springback_end") or args.springback:
        print(f"[{args.case}] final springback ...", flush=True)
        sb = sim.springback()
        if sb is not None:
            history_all.append(sb)

    save_load_curve(sim.load_curve, outdir,
                    tool_loads=sim.tool_load_history,
                    punch_name=stages[-1]["punch"])
    save_hdf5(history_all, outdir, sim=sim)
    # fixed global colour scale across ALL frames + stages, incl. step 0
    plot_history(history_all, sim.tools, outdir, every=args.plot_every)
    rec = history_all[-1]
    if args.flownet and sim.tracers is not None:
        plot_flownet(sim, rec, outdir, tag="end")
        print(f"[{args.case}] flow net -> {outdir}\\flownet_end.png")
    plot_step(rec, sim.tools, outdir, prefix="final", tag="end", field="mises",
              vmin=0.0, vmax=max(float(r.mises.max()) for r in history_all))
    plot_step(rec, sim.tools, outdir, prefix="final", tag="end", field="peeq",
              vmin=0.0, vmax=max(float(r.ep.max()) for r in history_all))
    print(f"[{args.case}] done: {len(history_all)} steps, "
          f"{time.time()-t0:.0f}s, remesh x{sim.remesh_count}, "
          f"final load {rec.punch_load/1000:.1f} kN")


if __name__ == "__main__":
    main()

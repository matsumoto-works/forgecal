"""Run one forming job into its own job directory and write a status file.

Reuses the validated run_case helpers (CASES / build_tool / material defs) so
the web path runs exactly the same solver as the CLI. A job is described by
`<jobdir>/config.json`:

    {"case": "s_forging_400", "flownet": true}          # predefined case, or
    {"mode": "axisymmetric", "material": "S45C", ...}    # custom (future)

Designed to run as a *subprocess* (see jobs.py) so its memory is isolated and a
runaway solve is killed without touching the rest of the host.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# never let a diagnostic print (Japanese / em-dash) crash the solve on a narrow
# console encoding (e.g. Windows cp932) -> replace unencodable chars instead.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import run_case  # noqa: E402  reuse CASES / build_tool / material defs
from plasticfem import fem  # noqa: E402
from plasticfem.geometry import read_dxf  # noqa: E402
from plasticfem.material import Material  # noqa: E402
from plasticfem.post import save_hdf5, save_load_curve  # noqa: E402
from plasticfem.solver import SimConfig, Simulation  # noqa: E402


def _write_status(jobdir, **kw):
    kw["t"] = time.time()
    p = os.path.join(jobdir, "status.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False)
    os.replace(tmp, p)            # atomic: readers never see a partial file


# auto-retry plan (custom jobs): (mesh_mult, step_mult, label). The remesh
# livelock has a NON-monotonic sweet spot, so we try BOTH directions: coarser
# first (fewer/larger elements -> fewer remesh storms, and faster), then finer
# (for under-resolution failures). First attempt that reaches full stroke wins;
# otherwise the attempt that got furthest is kept.
_RETRY_PLAN = [(1.0, 1.0, "推奨値"), (1.3, 1.4, "粗め"), (0.7, 0.7, "細かめ")]


def run(jobdir: str) -> int:
    cfg = json.load(open(os.path.join(jobdir, "config.json"), encoding="utf-8"))
    t0 = time.time()
    try:
        spec, dxf_dir = _resolve_spec(cfg, jobdir)
        mat = Material(**spec["mat"])
        blank = read_dxf(os.path.join(dxf_dir, "blank.dxf")).outline.points
        stages = spec.get("stages") or [dict(
            punch=spec.get("punch", "punch"), stroke=spec["stroke"],
            tools=spec["tools"])]
        total_stroke = sum(s["stroke"] for s in stages)
        base_mesh = spec["mesh"]
        base_step = spec.get("dstroke", 0.05)
        custom = bool(cfg.get("custom"))
        no_prog = max(2 * base_step, 0.01 * total_stroke)
        # attempt queue (mesh_mult, step_mult, label); escalated adaptively below
        queue = list(_RETRY_PLAN) if custom else [(1.0, 1.0, "")]
        n_fixed = len(queue)
        MAX_ATTEMPTS = 5

        best = None
        marginal = None       # recommended-value attempt's failure (if any)
        k = 0
        while queue and k < MAX_ATTEMPTS:
            mm, sm, lbl = queue.pop(0)
            k += 1
            mesh = round(base_mesh * mm, 3)
            step = round(base_step * sm, 4)
            history, sim = _solve_stages(jobdir, cfg, spec, dxf_dir, mat, blank,
                                         stages, total_stroke, mesh, step, t0,
                                         k, lbl)
            reached = float(history[-1].stroke) if history else 0.0
            full = reached >= total_stroke - 1e-6
            diag0 = getattr(sim, "diagnosis", None)
            if k == 1 and not full:           # recommended values didn't complete
                marginal = dict(mesh=mesh, step=step, reached=reached,
                                cause=(diag0.cause if diag0 else None),
                                location=(diag0.location if diag0 else None))
            improved = best is None or reached > best["reached"] + 1e-9
            if improved:
                # new best -> write its outputs now (one history in RAM at a time)
                _write_outputs(jobdir, history, sim, cfg, stages,
                               marginal_loc=(marginal["location"]
                                             if marginal else None))
                diag = getattr(sim, "diagnosis", None)
                best = dict(reached=reached, full=full, mesh=mesh, step=step,
                            attempt=k, steps=len(history), mult=mm, smult=sm,
                            load=float(history[-1].punch_load / 1000.0),
                            remesh=int(sim.remesh_count),
                            diag=(dict(cause=diag.cause, title=diag.title,
                                       suggestions=diag.suggestions,
                                       location=diag.location,
                                       design_ng=diag.design_ng)
                                  if diag is not None else None))
            del history, sim
            if full:
                break
            # no progress on the first try -> setup/contact issue, not mesh
            if k == 1 and custom and reached < no_prog:
                best["diag"] = dict(
                    cause="setup", design_ng=False,
                    title="最初の増分から進みません — 初期の接触/配置の問題の可能性",
                    location=None,
                    suggestions=[
                        "配置プレビューで パンチ/板押さえ が素材表面に接触しているか確認",
                        "板押さえの向き（緑矢印）が素材を押す向きか確認"
                        "（逆だと力制御が押し当てる相手を失い発散）",
                        "押さえ力が過大でないか／DXFの工具位置がずれていないか確認",
                    ])
                break
            # queue empty & still not full -> push FURTHER in the direction that
            # got furthest. After the fixed plan, escalate unconditionally once
            # (toward the best direction); for later escalations, continue only
            # while they keep improving. Capped by MAX_ATTEMPTS.
            if not queue and custom and (k == n_fixed or improved):
                bm, bs = best["mult"], best["smult"]
                if bm >= 1.0:
                    queue = [(round(bm * 1.3, 3), round(bs * 1.4, 4), "さらに粗め")]
                else:
                    queue = [(round(bm * 0.7, 3), round(bs * 0.7, 4), "さらに細かめ")]

        _write_status(
            jobdir, state="done", progress=1.0, elapsed=time.time() - t0,
            steps=best["steps"], reached_stroke=best["reached"],
            target_stroke=total_stroke, final_load_kN=best["load"],
            remesh=best["remesh"], used_mesh=best["mesh"], used_step=best["step"],
            attempts_tried=best["attempt"], total_attempts=k,
            diagnosis=best["diag"],
            marginal=(marginal if (marginal and best["full"]) else None))
        return 0
    except Exception as e:  # noqa: BLE001
        _write_status(jobdir, state="failed", error=str(e),
                      traceback=traceback.format_exc(),
                      elapsed=time.time() - t0)
        return 1


def _solve_stages(jobdir, cfg, spec, dxf_dir, mat, blank, stages, total_stroke,
                  mesh, step, t0, attempt_k, label):
    """Run all stages once at the given mesh/step. Returns (history, sim)."""
    note = f"試行 {attempt_k}（{label} mesh {mesh}mm）" if label else ""
    history, sim, base = [], None, 0.0

    def progress(stroke_done):
        _write_status(jobdir, state="running",
                      progress=min(stroke_done / total_stroke, 1.0),
                      stroke=stroke_done, total=total_stroke, note=note,
                      elapsed=time.time() - t0)

    for si, st in enumerate(stages):
        tools = [run_case.build_tool(dxf_dir, ts) for ts in st["tools"]]
        scfg = SimConfig(mode=spec["mode"], stroke=st["stroke"],
                         d_stroke=step, mesh_size=mesh, **spec.get("cfg", {}))
        prev = sim
        sim = Simulation(scfg, mat, blank, tools, punch_name=st["punch"])
        sim.stage = si
        if prev is not None:
            _carry_state(prev, sim)
        if cfg.get("flownet") and prev is None:
            sim.init_tracers(n_div=int(cfg.get("flownet_div", 20)))

        def cb(step_i, rec, _b=base):
            progress(_b + rec.stroke)
            return False
        history += sim.run(callback=cb, record_initial=True, base_stroke=base)
        base += st["stroke"]
    return history, sim


def _write_outputs(jobdir, history, sim, cfg, stages, marginal_loc=None):
    """Load curve + H5 + mesh GIF (+ flow net) for the chosen attempt.

    Mark: this attempt's own problem spot in RED; if this attempt completed
    fine but an earlier (recommended-value) attempt failed, mark THAT spot in
    ORANGE so the user sees where the design was marginal."""
    save_load_curve(sim.load_curve, jobdir, tool_loads=sim.tool_load_history,
                    punch_name=stages[-1]["punch"])
    save_hdf5(history, jobdir, sim=sim)
    import animate
    holder_names = [t.name for t in sim.tools
                    if getattr(t, "control", "rigid") == "force"]
    diag = getattr(sim, "diagnosis", None)
    if diag is not None and diag.location:
        mark, color = diag.location, "#e2231a"          # red: this attempt's spot
    elif marginal_loc:
        mark, color = marginal_loc, "#f39c12"           # orange: earlier marginal
    else:
        mark, color = None, "#e2231a"
    animate.render_mesh_gif(history, sim.load_curve,
                            os.path.join(jobdir, "mesh_anim.gif"),
                            tool_loads=sim.tool_load_history,
                            holder_names=holder_names, mark=mark, mark_color=color)
    if cfg.get("flownet") and sim.tracers is not None:
        from plasticfem.post import plot_flownet
        plot_flownet(sim, history[-1], jobdir, tag="end")


def _resolve_spec(cfg, jobdir):
    """Return (spec, dxf_dir).

    Predefined case: {"case": name}. Custom job: full params + DXFs already
    saved in jobdir (blank.dxf + one <name>.dxf per tool)."""
    if "case" in cfg:
        spec = run_case.CASES[cfg["case"]]
        dxf_dir = os.path.join(run_case.MC, spec.get("dxf_dir", cfg["case"]))
        return spec, dxf_dir

    # custom DXF job -- build a CASES-style spec from the request
    mode = (fem.AXISYMMETRIC if cfg.get("mode") == "axisymmetric"
            else fem.PLANE_STRAIN)
    # material: saturating power-law flow curve from the user's inputs
    # (forging: flattens at large strain via e_sat instead of rising forever)
    from plasticfem.material import SaturatingPowerLaw
    mp = cfg["mat"]
    mat = dict(E=float(mp["E"]), nu=float(mp["nu"]),
               flow=SaturatingPowerLaw(C=float(mp["C"]), n=float(mp["n"]),
                                       e_sat=float(mp.get("e_sat", 1.0)), e0=1e-3))
    tools = []
    for t in cfg["tools"]:
        ts = dict(file=t["file"],
                  friction_model=t.get("friction_model", "shear"),
                  m=float(t.get("m", 0.12)), mu=float(t.get("mu", 0.0)),
                  smooth_contact=bool(t.get("smooth_contact", False)),
                  control=t.get("control", "rigid"),
                  f_const=float(t.get("f_const", 0.0)),
                  k_spring=float(t.get("k_spring", 0.0)),
                  free_dir=tuple(t.get("free_dir", (0.0, -1.0))))
        tools.append(ts)
    spec = dict(mode=mode, mat=mat, stroke=float(cfg["stroke"]),
                dstroke=float(cfg.get("dstroke", 0.05)),
                mesh=float(cfg["mesh"]), punch=cfg["punch"], tools=tools,
                cfg=cfg.get("cfg", {}))
    return spec, jobdir          # uploaded DXFs live in the job directory


def _carry_state(prev, sim):
    sim.coords = prev.coords.copy()
    sim.elems = prev.elems.copy()
    sim.sigma = prev.sigma.copy()
    sim.eps_e = prev.eps_e.copy()
    sim.ep = prev.ep.copy()
    sim._update_boundary()
    sim._relax()


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))

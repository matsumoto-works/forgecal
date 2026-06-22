"""Post-processing: contour PNGs, load-stroke curve (CSV/PNG), HDF5 dump."""

from __future__ import annotations

import csv
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def _tool_segments(rec, tools):
    """Tool segments for a record: use the per-step snapshot if present
    (correct for post-run plotting), else the live tool positions."""
    if getattr(rec, "tool_segs", None) is not None:
        return rec.tool_segs
    return [t.current_segments() for t in tools]


def plot_step(rec, tools, outdir, prefix="step", tag="", field="mises",
              vmin=None, vmax=None, levels=None):
    """Contour PNG of one step (element field averaged to nodes).

    vmin/vmax fix the colour scale so frames are comparable across a run /
    across stages; if None they auto-scale per frame.
    """
    os.makedirs(outdir, exist_ok=True)
    coords, elems = rec.coords, rec.elems
    if field == "mises":
        ef, label = rec.mises, "von Mises [MPa]"
    elif field == "peeq":
        ef, label = rec.ep, "Equivalent plastic strain"
    elif field == "damage":
        ef = rec.damage if rec.damage is not None else np.zeros(len(elems))
        label = "Cockcroft-Latham damage"
    else:
        raise ValueError(field)
    # element -> node average
    nf = np.zeros(len(coords))
    cnt = np.zeros(len(coords))
    for e, tri in enumerate(elems):
        nf[tri] += ef[e]
        cnt[tri] += 1
    nf /= np.maximum(cnt, 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    tr = mtri.Triangulation(coords[:, 0], coords[:, 1], elems)
    if vmin is not None and vmax is not None and vmax > vmin:
        lv = np.linspace(vmin, vmax, levels or 21)
        nf = np.clip(nf, vmin, vmax - 1e-9 * (vmax - vmin))
        tc = ax.tricontourf(tr, nf, levels=lv, cmap="jet", extend="both")
    else:
        tc = ax.tricontourf(tr, nf, levels=20, cmap="jet")
    ax.triplot(tr, lw=0.15, color="k", alpha=0.4)
    for segs in _tool_segments(rec, tools):
        for a, b in segs:
            ax.plot([a[0], b[0]], [a[1], b[1]], "g-", lw=1.5)
    fig.colorbar(tc, ax=ax, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("r / x [mm]")
    ax.set_ylabel("z / y [mm]")
    stg = getattr(rec, "stage", 0)
    stag = f"  stage {stg+1}" if stg else ""
    ax.set_title(f"{field}  stroke={rec.stroke:.2f} mm{stag}")
    path = os.path.join(outdir, f"{prefix}_{tag}_{field}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_history(history, tools, outdir, every=25, fields=("mises", "peeq"),
                 fixed_scale=True):
    """Re-plot frames from a finished run with a GLOBAL fixed colour scale.

    Always includes step 0 (initial state) and the final frame. Tool
    positions are taken from each record's snapshot so every frame draws
    the tools where they actually were.
    """
    if not history:
        return
    ranges = {}
    if fixed_scale:
        ranges["mises"] = (0.0, max(float(r.mises.max()) for r in history) or 1.0)
        ranges["peeq"] = (0.0, max(float(r.ep.max()) for r in history) or 1.0)
    # always include the first frame of each stage (the carried/relaxed
    # inter-stage state) so multi-stage carry-over is visible
    stage_starts = [i for i, r in enumerate(history)
                    if i == 0 or getattr(r, "stage", 0)
                    != getattr(history[i - 1], "stage", 0)]
    idx = sorted(set([0, len(history) - 1] + stage_starts
                     + list(range(0, len(history), every))))
    for field in fields:
        vmin, vmax = ranges.get(field, (None, None))
        for i in idx:
            rec = history[i]
            tag = f"{i:05d}_s{rec.stroke:05.2f}mm"
            plot_step(rec, tools, outdir, prefix="frame", tag=tag,
                      field=field, vmin=vmin, vmax=vmax)


def plot_flownet(sim, rec, outdir, prefix="flownet", tag=""):
    """Flow net (DEFORM-style fiber flow): the deformed square tracer grid
    overlaid on the part outline. Reveals material flow, laps and defects.
    `sim` must have run with init_tracers(); the tracer positions are in
    sim.tracers (final) or pass via sim.tracer_history for a given step.
    """
    os.makedirs(outdir, exist_ok=True)
    pts = sim.tracers
    ny, nx = sim._tr_grid_shape
    inside = sim._tr_inside
    P = pts.reshape(ny, nx, 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    # element mesh outline (light)
    tr = mtri.Triangulation(rec.coords[:, 0], rec.coords[:, 1], rec.elems)
    ax.triplot(tr, lw=0.1, color="0.8")
    # grid lines: draw a segment only if both endpoints started inside
    for j in range(ny):           # horizontal lines
        for i in range(nx - 1):
            if inside[j, i] and inside[j, i + 1]:
                ax.plot(P[j, i:i+2, 0], P[j, i:i+2, 1], "b-", lw=0.8)
    for i in range(nx):           # vertical lines
        for j in range(ny - 1):
            if inside[j, i] and inside[j + 1, i]:
                ax.plot(P[j:j+2, i, 0], P[j:j+2, i, 1], "b-", lw=0.8)
    for segs in _tool_segments(rec, sim.tools):
        for a, b in segs:
            ax.plot([a[0], b[0]], [a[1], b[1]], "g-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("r / x [mm]")
    ax.set_ylabel("z / y [mm]")
    ax.set_title(f"flow net  stroke={rec.stroke:.2f} mm")
    path = os.path.join(outdir, f"{prefix}_{tag}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_load_curve(load_curve, outdir, deform_csv=None, tool_loads=None,
                    punch_name="punch"):
    """CSV + PNG of load-stroke curve. Load output in kN.

    tool_loads: optional list of per-step {tool: load_N, _stroke: mm} dicts;
    the blank-holder / counter-punch loads are overlaid on the graph."""
    os.makedirs(outdir, exist_ok=True)
    arr = np.asarray(load_curve)
    if arr.ndim != 2 or len(arr) == 0:
        return None, None                  # nothing converged: skip the plot
    csv_path = os.path.join(outdir, "load_stroke.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stroke_mm", "load_kN"])
        for s, p in arr:
            w.writerow([f"{s:.4f}", f"{p / 1000.0:.3f}"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(arr[:, 0], arr[:, 1] / 1000.0, "b-", lw=0.8, alpha=0.5, label="PlasticFEM raw")
    # moving average (0.5mm bin)
    if len(arr) > 10:
        s, p = arr[:, 0], arr[:, 1] / 1000.0
        # cap the window to the data length: np.convolve("same") returns the
        # LONGER of signal/kernel, so a window > len(p) (tiny-stroke partial
        # runs) would desync s and pm and crash the plot
        win = max(3, int(0.5 / max(np.median(np.diff(s)), 1e-6)))
        win = min(win, len(p))
        kern = np.ones(win) / win
        # edge-pad before averaging so the running mean does NOT droop toward
        # zero at the ends (plain "same" convolution averages against implicit
        # zeros at the boundaries, which made the curve dive at the final
        # stroke). Replicating the end values keeps the averaged line on the
        # real load right to the last point.
        pad = win - 1
        pp = np.pad(p, (pad // 2, pad - pad // 2), mode="edge")
        pm = np.convolve(pp, kern, mode="valid")[:len(s)]
        ax.plot(s, pm, "r-", lw=2, label="PlasticFEM (0.5mm avg)")
    if deform_csv and os.path.exists(deform_csv):
        d = np.loadtxt(deform_csv, delimiter=",", skiprows=1)
        ax.plot(d[:, 0], d[:, 1], "k--o", lw=1.5, ms=5, label="DEFORM")
    # overlay blank-holder / counter-punch loads (force-controlled tools)
    if tool_loads:
        names = [k for k in tool_loads[0] if k not in (punch_name, "_stroke")]
        ss = np.array([d["_stroke"] for d in tool_loads])
        for nm in names:
            vals = np.array([d.get(nm, 0.0) for d in tool_loads]) / 1000.0
            if np.max(np.abs(vals)) > 1e-6:
                ax.plot(ss, vals, "--", lw=1.2, label=f"{nm} load")
    ax.set_xlabel("Stroke [mm]")
    ax.set_ylabel("Load [kN]")
    ax.grid(alpha=0.3)
    ax.legend()
    png_path = os.path.join(outdir, "load_stroke.png")
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def save_hdf5(history, outdir, fname="results.h5", sim=None):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    with h5py.File(path, "w") as h:
        if history:
            h.attrs["kn"] = float(getattr(history[0], "kn", 0.0) or 0.0)
            h.attrs["mesh_size"] = float(getattr(history[0], "mesh_size", 0.0)
                                         or 0.0)
        for i, rec in enumerate(history):
            g = h.create_group(f"step_{i:05d}")
            g.attrs["stroke_mm"] = rec.stroke
            g.attrs["punch_load_N"] = rec.punch_load
            g.attrs["stage"] = getattr(rec, "stage", 0)
            g.create_dataset("coords", data=rec.coords, compression="gzip")
            g.create_dataset("elems", data=rec.elems, compression="gzip")
            g.create_dataset("sigma", data=rec.sigma, compression="gzip")
            g.create_dataset("peeq", data=rec.ep, compression="gzip")
            g.create_dataset("mises", data=rec.mises, compression="gzip")
            if rec.damage is not None:
                g.create_dataset("damage", data=rec.damage, compression="gzip")
            if getattr(rec, "eps_e", None) is not None:
                g.create_dataset("strain", data=rec.eps_e, compression="gzip")
            if getattr(rec, "vel", None) is not None:
                g.create_dataset("velocity", data=rec.vel, compression="gzip")
            if getattr(rec, "tool_loads", None):
                tl = g.create_group("tool_loads")
                for nm, val in rec.tool_loads.items():
                    tl.attrs[nm] = float(val)
            if rec.tool_segs is not None:
                tg = g.create_group("tools")
                for ti, segs in enumerate(rec.tool_segs):
                    tg.create_dataset(str(ti), data=segs, compression="gzip")
                # tool names in the SAME index order as the segment datasets,
                # so a viewer maps a name -> its geometry unambiguously (the
                # tool_loads attrs are unordered, so don't rely on their order)
                if getattr(rec, "tool_names", None):
                    tg.attrs["names"] = [str(n) for n in rec.tool_names]
        # flow-net tracer history (advected during the solve, saved so the
        # flow net can be RENDERED in post-processing at any chosen step)
        if sim is not None and getattr(sim, "tracers", None) is not None:
            fg = h.create_group("flownet")
            fg.attrs["grid_ny"], fg.attrs["grid_nx"] = sim._tr_grid_shape
            fg.create_dataset("inside", data=sim._tr_inside)
            fg.create_dataset("history",
                              data=np.array(sim.tracer_history),
                              compression="gzip")
    return path

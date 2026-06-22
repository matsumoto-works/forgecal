"""Mesh-deformation GIF (with the forming-load curve) for the web result view.

Shows the deforming MESH + tools animating through the stroke next to the
load-stroke curve with a marker at the current step. Stress fields are
intentionally NOT drawn here -- download the results.h5 and use the desktop
post-processing app for stress / strain / flow-net post-processing.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402


def _smooth(s, p, bin_mm=0.5):
    """Edge-padded moving average (~bin_mm window) -> no end droop, tames the
    remesh sawtooth for display."""
    if len(p) < 5:
        return np.asarray(p, float)
    win = max(3, int(bin_mm / max(np.median(np.diff(s)), 1e-6)))
    win = min(win, len(p))
    kern = np.ones(win) / win
    pad = win - 1
    pp = np.pad(p, (pad // 2, pad - pad // 2), mode="edge")
    return np.convolve(pp, kern, mode="valid")[:len(p)]


def render_mesh_gif(history, load_curve, out_path, every=3, fps=8,
                    tool_loads=None, holder_names=None, mark=None,
                    mark_color="#e2231a"):
    """Write a GIF of the deforming mesh + load curve. Returns out_path or None.

    history      : list of StepRecord (coords, elems, stroke, tool_segs)
    load_curve   : list of (stroke_mm, punch_load_N)
    tool_loads   : optional list of per-step {tool: load_N, _stroke: mm}
    holder_names : tool names to overlay (e.g. the force-controlled stripper)
    mark         : optional (r,z) of the diagnosed problem spot -> red circle
    """
    import imageio.v2 as imageio
    if len(history) < 2:
        return None

    lc = np.asarray(load_curve, float) if len(load_curve) else None
    # display-smoothed punch load (0.5mm moving average) -> tames the remesh
    # sawtooth in the curve. The H5 keeps the RAW load (accurate); this only
    # affects what the GIF draws (same approach as the desktop post app).
    lc_sm = _smooth(lc[:, 0], lc[:, 1] / 1000.0) if lc is not None else None
    # force-controlled holder (e.g. stripper) load curves to overlay
    holders = {}
    if tool_loads and holder_names:
        tss = np.array([d.get("_stroke", 0.0) for d in tool_loads])
        for nm in holder_names:
            vals = np.array([d.get(nm, 0.0) for d in tool_loads]) / 1000.0
            if np.max(np.abs(vals)) > 1e-6:
                holders[nm] = (tss, vals)
    # fixed frame = union of the material over all steps (+margin). Tools are
    # clipped to this box by the axis limits (avoids a huge die blowing up the
    # view -- see the fixed-frame lesson in the post app).
    xs = np.concatenate([r.coords[:, 0] for r in history])
    ys = np.concatenate([r.coords[:, 1] for r in history])
    pad = 0.06 * max(np.ptp(xs), np.ptp(ys), 1.0)
    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (ys.min() - pad, ys.max() + pad)
    lmax = float(lc[:, 1].max() / 1000.0) if lc is not None else 1.0

    idx = list(range(0, len(history), max(every, 1)))
    if idx[-1] != len(history) - 1:
        idx.append(len(history) - 1)

    frames = []
    for i in idx:
        r = history[i]
        fig = plt.figure(figsize=(8.2, 4.2), dpi=90)
        axm = fig.add_axes([0.04, 0.12, 0.54, 0.80])
        axl = fig.add_axes([0.67, 0.16, 0.30, 0.74])

        tri = Triangulation(r.coords[:, 0], r.coords[:, 1], r.elems)
        axm.triplot(tri, lw=0.3, color="#1f4e79")
        for segs in (r.tool_segs or []):
            axm.add_collection(LineCollection(list(segs), colors="#2e7d32",
                                              linewidths=1.0))
        if mark is not None:                # diagnosed / marginal problem spot
            rad = 0.06 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
            axm.add_patch(plt.Circle(mark, rad, fill=False,
                                     edgecolor=mark_color, lw=2.0))
        axm.set_xlim(*xlim)
        axm.set_ylim(*ylim)
        axm.set_aspect("equal")          # AFTER set_xlim/ylim (fixed frame)
        axm.set_title(f"mesh  stroke = {r.stroke:.2f} mm", fontsize=10)
        axm.set_xlabel("r / x [mm]", fontsize=8)
        axm.set_ylabel("z / y [mm]", fontsize=8)

        if lc is not None:
            axl.plot(lc[:, 0], lc[:, 1] / 1000.0, color="#9fc5e8", lw=0.6)  # raw
            axl.plot(lc[:, 0], lc_sm, "b-", lw=1.5, label="punch")          # smooth
            cur = float(np.interp(r.stroke, lc[:, 0], lc_sm))
            axl.plot(r.stroke, cur, "bo", ms=5)
            axl.annotate(f"{cur:.0f} kN", (r.stroke, cur), fontsize=8,
                         xytext=(4, -10), textcoords="offset points")
        for nm, (ts, vals) in holders.items():
            axl.plot(ts, vals, "--", lw=1.0, label=nm)
            axl.plot(r.stroke, float(np.interp(r.stroke, ts, vals)), "o", ms=4)
        if lc is not None:
            axl.set_xlim(0, float(lc[:, 0].max()) or 1.0)
            axl.set_ylim(0, lmax * 1.05)
        if holders:
            axl.legend(fontsize=7, loc="upper left")
        axl.set_title("load [kN]", fontsize=10)
        axl.set_xlabel("stroke [mm]", fontsize=8)
        axl.grid(alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return out_path

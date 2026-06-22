"""Rendering helpers for the post-processing viewer.

Pure matplotlib (no Streamlit dependency) so the same renderers drive the
browser app, batch animation export, and the future web service. Each function
returns a Matplotlib Figure (or writes a file for the animation).
"""

from __future__ import annotations

import numpy as np

from .h5data import element_field, field_unit


def _node_avg(coords, elems, ef):
    nf = np.zeros(len(coords))
    cnt = np.zeros(len(coords))
    for e, tri in enumerate(elems):
        nf[tri] += ef[e]
        cnt[tri] += 1
    return nf / np.maximum(cnt, 1)


def node_avg(coords, elems, ef):
    return _node_avg(coords, elems, ef)


def field_grid(coords, elems, nodal, n=240, extent=None):
    """Sample a nodal field onto a regular grid (NaN outside the deformed
    domain) so an interactive Plotly heatmap can zoom/pan it. Returns
    (xi, yi, Z) with Z shape (len(yi), len(xi)).

    `extent` = (xmin, xmax, ymin, ymax) pins the grid to a FIXED frame across
    steps (e.g. the whole-run bounding box). With a constant x/y axis the
    heatmap's uirevision reliably preserves the user's zoom when stepping —
    otherwise the per-step domain change makes Plotly re-autorange and the
    zoom is lost."""
    import matplotlib.tri as mtri
    x, y = coords[:, 0], coords[:, 1]
    tri = mtri.Triangulation(x, y, elems)
    interp = mtri.LinearTriInterpolator(tri, nodal)
    if extent is None:
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
    else:
        xmin, xmax, ymin, ymax = extent
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)
    m = int(np.clip(round(n * yspan / xspan), 40, 700))
    xi = np.linspace(xmin, xmax, n)
    yi = np.linspace(ymin, ymax, m)
    Z = np.asarray(interp(*np.meshgrid(xi, yi)).filled(np.nan))
    return xi, yi, Z


def render_field(step, field_name, vmin=None, vmax=None, cmap="jet",
                 show_mesh=True, show_tools=True, velocity=False,
                 vel_stride=2, vel_scale=1.0, figsize=(7, 7), title_extra="",
                 view_range=None, extent=None, flownet_frame=None,
                 tool_frame=None, load_curve=None):
    """Filled contour of a derived field on the deformed mesh.

    vmin/vmax fix the colour range (None -> auto). `velocity` overlays the nodal
    increment as a quiver. `flownet_frame` = (P, inside, (ny,nx)) overlays the
    fiber-flow grid. `view_range` = ([x0,x1],[y0,y1]) crops to a saved view;
    else `extent` = (xmin,xmax,ymin,ymax) pins a fixed frame across the run.
    Used both for the live GIF (matching the on-screen settings) and exports."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    coords, elems = step["coords"], step["elems"]
    ef = element_field(step, field_name)
    nf = _node_avg(coords, elems, ef)
    tr = mtri.Triangulation(coords[:, 0], coords[:, 1], elems)

    # FIXED figure + axes geometry (explicit add_axes, no constrained/tight
    # layout). Auto layout resized the field axes per frame as the part deformed
    # -> the scale drifted in later frames; fixed positions = pixel-identical
    # frames. The figure WIDTH hugs the part aspect (+ a colour-bar zone) so the
    # image stays compact.
    has_lc = load_curve is not None
    has_tool = tool_frame is not None
    fr = (list(view_range[0]) + list(view_range[1]) if view_range is not None
          else (list(extent) if extent is not None else
                [coords[:, 0].min(), coords[:, 0].max(),
                 coords[:, 1].min(), coords[:, 1].max()]))
    aw = max((fr[1] - fr[0]) / max(fr[3] - fr[2], 1e-9), 1e-3)
    fig_h = 8.8 if has_lc else 7.8
    H_field = 0.72 if has_lc else 0.88
    bottom = 0.055 if has_lc else 0.07
    # cb zone leaves ~3 mm clear right of the (outer) colour-bar tick labels
    left_in, cb_in = 0.75, (2.3 if has_tool else 1.3)
    # Cap the FIELD width (not the whole figure) so left margin + colour-bar
    # zone always stay on-figure. Clamping fig_w instead let a wide (zoomed)
    # frame push FW->~1.0 and the colour bars off the right edge.
    max_total_w = 13.0
    data_w_in = float(np.clip(H_field * fig_h * aw,
                              1.2, max_total_w - left_in - cb_in))
    fig_w = data_w_in + left_in + cb_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    FX, FW = left_in / fig_w, data_w_in / fig_w
    ax = fig.add_axes([FX, bottom, FW, H_field])
    if has_lc:
        ax_lc = fig.add_axes([FX, 0.84, FW, 0.14])   # ~2x taller panel
        strokes, loads_kn = load_curve
        cs, cl = step["stroke"], step["punch_load"] / 1e3
        smax = float(strokes.max())
        ax_lc.plot(strokes, loads_kn, color="royalblue", lw=1.0)
        ax_lc.axvline(cs, color="red", ls="--", lw=1.0)
        # value text at the TOP of the panel (clear of the curve), on the side
        # of the marker that has room
        leftside = cs < 0.55 * smax
        ax_lc.annotate(f"{cl:.0f} kN", (cs, float(loads_kn.max())), color="red",
                       fontsize=9, va="top", ha="left" if leftside else "right",
                       textcoords="offset points",
                       xytext=(4 if leftside else -4, -1),
                       bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                 ec="red", lw=0.7))
        ax_lc.set_xlim(0, smax)
        ax_lc.set_ylabel("load [kN]", fontsize=8)
        ax_lc.set_xlabel("stroke [mm]", fontsize=8, labelpad=1)
        ax_lc.tick_params(labelsize=7)
    if vmin is not None and vmax is not None and vmax > vmin:
        lv = np.linspace(vmin, vmax, 21)
        nfc = np.clip(nf, vmin, vmax - 1e-9 * (vmax - vmin))
        tc = ax.tricontourf(tr, nfc, levels=lv, cmap=cmap, extend="both")
    else:
        tc = ax.tricontourf(tr, nf, levels=20, cmap=cmap)
    if show_mesh:
        ax.triplot(tr, lw=0.12, color="k", alpha=0.30)
    if show_tools:
        for segs in step.get("tool_segs", []):
            for a, b in segs:
                ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="0.1", lw=1.5)
    if velocity and step.get("velocity") is not None:
        v = step["velocity"]
        sl = slice(None, None, max(1, int(vel_stride)))
        p = coords[sl]
        vv = v[sl]
        mag = np.linalg.norm(vv, axis=1)
        if mag.max() > 1e-12:
            ax.quiver(p[:, 0], p[:, 1], vv[:, 0], vv[:, 1], mag,
                      cmap="cool", scale=None if vel_scale == 0 else
                      (1.0 / vel_scale) * mag.max() * 20,
                      width=0.003, alpha=0.9)
    if flownet_frame is not None:
        P, inside, (ny, nx) = flownet_frame
        Pg = np.asarray(P).reshape(ny, nx, 2)
        for jj in range(ny):
            for ii in range(nx - 1):
                if inside[jj, ii] and inside[jj, ii + 1]:
                    ax.plot(Pg[jj, ii:ii + 2, 0], Pg[jj, ii:ii + 2, 1],
                            "-", color="0.15", lw=0.6, alpha=0.7)
        for ii in range(nx):
            for jj in range(ny - 1):
                if inside[jj, ii] and inside[jj + 1, ii]:
                    ax.plot(Pg[jj:jj + 2, ii, 0], Pg[jj:jj + 2, ii, 1],
                            "-", color="0.15", lw=0.6, alpha=0.7)
    pcm_tool = None
    if tool_frame is not None:
        grids, tcmap, tvmn, tvmx = tool_frame
        for (txi, tyi, tZ) in grids:
            pcm_tool = ax.pcolormesh(txi, tyi, np.ma.masked_invalid(tZ),
                                     cmap=tcmap, vmin=tvmn, vmax=tvmx,
                                     shading="auto")
    u = field_unit(field_name)
    cb_y, cb_h = 0.20, 0.55
    cb_x0 = FX + FW + 0.45 / fig_w   # small gap (in) right of the field
    bw = 0.18 / fig_w                # bar width ~0.18 in
    mat_title = f"{field_name}\n[{u}]" if u else field_name
    if pcm_tool is not None:
        # both bars use a horizontal TOP title (no side labels) -> they sit
        # close together; only the inner bar's tick labels go between them.
        caxm = fig.add_axes([cb_x0, cb_y, bw, cb_h])
        fig.colorbar(tc, cax=caxm).ax.set_title(mat_title, fontsize=8)
        caxt = fig.add_axes([cb_x0 + 1.0 / fig_w, cb_y, bw, cb_h])
        cbt = fig.colorbar(pcm_tool, cax=caxt)
        cbt.ax.set_title("tool σ\n[MPa]", fontsize=8)
    else:
        caxm = fig.add_axes([cb_x0, cb_y, 0.22 / fig_w, cb_h])
        fig.colorbar(tc, cax=caxm).ax.set_title(mat_title, fontsize=8)
    ax.set_xlabel("r / x [mm]")
    ax.set_ylabel("z / y [mm]")
    ax.set_title(f"{field_name}   stroke={step['stroke']:.2f} mm{title_extra}")
    if view_range is not None:
        ax.set_xlim(view_range[0])
        ax.set_ylim(view_range[1])
    elif extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    # equal aspect AFTER the limits are fixed, so the axes box is sized to the
    # fixed frame (not the per-frame autoscaled data) -> identical every frame
    ax.set_aspect("equal")
    return fig


def render_flownet(history, inside, grid_shape, step_idx, coords=None,
                   elems=None, figsize=(7, 7)):
    """Deformed tracer grid (fiber flow) at a given step index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    ny, nx = grid_shape
    si = min(step_idx, len(history) - 1)
    P = history[si].reshape(ny, nx, 2)

    fig, ax = plt.subplots(figsize=figsize)
    if coords is not None and elems is not None:
        tr = mtri.Triangulation(coords[:, 0], coords[:, 1], elems)
        ax.triplot(tr, lw=0.1, color="0.8")
    for j in range(ny):
        for i in range(nx - 1):
            if inside[j, i] and inside[j, i + 1]:
                ax.plot(P[j, i:i + 2, 0], P[j, i:i + 2, 1], "b-", lw=0.8)
    for i in range(nx):
        for j in range(ny - 1):
            if inside[j, i] and inside[j + 1, i]:
                ax.plot(P[j:j + 2, i, 0], P[j:j + 2, i, 1], "b-", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("r / x [mm]")
    ax.set_ylabel("z / y [mm]")
    ax.set_title(f"flow net (fiber flow)   frame {si}")
    fig.tight_layout()
    return fig


def global_range(result, field_name, sample=12):
    """Global min/max of a field over the run (sampled steps) for a fixed
    colour scale across the animation."""
    idx = np.unique(np.linspace(0, result.n - 1, sample).astype(int))
    lo, hi = np.inf, -np.inf
    for i in idx:
        ef = element_field(result.step(int(i)), field_name)
        lo = min(lo, float(ef.min()))
        hi = max(hi, float(ef.max()))
    if not np.isfinite(lo) or hi <= lo:
        return None, None
    return lo, hi


def export_animation(result, field_name, out_path, vmin=None, vmax=None,
                     cmap="jet", fps=10, every=1, velocity=False,
                     show_mesh=True, show_tools=True, vel_stride=3,
                     view_range=None, extent=None, flownet=None,
                     tool_grid_fn=None, tool_cmap="hot", tool_vmin=0.0,
                     tool_vmax=None, load_curve=None, frame_dir=None,
                     progress=None):
    """Render every `every`-th step to a GIF, reflecting the on-screen settings
    (field, colour range, tools, velocity, flow net, zoom, tool stress). mp4
    needs ffmpeg (not assumed), so GIF.

    tool_grid_fn(i) -> (grids, tvmax) | None solves the tool stress per frame
    (slow); tool_vmax fixes the tool colour scale across frames (else per-step).
    frame_dir: if given, also save every frame as a PNG there."""
    import io
    import os
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    if frame_dir:
        os.makedirs(frame_dir, exist_ok=True)
    idx = list(range(0, result.n, max(1, every)))
    frames = []
    for k, i in enumerate(idx):
        frame = None
        if flownet is not None:
            history, inside, shape = flownet
            frame = (history[min(i, len(history) - 1)], inside, shape)
        tframe = None
        if tool_grid_fn is not None:
            tg = tool_grid_fn(i)
            if tg is None:
                # unloaded step (no tool stress): skip so every frame keeps the
                # same 2-colour-bar layout (no odd 1-bar frame at stroke 0)
                if progress:
                    progress((k + 1) / len(idx))
                continue
            grids, tv = tg
            tvmx = tool_vmax if tool_vmax is not None else tv
            tframe = (grids, tool_cmap, tool_vmin, tvmx)
        fig = render_field(result.step(i), field_name, vmin=vmin, vmax=vmax,
                           cmap=cmap, show_mesh=show_mesh, show_tools=show_tools,
                           velocity=velocity, vel_stride=vel_stride,
                           view_range=view_range, extent=extent,
                           flownet_frame=frame, tool_frame=tframe,
                           load_curve=load_curve)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        if frame_dir:
            fig.savefig(os.path.join(
                frame_dir, f"frame_{i:05d}_s{result.step(i)['stroke']:05.2f}mm.png"),
                dpi=110)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        if progress:
            progress((k + 1) / len(idx))
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return out_path

"""PlasticFEM v3 — browser post-processor (Streamlit, interactive Plotly).

Run:  anaconda3\\python.exe -m streamlit run app/post_app.py

Layout: ALL controls live in the left sidebar; the centre shows only the
result, so the view never scrolls away during playback. Field view is Plotly
(native zoom / pan / hover) with auto-play, 5 saveable viewports, and the
load-stroke curve underneath. Also: flow net, elastic tool-stress post-analysis
and GIF export. Renderers (plasticfem.viewer) / loader (plasticfem.h5data) are
shared so the same code can back the future VPS service.
"""

import glob
import os
import sys
import tempfile
import time

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plasticfem import fem
from plasticfem.h5data import (H5Result, element_field, STRESS_FIELDS,
                               STRAIN_FIELDS, OTHER_FIELDS, field_unit)
from plasticfem.viewer import (node_avg, field_grid, render_flownet,
                               global_range, export_animation)
from plasticfem.contact import (RigidTool, build_contact_points,
                                detect_contacts)
from plasticfem.tool_stress import solve_tool_stress, plot_tool_stress

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="PlasticFEM Post", layout="wide",
                   initial_sidebar_state="expanded")
# hide the hamburger menu / deploy toolbar / footer, but KEEP the top-right
# "running" indicator (it should show only while actually computing).
st.markdown("""<style>
#MainMenu{visibility:hidden;} [data-testid="stToolbar"]{display:none;}
footer{visibility:hidden;} [data-testid="stSidebar"]{min-width:330px;}
/* trim only the BOTTOM padding; keep enough top padding so the tabs stay
   clear of Streamlit's fixed header (over-trimming hid them earlier) */
[data-testid="stMain"] .block-container{padding-top:3.5rem; padding-bottom:0.5rem;}
/* hide the per-element hover toolbar (the confusing top-right fullscreen menu
   on the matplotlib / image panels) */
[data-testid="stElementToolbar"]{display:none;}
/* pan/zoom buttons compact (keyboard shortcuts may not work in every browser
   sandbox, so keep the buttons visible as the reliable control) */
.st-key-pfem_pan button{padding:0.1rem 0; min-height:1.8rem;}
</style>""", unsafe_allow_html=True)

MPL_CMAP = {"Jet": "jet", "Viridis": "viridis", "Turbo": "turbo",
            "RdBu": "RdBu", "Greys": "Greys", "Hot": "hot"}


@st.cache_resource(show_spinner=False)
def load_result(path, mtime):
    return H5Result(path)


@st.cache_data(show_spinner=False)
def global_extent(path, mtime, sample=15):
    """Whole-run bounding box (sampled) -> a FIXED grid frame so the heatmap's
    axes are identical every step and uirevision keeps the user's zoom."""
    r = load_result(path, mtime)
    idx = np.unique(np.linspace(0, r.n - 1, sample).astype(int))
    xmn = ymn = 1e30
    xmx = ymx = -1e30
    for i in idx:
        c = r.step(int(i))["coords"]
        xmn = min(xmn, float(c[:, 0].min())); xmx = max(xmx, float(c[:, 0].max()))
        ymn = min(ymn, float(c[:, 1].min())); ymx = max(ymx, float(c[:, 1].max()))
    pad = 0.02 * max(xmx - xmn, ymx - ymn)
    return (xmn - pad, xmx + pad, ymn - pad, ymx + pad)


@st.cache_data(show_spinner=False)
def global_extent_full(path, mtime, sample=15):
    """Whole-run frame containing the workpiece (every step) plus the die FACES
    around it — but NOT the full die bodies, which often extend far past the part
    (e.g. a die modelled as a big rectangle) and would blow the frame up so much
    the user must zoom in, which then clips the (taller) early frames. We include
    tool points only within a die-thick margin of the workpiece box, so the frame
    stays tight to the action and contains the whole deformation without clipping."""
    r = load_result(path, mtime)
    idx = np.unique(np.linspace(0, r.n - 1, sample).astype(int))
    wx0 = wy0 = 1e30
    wx1 = wy1 = -1e30
    tool_pts = []
    for i in idx:
        s = r.step(int(i))
        c = s["coords"]
        wx0 = min(wx0, float(c[:, 0].min())); wx1 = max(wx1, float(c[:, 0].max()))
        wy0 = min(wy0, float(c[:, 1].min())); wy1 = max(wy1, float(c[:, 1].max()))
        for segs in s.get("tool_segs", []):
            tool_pts.append(np.asarray(segs).reshape(-1, 2))
    xmn, xmx, ymn, ymx = wx0, wx1, wy0, wy1
    rng = max(wx1 - wx0, wy1 - wy0)
    if tool_pts:
        P = np.vstack(tool_pts)
        m = 0.15 * rng
        near = ((P[:, 0] >= wx0 - m) & (P[:, 0] <= wx1 + m) &
                (P[:, 1] >= wy0 - m) & (P[:, 1] <= wy1 + m))
        if near.any():
            Pn = P[near]
            xmn = min(xmn, float(Pn[:, 0].min())); xmx = max(xmx, float(Pn[:, 0].max()))
            ymn = min(ymn, float(Pn[:, 1].min())); ymx = max(ymx, float(Pn[:, 1].max()))
    pad = 0.04 * rng
    return (xmn - pad, xmx + pad, ymn - pad, ymx + pad)


@st.cache_data(show_spinner=False)
def grid_cached(path, mtime, i, field, n, extent):
    r = load_result(path, mtime)
    s = r.step(i)
    nf = node_avg(s["coords"], s["elems"], element_field(s, field))
    return field_grid(s["coords"], s["elems"], nf, n=n, extent=extent)


@st.cache_data(show_spinner=False)
def flownet_cached(path, mtime):
    return load_result(path, mtime).flownet()


@st.cache_data(show_spinner=False)
def global_range_cached(path, mtime, field):
    return global_range(load_result(path, mtime), field)


@st.cache_data(show_spinner=False)
def tool_load_history(path, mtime):
    """Per-tool reaction load over the run (read once, cached) — avoids reading
    the H5 for every step on every rerun, which made the load-curve view slow."""
    res = load_result(path, mtime)
    out = {nm: np.full(res.n, np.nan) for nm in res.tool_names}
    for k in range(res.n):
        tl = res.step(k)["tool_loads"]
        for nm in res.tool_names:
            out[nm][k] = tl.get(nm, np.nan)
    return out


def _boundary(elems):
    from collections import defaultdict
    cnt = defaultdict(int)
    for tri in elems:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            cnt[(min(a, b), max(a, b))] += 1
    nodes, edges = set(), []
    for (a, b), c in cnt.items():
        if c == 1:
            nodes.add(a); nodes.add(b); edges.append((a, b))
    return np.array(sorted(nodes), int), np.array(edges, int)


def build_field_fig(res, path, mtime, i, field, vmin, vmax, cmap, show_tools,
                    show_vel, vel_stride, view_range, fast=False, height=470,
                    flownet_data=None, tool_grids=None, tool_cmap="Hot",
                    tool_vmin=None, tool_vmax=None, show_mesh=False):
    import plotly.graph_objects as go
    s = res.step(i)
    n = 130 if fast else 190            # grid resolution (lower = faster render)
    extent = global_extent(path, mtime)   # fixed frame -> zoom persists on step
    xi, yi, Z = grid_cached(path, mtime, i, field, n, extent)
    fig = go.Figure(go.Heatmap(
        x=xi, y=yi, z=Z, colorscale=cmap, zmin=vmin, zmax=vmax,
        zsmooth=False if fast else "best", connectgaps=False,
        colorbar=dict(title=field_unit(field) or "", x=1.0),
        hovertemplate="x=%{x:.2f}  y=%{y:.2f}<br>" + field +
        "=%{z:.4g}<extra></extra>"))
    # tool stresses overlaid as heatmaps (own colourscale + colour bar) so the
    # material field and the tool stresses appear in ONE interactive figure
    if tool_grids is not None:
        grids, tvmax = tool_grids
        zmn = tool_vmin if tool_vmin is not None else 0.0
        zmx = tool_vmax if tool_vmax is not None else tvmax
        for gi, (txi, tyi, tZ) in enumerate(grids):
            fig.add_heatmap(x=txi, y=tyi, z=tZ, colorscale=tool_cmap,
                            zmin=zmn, zmax=zmx, zsmooth="best",
                            connectgaps=False, showscale=(gi == 0),
                            hovertemplate="tool σ=%{z:.0f} MPa<extra></extra>",
                            colorbar=dict(title="tool σ", x=1.16,
                                          thickness=12) if gi == 0 else None)
    if show_mesh:
        c, el = s["coords"], s["elems"]
        e = np.unique(np.sort(np.vstack([el[:, [0, 1]], el[:, [1, 2]],
                                         el[:, [2, 0]]]), axis=1), axis=0)
        mx = np.empty(e.shape[0] * 3); my = np.empty(e.shape[0] * 3)
        mx[0::3], mx[1::3], mx[2::3] = c[e[:, 0], 0], c[e[:, 1], 0], np.nan
        my[0::3], my[1::3], my[2::3] = c[e[:, 0], 1], c[e[:, 1], 1], np.nan
        fig.add_scatter(x=mx, y=my, mode="lines", hoverinfo="skip",
                        line=dict(color="rgba(0,0,0,0.28)", width=0.5),
                        showlegend=False)
    if show_tools:
        for segs in s.get("tool_segs", []):
            xs, ys = [], []
            for a, b in segs:
                xs += [a[0], b[0], None]; ys += [a[1], b[1], None]
            fig.add_scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                            line=dict(color="black", width=2), showlegend=False)
    if show_vel and not fast and s.get("velocity") is not None:
        import plotly.figure_factory as ff
        v = s["velocity"]; c = s["coords"]
        sl = slice(None, None, max(1, int(vel_stride)))
        u, w = v[sl, 0], v[sl, 1]
        mag = np.linalg.norm(np.c_[u, w], axis=1)
        if mag.max() > 1e-12:
            q = ff.create_quiver(c[sl, 0], c[sl, 1], u, w,
                                 scale=0.6 / mag.max() * (xi[-1] - xi[0]) / 20,
                                 arrow_scale=0.35,
                                 line=dict(width=1, color="#00d0ff"))
            fig.add_traces(q.data)
    if flownet_data is not None:
        history, inside, (ny, nx) = flownet_data
        P = history[min(i, len(history) - 1)].reshape(ny, nx, 2)
        xs, ys = [], []
        for jj in range(ny):
            for ii in range(nx - 1):
                if inside[jj, ii] and inside[jj, ii + 1]:
                    xs += [P[jj, ii, 0], P[jj, ii + 1, 0], None]
                    ys += [P[jj, ii, 1], P[jj, ii + 1, 1], None]
        for ii in range(nx):
            for jj in range(ny - 1):
                if inside[jj, ii] and inside[jj + 1, ii]:
                    xs += [P[jj, ii, 0], P[jj + 1, ii, 0], None]
                    ys += [P[jj, ii, 1], P[jj + 1, ii, 1], None]
        fig.add_scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                        line=dict(color="rgba(0,0,0,0.55)", width=0.8),
                        showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="z / y [mm]")
    fig.update_xaxes(title="r / x [mm]")
    # field/stroke label INSIDE the plot frame (a layout title sat in the top
    # margin and got clipped); a paper-coord annotation never clips
    fig.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper",
                       xanchor="left", yanchor="top", showarrow=False,
                       text=f"{field}　{s['stroke']:.2f} mm",
                       font=dict(size=13, color="#111"),
                       bgcolor="rgba(255,255,255,0.7)", borderpad=2)
    fig.update_layout(height=height,
                      margin=dict(l=0, r=75 if tool_grids else 0, t=8, b=0),
                      dragmode="select", uirevision="keep", showlegend=False)
    if view_range is not None:
        # keep the equal-aspect lock (scaleanchor) when zooming to a saved view
        # so the native autoscale button doesn't stretch the part horizontally
        fig.update_xaxes(range=view_range[0])
        fig.update_yaxes(range=view_range[1])
        fig.update_layout(uirevision=str(view_range))
    else:
        # fixed frame so the view doesn't jitter between steps; include the
        # tools (which extend past the workpiece) when they are shown
        fe = (global_extent_full(path, mtime)
              if (show_tools or tool_grids is not None) else extent)
        fig.update_xaxes(range=[fe[0], fe[1]])
        fig.update_yaxes(range=[fe[2], fe[3]])
    return fig


def _solve_tool_result(res, step, tool_name, tool_mesh):
    """Solve one tool's elastic stress at this step; returns a ToolStressResult."""
    ti = res.tool_names.index(tool_name)
    poly = step["tool_segs"][ti][:, 0, :]
    tool = RigidTool.from_polygon(tool_name, poly)
    coords, elems = step["coords"], step["elems"]
    axisym = coords[:, 0].min() >= -1e-6 and (np.abs(coords[:, 0]) < 1e-6).any()
    mode = fem.AXISYMMETRIC if axisym else fem.PLANE_STRAIN
    sn, se = _boundary(elems)
    cpoints = [c for c in build_contact_points(coords, sn, se, axisym=axisym)
               if c[0] == c[1]]
    contacts = detect_contacts(coords, cpoints, tool, g_max=res.mesh_size or 1.5)
    if not contacts:
        raise RuntimeError(f"no contact on '{tool_name}'")
    kn = res.kn if res.kn > 0 else 2.0 * 210000.0 / (res.mesh_size or 1.5)
    return solve_tool_stress(tool, contacts, cpoints, coords, kn, mode=mode,
                             mesh_size=tool_mesh, tool_name=tool_name,
                             stroke=step["stroke"])


@st.cache_data(show_spinner=False)
def tool_stress_all_cached(path, mtime, step_idx, tool_mesh):
    """Solve ALL tools at this step and render them together (with the material)
    in one figure — no tool selection needed."""
    from plasticfem.tool_stress import plot_all_tools
    res = load_result(path, mtime)
    step = res.step(step_idx)
    results, infos, errs = [], [], []
    for tn in res.tool_names:
        try:
            r = _solve_tool_result(res, step, tn, tool_mesh)
            results.append(r)
            F = r.load_vecs.sum(axis=0)
            infos.append(f"{tn}: |F|={np.linalg.norm(F)/1e3:.0f} kN, "
                         f"max {r.mises.max():.0f} MPa")
        except Exception as e:
            errs.append(f"{tn}: {type(e).__name__}: {e}")
    if not results:
        raise RuntimeError("全工具で失敗 → " + " / ".join(errs))
    out = os.path.join(tempfile.gettempdir(), f"_tsall_{os.getpid()}.png")
    plot_all_tools(results, step["coords"], step["elems"], out,
                   work_mises=step["mises"])
    with open(out, "rb") as f:
        return f.read(), " · ".join(infos)


@st.cache_data(show_spinner=False)
def tool_stress_grids_cached(path, mtime, step_idx, tool_mesh):
    """Solve all tools and return each tool's von Mises sampled on a grid, so it
    can be drawn as a heatmap overlay INSIDE the interactive field figure
    (material field + tool stresses in one zoomable view)."""
    import matplotlib.tri as mtri
    res = load_result(path, mtime)
    step = res.step(step_idx)
    grids, infos, tvmax = [], [], 0.0
    for tn in res.tool_names:
        try:
            r = _solve_tool_result(res, step, tn, tool_mesh)
        except Exception:
            continue
        nf = node_avg(r.coords, r.elems, r.mises)
        x0, x1 = float(r.coords[:, 0].min()), float(r.coords[:, 0].max())
        y0, y1 = float(r.coords[:, 1].min()), float(r.coords[:, 1].max())
        nx = 120
        ny = int(np.clip(round(nx * (y1 - y0) / max(x1 - x0, 1e-9)), 30, 320))
        xi = np.linspace(x0, x1, nx)
        yi = np.linspace(y0, y1, ny)
        tri = mtri.Triangulation(r.coords[:, 0], r.coords[:, 1], r.elems)
        Z = np.asarray(mtri.LinearTriInterpolator(tri, nf)(
            *np.meshgrid(xi, yi)).filled(np.nan))
        grids.append((xi, yi, Z))
        infos.append(f"{tn}: max {r.mises.max():.0f} MPa")
        tvmax = max(tvmax, float(r.mises.max()))
    if not grids:
        raise RuntimeError("接触している工具がありません")
    return grids, tvmax, " · ".join(infos)


@st.cache_data(show_spinner=False)
def tool_global_vmax(path, mtime, tool_mesh, sample=8):
    """Global max tool von Mises over the run (sampled) -> a FIXED tool colour
    scale for comparing steps (like the material's 'whole run')."""
    res = load_result(path, mtime)
    idx = np.unique(np.linspace(0, res.n - 1, sample).astype(int))
    vmax = 0.0
    for i in idx:
        try:
            _, tv, _ = tool_stress_grids_cached(path, mtime, int(i), tool_mesh)
            vmax = max(vmax, tv)
        except Exception:
            pass
    return vmax or None


# ════════════════════════ SIDEBAR — all controls ════════════════════════════
sb = st.sidebar
sb.title("PlasticFEM Post")
# primary input: drop in a results.h5 downloaded from ForgeCal (works in the
# standalone EXE, where there is no local results/ folder)
up = sb.file_uploader("results.h5 を選択（ForgeCal でDLしたファイル）",
                      type=["h5"])
path = None
if up is not None:
    _td = os.path.join(tempfile.gettempdir(), "plasticfem_post")
    os.makedirs(_td, exist_ok=True)
    path = os.path.join(_td, up.name)
    _buf = up.getbuffer()
    if not os.path.exists(path) or os.path.getsize(path) != len(_buf):
        with open(path, "wb") as _f:        # write once -> stable mtime -> cache
            _f.write(_buf)
else:
    files = sorted(glob.glob(os.path.join(HERE, "results", "**", "results.h5"),
                             recursive=True), key=os.path.getmtime, reverse=True)
    choice = sb.selectbox("results.h5（ローカル results/ 内）", files,
                          format_func=lambda p: os.path.relpath(p, HERE)) \
        if files else None
    path = sb.text_input("…or path", value=choice or "") or choice
if not path or not os.path.exists(path):
    st.info("Pick a results.h5 (e.g. `run_case.py upsetting --flownet`).")
    st.stop()
mtime = os.path.getmtime(path)
res = load_result(path, mtime)
sb.caption(f"{res.n} steps · 0–{res.strokes.max():.1f} mm · "
           f"tools: {', '.join(res.tool_names) or '—'} · "
           f"flownet: {'✓' if res.has_flownet else '✗'} · "
           f"velocity: {'✓' if res.has_velocity else '✗'}")

ss = st.session_state
ss.setdefault("step", res.n - 1)
ss.setdefault("step_idx", res.n - 1)
ss.setdefault("playing", False)
ss.setdefault("viewports", [None] * 5)
ss.setdefault("view_range", None)
ss.setdefault("last_box", None)
# saved viewports / zoom are absolute mm boxes tied to ONE part's geometry, so
# they make no sense for a different H5. Reset them whenever the file changes.
if ss.get("loaded_path") != path:
    ss.loaded_path = path
    ss.viewports = [None] * 5
    ss.view_range = None
    ss.last_box = None
    ss.step = ss.step_idx = res.n - 1
ss.step = min(ss.step, res.n - 1)
ss.step_idx = min(ss.step_idx, res.n - 1)   # keep the slider key in range

sb.divider()
sb.markdown("**① ステップ移動**")
# Step(index) sits directly under the header; the 送り幅 + ◀▶ controls below
# drive it through its session key (ss.step_idx). The buttons use on_click
# callbacks (which run BEFORE the rerun, so they may set the slider's key).
ss.step = sb.slider("Step (index)", 0, res.n - 1, key="step_idx")
pitch = sb.select_slider("送り幅 [mm]", [0.05, 0.1, 0.2, 0.5, 1.0], value=0.1)


def _goto(j):
    ss.step_idx = int(min(max(j, 0), res.n - 1))


def _step_by(d_mm):
    """Jump to the step nearest the current stroke ± d_mm (so ◀▶ move by a
    chosen mm pitch, not one raw substep)."""
    cur = ss.step_idx
    j = int(np.argmin(np.abs(res.strokes - (res.strokes[cur] + d_mm))))
    if j == cur:                           # guarantee at least one step
        j = min(res.n - 1, cur + 1) if d_mm > 0 else max(0, cur - 1)
    ss.step_idx = j


b = sb.columns(4)
b[0].button("⏮", help="最初へ", use_container_width=True,
            on_click=_goto, args=(0,))
b[1].button("◀", help=f"{pitch}mm 戻る", use_container_width=True,
            on_click=_step_by, args=(-pitch,))
b[2].button("▶", help=f"{pitch}mm 進む", use_container_width=True,
            on_click=_step_by, args=(pitch,))
b[3].button("⏭", help="最後へ", use_container_width=True,
            on_click=_goto, args=(res.n - 1,))
sb.caption(f"◀ ▶ と , . は {pitch}mm 刻みで移動します。")

sb.divider()
sb.markdown("**② フィールド**")
group = sb.radio("Field group", ["Stress", "Strain", "Damage"], horizontal=True)
opts = {"Stress": STRESS_FIELDS, "Strain": STRAIN_FIELDS,
        "Damage": OTHER_FIELDS}[group]
field = sb.selectbox("Field", opts,
                     help="σxx/σyy/σzz/σxy are stress COMPONENTS; σ1/σ2 are the "
                          "principal stresses. Axisymmetric: x=r, y=z, σzz=hoop.")
cmap = sb.selectbox("Colormap", list(MPL_CMAP))
rng_mode = sb.radio(
    "Colour range — 色スケールの範囲",
    ["Auto (this step)", "Auto (whole run)", "Manual"],
    index=1,
    help="値→色の対応の最小/最大を決めます。\n"
         "・Auto (this step): その工程の最小〜最大に自動。\n"
         "・Auto (whole run): 全工程で同じ範囲に固定（工程間やGIFの比較に最適）。\n"
         "・Manual: vmin/vmax を入力し、特定の応力/ひずみ帯だけを強調。")
vmin = vmax = None
if rng_mode == "Auto (whole run)":
    # spinner shows only while actually computing (cached -> instant, no flash)
    with st.spinner("全工程のカラーレンジを計算中…（初回のみ）"):
        vmin, vmax = global_range_cached(path, mtime, field)
elif rng_mode == "Manual":
    ef0 = element_field(res.step(ss.step), field)
    lo, hi = float(ef0.min()), float(ef0.max())
    pad = 0.5 * (hi - lo + 1e-9)
    cA, cB = sb.columns(2)
    vmin = cA.number_input("vmin", value=round(lo - pad, 3))
    vmax = cB.number_input("vmax", value=round(hi + pad, 3))
sb.divider()
sb.markdown("**③ オーバーレイ（重ね表示）**")
c1, c2 = sb.columns(2)
show_tools = c1.checkbox("工具線", True)
show_vel = c2.checkbox("速度ベクトル", res.has_velocity,
                       disabled=not res.has_velocity)
vel_stride = sb.slider("ベクトル間引き", 1, 8, 3) if show_vel else 3
c3, c4 = sb.columns(2)
show_flow = c3.checkbox("フローネット", res.has_flownet,
                        disabled=not res.has_flownet,
                        help="繊維流（tracer格子）を場に重ねます")
show_mesh = c4.checkbox("メッシュ表示", False,
                        help="FEMメッシュ（要素の辺）を場に重ねます")
# 工具応力 on its own row so it reads together with the 工具メッシュ寸法 slider
show_ts = sb.checkbox("工具応力", True,
                      help="工具のvon Misesを場に重ねて1つの図に統合")
ts_mesh, ts_cmap, ts_vmin, ts_vmax = 0.9, "Hot", None, None
if show_ts:
    ts_mesh = sb.slider("工具メッシュ寸法", 0.4, 2.0, 0.9)
    ts_cmap = sb.selectbox("工具応力 Colormap", list(MPL_CMAP),
                           index=list(MPL_CMAP).index("Hot"))
    ts_rng = sb.radio("工具応力 Colour range",
                      ["Auto (this step)", "Auto (whole run)", "Manual"],
                      index=1)
    if ts_rng == "Manual":
        tcA, tcB = sb.columns(2)
        ts_vmin = tcA.number_input("工具 vmin", value=0.0)
        ts_vmax = tcB.number_input("工具 vmax", value=6000.0)
    elif ts_rng == "Auto (whole run)":
        with st.spinner("工具応力の全工程レンジを計算中…"):
            ts_vmin, ts_vmax = 0.0, tool_global_vmax(path, mtime, ts_mesh)
sb.divider()
sb.markdown("**④ レイアウト**")
show_lc = sb.checkbox("荷重曲線を上に表示", True,
                      help="場の上に荷重曲線を表示（既定ON）")
chart_h = sb.slider("図の高さ [px]", 320, 680, 470, 10,
                    help="場の図の縦サイズ。画面に収まらずスクロールが出る時は"
                         "小さく、大きく見たい時は大きくしてください。")

sb.divider()
sb.markdown("**⑤ 画角を記憶**")
sb.caption("移動 ↓（番号ボタン）")
v = sb.columns(5)
for k in range(5):
    if v[k].button(f"{k+1}", disabled=ss.viewports[k] is None,
                   help="この保存ビューへ移動" if ss.viewports[k]
                   else "空き", use_container_width=True):
        ss.view_range = ss.viewports[k]
sb.caption("保存 ↓（枠を選んで Save）")
v2 = sb.columns([2, 2])
slot = v2[0].selectbox("slot", ["1", "2", "3", "4", "5"],
                       label_visibility="collapsed")
if v2[1].button("Save", help="現在ズームしている画角をこの枠に保存",
                use_container_width=True):
    if ss.view_range is not None:
        ss.viewports[int(slot) - 1] = ss.view_range
        st.toast(f"現在の画角を View {slot} に保存しました")
        st.rerun()                       # refresh so the View button enables now
    else:
        st.toast("先に図をドラッグでズームしてください（全体表示は保存対象外）")


def _shift_view(dx=0.0, dy=0.0, zoom=1.0):
    """Pan/zoom the current view by fractions of its size (persists across steps,
    unlike native pan/scroll which Streamlit cannot read back). If not zoomed
    yet, start from the whole-run frame so the buttons always work."""
    if ss.view_range is None:
        e = global_extent(path, mtime)
        ss.view_range = ([e[0], e[1]], [e[2], e[3]])
    (x0, x1), (y0, y1) = ss.view_range
    w, h = (x1 - x0) * zoom, (y1 - y0) * zoom
    cx, cy = 0.5 * (x0 + x1) + dx * (x1 - x0), 0.5 * (y0 + y1) + dy * (y1 - y0)
    ss.view_range = ([cx - w / 2, cx + w / 2], [cy - h / 2, cy + h / 2])


sb.markdown("**⑥ 表示の操作（パン/ズーム）**")
sb.caption("ズーム=図をドラッグで囲む or 下の＋−。ボタンでパン/拡縮/全体。"
           "キー（環境により可）: 矢印=パン, PgUp/PgDn=拡縮, 0=全体, , .=step。")
# buttons always shown (also kept in the DOM so the keyboard shortcuts can find
# them); they work even before a drag-zoom (start from the whole-run frame)
with sb.container(key="pfem_pan"):
    p = st.columns(4)
    if p[0].button("←"):
        _shift_view(dx=-0.15)
    if p[1].button("→"):
        _shift_view(dx=0.15)
    if p[2].button("↑"):
        _shift_view(dy=0.15)
    if p[3].button("↓"):
        _shift_view(dy=-0.15)
    z = st.columns(3)
    if z[0].button("－ 縮小"):
        _shift_view(zoom=1.3)
    if z[1].button("＋ 拡大"):
        _shift_view(zoom=1 / 1.3)
    if z[2].button("全体"):
        ss.view_range = None

# ── quit (bottom of the sidebar) — EXE only ──────────────────────────────────
if getattr(sys, "frozen", False):
    sb.divider()
    if sb.button("⏻ アプリを終了", use_container_width=True,
                 help="サーバーを停止してアプリを終了します"):
        import threading
        # Cover the whole page with a clean "終了しました" overlay BEFORE the
        # server dies, so the user never sees Streamlit's scary "Connection
        # error / restart streamlit" screen. Also try to close the tab (browsers
        # only allow closing a window the page scripted open, so this is a no-op
        # in Chrome etc.). The server is killed a beat later, after the overlay
        # has painted.
        components.html("""
        <script>
        const d = window.parent.document;
        const o = d.createElement('div');
        o.style.cssText = 'position:fixed;inset:0;z-index:2147483647;'+
          'background:#0e1117;color:#e6e6e6;display:flex;flex-direction:column;'+
          'align-items:center;justify-content:center;font-family:sans-serif;'+
          'font-size:20px;line-height:1.8;text-align:center;';
        o.innerHTML = '✅ アプリを終了しました。<br>'+
          '<span style="font-size:15px;color:#9aa0a6;">'+
          'このタブを閉じてください。</span>';
        d.body.appendChild(o);
        window.open('','_self'); window.close();   // close tab if allowed
        </script>
        """, height=0)
        threading.Timer(1.2, lambda: os._exit(0)).start()
        st.stop()

# ════════════════════════ MAIN — the field view (simplified, single view) ═══
if True:
    import plotly.graph_objects as go
    s = res.step(ss.step)
    ef = element_field(s, field)
    # ── load curve fixed ABOVE the field ──
    if show_lc:
        cur_load = res.loads[ss.step] / 1e3
        lc = go.Figure(go.Scatter(x=res.strokes, y=res.loads / 1e3,
                                  line=dict(color="royalblue")))
        lc.add_vline(x=s["stroke"], line=dict(color="red", dash="dash"))
        # put the value text up at the top of the panel (clear of the curve),
        # left/right of the marker depending on which side has room
        right_side = s["stroke"] < 0.6 * res.strokes.max()
        lc.add_annotation(x=s["stroke"], y=res.loads.max() / 1e3,
                          text=f"{cur_load:.0f} kN", showarrow=False,
                          xanchor="left" if right_side else "right",
                          yanchor="top", xshift=6 if right_side else -6,
                          font=dict(color="red", size=12),
                          bgcolor="rgba(255,255,255,0.9)",
                          bordercolor="red", borderwidth=1)
        lc.update_layout(height=140, margin=dict(l=0, r=0, t=2, b=2),
                         xaxis_title=None, yaxis_title="kN", showlegend=False)
        st.plotly_chart(lc, use_container_width=True, key="lc_field")
    # ── info line ABOVE the field (no zoom hint) ──
    st.caption(f"step {ss.step}/{res.n-1} · stroke {s['stroke']:.3f} mm · "
               f"punch {s['punch_load']/1e3:.0f} kN · {field}: "
               f"{ef.min():.3g} … {ef.max():.3g} {field_unit(field)}")
    if ef.max() - ef.min() < 1e-9:
        st.info(f"'{field}' is uniform ({ef.max():.3g}) here — e.g. damage is "
                f"zero unless this is a shearing/blanking case.")
    # ── tool stress: solve and overlay INTO the field figure ──
    tool_grids = None
    ts_info = None
    if show_ts and res.tool_names:
        try:
            with st.spinner("工具応力を計算中…"):
                grids, tvmax, ts_info = tool_stress_grids_cached(
                    path, mtime, ss.step, ts_mesh)
            tool_grids = (grids, tvmax)
        except Exception as e:
            st.warning(f"工具応力を計算できません: {e}")
    # ── the field (with optional tool-stress overlay) ──
    with st.spinner("場を描画中…"):
        fig = build_field_fig(res, path, mtime, ss.step, field, vmin, vmax,
                              cmap, show_tools, show_vel, vel_stride,
                              ss.view_range, height=chart_h,
                              flownet_data=flownet_cached(path, mtime)
                              if show_flow else None, tool_grids=tool_grids,
                              tool_cmap=ts_cmap, tool_vmin=ts_vmin,
                              tool_vmax=ts_vmax, show_mesh=show_mesh)
    ev = st.plotly_chart(fig, use_container_width=True, key="fchart",
                         on_select="rerun", selection_mode="box",
                         config={"scrollZoom": False, "displayModeBar": False})
    if ts_info:
        st.caption("工具応力 " + ts_info)
    # dragging a box ZOOMS to it and that framing becomes the saved "view"
    try:
        box = ev["selection"]["box"]
        if box:
            bx, by = box[0]["x"], box[0]["y"]
            rng = ([min(bx), max(bx)], [min(by), max(by)])
            if rng != ss.view_range:
                ss.view_range = rng
                st.rerun()
    except (KeyError, IndexError, TypeError):
        pass
    # ── GIF export reflecting the CURRENT field settings ──
    st.divider()
    with st.expander("アニメGIF出力（今の表示設定・配色・画角を反映）"):
        ga = st.columns(3)
        g_every = ga[0].slider("何step毎", 1, 10, 2, key="gif_every")
        g_fps = ga[1].slider("FPS", 2, 20, 8, key="gif_fps")
        g_fixed = ga[2].checkbox("色スケール固定", True, key="gif_fixed",
                                 help="全フレームで同じ配色範囲（工程比較向き）")
        st.caption("反映: 応力/ひずみ・工具線・速度・フローネット・ズーム画角"
                   "・荷重曲線（『荷重曲線を上に表示』ON時）"
                   "・工具応力（『工具応力を重ね』ON時。各フレームで解くため低速）。"
                   "GIFと同時に、各フレームのPNGも新規フォルダ（H5と同じ場所）へ保存します。")
        if st.button("GIFを出力", type="primary", key="gif_btn"):
            gmn, gmx = (global_range_cached(path, mtime, field) if g_fixed
                        else (vmin, vmax))

            tg_fn = None
            if show_ts:
                def tg_fn(j):
                    try:
                        g, tv, _ = tool_stress_grids_cached(path, mtime, j, ts_mesh)
                        return g, tv
                    except Exception:
                        return None
            tvmax_fixed = (ts_vmax if ts_vmax is not None
                           else (tool_grids[1] if tool_grids else None))
            # where to drop the GIF + per-frame PNGs. An UPLOADED H5 lives in a
            # temp dir (the EXE case) -> users can't find %TEMP%, so write to
            # their Downloads folder instead. A real local results.h5 (dev) ->
            # keep the output next to it.
            safe = "".join(c if c.isalnum() else "_" for c in field)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            _h5dir = os.path.dirname(os.path.abspath(path))
            _tmp = os.path.normcase(os.path.abspath(tempfile.gettempdir()))
            if os.path.normcase(_h5dir).startswith(_tmp):
                out_base = os.path.join(os.path.expanduser("~"), "Downloads",
                                        "PlasticFEM_Post")
            else:
                out_base = _h5dir
            frame_dir = os.path.join(out_base, f"gif_frames_{safe}_{stamp}")
            os.makedirs(frame_dir, exist_ok=True)
            bar = st.progress(0.0, "レンダリング中…")
            # save the GIF in the SAME folder as the frame PNGs (easy to find)
            out = os.path.join(frame_dir, f"animation_{safe}.gif")
            export_animation(
                res, field, out, vmin=gmn, vmax=gmx, cmap=MPL_CMAP[cmap],
                fps=g_fps, every=g_every, velocity=show_vel,
                show_mesh=show_mesh, show_tools=show_tools, vel_stride=vel_stride,
                view_range=ss.view_range,
                extent=(global_extent_full(path, mtime)
                        if (show_tools or show_ts) else global_extent(path, mtime)),
                flownet=flownet_cached(path, mtime) if show_flow else None,
                tool_grid_fn=tg_fn, tool_cmap=MPL_CMAP[ts_cmap],
                tool_vmin=(ts_vmin if ts_vmin is not None else 0.0),
                tool_vmax=tvmax_fixed,
                load_curve=((res.strokes, res.loads / 1e3) if show_lc else None),
                frame_dir=frame_dir, progress=lambda f: bar.progress(f))
            bar.empty()
            st.image(out)
            nfr = len([f for f in os.listdir(frame_dir)
                       if f.endswith(".png")]) if os.path.isdir(frame_dir) else 0
            # GIF (+ PNG frames) are written straight to disk -> no download
            # button needed. Show the exact folder so the user can open it.
            st.success(f"💾 保存しました → {out}\n\n"
                       f"（同じフォルダにフレーム画像 {nfr} 枚も保存）")

# ── keyboard shortcuts ───────────────────────────────────────────────────────
# components.html runs as a same-origin srcdoc iframe, so its JS can reach the
# parent document and click the sidebar buttons by their label. This adds key
# bindings WITHOUT a custom component: arrows = pan, +/- = zoom, 0 = fit-all,
# , / . = prev/next step. (Registered once; skipped while typing in a field.)
components.html("""
<script>
const doc = window.parent.document;
function click(test){
  // textContent (not innerText): the pan/zoom buttons are display:none, and
  // innerText returns '' for hidden elements, so the shortcuts could not find
  // them. textContent works regardless of visibility.
  for (const b of doc.querySelectorAll('button')) {
    if (test((b.textContent||'').trim())) { b.click(); return true; }
  }
  return false;
}
if (!window.parent.__pfemKeys) {
  window.parent.__pfemKeys = true;
  doc.addEventListener('keydown', function(e){
    const t = e.target;
    if (t && (t.tagName==='INPUT' || t.tagName==='TEXTAREA' || t.isContentEditable
        || (t.closest && t.closest('[data-testid=\\"stSlider\\"]')))) return;
    const eq = s => (x => x===s);
    const has = s => (x => x.indexOf(s)>=0);
    let hit = true;
    switch (e.key) {
      case 'ArrowLeft':  click(eq('←')); break;
      case 'ArrowRight': click(eq('→')); break;
      case 'ArrowUp':    click(eq('↑')); break;
      case 'ArrowDown':  click(eq('↓')); break;
      case 'PageUp':     click(has('拡大')); break;
      case 'PageDown':   click(has('縮小')); break;
      case '0':          click(eq('全体')); break;
      case ',':          click(eq('◀')); break;
      case '.':          click(eq('▶')); break;
      default: hit = false;
    }
    if (hit) e.preventDefault();
  });
}
</script>
""", height=0)

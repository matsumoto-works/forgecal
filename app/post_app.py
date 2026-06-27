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

_FIELD_HELP = {
    "Stress": (
        "von Mises: 相当応力（降伏・塑性流れの評価）\n"
        "σ1 (max principal): 最大主応力（正=引張。延性損傷の駆動力）\n"
        "σ2 (min principal): 最小主応力（負=圧縮）\n"
        "σxx / σyy / σzz / σxy: 応力テンソルの各成分（主応力とは別物）\n"
        "  軸対称モデル: x=r（径方向）, y=z（軸方向）, σzz=フープ応力\n"
        "mean (hydrostatic): 静水圧応力 = (σxx+σyy+σzz)/3\n"
        "max shear: 最大せん断応力 = (σ1−σ2)/2\n"
        "triaxiality: 三軸度 = mean/von Mises（正が高いほど延性破壊しやすい）"
    ),
    "Strain": (
        "PEEQ: 相当塑性ひずみ（塑性変形の累積量。加工度の目安）\n"
        "εxx / εyy / εzz / γxy: 弾性ひずみテンソルの各成分"
    ),
    "Damage": (
        "damage: Cockcroft-Latham 延性損傷指数\n"
        "  D = ∫ max(σ1, 0) / σ_eq · dεp\n"
        "  最大主応力 σ1 が引張（正）のときのみ損傷が蓄積します。\n"
        "  圧縮が支配的な鍛造では σ1≤0 の領域が多く damage≈0 は正常。\n"
        "  blanking（打ち抜き）・押し出し端部・バリ部で顕著に増大します。"
    ),
}

# ── i18n ─────────────────────────────────────────────────────────────────────
STRINGS: dict[str, dict[str, str]] = {
    "upload_label": {
        "jp": "results.h5 を選択（ForgeCal でDLしたファイル）",
        "en": "Select results.h5 (downloaded from ForgeCal)"},
    "local_select": {
        "jp": "results.h5（ローカル results/ 内）",
        "en": "results.h5 (local results/ folder)"},
    "or_path": {"jp": "…or path", "en": "…or path"},
    "no_file_msg": {
        "jp": "Pick a results.h5 (e.g. `run_case.py upsetting --flownet`).",
        "en": "Pick a results.h5 (e.g. `run_case.py upsetting --flownet`)."},
    "s1_header": {"jp": "**① ステップ移動**", "en": "**① Step Navigation**"},
    "pitch_label": {"jp": "送り幅 [mm]", "en": "Pitch [mm]"},
    "go_first": {"jp": "最初へ", "en": "First"},
    "go_prev": {"jp": "{p}mm 戻る", "en": "Back {p}mm"},
    "go_next": {"jp": "{p}mm 進む", "en": "Fwd {p}mm"},
    "go_last": {"jp": "最後へ", "en": "Last"},
    "pitch_caption": {
        "jp": "◀ ▶ と , . は {p}mm 刻みで移動します。",
        "en": "◀ ▶ and , . move by {p}mm steps."},
    "s2_header": {"jp": "**② フィールド**", "en": "**② Field**"},
    "rng_label": {
        "jp": "Colour range — 色スケールの範囲",
        "en": "Colour range"},
    "rng_help": {
        "jp": ("値→色の対応の最小/最大を決めます。\n"
               "・Auto (this step): その工程の最小〜最大に自動。\n"
               "・Auto (whole run): 全工程で同じ範囲に固定（工程間やGIFの比較に最適）。\n"
               "・Manual: vmin/vmax を入力し、特定の応力/ひずみ帯だけを強調。"),
        "en": ("Set min/max for color mapping.\n"
               "· Auto (this step): auto-scale per step.\n"
               "· Auto (whole run): fixed range across all steps (best for GIF / step comparison).\n"
               "· Manual: enter vmin/vmax to highlight a specific stress/strain band.")},
    "spinner_global_range": {
        "jp": "全工程のカラーレンジを計算中…（初回のみ）",
        "en": "Computing whole-run colour range… (first time only)"},
    "s3_header": {
        "jp": "**③ オーバーレイ（重ね表示）**",
        "en": "**③ Overlays**"},
    "show_tools": {"jp": "工具線", "en": "Tool outline"},
    "show_vel": {"jp": "速度ベクトル", "en": "Velocity vectors"},
    "vel_stride": {"jp": "ベクトル間引き", "en": "Vector stride"},
    "show_flow": {"jp": "フローネット", "en": "Flow net"},
    "show_flow_help": {
        "jp": "繊維流（tracer格子）を場に重ねます",
        "en": "Overlay material flow tracer grid"},
    "show_mesh": {"jp": "メッシュ表示", "en": "Show mesh"},
    "show_mesh_help": {
        "jp": "FEMメッシュ（要素の辺）を場に重ねます",
        "en": "Overlay FEM mesh edges"},
    "show_ts": {"jp": "工具応力", "en": "Tool stress"},
    "show_ts_help": {
        "jp": "工具のvon Misesを場に重ねて1つの図に統合",
        "en": "Overlay tool von Mises in same figure"},
    "ts_mesh_label": {"jp": "工具メッシュ寸法", "en": "Tool mesh size"},
    "ts_cmap_label": {"jp": "工具応力 Colormap", "en": "Tool stress colormap"},
    "ts_rng_label": {
        "jp": "工具応力 Colour range",
        "en": "Tool stress colour range"},
    "ts_vmin_label": {"jp": "工具 vmin", "en": "Tool vmin"},
    "ts_vmax_label": {"jp": "工具 vmax", "en": "Tool vmax"},
    "spinner_ts_global": {
        "jp": "工具応力の全工程レンジを計算中…",
        "en": "Computing whole-run tool stress range…"},
    "s4_header": {"jp": "**④ レイアウト**", "en": "**④ Layout**"},
    "show_lc": {"jp": "荷重曲線を上に表示", "en": "Show load curve above"},
    "show_lc_help": {
        "jp": "場の上に荷重曲線を表示（既定ON）",
        "en": "Show load curve above the field (default ON)"},
    "chart_h_label": {"jp": "図の高さ [px]", "en": "Figure height [px]"},
    "chart_h_help": {
        "jp": ("場の図の縦サイズ。画面に収まらずスクロールが出る時は"
               "小さく、大きく見たい時は大きくしてください。"),
        "en": ("Height of the field figure. "
               "Reduce if scrolling appears, increase for a larger view.")},
    "s5_header": {"jp": "**⑤ 画角を記憶**", "en": "**⑤ Save Viewports**"},
    "s5_move_cap": {"jp": "移動 ↓（番号ボタン）", "en": "Jump to ↓ (number buttons)"},
    "s5_save_cap": {"jp": "保存 ↓（枠を選んで Save）", "en": "Save ↓ (pick slot then Save)"},
    "view_btn_help": {
        "jp": "この保存ビューへ移動",
        "en": "Jump to this saved view"},
    "view_btn_empty": {"jp": "空き", "en": "Empty"},
    "slot_save_help": {
        "jp": "現在ズームしている画角をこの枠に保存",
        "en": "Save current zoom to this slot"},
    "toast_view_saved": {
        "jp": "現在の画角を View {slot} に保存しました",
        "en": "Saved current view to slot {slot}"},
    "toast_no_zoom": {
        "jp": "先に図をドラッグでズームしてください（全体表示は保存対象外）",
        "en": "Please drag to zoom the figure first (full view cannot be saved)"},
    "s6_header": {
        "jp": "**⑥ 表示の操作（パン/ズーム）**",
        "en": "**⑥ Pan / Zoom**"},
    "s6_caption": {
        "jp": ("ズーム=図をドラッグで囲む or 下の＋−。ボタンでパン/拡縮/全体。"
               "キー（環境により可）: 矢印=パン, PgUp/PgDn=拡縮, 0=全体, , .=step。"),
        "en": ("Zoom=drag to select region or use ＋− buttons. "
               "Keys (may depend on browser): arrows=pan, PgUp/PgDn=zoom, 0=fit all, , .=step.")},
    "btn_zoom_in":  {"jp": "＋ 拡大", "en": "Zoom In"},
    "btn_zoom_out": {"jp": "－ 縮小", "en": "Zoom Out"},
    "btn_fit_all":  {"jp": "全体",   "en": "Fit All"},
    "quit_btn":  {"jp": "⏻ アプリを終了", "en": "⏻ Quit App"},
    "quit_help": {
        "jp": "サーバーを停止してアプリを終了します",
        "en": "Stop the server and exit the app"},
    "quit_msg": {
        "jp": ("✅ アプリを終了しました。<br>"
               "<span style=\"font-size:15px;color:#9aa0a6;\">このタブを閉じてください。</span>"),
        "en": ("✅ App has exited.<br>"
               "<span style=\"font-size:15px;color:#9aa0a6;\">You can close this tab.</span>")},
    "spinner_field": {"jp": "場を描画中…", "en": "Rendering field…"},
    "spinner_ts_compute": {"jp": "工具応力を計算中…", "en": "Computing tool stress…"},
    "ts_info_cap": {"jp": "工具応力 ", "en": "Tool stress "},
    "ts_warn": {"jp": "工具応力を計算できません: ", "en": "Cannot compute tool stress: "},
    "uniform_damage": {
        "jp": ("damage = {v:.3g}（均一）— "
               "圧縮が支配的な工程では σ1≤0 のため damage≈0 は正常。"
               "blanking や押し出し端部では増大します。"),
        "en": ("damage = {v:.3g} (uniform) — normal in compression-dominated forming (σ1≤0). "
               "Increases at blanking shear bands and extrusion tips.")},
    "gif_expander": {
        "jp": "アニメGIF出力（今の表示設定・配色・画角を反映）",
        "en": "Export Animated GIF (uses current settings)"},
    "gif_every": {"jp": "何step毎", "en": "Every N steps"},
    "gif_fixed": {"jp": "色スケール固定", "en": "Fixed colour scale"},
    "gif_fixed_help": {
        "jp": "全フレームで同じ配色範囲（工程比較向き）",
        "en": "Same colour range for all frames (good for step comparison)"},
    "gif_caption": {
        "jp": ("反映: 応力/ひずみ・工具線・速度・フローネット・ズーム画角"
               "・荷重曲線（『荷重曲線を上に表示』ON時）"
               "・工具応力（『工具応力を重ね』ON時。各フレームで解くため低速）。"
               "GIFと同時に、各フレームのPNGも新規フォルダ（H5と同じ場所）へ保存します。"),
        "en": ("Reflects: stress/strain, tool outline, velocity, flow net, zoom, "
               "load curve (when enabled), tool stress (when enabled; slow — solved per frame). "
               "Frame PNGs are also saved to a new folder next to the H5.")},
    "gif_btn": {"jp": "GIFを出力", "en": "Export GIF"},
    "gif_render": {"jp": "レンダリング中…", "en": "Rendering…"},
    "gif_saved": {
        "jp": "💾 保存しました → {path}\n\n（同じフォルダにフレーム画像 {n} 枚も保存）",
        "en": "💾 Saved → {path}\n\n({n} frame images also saved to the same folder)"},
    "3d_expander": {
        "jp": "3D最終形状（ホールケーキ表示）",
        "en": "3D Final Shape (cake-slice view)"},
    "3d_sweep": {"jp": "表示角度（°）", "en": "Sweep angle (°)"},
    "3d_ntheta": {"jp": "角分割数", "en": "Angular divisions"},
    "3d_step": {"jp": "ステップ", "en": "Step"},
    "3d_step_help": {
        "jp": "0=初期形状、最大値=最終形状",
        "en": "0=initial shape, max=final shape"},
    "3d_fail": {
        "jp": "境界ポリゴンの抽出に失敗しました。",
        "en": "Failed to extract boundary polygon."},
}


def T(key: str, **kw) -> str:
    lang = st.session_state.get("lang", "jp")
    s = STRINGS.get(key, {}).get(lang) or STRINGS.get(key, {}).get("jp", key)
    return s.format(**kw) if kw else s
# ─────────────────────────────────────────────────────────────────────────────


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
    _unit = field_unit(field) or ""
    _htpl = (f"x=%{{x:.2f}} mm  y=%{{y:.2f}} mm<br>{field}=%{{z:.4g}}"
             + (f" {_unit}" if _unit else "") + "<extra></extra>")
    fig = go.Figure(go.Heatmap(
        x=xi, y=yi, z=Z, colorscale=cmap, zmin=vmin, zmax=vmax,
        zsmooth=False if fast else "best", connectgaps=False,
        colorbar=dict(title=_unit, x=1.0),
        hovertemplate=_htpl))
    # tool stresses overlaid as heatmaps (own colourscale + colour bar) so the
    # material field and the tool stresses appear in ONE interactive figure
    if tool_grids is not None:
        grids, tvmax = tool_grids
        zmn = tool_vmin if tool_vmin is not None else 0.0
        zmx = tool_vmax if tool_vmax is not None else tvmax
        # Mask out low-stress cells (bottom 6% of range) so only the contact
        # zones are visible; prevents bulk die/punch body from darkening the view
        _clip = zmn + 0.06 * max(zmx - zmn, 1.0)
        for gi, (txi, tyi, tZ) in enumerate(grids):
            tZ_disp = np.where(tZ >= _clip, tZ, np.nan)
            fig.add_heatmap(x=txi, y=tyi, z=tZ_disp, colorscale=tool_cmap,
                            zmin=zmn, zmax=zmx, zsmooth="best",
                            connectgaps=False, showscale=(gi == 0),
                            hovertemplate="tool σ=%{z:.0f} MPa<extra></extra>",
                            colorbar=dict(title="tool σ", x=1.1,
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
                      margin=dict(l=0, r=120 if tool_grids else 0, t=8, b=0),
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
    grids, infos, errs, tvmax = [], [], [], 0.0
    for tn in res.tool_names:
        try:
            r = _solve_tool_result(res, step, tn, tool_mesh)
        except Exception as _e:
            errs.append(f"{tn}: {type(_e).__name__}: {_e}")
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
        raise RuntimeError("no tools in contact at this step — " + "; ".join(errs))
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


@st.cache_data(max_entries=4, show_spinner=False)
def build_3d_revolution(path, mtime, step_idx, sweep_deg=300.0, n_theta=60):
    """Axisymmetric 2D boundary → 3D revolution mesh (Plotly Mesh3d kwargs).

    Returns dict(x,y,z,i,j,k) or None on failure.
    sweep_deg: arc angle (300 = cake with 60° slice removed, opening at θ=0°).
    """
    from collections import Counter
    res = H5Result(path)
    s = res.step(step_idx)
    coords = s["coords"]   # (Nn,2): col0=r, col1=z
    elems  = s["elems"]    # (Ne,3) triangles

    # ── boundary edges (appear exactly once) ─────────────────────────────────
    ec = Counter()
    for el in elems:
        for k in range(3):
            a, b = int(el[k]), int(el[(k + 1) % 3])
            ec[(min(a, b), max(a, b))] += 1
    bedges = [e for e, c in ec.items() if c == 1]
    if not bedges:
        return None

    adj: dict = {}
    for a, b in bedges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # ── trace closed loops ────────────────────────────────────────────────────
    def _trace(start):
        chain, seen = [start], {start}
        prev, cur = -1, start
        for _ in range(len(adj) * 2):
            cands = [n for n in adj.get(cur, []) if n != prev and n not in seen]
            if not cands:
                break
            nxt = cands[0]
            chain.append(nxt); seen.add(nxt)
            prev, cur = cur, nxt
        return chain

    visited: set = set()
    polys = []
    for s0 in sorted(adj):
        if s0 in visited:
            continue
        poly = _trace(s0)
        visited.update(poly)
        if len(poly) >= 3:
            polys.append(poly)
    if not polys:
        return None

    ring = coords[[int(i) for i in max(polys, key=len)]]   # (P,2)
    P = len(ring)
    r_p, z_p = ring[:, 0], ring[:, 1]

    # ── revolution angles: sweep centred so the opening faces +x ─────────────
    gap = 360.0 - sweep_deg
    theta = np.linspace(np.radians(gap / 2.0),
                        np.radians(gap / 2.0 + sweep_deg), n_theta + 1)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    NT1 = n_theta + 1

    xv = (r_p[:, None] * cos_t).ravel().tolist()
    yv = (r_p[:, None] * sin_t).ravel().tolist()
    zv = np.repeat(z_p, NT1).tolist()

    ti_l: list = []; tj_l: list = []; tk_l: list = []

    # ── side-surface quads → triangles ───────────────────────────────────────
    for p in range(P):
        p1 = (p + 1) % P
        ax0 = abs(r_p[p])  < 1e-9
        ax1 = abs(r_p[p1]) < 1e-9
        for t in range(n_theta):
            v00 = p  * NT1 + t;   v01 = p  * NT1 + t + 1
            v10 = p1 * NT1 + t;   v11 = p1 * NT1 + t + 1
            if ax0 and ax1:
                continue
            elif ax0:
                ti_l.append(v00); tj_l.append(v10); tk_l.append(v11)
            elif ax1:
                ti_l.append(v00); tj_l.append(v01); tk_l.append(v10)
            else:
                ti_l += [v00, v00]; tj_l += [v01, v11]; tk_l += [v11, v10]

    # ── flat cap faces at both cut angles (fan from centroid) ─────────────────
    for a_idx in [0, n_theta]:
        rvi = [p * NT1 + a_idx for p in range(P)]
        cx = float(np.mean([xv[v] for v in rvi]))
        cy = float(np.mean([yv[v] for v in rvi]))
        cz = float(np.mean([zv[v] for v in rvi]))
        c_v = len(xv)
        xv.append(cx); yv.append(cy); zv.append(cz)
        for p in range(P):
            va, vb = rvi[p], rvi[(p + 1) % P]
            if a_idx == 0:
                ti_l.append(va); tj_l.append(c_v); tk_l.append(vb)
            else:
                ti_l.append(va); tj_l.append(vb);  tk_l.append(c_v)

    return dict(x=xv, y=yv, z=zv, i=ti_l, j=tj_l, k=tk_l)


# ════════════════════════ SIDEBAR — all controls ════════════════════════════
sb = st.sidebar

# ── language toggle ──────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state["lang"] = "jp"
_lc1, _lc2 = sb.columns(2)
if _lc1.button("日本語", use_container_width=True,
               type="primary" if st.session_state["lang"] == "jp" else "secondary"):
    st.session_state["lang"] = "jp"
    st.rerun()
if _lc2.button("English", use_container_width=True,
               type="primary" if st.session_state["lang"] == "en" else "secondary"):
    st.session_state["lang"] = "en"
    st.rerun()

sb.title("PlasticFEM Post")

# primary input: drop in a results.h5 downloaded from ForgeCal (works in the
# standalone EXE, where there is no local results/ folder)
up = sb.file_uploader(T("upload_label"), type=["h5"])
path = None
if up is not None:
    _td = os.path.join(tempfile.gettempdir(), "plasticfem_post")
    os.makedirs(_td, exist_ok=True)
    path = os.path.join(_td, up.name)
    _buf = up.getbuffer()
    if not os.path.exists(path) or os.path.getsize(path) != len(_buf):
        with open(path, "wb") as _f:        # write once -> stable mtime -> cache
            _f.write(_buf)
    del _buf  # release memoryview so Streamlit can GC the UploadedFile buffer
else:
    files = sorted(glob.glob(os.path.join(HERE, "results", "**", "results.h5"),
                             recursive=True), key=os.path.getmtime, reverse=True)
    choice = sb.selectbox(T("local_select"), files,
                          format_func=lambda p: os.path.relpath(p, HERE)) \
        if files else None
    path = sb.text_input(T("or_path"), value=choice or "") or choice
if not path or not os.path.exists(path):
    st.info(T("no_file_msg"))
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
sb.markdown(T("s1_header"))
# Step(index) sits directly under the header; the 送り幅 + ◀▶ controls below
# drive it through its session key (ss.step_idx). The buttons use on_click
# callbacks (which run BEFORE the rerun, so they may set the slider's key).
ss.step = sb.slider("Step (index)", 0, res.n - 1, key="step_idx")
pitch = sb.select_slider(T("pitch_label"), [0.05, 0.1, 0.2, 0.5, 1.0], value=0.1)


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
b[0].button("⏮", help=T("go_first"), use_container_width=True,
            on_click=_goto, args=(0,))
b[1].button("◀", help=T("go_prev", p=pitch), use_container_width=True,
            on_click=_step_by, args=(-pitch,))
b[2].button("▶", help=T("go_next", p=pitch), use_container_width=True,
            on_click=_step_by, args=(pitch,))
b[3].button("⏭", help=T("go_last"), use_container_width=True,
            on_click=_goto, args=(res.n - 1,))
sb.caption(T("pitch_caption", p=pitch))

sb.divider()
sb.markdown(T("s2_header"))
group = sb.radio("Field group", ["Stress", "Strain", "Damage"], horizontal=True)
opts = {"Stress": STRESS_FIELDS, "Strain": STRAIN_FIELDS,
        "Damage": OTHER_FIELDS}[group]
field = sb.selectbox("Field", opts, help=_FIELD_HELP[group])
cmap = sb.selectbox("Colormap", list(MPL_CMAP))
_rng_opts = ["Auto (this step)", "Auto (whole run)", "Manual"]
rng_mode = sb.radio(T("rng_label"), _rng_opts, index=1, help=T("rng_help"))
vmin = vmax = None
if rng_mode == "Auto (whole run)":
    # spinner shows only while actually computing (cached -> instant, no flash)
    with st.spinner(T("spinner_global_range")):
        vmin, vmax = global_range_cached(path, mtime, field)
elif rng_mode == "Manual":
    ef0 = element_field(res.step(ss.step), field)
    lo, hi = float(ef0.min()), float(ef0.max())
    pad = 0.5 * (hi - lo + 1e-9)
    if ss.get("manual_field") != field:
        ss["manual_vmin_inp"] = round(lo - pad, 3)
        ss["manual_vmax_inp"] = round(hi + pad, 3)
        ss["manual_field"] = field
    cA, cB = sb.columns(2)
    vmin = cA.number_input("vmin", key="manual_vmin_inp")
    vmax = cB.number_input("vmax", key="manual_vmax_inp")
sb.divider()
sb.markdown(T("s3_header"))
c1, c2 = sb.columns(2)
show_tools = c1.checkbox(T("show_tools"), True)
show_vel = c2.checkbox(T("show_vel"), res.has_velocity,
                       disabled=not res.has_velocity)
vel_stride = sb.slider(T("vel_stride"), 1, 8, 3) if show_vel else 3
c3, c4 = sb.columns(2)
show_flow = c3.checkbox(T("show_flow"), res.has_flownet,
                        disabled=not res.has_flownet,
                        help=T("show_flow_help"))
show_mesh = c4.checkbox(T("show_mesh"), False, help=T("show_mesh_help"))
# 工具応力 on its own row so it reads together with the 工具メッシュ寸法 slider
show_ts = sb.checkbox(T("show_ts"), True, help=T("show_ts_help"))
ts_mesh, ts_cmap, ts_vmin, ts_vmax = 0.9, "Hot", None, None
if show_ts:
    ts_mesh = sb.slider(T("ts_mesh_label"), 0.4, 2.0, 0.9)
    ts_cmap = sb.selectbox(T("ts_cmap_label"), list(MPL_CMAP),
                           index=list(MPL_CMAP).index("Hot"))
    ts_rng = sb.radio(T("ts_rng_label"), _rng_opts, index=1)
    if ts_rng == "Manual":
        tcA, tcB = sb.columns(2)
        ts_vmin = tcA.number_input(T("ts_vmin_label"), value=0.0)
        ts_vmax = tcB.number_input(T("ts_vmax_label"), value=6000.0)
    elif ts_rng == "Auto (whole run)":
        with st.spinner(T("spinner_ts_global")):
            ts_vmin, ts_vmax = 0.0, tool_global_vmax(path, mtime, ts_mesh)
sb.divider()
sb.markdown(T("s4_header"))
show_lc = sb.checkbox(T("show_lc"), True, help=T("show_lc_help"))
chart_h = sb.slider(T("chart_h_label"), 320, 680, 470, 10, help=T("chart_h_help"))

sb.divider()
sb.markdown(T("s5_header"))
sb.caption(T("s5_move_cap"))
v = sb.columns(5)
for k in range(5):
    if v[k].button(f"{k+1}", disabled=ss.viewports[k] is None,
                   help=T("view_btn_help") if ss.viewports[k]
                   else T("view_btn_empty"), use_container_width=True):
        ss.view_range = ss.viewports[k]
sb.caption(T("s5_save_cap"))
v2 = sb.columns([2, 2])
slot = v2[0].selectbox("slot", ["1", "2", "3", "4", "5"],
                       label_visibility="collapsed")
if v2[1].button("Save", help=T("slot_save_help"), use_container_width=True):
    if ss.view_range is not None:
        ss.viewports[int(slot) - 1] = ss.view_range
        st.toast(T("toast_view_saved", slot=slot))
        st.rerun()                       # refresh so the View button enables now
    else:
        st.toast(T("toast_no_zoom"))


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


sb.markdown(T("s6_header"))
sb.caption(T("s6_caption"))
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
    if z[0].button(T("btn_zoom_out")):
        _shift_view(zoom=1.3)
    if z[1].button(T("btn_zoom_in")):
        _shift_view(zoom=1 / 1.3)
    if z[2].button(T("btn_fit_all")):
        ss.view_range = None

# ── quit (bottom of the sidebar) — EXE only ──────────────────────────────────
if getattr(sys, "frozen", False):
    sb.divider()
    if sb.button(T("quit_btn"), use_container_width=True, help=T("quit_help")):
        import threading
        # Cover the whole page with a clean overlay BEFORE the server dies, so
        # the user never sees Streamlit's scary "Connection error" screen.
        _quit_msg = T("quit_msg")
        components.html(f"""
        <script>
        const d = window.parent.document;
        const o = d.createElement('div');
        o.style.cssText = 'position:fixed;inset:0;z-index:2147483647;'+
          'background:#0e1117;color:#e6e6e6;display:flex;flex-direction:column;'+
          'align-items:center;justify-content:center;font-family:sans-serif;'+
          'font-size:20px;line-height:1.8;text-align:center;';
        o.innerHTML = '{_quit_msg}';
        d.body.appendChild(o);
        window.open('','_self'); window.close();
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
        _tl = tool_load_history(path, mtime)
        _COLORS = ["royalblue", "green", "darkorange", "purple", "brown"]
        # check if any non-punch tool has meaningful load data
        _extra = {nm: v for nm, v in _tl.items()
                  if 'punch' not in nm.lower()
                  and not np.all(np.isnan(v))
                  and np.nanmax(np.abs(v)) > 1.0}
        _show_legend = bool(_extra)
        lc = go.Figure(go.Scatter(x=res.strokes, y=res.loads / 1e3,
                                  line=dict(color="royalblue"),
                                  name="punch", showlegend=_show_legend))
        for ci, (nm, loads) in enumerate(_extra.items()):
            lc.add_scatter(x=res.strokes, y=np.abs(np.where(np.isnan(loads), 0, loads)) / 1e3,
                           line=dict(color=_COLORS[(ci + 1) % len(_COLORS)]),
                           name=nm, showlegend=True)
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
                         xaxis_title=None, yaxis_title="kN",
                         showlegend=_show_legend,
                         legend=dict(orientation="h", x=1, y=1,
                                     xanchor="right", yanchor="bottom",
                                     font=dict(size=10)) if _show_legend else None)
        st.plotly_chart(lc, use_container_width=True, key="lc_field")
    # ── info line ABOVE the field (no zoom hint) ──
    st.caption(f"step {ss.step}/{res.n-1} · stroke {s['stroke']:.3f} mm · "
               f"punch {s['punch_load']/1e3:.0f} kN · {field}: "
               f"{ef.min():.3g} … {ef.max():.3g} {field_unit(field)}")
    if ef.max() - ef.min() < 1e-9:
        if field == "damage":
            st.info(T("uniform_damage", v=ef.max()))
        else:
            st.info(f"'{field}' is uniform ({ef.max():.3g}) at this step.")
    # ── tool stress: solve and overlay INTO the field figure ──
    tool_grids = None
    ts_info = None
    if show_ts and res.tool_names:
        try:
            with st.spinner(T("spinner_ts_compute")):
                grids, tvmax, ts_info = tool_stress_grids_cached(
                    path, mtime, ss.step, ts_mesh)
            tool_grids = (grids, tvmax)
        except Exception as e:
            st.warning(T("ts_warn") + str(e))
    # ── the field (with optional tool-stress overlay) ──
    with st.spinner(T("spinner_field")):
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
        st.caption(T("ts_info_cap") + ts_info)
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
    with st.expander(T("gif_expander")):
        ga = st.columns(3)
        g_every = ga[0].slider(T("gif_every"), 1, 10, 2, key="gif_every")
        g_fps = ga[1].slider("FPS", 2, 20, 8, key="gif_fps")
        g_fixed = ga[2].checkbox(T("gif_fixed"), True, key="gif_fixed",
                                 help=T("gif_fixed_help"))
        st.caption(T("gif_caption"))
        if st.button(T("gif_btn"), type="primary", key="gif_btn"):
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
            bar = st.progress(0.0, T("gif_render"))
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
            st.success(T("gif_saved", path=out, n=nfr))

# ── 3D最終形状（軸対称のみ） ──────────────────────────────────────────────────
if res.mode == "axisymmetric":
    st.divider()
    with st.expander(T("3d_expander")):
        _c3a, _c3b, _c3c = st.columns([2, 2, 2])
        _sweep  = _c3a.slider(T("3d_sweep"), 90, 360, 300, 30, key="3d_sweep")
        _ntheta = _c3b.selectbox(T("3d_ntheta"), [36, 60, 90, 120], 1, key="3d_ntheta")
        _step3d = _c3c.number_input(
            T("3d_step"), 0, res.n - 1, res.n - 1, key="3d_step",
            help=T("3d_step_help"))

        _m3d = build_3d_revolution(path, mtime, int(_step3d), _sweep, _ntheta)
        if _m3d:
            import plotly.graph_objects as _pgo
            _fig3d = _pgo.Figure(_pgo.Mesh3d(
                x=_m3d["x"], y=_m3d["y"], z=_m3d["z"],
                i=_m3d["i"], j=_m3d["j"], k=_m3d["k"],
                color="#4a8fc4",
                opacity=1.0,
                flatshading=False,
                lighting=dict(ambient=0.55, diffuse=0.85, roughness=0.5,
                              specular=0.4, fresnel=0.2),
                lightposition=dict(x=3, y=2, z=6),
                hoverinfo="skip",
            ))
            _fig3d.update_layout(
                height=520,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    aspectmode="data",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(title="z (mm)", showgrid=True,
                               gridcolor="#cccccc", zeroline=False),
                    bgcolor="rgba(240,244,248,1)",
                    camera=dict(
                        eye=dict(x=1.6, y=1.1, z=0.7),
                        up=dict(x=0, y=0, z=1)),
                ))
            st.plotly_chart(_fig3d, use_container_width=True, key="3d_fig")
        else:
            st.info(T("3d_fail"))

# ── keyboard shortcuts ───────────────────────────────────────────────────────
# components.html runs as a same-origin srcdoc iframe, so its JS can reach the
# parent document and click the sidebar buttons by their label. This adds key
# bindings WITHOUT a custom component: arrows = pan, +/- = zoom, 0 = fit-all,
# , / . = prev/next step. (Registered once; skipped while typing in a field.)
_zoom_in_label  = T("btn_zoom_in")
_zoom_out_label = T("btn_zoom_out")
_fit_all_label  = T("btn_fit_all")
components.html(f"""
<script>
const doc = window.parent.document;
function click(test){{
  // textContent (not innerText): the pan/zoom buttons are display:none, and
  // innerText returns '' for hidden elements, so the shortcuts could not find
  // them. textContent works regardless of visibility.
  for (const b of doc.querySelectorAll('button')) {{
    if (test((b.textContent||'').trim())) {{ b.click(); return true; }}
  }}
  return false;
}}
if (!window.parent.__pfemKeys) {{
  window.parent.__pfemKeys = true;
  doc.addEventListener('keydown', function(e){{
    const t = e.target;
    if (t && (t.tagName==='INPUT' || t.tagName==='TEXTAREA' || t.isContentEditable
        || (t.closest && t.closest('[data-testid=\\"stSlider\\"]')))) return;
    const eq = s => (x => x===s);
    let hit = true;
    switch (e.key) {{
      case 'ArrowLeft':  click(eq('←')); break;
      case 'ArrowRight': click(eq('→')); break;
      case 'ArrowUp':    click(eq('↑')); break;
      case 'ArrowDown':  click(eq('↓')); break;
      case 'PageUp':     click(eq('{_zoom_in_label}')); break;
      case 'PageDown':   click(eq('{_zoom_out_label}')); break;
      case '0':          click(eq('{_fit_all_label}')); break;
      case ',':          click(eq('◀')); break;
      case '.':          click(eq('▶')); break;
      default: hit = false;
    }}
    if (hit) e.preventDefault();
  }});
}}
</script>
""", height=0)

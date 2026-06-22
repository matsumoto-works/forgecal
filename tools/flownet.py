"""Flow-net (fiber-flow) POST-PROCESSING viewer.

Renders the deformed tracer grid from a finished analysis (results.h5) at a
user-chosen step. The tracers were advected with the material during the
solve (recorded only if the analysis was run with flow-net enabled), so the
grid survives remeshing; here the user simply selects WHICH step to view and
how to style it — no re-analysis.

usage:
    python tools/flownet.py <results.h5> [--step N] [--all] [--every K]
                            [--out DIR]
    --step N   render step N (default: last)
    --all      render every --every-th step (animation frames)
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def _load(path):
    with h5py.File(path, "r") as h:
        if "flownet" not in h:
            raise SystemExit(
                "this results file has no flow-net data — re-run the analysis "
                "with flow-net enabled (run_case.py ... --flownet)")
        fg = h["flownet"]
        ny, nx = int(fg.attrs["grid_ny"]), int(fg.attrs["grid_nx"])
        inside = fg["inside"][:]
        tracer_hist = fg["history"][:]              # (Nsteps, Npts, 2)
        steps = sorted(k for k in h.keys() if k.startswith("step_"))
        meshes = []
        for k in steps:
            g = h[k]
            tools = []
            if "tools" in g:
                tools = [g["tools"][t][:] for t in sorted(g["tools"].keys())]
            meshes.append((g["coords"][:], g["elems"][:],
                           float(g.attrs["stroke_mm"]), tools))
    return ny, nx, inside, tracer_hist, meshes


def render(path, step, outdir):
    ny, nx, inside, hist, meshes = _load(path)
    n = min(len(hist), len(meshes))
    step = (step if step is not None else n - 1) % n
    pts = hist[step]
    coords, elems, stroke, tools = meshes[step]
    P = pts.reshape(ny, nx, 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    tr = mtri.Triangulation(coords[:, 0], coords[:, 1], elems)
    ax.triplot(tr, lw=0.1, color="0.85")
    for j in range(ny):
        for i in range(nx - 1):
            if inside[j, i] and inside[j, i + 1]:
                ax.plot(P[j, i:i+2, 0], P[j, i:i+2, 1], "b-", lw=0.8)
    for i in range(nx):
        for j in range(ny - 1):
            if inside[j, i] and inside[j + 1, i]:
                ax.plot(P[j:j+2, i, 0], P[j:j+2, i, 1], "b-", lw=0.8)
    for segs in tools:
        for a, b in segs:
            ax.plot([a[0], b[0]], [a[1], b[1]], "g-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("r / x [mm]")
    ax.set_ylabel("z / y [mm]")
    ax.set_title(f"flow net  step {step}  stroke={stroke:.2f} mm")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"flownet_step{step:05d}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    outdir = args.out or os.path.join(os.path.dirname(args.h5), "flownet")
    if args.all:
        ny, nx, inside, hist, meshes = _load(args.h5)
        n = min(len(hist), len(meshes))
        for s in list(range(0, n, args.every)) + [n - 1]:
            print(render(args.h5, s, outdir))
    else:
        print(render(args.h5, args.step, outdir))


if __name__ == "__main__":
    main()

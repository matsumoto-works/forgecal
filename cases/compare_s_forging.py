"""Compare PlasticFEM s_forging load curves vs DEFORM (tons-SI -> kN).

DEFORM "Top Die" is the total top-ram load = punch coining load + stripper
hold-down force. PlasticFEM tracks the punch contact load separately and the
stripper is force-controlled at its setpoint, so the comparable ram load is
punch_load + stripper_setpoint.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEF = os.path.join(ROOT, "model_cases", "s_forging", "DEFORM_result")
TON = 9.80665  # tons-SI -> kN


def load_deform(fn):
    rows = []
    with open(fn) as f:
        for ln in f:
            p = ln.split()
            if len(p) == 3:
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    pass
    a = np.array(rows)
    return a[:, 0], a[:, 1] * TON, a[:, 2] * TON  # stroke, topdie kN, stripper kN


def load_pf(case):
    fn = os.path.join(ROOT, "results", "cases", case, "load_stroke.csv")
    a = np.loadtxt(fn, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1]


fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, kn in zip(axes, (20, 200, 400)):
    ds, dtop, dstr = load_deform(os.path.join(DEF, f"graph_{kn}kN.txt"))
    ax.plot(ds, dtop, "k--", lw=1.6, label="DEFORM top die (ram)")
    ax.plot(ds, dstr, "0.6", ls=":", lw=1.2, label="DEFORM stripper")
    case = f"s_forging_{kn}"
    try:
        ps, pl = load_pf(case)
        ram = pl + kn          # punch + stripper setpoint
        ax.plot(ps, pl, "b-", lw=1.0, alpha=0.5, label="PFEM punch only")
        ax.plot(ps, ram, "r-", lw=1.4, label="PFEM punch+stripper")
        d_end = dtop[-1]
        r_end = np.interp(ds[-1], ps, ram)
        p_end = np.interp(ds[-1], ps, pl)
        print(f"[{kn:>3}kN] @6mm  DEFORM={d_end:7.1f}  "
              f"PFEM(punch+strip)={r_end:7.1f} ({100*(r_end-d_end)/d_end:+5.1f}%)  "
              f"| punch only={p_end:7.1f}")
    except OSError:
        print(f"[{kn}kN] PlasticFEM result not found ({case})")
    ax.set_title(f"stripper {kn} kN")
    ax.set_xlabel("Punch stroke [mm]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Load [kN]")
fig.suptitle("s_forging stage-1: PlasticFEM vs DEFORM (top-die ram load)")
fig.tight_layout()
out = os.path.join(ROOT, "results", "cases", "s_forging_compare.png")
fig.savefig(out, dpi=110)
print("saved", out)

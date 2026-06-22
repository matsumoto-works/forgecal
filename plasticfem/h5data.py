"""Read a PlasticFEM results.h5 and derive post-processing fields.

A thin, dependency-light loader used by the Streamlit post-processor (and
reusable by the future web/VPS service). Steps are read lazily so animations
over a long run do not have to hold every mesh in memory at once.

Per step the solver stores: coords (Nn,2), elems (Ne,3), sigma (Ne,4)
= [sxx,syy,szz,sxy], peeq (Ne,), mises (Ne,), damage (Ne,), and optionally
strain (Ne,4 elastic), velocity (Nn,2 nodal increment), tool segments and
tool reaction loads. A flownet group (tracer history) is present when the run
was seeded with --flownet.
"""

from __future__ import annotations

import numpy as np

# element scalar fields the viewer can show, grouped for the UI. The names are
# explicit so a COMPONENT (σxx = the x/normal component) is not mistaken for a
# PRINCIPAL stress (σ1 = the largest principal). Axisymmetric: x = r, y = z.
STRESS_FIELDS = ["von Mises", "σ1 (max principal)", "σ2 (min principal)",
                 "σxx (x / r-normal)", "σyy (y / z-normal)", "σzz (hoop)",
                 "σxy (shear)", "mean (hydrostatic)", "max shear",
                 "triaxiality"]
STRAIN_FIELDS = ["PEEQ", "εxx (elastic)", "εyy (elastic)", "εzz (elastic)",
                 "γxy (elastic)"]
OTHER_FIELDS = ["damage"]


class H5Result:
    def __init__(self, path: str):
        import h5py
        self.path = path
        self._h5py = h5py
        with h5py.File(path, "r") as h:
            self._steps = sorted(k for k in h.keys() if k.startswith("step_"))
            self.n = len(self._steps)
            self.kn = float(h.attrs.get("kn", 0.0))
            self.mesh_size = float(h.attrs.get("mesh_size", 0.0))
            self.strokes = np.array([h[k].attrs.get("stroke_mm", i)
                                     for i, k in enumerate(self._steps)])
            self.loads = np.array([h[k].attrs.get("punch_load_N", 0.0)
                                   for k in self._steps])
            self.stages = np.array([int(h[k].attrs.get("stage", 0))
                                    for k in self._steps])
            g0 = h[self._steps[0]]
            self.has_strain = any("strain" in h[k] for k in self._steps[:3])
            self.has_velocity = any("velocity" in h[k] for k in self._steps)
            self.has_flownet = "flownet" in h
            self.tool_names = self._infer_tool_names(h)

    def _infer_tool_names(self, h):
        # ORDERED names stored alongside the segment datasets (the tool_loads
        # attrs are unordered, so must NOT be used to map name->geometry)
        g = h[self._steps[0]]
        if "tools" in g and "names" in g["tools"].attrs:
            return [str(n) for n in g["tools"].attrs["names"]]
        if "tools" in g:
            return [str(i) for i in range(len(g["tools"].keys()))]
        return []

    # ── per-step access (lazy) ──────────────────────────────────────────────
    def step(self, i: int) -> dict:
        with self._h5py.File(self.path, "r") as h:
            g = h[self._steps[i]]
            d = dict(stroke=float(g.attrs.get("stroke_mm", i)),
                     punch_load=float(g.attrs.get("punch_load_N", 0.0)),
                     stage=int(g.attrs.get("stage", 0)),
                     coords=g["coords"][:], elems=g["elems"][:],
                     sigma=g["sigma"][:], peeq=g["peeq"][:], mises=g["mises"][:])
            d["damage"] = g["damage"][:] if "damage" in g else None
            d["strain"] = g["strain"][:] if "strain" in g else None
            d["velocity"] = g["velocity"][:] if "velocity" in g else None
            d["tool_segs"] = ([g["tools"][str(t)][:]
                               for t in range(len(g["tools"].keys()))]
                              if "tools" in g else [])
            d["tool_loads"] = (dict(g["tool_loads"].attrs)
                               if "tool_loads" in g else {})
            return d

    def flownet(self):
        """Return (tracer_history (S,P,2), inside (ny,nx), grid_shape)."""
        with self._h5py.File(self.path, "r") as h:
            if "flownet" not in h:
                return None
            fn = h["flownet"]
            return (fn["history"][:], fn["inside"][:],
                    (int(fn.attrs["grid_ny"]), int(fn.attrs["grid_nx"])))


def element_field(step: dict, name: str) -> np.ndarray:
    """Derive a per-element scalar field from a step dict (matched by the
    leading token of the label, so display text can carry hints)."""
    s = step["sigma"]
    sxx, syy, szz, sxy = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    if name.startswith("von Mises"):
        return step["mises"]
    if name.startswith("σ1") or name.startswith("σ2") \
            or name.startswith("max shear"):
        c = 0.5 * (sxx + syy)
        r = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
        if name.startswith("max shear"):
            return r
        return c + r if name.startswith("σ1") else c - r
    if name.startswith("σxx"):
        return sxx
    if name.startswith("σyy"):
        return syy
    if name.startswith("σzz"):
        return szz
    if name.startswith("σxy"):
        return sxy
    if name.startswith("mean"):
        return (sxx + syy + szz) / 3.0
    if name.startswith("triax"):
        return (sxx + syy + szz) / 3.0 / np.maximum(step["mises"], 1e-9)
    if name.startswith("PEEQ"):
        return step["peeq"]
    if name.startswith("damage"):
        return step["damage"] if step["damage"] is not None \
            else np.zeros(len(step["elems"]))
    if name.startswith("ε") or name.startswith("γ"):
        e = step["strain"]
        if e is None:
            return np.zeros(len(step["elems"]))
        idx = {"εxx": 0, "εyy": 1, "εzz": 2, "γxy": 3}[name[:3]]
        return e[:, idx]
    raise ValueError(f"unknown field '{name}'")


def field_unit(name: str) -> str:
    if name == "triaxiality":
        return ""
    if name == "PEEQ" or name.startswith("ε") or name.startswith("γ") \
            or name == "damage":
        return "" if name in ("PEEQ", "damage") else "-"
    return "MPa"

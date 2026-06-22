"""DXF geometry input.

Reads LINE / ARC / LWPOLYLINE / POLYLINE / CIRCLE entities from a DXF file and
chains them into ordered point loops (closed -> blank outline; open chains are
also returned for tool surfaces).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import ezdxf
import numpy as np

_TOL = 1e-6


@dataclass
class Contour:
    """Ordered 2D point chain. closed=True means last point connects to first."""
    points: np.ndarray  # (N, 2)
    closed: bool = False

    @property
    def segments(self) -> np.ndarray:
        """(M, 2, 2) array of segments [[p0, p1], ...]."""
        pts = self.points
        if self.closed:
            pts = np.vstack([pts, pts[:1]])
        return np.stack([pts[:-1], pts[1:]], axis=1)

    def bbox(self):
        return self.points.min(axis=0), self.points.max(axis=0)


@dataclass
class DxfShape:
    """All contours found in one DXF file."""
    contours: list[Contour] = field(default_factory=list)

    @property
    def outline(self) -> Contour:
        """Largest closed contour (blank outline)."""
        closed = [c for c in self.contours if c.closed]
        if not closed:
            raise ValueError("no closed contour in DXF")
        return max(closed, key=lambda c: abs(_polygon_area(c.points)))

    def all_segments(self) -> np.ndarray:
        segs = [c.segments for c in self.contours if len(c.points) >= 2]
        return np.concatenate(segs, axis=0) if segs else np.empty((0, 2, 2))


def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _arc_points(cx, cy, r, a0_deg, a1_deg, max_chord_deg=6.0):
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    while a1 <= a0:
        a1 += 2 * math.pi
    n = max(2, int(math.ceil((a1 - a0) / math.radians(max_chord_deg))) + 1)
    ang = np.linspace(a0, a1, n)
    return np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])


def _bulge_arc(p0, p1, bulge, max_chord_deg=6.0):
    """LWPOLYLINE bulge segment -> intermediate points (excluding p0, including p1)."""
    if abs(bulge) < 1e-12:
        return np.array([p1])
    theta = 4.0 * math.atan(bulge)
    chord = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    r = chord / (2.0 * math.sin(abs(theta) / 2.0))
    mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    d = math.sqrt(max(r * r - (chord / 2.0) ** 2, 0.0))
    nx, ny = -(p1[1] - p0[1]) / chord, (p1[0] - p0[0]) / chord
    if bulge < 0:
        nx, ny = -nx, -ny
    cx, cy = mx + d * nx, my + d * ny
    a0 = math.atan2(p0[1] - cy, p0[0] - cx)
    n = max(2, int(math.ceil(abs(theta) / math.radians(max_chord_deg))) + 1)
    ang = a0 + np.linspace(0, theta, n)[1:]
    return np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])


def read_dxf(path: str, max_chord_deg: float = 6.0) -> DxfShape:
    """Read a DXF file and chain entities into contours."""
    doc = ezdxf.readfile(path)
    pieces: list[np.ndarray] = []  # each an (N,2) open polyline piece

    for e in doc.modelspace():
        t = e.dxftype()
        if t == "LINE":
            s, en = e.dxf.start, e.dxf.end
            pieces.append(np.array([[s.x, s.y], [en.x, en.y]]))
        elif t == "ARC":
            c = e.dxf.center
            pieces.append(_arc_points(c.x, c.y, e.dxf.radius,
                                      e.dxf.start_angle, e.dxf.end_angle, max_chord_deg))
        elif t == "CIRCLE":
            c = e.dxf.center
            pts = _arc_points(c.x, c.y, e.dxf.radius, 0.0, 360.0, max_chord_deg)
            pieces.append(pts)
        elif t == "LWPOLYLINE":
            raw = list(e.get_points())  # (x, y, start_w, end_w, bulge)
            pts = [np.array([raw[0][0], raw[0][1]])]
            for i in range(len(raw) - 1):
                p0 = (raw[i][0], raw[i][1])
                p1 = (raw[i + 1][0], raw[i + 1][1])
                pts.extend(_bulge_arc(p0, p1, raw[i][4], max_chord_deg))
            if e.closed:
                p0 = (raw[-1][0], raw[-1][1])
                p1 = (raw[0][0], raw[0][1])
                pts.extend(_bulge_arc(p0, p1, raw[-1][4], max_chord_deg))
            pieces.append(np.array([np.asarray(p).ravel()[:2] for p in pts]))
        elif t == "POLYLINE":
            pts = np.array([[v.dxf.location.x, v.dxf.location.y] for v in e.vertices])
            if e.is_closed and len(pts) > 0:
                pts = np.vstack([pts, pts[:1]])
            pieces.append(pts)

    return DxfShape(contours=_chain(pieces))


def _chain(pieces: list[np.ndarray], tol: float = 1e-4) -> list[Contour]:
    """Connect pieces end-to-end into contours."""
    pieces = [p.astype(float) for p in pieces if len(p) >= 2]
    contours = []
    while pieces:
        chain = pieces.pop(0)
        grew = True
        while grew:
            grew = False
            for i, p in enumerate(pieces):
                if np.linalg.norm(chain[-1] - p[0]) < tol:
                    chain = np.vstack([chain, p[1:]])
                elif np.linalg.norm(chain[-1] - p[-1]) < tol:
                    chain = np.vstack([chain, p[::-1][1:]])
                elif np.linalg.norm(chain[0] - p[-1]) < tol:
                    chain = np.vstack([p[:-1], chain])
                elif np.linalg.norm(chain[0] - p[0]) < tol:
                    chain = np.vstack([p[::-1][:-1], chain])
                else:
                    continue
                pieces.pop(i)
                grew = True
                break
        closed = bool(np.linalg.norm(chain[0] - chain[-1]) < tol)
        if closed:
            chain = chain[:-1]
        # drop consecutive duplicates
        keep = np.ones(len(chain), bool)
        keep[1:] = np.linalg.norm(np.diff(chain, axis=0), axis=1) > _TOL
        contours.append(Contour(points=chain[keep], closed=closed))
    return contours

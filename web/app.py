"""forge-calc web API — upload/select a forming case, run it on the server
(single isolated worker), poll progress, view results.

v0 exposes the validated predefined cases so the full pipeline + deployment
isolation can be proven end-to-end; custom-DXF upload plugs into solve_job
(_resolve_spec) next.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import jobs  # noqa: E402
import run_case  # noqa: E402  for the case catalogue
from plasticfem.geometry import read_dxf  # noqa: E402

app = FastAPI(title="ForgeCal", version="0.1")

_RESULT_WHITELIST = {
    "mesh_anim.gif", "load_stroke.png", "load_stroke.csv",
    "flownet_end.png", "results.h5", "run.log",
}


@app.on_event("startup")
def _startup():
    jobs.start_worker()


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/cases")
def list_cases():
    return {"cases": sorted(run_case.CASES.keys())}


@app.get("/api/queue")
def queue():
    return jobs.queue_info()


class JobRequest(BaseModel):
    case: str
    flownet: bool = False


@app.post("/api/jobs")
def create_job(req: JobRequest):
    if req.case not in run_case.CASES:
        raise HTTPException(404, f"unknown case: {req.case}")
    job_id = jobs.submit({"case": req.case, "flownet": req.flownet})
    return {"job_id": job_id, "queue_depth": jobs.queue_depth()}


_MAX_DXF = 5_000_000          # 5 MB per file
_MAX_FILES = 8


def _tok(name: str) -> str:
    """A filesystem-safe tool token from an arbitrary DXF name."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name) or "tool"


@app.post("/api/preview")
async def preview(files: list[UploadFile] = File(...)):
    """Parse uploaded DXFs and return each file's contour polylines so the
    client can draw the geometry/arrangement BEFORE running a (long) solve."""
    out: dict[str, list] = {}
    with tempfile.TemporaryDirectory() as td:
        for uf in files:
            stem = os.path.splitext(os.path.basename(uf.filename or ""))[0]
            if not stem:
                continue
            data = await uf.read()
            if len(data) > _MAX_DXF:
                raise HTTPException(400, f"DXFが大きすぎます(>5MB): {uf.filename}")
            p = os.path.join(td, "f.dxf")
            with open(p, "wb") as f:
                f.write(data)
            try:
                shape = read_dxf(p)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(400, f"{uf.filename}: DXF読込に失敗 ({e})")
            out[stem] = [{"points": c.points.tolist(), "closed": bool(c.closed)}
                         for c in shape.contours if len(c.points) >= 2]
    return {"outlines": out}


@app.post("/api/jobs/custom")
async def create_custom_job(config: str = Form(...),
                            files: list[UploadFile] = File(...)):
    """Custom job: uploaded DXFs (any names) + a JSON `config` that names which
    file is the blank/workpiece and the per-tool roles. Tool names are
    sanitized server-side, so the uploaded filenames can be arbitrary."""
    try:
        cfg = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(400, "config が不正な JSON です")
    if len(files) > _MAX_FILES:
        raise HTTPException(400, f"ファイルが多すぎます（最大{_MAX_FILES}）")

    # read uploads keyed by their original stem (filename without .dxf)
    uploads: dict[str, bytes] = {}
    for uf in files:
        stem = os.path.splitext(os.path.basename(uf.filename or ""))[0]
        if not stem:
            raise HTTPException(400, f"ファイル名が不正です: {uf.filename}")
        data = await uf.read()
        if len(data) > _MAX_DXF:
            raise HTTPException(400, f"DXFが大きすぎます(>5MB): {uf.filename}")
        uploads[stem] = data

    if cfg.get("mode") not in ("axisymmetric", "plane_strain"):
        raise HTTPException(400, "mode が不正です")
    for key, lo, hi in (("stroke", 0.0, 1e4), ("mesh", 0.1, 50.0),
                        ("dstroke", 1e-4, 10.0)):
        v = float(cfg.get(key, 0))
        if not (lo < v <= hi):
            raise HTTPException(400, f"{key} の値が範囲外です: {v}")
    mp = cfg.get("mat") or {}
    for key, lo, hi in (("E", 1e3, 1e6), ("nu", 0.0, 0.49),
                        ("C", 1.0, 1e5), ("n", 0.0, 1.0), ("e_sat", 0.05, 10.0)):
        v = float(mp.get(key, -1))
        if not (lo <= v <= hi):
            raise HTTPException(400, f"材料 {key} の値が不正です: {v}")

    blank = cfg.get("blank")
    if not blank or blank not in uploads:
        raise HTTPException(400, "素材(blank)を指定してください")
    tools = cfg.get("tools") or []
    missing = {t["file"] for t in tools} - set(uploads)
    if missing:
        raise HTTPException(400, f"工具DXFが不足: {sorted(missing)}")
    if not cfg.get("punch") or cfg["punch"] not in {t["file"] for t in tools}:
        raise HTTPException(400, "パンチを指定してください")
    if jobs.queue_full():
        raise HTTPException(429, "サーバが混雑しています（順番待ちが上限）。"
                                 "しばらくしてから再投入してください。")

    # save: workpiece -> blank.dxf; each tool -> a unique safe <token>.dxf
    saved = {"blank.dxf": uploads[blank]}
    used, remap = {"blank"}, {}
    for t in tools:
        base = tk = _tok(t["file"])
        i = 1
        while tk in used:
            tk = f"{base}_{i}"
            i += 1
        used.add(tk)
        remap[t["file"]] = tk
        saved[tk + ".dxf"] = uploads[t["file"]]
        t["file"] = tk
    cfg["punch"] = remap[cfg["punch"]]
    cfg["custom"] = True

    job_id = jobs.submit(cfg, files=saved)
    return {"job_id": job_id, "queue_depth": jobs.queue_depth()}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    s = jobs.get_status(job_id)
    if s is None:
        raise HTTPException(404, "no such job")
    d = jobs.jobdir(job_id)
    s["files"] = [f for f in _RESULT_WHITELIST
                  if os.path.exists(os.path.join(d, f))]
    return s


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if jobs.get_status(job_id) is None:
        raise HTTPException(404, "no such job")
    return {"cancelled": jobs.cancel(job_id)}


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    if name not in _RESULT_WHITELIST:
        raise HTTPException(403, "not allowed")
    p = os.path.join(jobs.jobdir(job_id), name)
    if not os.path.exists(p):
        raise HTTPException(404, "not found")
    return FileResponse(p)


# static assets (index.html lives here; served at "/")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")),
          name="static")

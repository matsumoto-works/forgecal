"""Single-worker job queue with subprocess isolation + a memory watchdog.

Only one solve runs at a time (bounds CPU and RAM). Each solve is a child
process running solve_job.py, so a runaway solve can be killed without taking
down the API. A psutil watchdog samples the child's RSS and terminates it
cleanly with a "memory exceeded" status if it crosses MEM_SOFT_MB -> the
forming load that matters (the host's IoT workload) is never starved into OOM.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.environ.get("JOBS_DIR", os.path.join(HERE, "jobs"))
MEM_SOFT_MB = int(os.environ.get("MEM_SOFT_MB", "700"))   # kill the solve above
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))   # wall-clock seconds
# results are transient: keep only the most recent jobs and drop old ones so
# the H5s do not accumulate. The user downloads right after the run; nothing is
# stored permanently.
MAX_JOBS = int(os.environ.get("MAX_JOBS", "20"))
JOB_TTL_SEC = int(os.environ.get("JOB_TTL_SEC", str(24 * 3600)))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "5"))   # reject new jobs past this
PY = os.environ.get("SOLVER_PYTHON", sys.executable)

_current: str | None = None      # job id currently solving (None if idle)

os.makedirs(JOBS_DIR, exist_ok=True)
_q: "queue.Queue[str]" = queue.Queue()
_cancel: set[str] = set()        # job ids the user asked to stop


def cancel(job_id: str) -> bool:
    """Request a running/queued job be stopped. Returns False if it's already
    finished."""
    s = get_status(job_id)
    if not s or s.get("state") in ("done", "failed"):
        return False
    _cancel.add(job_id)
    return True


def jobdir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, job_id)


def _status_path(job_id: str) -> str:
    return os.path.join(jobdir(job_id), "status.json")


def _set_status(job_id: str, **kw):
    kw["t"] = time.time()
    p = _status_path(job_id)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False)
    os.replace(tmp, p)            # atomic: readers never see a partial file


def _prune():
    """Drop old job directories so results.h5 files don't accumulate: keep the
    most recent MAX_JOBS and delete anything older than JOB_TTL_SEC. Active
    (running/queued) jobs are never removed."""
    items = []
    for d in os.listdir(JOBS_DIR):
        p = os.path.join(JOBS_DIR, d)
        if not os.path.isdir(p):
            continue
        st = get_status(d) or {}
        if st.get("state") in ("running", "queued"):
            continue
        try:
            items.append((os.path.getmtime(p), p))
        except OSError:
            pass
    items.sort(reverse=True)                 # newest first
    now = time.time()
    for i, (mt, p) in enumerate(items):
        if i >= MAX_JOBS or (now - mt) > JOB_TTL_SEC:
            shutil.rmtree(p, ignore_errors=True)


def submit(config: dict, files: dict | None = None) -> str:
    """Enqueue a job. `files` maps a filename -> bytes (custom-DXF uploads),
    saved into the job directory alongside config.json."""
    _prune()
    job_id = uuid.uuid4().hex[:12]
    d = jobdir(job_id)
    os.makedirs(d, exist_ok=True)
    for name, data in (files or {}).items():
        with open(os.path.join(d, os.path.basename(name)), "wb") as f:
            f.write(data)
    json.dump(config, open(os.path.join(d, "config.json"), "w",
                           encoding="utf-8"), ensure_ascii=False)
    _set_status(job_id, state="queued", queued_at=time.time())
    _q.put(job_id)
    return job_id


def get_status(job_id: str) -> dict | None:
    p = _status_path(job_id)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"state": "running"}        # mid-write; treat as running


def queue_depth() -> int:
    return _q.qsize()


def queue_full() -> bool:
    return _q.qsize() >= MAX_QUEUE


def queue_info() -> dict:
    """Waiting count + whether a solve is in progress (for the UI)."""
    return {"waiting": _q.qsize(), "running": _current is not None,
            "max_queue": MAX_QUEUE}


def _run_one(job_id: str):
    d = jobdir(job_id)
    if job_id in _cancel:                       # cancelled while still queued
        _cancel.discard(job_id)
        _set_status(job_id, state="failed", error="cancelled")
        return
    _set_status(job_id, state="running", progress=0.0, started_at=time.time())
    log = open(os.path.join(d, "run.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([PY, os.path.join(HERE, "solve_job.py"), d],
                            stdout=log, stderr=subprocess.STDOUT)
    ps = psutil.Process(proc.pid)
    t0 = time.time()
    killed = None
    while proc.poll() is None:
        time.sleep(0.5)
        if job_id in _cancel:
            killed = "cancelled"
            break
        try:
            rss = ps.memory_info().rss
            for c in ps.children(recursive=True):
                try:
                    rss += c.memory_info().rss
                except psutil.Error:
                    pass
            if rss / 1e6 > MEM_SOFT_MB:
                killed = "memory"
                break
        except psutil.Error:
            pass
        if time.time() - t0 > JOB_TIMEOUT:
            killed = "timeout"
            break
    if killed:
        _kill_tree(ps)
        proc.wait(timeout=10)
        log.close()
        _cancel.discard(job_id)
        if killed == "cancelled":
            _set_status(job_id, state="failed", error="cancelled",
                        diagnosis=dict(cause="cancelled", design_ng=False,
                                       title="ユーザー操作で解析を停止しました",
                                       location=None, suggestions=[]))
        elif killed == "memory":
            _set_status(job_id, state="failed", error="memory_exceeded",
                        diagnosis=dict(
                            cause="memory", design_ng=False,
                            title=f"メモリ上限超過（>{MEM_SOFT_MB}MB）で停止",
                            location=None,
                            suggestions=["メッシュを粗くする（要素寸法を大きく）",
                                         "尖った工具は先端で要素が細かくなりメモリ増："
                                         "先端に R（丸み）を付けると改善（実形状にも近い）",
                                         "ストローク量を減らす",
                                         "サーバ管理者に mem_limit 引き上げを相談"]))
        else:
            _set_status(job_id, state="failed", error="timeout",
                        diagnosis=dict(
                            cause="timeout", design_ng=False,
                            title=f"実行時間が上限（{JOB_TIMEOUT}s）を超過",
                            location=None,
                            suggestions=["メッシュを粗く／ストロークを減らす"]))
        return
    log.close()
    # solve_job.py wrote the final status.json itself (done / failed)
    if get_status(job_id).get("state") not in ("done", "failed"):
        _set_status(job_id, state="failed", error="solver_exited_unexpectedly")


def _kill_tree(ps: psutil.Process):
    try:
        for c in ps.children(recursive=True):
            c.kill()
        ps.kill()
    except psutil.Error:
        pass


def _worker():
    # any job left "running" from a previous crash/restart -> mark failed
    for jid in os.listdir(JOBS_DIR):
        s = get_status(jid)
        if s and s.get("state") in ("running", "queued"):
            _set_status(jid, state="failed", error="server_restarted")
    global _current
    while True:
        job_id = _q.get()
        _current = job_id
        try:
            _run_one(job_id)
        except Exception as e:  # noqa: BLE001
            _set_status(job_id, state="failed", error=f"worker:{e}")
        finally:
            _current = None
            _q.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)


def start_worker():
    if not _worker_thread.is_alive():
        _worker_thread.start()

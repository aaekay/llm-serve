"""Batch job server: upload a JSONL batch over HTTP, let it run on the GPU, download the results later.

The client only needs the tunnel while uploading and downloading; the job runs on its own in between.
Jobs run one at a time, in submission order, through run-vllm-batch-file.sh (vllm run-batch, one GPU), so every
request body (JSON schema, seed, chat_template_kwargs, ...) reaches vLLM unchanged.

Endpoints (127.0.0.1 only):
  POST   /jobs?preset=<preset>&name=<label>   body = OpenAI batch JSONL      -> {"id", "status", "n_requests"}
  GET    /jobs                                                              -> list of jobs
  GET    /jobs/<id>                                                         -> status: queued|running|completed|failed
  GET    /jobs/<id>/output                                                  -> results JSONL (when completed)
  DELETE /jobs/<id>                                                         -> deletes the job's input and output files
  GET    /healthz
State lives in batch_io/jobs/<id>/ (git-ignored). Inputs may contain PHI: clients should DELETE a job after download.
On restart, queued and interrupted jobs are re-queued (run-vllm-batch-file.sh skips finished files).

Started by run-batch-jobs.sh; standard library only.
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "batch_io" / "jobs"
PRESETS = {"gemma4-31b", "qwen36-35b-a3b-fp8", "qwen36-35b-a3b", "qwen36-27b-fp8"}
MAX_UPLOAD = int(os.getenv("BATCH_MAX_UPLOAD_MB", "512")) * 1024 * 1024
GPU = os.getenv("GPU", "0")
_lock = threading.Lock()
_wake = threading.Event()


def _meta_path(jid): return JOBS / jid / "job.json"


def _load(jid):
    return json.loads(_meta_path(jid).read_text())


def _save(meta):
    p = _meta_path(meta["id"]); tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=1)); tmp.replace(p)


def _all():
    out = []
    for d in sorted(JOBS.glob("*/job.json")):
        try: out.append(json.loads(d.read_text()))
        except Exception: pass
    return sorted(out, key=lambda m: m["submitted_at"])


def _public(m):
    keys = ("id", "name", "preset", "status", "n_requests", "n_results", "submitted_at", "started_at", "finished_at", "error")
    return {k: m.get(k) for k in keys}


def worker():
    while True:
        with _lock:
            queued = [m for m in _all() if m["status"] == "queued"]
        if not queued:
            _wake.wait(10); _wake.clear(); continue
        m = queued[0]; jdir = JOBS / m["id"]
        m.update(status="running", started_at=time.time(), error=None); _save(m)
        env = dict(os.environ, GPU=GPU, BATCH_IN_DIR=str(jdir / "in"), BATCH_OUT_DIR=str(jdir / "out"))
        with open(jdir / "run.log", "ab") as log:
            rc = subprocess.call([str(ROOT / "run-vllm-batch-file.sh"), m["preset"]], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        outs = sorted((jdir / "out").glob("*.out.jsonl"))
        n_res = sum(sum(1 for line in f.open() if line.strip()) for f in outs)
        with _lock:
            m = _load(m["id"])
            if m["status"] == "deleted":
                continue
            m.update(finished_at=time.time(), n_results=n_res)
            if rc == 0 and n_res >= m["n_requests"]:
                m["status"] = "completed"
            else:
                m.update(status="failed", error=f"exit code {rc}; {n_res}/{m['n_requests']} results; see batch_io/jobs/{m['id']}/run.log")
            _save(m)


class H(BaseHTTPRequestHandler):
    server_version = "batch-jobs/1"

    def log_message(self, fmt, *args):   # never log bodies; method + path only
        sys.stderr.write(f"[batch-jobs] {self.command} {urlparse(self.path).path} -> {args[1] if len(args) > 1 else ''}\n")

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _job(self, jid):
        if not re.fullmatch(r"[0-9a-f]{12}", jid or "") or not _meta_path(jid).exists():
            return None
        return _load(jid)

    def do_GET(self):
        u = urlparse(self.path); parts = [p for p in u.path.split("/") if p]
        if parts == ["healthz"]:
            return self._json(200, {"ok": True, "gpu": GPU})
        if parts == ["jobs"]:
            return self._json(200, [_public(m) for m in _all() if m["status"] != "deleted"])
        if len(parts) >= 2 and parts[0] == "jobs":
            m = self._job(parts[1])
            if not m or m["status"] == "deleted":
                return self._json(404, {"error": "no such job"})
            if len(parts) == 2:
                return self._json(200, _public(m))
            if parts[2:] == ["output"]:
                if m["status"] != "completed":
                    return self._json(409, {"error": f"job is {m['status']}"})
                outs = sorted((JOBS / m["id"] / "out").glob("*.out.jsonl"))
                size = sum(f.stat().st_size for f in outs)
                self.send_response(200); self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(size)); self.end_headers()
                for f in outs:
                    with f.open("rb") as fh: shutil.copyfileobj(fh, self.wfile)
                return
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path.rstrip("/") != "/jobs":
            return self._json(404, {"error": "not found"})
        preset = (q.get("preset") or [""])[0]; name = re.sub(r"[^A-Za-z0-9_.-]", "_", (q.get("name") or ["job"])[0])[:80]
        if preset not in PRESETS:
            return self._json(400, {"error": f"preset must be one of {sorted(PRESETS)}"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_UPLOAD:
            return self._json(413, {"error": f"body must be 1..{MAX_UPLOAD} bytes"})
        jid = uuid.uuid4().hex[:12]; jdir = JOBS / jid
        (jdir / "in").mkdir(parents=True); (jdir / "out").mkdir()
        dest = jdir / "in" / f"{name}.jsonl"; left = n; n_lines = 0
        with dest.open("wb") as fh:
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk: break
                fh.write(chunk); left -= len(chunk); n_lines += chunk.count(b"\n")
        if left:
            shutil.rmtree(jdir); return self._json(400, {"error": "upload truncated"})
        with dest.open() as fh:   # validate every line before queuing
            for i, line in enumerate(fh, 1):
                if not line.strip(): continue
                try:
                    r = json.loads(line); assert r["url"] == "/v1/chat/completions" and "custom_id" in r and "body" in r
                except Exception:
                    shutil.rmtree(jdir); return self._json(400, {"error": f"line {i} is not an OpenAI batch request"})
        n_req = sum(1 for line in dest.open() if line.strip())
        meta = {"id": jid, "name": name, "preset": preset, "status": "queued", "n_requests": n_req, "n_results": 0,
                "submitted_at": time.time(), "started_at": None, "finished_at": None, "error": None}
        with _lock: _save(meta)
        _wake.set()
        return self._json(201, _public(meta))

    def do_DELETE(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if len(parts) == 2 and parts[0] == "jobs":
            m = self._job(parts[1])
            if not m:
                return self._json(404, {"error": "no such job"})
            if m["status"] == "running":
                return self._json(409, {"error": "job is running; wait for it to finish"})
            with _lock:
                for sub in ("in", "out"):
                    shutil.rmtree(JOBS / m["id"] / sub, ignore_errors=True)
                m["status"] = "deleted"; _save(m)
            return self._json(200, {"id": m["id"], "status": "deleted"})
        return self._json(404, {"error": "not found"})


def main():
    JOBS.mkdir(parents=True, exist_ok=True)
    for m in _all():                       # recover after a restart
        if m["status"] == "running":
            m["status"] = "queued"; _save(m)
    threading.Thread(target=worker, daemon=True).start()
    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
    assert host in ("127.0.0.1", "localhost"), "bind to localhost only; reach it through an SSH tunnel"
    print(f"[batch-jobs] listening on http://{host}:{port}, GPU {GPU}, jobs in {JOBS}", flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()


if __name__ == "__main__":
    main()

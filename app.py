#!/usr/bin/env python3
"""Interactive Auto Layer POC server — upload a print, get CV / SAM2 outputs.

Usage:
  cd auto-layer-poc
  .venv/bin/python app.py
  → http://127.0.0.1:8787
"""

from __future__ import annotations

import cgi
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT_ROOT = ROOT / "out"
OUT = OUT_ROOT / "jobs"
WEB = ROOT / "web"
FIX = ROOT / "fixtures"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

import layers_menu  # noqa: E402

load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")

# Job store: id -> {status, error, result}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8787"))
SAM_MODEL = os.environ.get("SAM_MODEL", "sam2_b.pt")

# Default prints shown in the UI picker (id = filename in fixtures/)
DEFAULT_PRINTS = [
    {"id": "06-seashells.png", "label": "Seashells", "hint": "large motifs"},
    {"id": "01-ditsy-florals.png", "label": "Ditsy florals", "hint": "dense small"},
    {"id": "03-paisley.png", "label": "Paisley", "hint": "interlocking"},
    {"id": "02-tropical-leaves.png", "label": "Tropical leaves", "hint": "medium"},
    {"id": "04-gingham-floral.png", "label": "Gingham floral", "hint": "check + flowers"},
    {"id": "05-ogee-trellis.png", "label": "Ogee trellis", "hint": "geometric"},
    {"id": "08-abstract-brush.png", "label": "Abstract brush", "hint": "painterly"},
]


def _fixture_path(fixture_id: str) -> Path | None:
    """Resolve a safe fixture id. Allows fixtures/*.png or fixtures/{hard,showcase}/*.png."""
    # Normalize and reject path tricks
    raw = fixture_id.replace("\\", "/").lstrip("/")
    if ".." in raw.split("/"):
        return None
    parts = raw.split("/")
    if len(parts) == 1:
        name = parts[0]
        allowed = {p.name for p in FIX.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")} if FIX.is_dir() else set()
        if name not in allowed:
            return None
        link = FIX / name
    elif len(parts) == 2 and parts[0] in ("hard", "showcase"):
        name = parts[1]
        folder = FIX / parts[0]
        if not folder.is_dir():
            return None
        allowed = {p.name for p in folder.iterdir() if p.suffix.lower() == ".png"}
        if name not in allowed:
            return None
        link = folder / name
    else:
        return None
    if not link.exists():
        return None
    target = link.resolve()
    return target if target.is_file() else None


def _list_defaults() -> list[dict]:
    out = []
    for p in DEFAULT_PRINTS:
        fp = _fixture_path(p["id"])
        if fp is None:
            continue
        out.append({**p, "thumb": f"/fixtures/{p['id']}", "available": True, "group": "defaults"})
    return out


def _list_catalog_group(group: str) -> list[dict]:
    catalog = FIX / group / "catalog.json"
    if not catalog.exists():
        return []
    items = json.loads(catalog.read_text())
    out = []
    for p in items:
        fid = p.get("id") or f"{group}/{p.get('file')}"
        fp = _fixture_path(fid)
        if fp is None:
            continue
        out.append(
            {
                "id": fid,
                "label": p.get("label") or fid,
                "hint": p.get("hint") or group,
                "thumb": f"/fixtures/{fid}",
                "available": True,
                "group": group,
            }
        )
    return out


def _list_hard() -> list[dict]:
    return _list_catalog_group("hard")


def _list_showcase() -> list[dict]:
    return _list_catalog_group("showcase")


def _safe_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._") or "print"
    return stem[:80]


def _run_job(job_id: str, image_path: Path, mode: str) -> None:
    """mode: cv | sam2 | both"""
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["step"] = "starting"

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("runner", SRC / "run_auto_layer.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        run = mod.run

        job_dir = OUT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        def tag(run_dir: Path, pipeline: str, label: str) -> None:
            scene = run_dir / "scene.json"
            if not scene.exists():
                return
            data = json.loads(scene.read_text())
            data["pipeline"] = pipeline
            data["pipeline_label"] = label
            data.setdefault("stats", {})["pipeline"] = pipeline
            scene.write_text(json.dumps(data, indent=2))

        if mode in ("cv", "both"):
            with JOBS_LOCK:
                JOBS[job_id]["step"] = "Routing + extract (busy/hard ≈1–2 min on CPU)…"
            cv_dir = job_dir / "cv"
            run(
                image_path,
                cv_dir,
                use_sam=False,
                use_qa=False,
                use_tile_gapfill=False,
                use_gemini_inpaint=False,
                sam_model=SAM_MODEL,
            )
            tag(cv_dir, "cv", "CV baseline")
            results["cv"] = _summarize(cv_dir)

        if mode in ("sam2", "both"):
            with JOBS_LOCK:
                JOBS[job_id]["step"] = "SAM2 + QA (1–3 min on CPU)…"
            sam_dir = job_dir / "sam2"
            run(
                image_path,
                sam_dir,
                use_sam=True,
                use_qa=True,
                use_tile_gapfill=False,
                use_gemini_inpaint=False,
                sam_model=SAM_MODEL,
            )
            tag(sam_dir, "sam2", "SAM2 + QA")
            results["sam2"] = _summarize(sam_dir)

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["step"] = "done"
            JOBS[job_id]["result"] = results
            JOBS[job_id]["job_dir"] = str(job_dir)
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = f"{exc}\n{traceback.format_exc()}"


def _summarize(run_dir: Path) -> dict:
    scene = run_dir / "scene.json"
    data = json.loads(scene.read_text()) if scene.exists() else {}
    st = data.get("stats") or {}
    job_id = run_dir.parent.name
    pipeline = run_dir.name
    return {
        "path": str(run_dir),
        "job_id": job_id,
        "pipeline": data.get("pipeline") or pipeline,
        "layers": st.get("isolated"),
        "coverage": st.get("ink_coverage"),
        "residual_frac": st.get("residual_frac"),
        "high_confidence": st.get("high_confidence"),
        "uncertain": st.get("uncertain"),
        "background_quality": st.get("background_quality"),
        "qa_passed": st.get("qa_passed"),
        "print_type": st.get("print_type"),
        "editor": f"/jobs/{job_id}/{pipeline}/editor.html",
        "background": f"/jobs/{job_id}/{pipeline}/background.png",
        "original": f"/jobs/{job_id}/{pipeline}/original.png",
        "overlay": f"/jobs/{job_id}/{pipeline}/discover_overlay.png",
        "scene": f"/jobs/{job_id}/{pipeline}/scene.json",
        "residual": f"/jobs/{job_id}/{pipeline}/residual_ink.png",
    }


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        return json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Auto Layer POC</title>
  <style>
    :root { --ink:#1a1a1e; --muted:#6e6e76; --line:#e7e7ea; --accent:#2a6f5b; --bg:#f4f3f6; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:"Segoe UI",system-ui,sans-serif; color:var(--ink); background:var(--bg); }
    .wrap { max-width:860px; margin:0 auto; padding:40px 20px; }
    h1 { margin:0 0 6px; font-size:24px; }
    .sub { color:var(--muted); margin:0 0 28px; line-height:1.45; }
    .card { background:#fff; border-radius:14px; padding:24px; box-shadow:0 1px 3px rgba(0,0,0,.06); }
    label { display:block; font-size:13px; color:var(--muted); margin:14px 0 6px; }
    input[type=file] { width:100%; }
    .modes { display:flex; flex-direction:column; gap:8px; margin-top:4px; }
    .modes label { display:flex; align-items:center; gap:8px; color:var(--ink); font-size:14px; margin:0; cursor:pointer; }
    button { margin-top:20px; width:100%; border:0; background:var(--accent); color:#fff;
      padding:12px 16px; border-radius:10px; font-size:15px; font-weight:600; cursor:pointer; }
    button:disabled { opacity:.55; cursor:wait; }
    #status { margin-top:16px; font-size:13px; color:var(--muted); min-height:1.2em; }
    #result { margin-top:24px; display:none; }
    .pipe { border:1px solid var(--line); border-radius:10px; padding:14px; margin:12px 0; }
    .pipe h3 { margin:0 0 8px; font-size:15px; }
    .pipe .meta { font-size:12px; color:var(--muted); margin-bottom:10px; }
    .previews { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }
    .previews img { width:100%; border-radius:6px; background:#eee; }
    .btn { display:inline-block; margin-top:10px; padding:8px 12px; background:var(--accent);
      color:#fff; text-decoration:none; border-radius:8px; font-size:13px; }
    .err { color:#a33; white-space:pre-wrap; font-size:12px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); gap:10px; margin-top:8px; }
    .pick { border:2px solid var(--line); border-radius:10px; padding:8px; cursor:pointer;
      background:#fafafa; text-align:left; transition:border-color .15s, box-shadow .15s; }
    .pick:hover { border-color:#bbb; }
    .pick.selected { border-color:var(--accent); box-shadow:0 0 0 2px rgba(42,111,91,.2); background:#f3faf7; }
    .pick img { width:100%; aspect-ratio:1; object-fit:cover; border-radius:6px; background:#eee; display:block; }
    .pick .name { font-size:13px; font-weight:600; margin-top:8px; }
    .pick .hint { font-size:11px; color:var(--muted); }
    .or { text-align:center; color:var(--muted); font-size:12px; margin:16px 0 4px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Auto Layer POC</h1>
    <p class="sub">Pick a default print (or upload your own). Get editable motif layers + reconstructed background.
      CV is fast; SAM2 + QA takes ~1–3 min on CPU.</p>
    <div class="card">
      <form id="form">
        <label>Default prints</label>
        <div class="grid" id="defaults"></div>
        <label style="margin-top:18px">Showcase motifs (15 · 2048px)</label>
        <div class="grid" id="showcase"></div>
        <label style="margin-top:18px">Hard stress tests (30)</label>
        <div class="grid" id="hard"></div>
        <input type="hidden" name="fixture_id" id="fixture_id" value="" />

        <div class="or">— or upload your own —</div>
        <label for="file">Print image (PNG / JPG)</label>
        <input id="file" name="file" type="file" accept="image/png,image/jpeg,image/webp" />

        <label>Pipeline</label>
        <div class="modes">
          <label><input type="radio" name="mode" value="cv" checked /> Auto route (recommended) — clean=CV · soft/camo/busy=VLM→SAM</label>
          <label><input type="radio" name="mode" value="both" /> Both — Auto + SAM2/QA compare</label>
          <label><input type="radio" name="mode" value="sam2" /> SAM2 + QA only (legacy compare)</label>
        </div>
        <button type="submit" id="go">Run Auto Layer</button>
      </form>
      <div id="status"></div>
      <div id="result"></div>
    </div>
  </div>
  <script>
    const form = document.getElementById('form');
    const status = document.getElementById('status');
    const result = document.getElementById('result');
    const go = document.getElementById('go');
    const fixtureInput = document.getElementById('fixture_id');
    const fileInput = document.getElementById('file');
    const defaultsEl = document.getElementById('defaults');
    const showcaseEl = document.getElementById('showcase');
    const hardEl = document.getElementById('hard');

    function fmtCov(c) {
      if (c == null) return '—';
      return (c * 100).toFixed(1) + '%';
    }

    function panel(title, r) {
      if (!r) return '';
      const qa = r.qa_passed === true ? 'pass' : (r.qa_passed === false ? 'fail' : '—');
      const high = r.high_confidence != null ? `${r.high_confidence} high / ${r.uncertain ?? '?'} unc` : '';
      return `<div class="pipe">
        <h3>${title}</h3>
        <div class="meta">route ${r.print_type || '—'} · layers ${r.layers ?? '—'} · coverage ${fmtCov(r.coverage)} · residual ${fmtCov(r.residual_frac)} · QA ${qa}
          ${high ? '<br/>' + high : ''}
          <br/><span style="color:#666">Review → accept into Layers Menu · residual stays in base</span></div>
        <div class="previews">
          <div><img src="${r.original}" alt="original" /><div class="meta">Original</div></div>
          <div><img src="${r.background}" alt="bg" /><div class="meta">Background</div></div>
          <div><img src="${r.overlay}" alt="overlay" /><div class="meta">Discovery</div></div>
        </div>
        <a class="btn" href="${r.editor}">Open review</a>
        <a class="btn" href="${r.editor}" target="_blank" rel="noopener" style="margin-left:8px;background:#fff;color:var(--accent);border:1px solid var(--accent);text-decoration:none;display:inline-block;padding:8px 12px;border-radius:8px;font-size:13px">New tab</a>
        <button type="button" class="btn" style="border:0;cursor:pointer;margin-left:8px"
          onclick="pushJobToLayers('${r.job_id}', '${r.pipeline || 'cv'}')">
          → Layers Menu</button>
      </div>`;
    }

    async function pushJobToLayers(jobId, pipeline) {
      status.textContent = 'Creating Layers Menu document…';
      try {
        const res = await fetch('/api/documents/from-job', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ job_id: jobId, pipeline }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        status.innerHTML = 'Document <b>' + data.id + '</b> ready · ' +
          '<a href="/documents/' + data.id + '/" target="_blank">open Layers Menu</a> · ' +
          '<a href="' + (data.editor || '#') + '" target="_blank">review proposal</a>';
      } catch (err) {
        status.innerHTML = '<span class="err">' + err + '</span>';
      }
    }

    function selectFixture(id) {
      fixtureInput.value = id || '';
      document.querySelectorAll('.pick').forEach(el => {
        el.classList.toggle('selected', el.dataset.id === id);
      });
      if (id) fileInput.value = '';
    }

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length) selectFixture('');
    });

    function renderPicks(el, prints, selectFirst) {
      el.innerHTML = '';
      prints.forEach((p, i) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'pick' + (selectFirst && i === 0 ? ' selected' : '');
        btn.dataset.id = p.id;
        btn.innerHTML = `<img src="${p.thumb}" alt="${p.label}" /><div class="name">${p.label}</div><div class="hint">${p.hint || ''}</div>`;
        btn.addEventListener('click', () => selectFixture(p.id));
        el.appendChild(btn);
        if (selectFirst && i === 0) fixtureInput.value = p.id;
      });
    }

    async function loadDefaults() {
      const list = await fetch('/api/defaults').then(r => r.json());
      renderPicks(defaultsEl, list.prints || [], true);
      renderPicks(showcaseEl, list.showcase || [], false);
      renderPicks(hardEl, list.hard || [], false);
    }
    loadDefaults();

    function pollJob(jobId) {
      status.textContent = 'Job ' + jobId + ' started…';
      const poll = setInterval(async () => {
        const s = await fetch('/api/status?id=' + encodeURIComponent(jobId)).then(r => r.json());
        if (s.status === 'running' || s.status === 'queued') {
          status.textContent = s.step || 'Running…';
        } else if (s.status === 'done') {
          clearInterval(poll);
          status.textContent = 'Done.';
          go.disabled = false;
          const r = s.result || {};
          result.style.display = 'block';
          result.innerHTML =
            panel('Auto route', r.cv) +
            panel('SAM2 + QA', r.sam2);
        } else if (s.status === 'error') {
          clearInterval(poll);
          status.innerHTML = '<span class="err">' + (s.error || 'Error') + '</span>';
          go.disabled = false;
        }
      }, 1200);
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const mode = (form.querySelector('input[name=mode]:checked') || {}).value || 'cv';
      const fixtureId = fixtureInput.value;
      const hasFile = fileInput.files && fileInput.files.length > 0;
      if (!fixtureId && !hasFile) {
        status.innerHTML = '<span class="err">Pick a default print or upload a file.</span>';
        return;
      }
      go.disabled = true;
      result.style.display = 'none';
      result.innerHTML = '';
      status.textContent = 'Starting…';
      try {
        let res, data;
        if (hasFile) {
          const fd = new FormData();
          fd.append('file', fileInput.files[0]);
          fd.append('mode', mode);
          res = await fetch('/api/run', { method: 'POST', body: fd });
        } else {
          res = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fixture_id: fixtureId, mode }),
          });
        }
        data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        pollJob(data.job_id);
      } catch (err) {
        status.innerHTML = '<span class="err">' + err + '</span>';
        go.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self._send(code, raw, "application/json")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            weights = Path(SAM_MODEL)
            if not weights.is_file():
                weights = ROOT / SAM_MODEL
            self._json(200, {"ok": True, "sam_model": SAM_MODEL, "sam_weights": weights.is_file()})
            return

        if path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode("utf-8"))
            return

        if path == "/api/defaults":
            self._json(200, {"prints": _list_defaults(), "showcase": _list_showcase(), "hard": _list_hard()})
            return

        if path == "/api/status":
            qs = parse_qs(parsed.query)
            job_id = (qs.get("id") or [""])[0]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self._json(404, {"error": "unknown job"})
                return
            self._json(
                200,
                {
                    "id": job_id,
                    "status": job.get("status"),
                    "step": job.get("step"),
                    "error": job.get("error"),
                    "result": job.get("result"),
                },
            )
            return

        if path == "/api/documents":
            self._json(200, {"documents": layers_menu.list_documents(OUT_ROOT)})
            return

        # GET /api/documents/<id> or /api/documents/<id>/layers-menu
        if path.startswith("/api/documents/"):
            parts = [p for p in path[len("/api/documents/") :].split("/") if p]
            if not parts:
                self._json(200, {"documents": layers_menu.list_documents(OUT_ROOT)})
                return
            doc_id = parts[0]
            try:
                if len(parts) == 1:
                    doc = layers_menu.load_document(OUT_ROOT, doc_id)
                    if not doc:
                        self._json(404, {"error": "unknown document"})
                        return
                    self._json(200, doc)
                    return
                if parts[1] == "layers-menu":
                    self._json(200, layers_menu.layers_menu_view(OUT_ROOT, doc_id))
                    return
                if parts[1] == "motif-pack":
                    d = layers_menu.doc_dir(OUT_ROOT, doc_id)
                    zip_path = d / "motif_pack.zip"
                    layers_menu.build_motif_pack_zip(OUT_ROOT, doc_id, zip_path)
                    data = zip_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header(
                        "Content-Disposition",
                        f'attachment; filename="motif_pack_{doc_id}.zip"',
                    )
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except FileNotFoundError as exc:
                self._json(404, {"error": str(exc)})
                return
            self._json(404, {"error": "not found"})
            return

        # Default / hard print thumbs
        if path.startswith("/fixtures/"):
            rel = path[len("/fixtures/") :]
            file_path = _fixture_path(rel)
            if file_path is None:
                self._send(404, b"not found")
                return
            ctype = mimetypes.guess_type(str(file_path))[0] or "image/png"
            self._send(200, file_path.read_bytes(), ctype)
            return

        # Document assets: /documents/<id>/...
        if path.startswith("/documents/"):
            rel = path[len("/documents/") :]
            parts = [p for p in rel.split("/") if p]
            docs_root = (OUT_ROOT / "documents").resolve()
            file_path = (OUT_ROOT / "documents" / rel).resolve()
            try:
                file_path.relative_to(docs_root)
            except ValueError:
                self._send(403, b"forbidden")
                return
            # /documents/<id> or /documents/<id>/ → Layers Menu viewer
            if len(parts) == 1 and (rel.endswith("/") or file_path.is_dir() or not file_path.exists()):
                doc_id = parts[0]
                if not (docs_root / doc_id).is_dir():
                    self._send(404, b"unknown document")
                    return
                html = (WEB / "layers_menu.html").read_text(encoding="utf-8")
                html = html.replace("{{DOC_ID}}", doc_id)
                self._send(200, html.encode("utf-8"))
                return
            if file_path.is_dir():
                self._send(404, b"not found")
                return
            if not file_path.is_file():
                if file_path.name in ("editor.html", "editor.js"):
                    src = WEB / file_path.name
                    if src.exists():
                        ctype = mimetypes.guess_type(str(src))[0] or "text/html"
                        self._send(200, src.read_bytes(), ctype)
                        return
                self._send(404, b"not found")
                return
            ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            self._send(200, file_path.read_bytes(), ctype)
            return

        # Static job outputs: /jobs/<id>/<cv|sam2>/...
        if path.startswith("/jobs/"):
            rel = path[len("/jobs/") :]
            file_path = (OUT / rel).resolve()
            try:
                file_path.relative_to(OUT.resolve())
            except ValueError:
                self._send(403, b"forbidden")
                return
            # Always serve live editor assets from web/ so review UX fixes ship
            # without re-running the job (job-local copies stay for offline ZIP).
            if file_path.name in ("editor.html", "editor.js"):
                src = WEB / file_path.name
                if src.exists():
                    file_path = src
            if not file_path.is_file():
                self._send(404, b"not found")
                return
            ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            data = file_path.read_bytes()
            self._send(200, data, ctype)
            return

        if path.startswith("/web/"):
            file_path = (WEB / path[len("/web/") :]).resolve()
            try:
                file_path.relative_to(WEB.resolve())
            except ValueError:
                self._send(403, b"forbidden")
                return
            if file_path.is_file():
                ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
                self._send(200, file_path.read_bytes(), ctype)
                return

        self._send(404, b"not found")

    def _start_job(self, image_path: Path, mode: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        job_dir = OUT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        dest = job_dir / f"input{image_path.suffix.lower() or '.png'}"
        if image_path.resolve() != dest.resolve():
            shutil.copy2(image_path, dest)
            image_path = dest
        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "queued",
                "step": "queued",
                "input": str(image_path),
                "mode": mode,
            }
        t = threading.Thread(target=_run_job, args=(job_id, image_path, mode), daemon=True)
        t.start()
        return job_id

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # --- Layers Menu document APIs ---
        if path == "/api/documents" or path == "/api/documents/":
            try:
                payload = _read_json_body(self)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            doc = layers_menu.create_document(
                OUT_ROOT,
                name=str(payload.get("name") or "Untitled print"),
                width=int(payload.get("width") or 0),
                height=int(payload.get("height") or 0),
                source=payload.get("source"),
            )
            self._json(200, doc)
            return

        if path == "/api/documents/from-job":
            try:
                payload = _read_json_body(self)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            job_id = payload.get("job_id")
            pipeline = str(payload.get("pipeline") or "cv")
            scene_path = payload.get("scene_path")
            scene_dir: Path | None = None
            if scene_path:
                scene_dir = Path(str(scene_path))
                if not scene_dir.is_absolute():
                    scene_dir = (OUT_ROOT / scene_dir).resolve()
            elif job_id:
                candidates = [
                    OUT / str(job_id) / pipeline,
                    OUT_ROOT / str(job_id) / pipeline,
                    OUT_ROOT / "smoke_handoff" / pipeline if str(job_id) == "smoke" else None,
                ]
                for c in candidates:
                    if c is not None and (c / "scene.json").is_file():
                        scene_dir = c
                        break
                if scene_dir is None:
                    scene_dir = OUT / str(job_id) / pipeline
            else:
                self._json(400, {"error": "job_id or scene_path required"})
                return
            if not (scene_dir / "scene.json").is_file():
                self._json(404, {"error": f"no scene at {scene_dir}"})
                return
            try:
                doc = layers_menu.create_document_from_scene(
                    OUT_ROOT,
                    scene_dir,
                    name=payload.get("name"),
                    pipeline=pipeline,
                    doc_id=payload.get("doc_id"),
                )
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": str(exc)})
                return
            prop = doc.get("active_proposal") or {}
            self._json(
                200,
                {
                    "id": doc["id"],
                    "document": doc,
                    "editor": prop.get("document_editor") or prop.get("editor"),
                    "layers_menu": f"/documents/{doc['id']}/",
                    "api": f"/api/documents/{doc['id']}/layers-menu",
                },
            )
            return

        if path.startswith("/api/documents/"):
            parts = [p for p in path[len("/api/documents/") :].split("/") if p]
            if len(parts) < 2:
                self._json(404, {"error": "not found"})
                return
            doc_id, action = parts[0], parts[1]
            try:
                payload = _read_json_body(self)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            try:
                if action == "attach-proposal":
                    job_id = payload.get("job_id")
                    pipeline = str(payload.get("pipeline") or "cv")
                    if job_id:
                        scene_dir = OUT / str(job_id) / pipeline
                    elif payload.get("scene_path"):
                        scene_dir = Path(str(payload["scene_path"]))
                    else:
                        self._json(400, {"error": "job_id or scene_path required"})
                        return
                    doc = layers_menu.attach_proposal_from_scene(
                        OUT_ROOT, doc_id, scene_dir, pipeline=pipeline
                    )
                    self._json(200, doc)
                    return
                if action == "accept":
                    motifs = payload.get("motifs") or payload.get("layers") or []
                    if not isinstance(motifs, list) or not motifs:
                        self._json(400, {"error": "motifs[] required"})
                        return
                    scene_dir = None
                    if payload.get("scene_path"):
                        scene_dir = Path(str(payload["scene_path"]))
                    elif payload.get("job_id"):
                        scene_dir = OUT / str(payload["job_id"]) / str(
                            payload.get("pipeline") or "cv"
                        )
                    result = layers_menu.accept_motifs(
                        OUT_ROOT,
                        doc_id,
                        motifs,
                        scene_dir=scene_dir,
                        proposal_id=payload.get("proposal_id"),
                        replace=bool(payload.get("replace")),
                    )
                    self._json(200, result)
                    return
                if action == "remove":
                    ids = payload.get("layer_ids") or payload.get("ids") or []
                    doc = layers_menu.remove_layers(OUT_ROOT, doc_id, list(ids))
                    self._json(200, doc)
                    return
                if action == "motif-pack":
                    d = layers_menu.doc_dir(OUT_ROOT, doc_id)
                    zip_path = d / "motif_pack.zip"
                    layers_menu.build_motif_pack_zip(OUT_ROOT, doc_id, zip_path)
                    self._json(
                        200,
                        {
                            "url": f"/api/documents/{doc_id}/motif-pack",
                            "path": str(zip_path),
                            "count": len((layers_menu.load_document(OUT_ROOT, doc_id) or {}).get("layers") or []),
                        },
                    )
                    return
            except FileNotFoundError as exc:
                self._json(404, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": f"{exc}\n{traceback.format_exc()}"})
                return
            self._json(404, {"error": f"unknown action {action}"})
            return

        if path != "/api/run":
            self._json(404, {"error": "not found"})
            return

        ctype = self.headers.get("Content-Type", "")
        mode = "cv"
        image_path: Path | None = None

        if "application/json" in ctype:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid JSON"})
                return
            mode = str(payload.get("mode") or "cv")
            if mode not in ("cv", "sam2", "both"):
                mode = "cv"
            fixture_id = payload.get("fixture_id")
            if not fixture_id:
                self._json(400, {"error": "missing fixture_id"})
                return
            image_path = _fixture_path(str(fixture_id))
            if image_path is None:
                self._json(404, {"error": f"unknown fixture: {fixture_id}"})
                return
            job_id = self._start_job(image_path, mode)
            self._json(200, {"job_id": job_id, "mode": mode, "fixture_id": fixture_id})
            return

        if "multipart/form-data" not in ctype:
            self._json(400, {"error": "expected multipart form or JSON"})
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": ctype,
            },
        )
        mode = str(form.getvalue("mode") or "cv")
        if mode not in ("cv", "sam2", "both"):
            mode = "cv"

        fixture_id = form.getvalue("fixture_id")
        if fixture_id and (not form["file"].filename if "file" in form else True):
            image_path = _fixture_path(str(fixture_id))
            if image_path is None:
                self._json(404, {"error": f"unknown fixture: {fixture_id}"})
                return
            job_id = self._start_job(image_path, mode)
            self._json(200, {"job_id": job_id, "mode": mode, "fixture_id": str(fixture_id)})
            return

        if "file" not in form:
            self._json(400, {"error": "missing file or fixture_id"})
            return
        file_item = form["file"]
        if not getattr(file_item, "file", None) or not getattr(file_item, "filename", None):
            self._json(400, {"error": "invalid file"})
            return

        job_id = uuid.uuid4().hex[:12]
        job_dir = OUT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(getattr(file_item, "filename", "") or "").suffix.lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        image_path = job_dir / f"input{ext}"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file_item.file, f)

        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "queued",
                "step": "queued",
                "input": str(image_path),
                "mode": mode,
            }
        t = threading.Thread(target=_run_job, args=(job_id, image_path, mode), daemon=True)
        t.start()
        self._json(200, {"job_id": job_id, "mode": mode})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if not path.startswith("/api/documents/"):
            self._json(404, {"error": "not found"})
            return
        parts = [p for p in path[len("/api/documents/") :].split("/") if p]
        if len(parts) != 1:
            self._json(404, {"error": "not found"})
            return
        doc_id = parts[0]
        d = layers_menu.doc_dir(OUT_ROOT, doc_id)
        if not d.exists():
            self._json(404, {"error": "unknown document"})
            return
        shutil.rmtree(d)
        self._json(200, {"deleted": doc_id})


def _preload_sam() -> None:
    """Warm SAM2 in the background after bind."""
    try:
        from sam_refine import _get_sam

        model = _get_sam(SAM_MODEL)
        print("SAM2 ready" if model is not None else "SAM2 preload skipped")
    except Exception as exc:  # noqa: BLE001
        print(f"SAM2 preload failed: {exc}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    layers_menu.documents_root(OUT_ROOT)
    if not (WEB / "editor.html").exists():
        print("WARNING: web/editor.html missing")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    if os.environ.get("PRELOAD_SAM", "0") == "1":
        threading.Thread(target=_preload_sam, daemon=True).start()
    print(f"Auto Layer POC → http://{HOST}:{PORT}")
    print(f"Default prints: {len(_list_defaults())} available under /fixtures/")
    print("Layers Menu API: GET/POST /api/documents …")
    print("Pick a print in the UI, or upload your own.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Garment segmentation comparison grid: dismantle SAM / SAM2 / alternatives.

Methods:
  - dismantle_sam   SageMaker classic SAM auto-masks (production parts)
  - sam2            Ultralytics SAM2 bbox+point (Docker worker)
  - mobile_sam      Ultralytics MobileSAM bbox+point (Docker worker)
  - fastsam         FastSAM everything mode (Docker worker)
  - yolo_seg        YOLO11n-seg (Docker worker; COCO classes)
  - gpt25_flare     gpt-image-2.5-flare → BW mask edit
  - gpt25_sunburst  gpt-image-2.5-sunburst → BW mask edit
  - gemini_box_matte Gemini Flash boxes + Lab matte
  - cv_silhouette   flood-fill / Lab FG silhouette

Usage:
  .venv/bin/python scripts/bakeoff_garment_seg_grid.py
  .venv/bin/python scripts/bakeoff_garment_seg_grid.py --append-sam
      # keep existing API results; only add/refresh SAM-family columns via Docker
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "garment_seg_grid"
MAX_SIDE = 1024
BUCKET = os.environ.get("GARMENT_SEG_BUCKET", "makethedot-originals-test")
PREFIX = "garment-seg-bakeoff"

from dotenv import load_dotenv

for env_path in (
    ROOT / ".env",
    ROOT.parent / ".env",
    ROOT.parent / "scraper" / ".env",
    ROOT.parent / "consistency-eval-app" / ".env.local",
):
    load_dotenv(env_path)

# Do NOT load sketch-adherence AWS keys — they override ~/.aws and break SageMaker/S3.
# Prefer the default AWS CLI credential chain for dismantle.
for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    # If dotenv injected stale keys, drop them so aws CLI uses ~/.aws/credentials.
    if os.environ.get(k) and not os.environ.get("GARMENT_SEG_KEEP_ENV_AWS"):
        # Only clear if they look like they came from project .env pollution during this script.
        pass

def _aws_env() -> dict[str, str]:
    """Child-process env for aws CLI: strip project-dotenv AWS keys."""
    env = os.environ.copy()
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        env.pop(k, None)
    env.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    env.setdefault("AWS_REGION", "us-east-1")
    return env

# Diverse garment cases: flat / photo / sketch
CASES = [
    {
        "id": "knit_flat_clean",
        "path": ROOT.parent / "knit-application-prototype/out/garment_clean.png",
        "hint": "knit flat (clean)",
    },
    {
        "id": "knit_flat_pockets",
        "path": ROOT.parent / "knit-application-prototype/out/garment_with_pockets.png",
        "hint": "knit flat + pockets",
    },
    {
        "id": "hoodie_sketch",
        "path": ROOT.parent / "sketches/crop-hoodie-sketch.png",
        "hint": "hoodie tech sketch",
    },
    {
        "id": "onmodel_lucille",
        "path": ROOT.parent / "dot-shot-tryon-share-20260714/05_lucille/02_garment.png",
        "hint": "on-model / product",
    },
    {
        "id": "cardigan_vneck",
        "path": ROOT.parent
        / "knit-application-prototype/out/yoke_sketches/yoke_02_circular_cardigan.png",
        "hint": "cardigan flat sketch",
    },
]


def _resize(im: Image.Image) -> Image.Image:
    if max(im.size) <= MAX_SIDE:
        return im
    s = MAX_SIDE / max(im.size)
    return im.resize((int(im.width * s), int(im.height * s)), Image.Resampling.LANCZOS)


def _overlay(image: Image.Image, union: np.ndarray, color=(255, 60, 60)) -> Image.Image:
    base = image.convert("RGBA")
    tint = np.zeros((*union.shape, 4), dtype=np.uint8)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    tint[..., 3] = (np.clip(union.astype(np.float32) * 0.5, 0, 180)).astype(np.uint8)
    return Image.alpha_composite(base, Image.fromarray(tint, "RGBA")).convert("RGB")


def _colored_parts(image: Image.Image, masks: list[np.ndarray]) -> Image.Image:
    """Tint each part mask a different color for dismantle visualization."""
    base = np.array(image.convert("RGB")).astype(np.float32)
    palette = [
        (255, 70, 70),
        (70, 140, 255),
        (60, 200, 120),
        (255, 200, 60),
        (200, 80, 220),
        (80, 220, 220),
        (255, 140, 60),
        (160, 160, 255),
    ]
    out = base.copy()
    for i, m in enumerate(masks):
        if m.shape[:2] != out.shape[:2]:
            m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = palette[i % len(palette)]
        alpha = (m.astype(np.float32) / 255.0)[..., None] * 0.55
        color = np.array(c, dtype=np.float32).reshape(1, 1, 3)
        out = out * (1 - alpha) + color * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _metrics(union: np.ndarray, n_parts: int) -> dict:
    fg = float((union > 40).mean())
    return {
        "parts": n_parts,
        "fg_frac": round(fg, 4),
        "ok": n_parts > 0 and fg > 0.005,
    }


def _result(method: str, sec: float, meta: dict, error: str | None = None) -> dict:
    return {
        "method": method,
        "sec": round(sec, 2),
        "parts": meta.get("parts", 0),
        "fg_frac": meta.get("fg_frac", 0.0),
        "ok": bool(meta.get("ok")) and error is None,
        "error": error,
    }


# ---------- CV silhouette ----------
def run_cv(image: Image.Image, out: Path) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    rgb = np.array(image.convert("RGB"))
    # Prefer alpha if present
    if image.mode == "RGBA":
        a = np.array(image.split()[-1])
        if float((a > 10).mean()) > 0.01 and float((a < 250).mean()) > 0.01:
            union = a
            n, _, st, _ = cv2.connectedComponentsWithStats((union > 40).astype(np.uint8), 8)
            parts = max(0, n - 1)
            Image.fromarray(union).save(out / "union_mask.png")
            _overlay(image.convert("RGB"), union).save(out / "overlay.png")
            return _result("cv_silhouette", time.time() - t0, _metrics(union, parts))

    # Lab distance from border-estimated bg + flood from corners
    h, w = rgb.shape[:2]
    border = np.concatenate(
        [rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3), rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    ink = (dist > 12).astype(np.uint8) * 255
    # Flood fill background from corners on inverted
    ff = ink.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if ff[seed[1], seed[0]] < 40:
            cv2.floodFill(ff, mask, seed, 128)
    # Garment = ink that wasn't flooded as bg... simpler: largest CC of ink
    n, labc, st, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    union = np.zeros_like(ink)
    parts = 0
    if n > 1:
        order = np.argsort(-st[1:, cv2.CC_STAT_AREA])
        keep = order[: min(12, len(order))]
        for idx in keep:
            union = np.maximum(union, ((labc == idx + 1) * 255).astype(np.uint8))
        parts = len(keep)
    Image.fromarray(union).save(out / "union_mask.png")
    _overlay(image.convert("RGB"), union).save(out / "overlay.png")
    return _result("cv_silhouette", time.time() - t0, _metrics(union, parts))


# ---------- GPT image mask ----------
def _gpt_mask(image: Image.Image, model: str, prompt: str) -> Image.Image:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    buf = BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    res = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {api_key}"},
        data={"model": model, "prompt": prompt, "size": "1024x1024", "quality": "high"},
        files={"image": ("in.png", buf.getvalue(), "image/png")},
        timeout=300,
    )
    payload = res.json()
    if not res.ok:
        raise RuntimeError(payload.get("error", {}).get("message") or res.text[:300])
    raw = base64.b64decode(payload["data"][0]["b64_json"])
    return Image.open(BytesIO(raw)).convert("RGB")


GPT_PROMPT = """Create a pure black-and-white garment segmentation mask of this image.
WHITE = garment / clothing regions (all parts: body, sleeves, hood, collar, pockets as ONE garment silhouette unless clearly separate floating pieces).
BLACK = background only.
No colors, labels, shadows, or gray fills. Same framing as input. Output ONLY the mask."""


def run_gpt(image: Image.Image, out: Path, method: str, model: str) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    try:
        gen = _gpt_mask(image, model, GPT_PROMPT)
        gen.save(out / "gpt_raw.png")
        arr = np.array(gen.convert("L").resize(image.size, Image.Resampling.BILINEAR))
        union = ((arr > 128).astype(np.uint8) * 255)
        n, _, _, _ = cv2.connectedComponentsWithStats((union > 40).astype(np.uint8), 8)
        Image.fromarray(union).save(out / "union_mask.png")
        _overlay(image.convert("RGB"), union).save(out / "overlay.png")
        return _result(method, time.time() - t0, _metrics(union, max(0, n - 1)))
    except Exception as exc:  # noqa: BLE001
        return _result(method, time.time() - t0, {"parts": 0, "fg_frac": 0, "ok": False}, error=str(exc)[:300])


# ---------- Gemini boxes + matte ----------
def run_gemini(image: Image.Image, out: Path) -> dict:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return _result("gemini_box_matte", 0, {"parts": 0, "fg_frac": 0, "ok": False}, error="no GOOGLE_API_KEY")
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = (
            "Detect garment parts in this fashion image (body, left sleeve, right sleeve, "
            "collar/hood, pockets, hem if distinct). Return JSON list "
            '[{"box_2d":[ymin,xmin,ymax,xmax],"label":"..."}] coords 0-1000. At most 12.'
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
        try:
            cfg = types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        except Exception:  # noqa: BLE001
            cfg = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        resp = client.models.generate_content(
            model=os.environ.get("AI_SEG_MODEL", "gemini-2.5-flash"),
            contents=[prompt, part],
            config=cfg,
        )
        text = (resp.text or "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\[[\s\S]*\]", text)
            data = json.loads(m.group(0)) if m else []
        if isinstance(data, dict):
            data = data.get("boxes") or data.get("parts") or []
        boxes = [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []

        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        border = np.concatenate(
            [rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3), rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)]
        )
        bg = np.median(border, axis=0).astype(np.float32)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        dist = np.linalg.norm(lab - bg_lab, axis=2)
        union = np.zeros((h, w), dtype=np.uint8)
        part_masks: list[np.ndarray] = []
        for item in boxes:
            box = item.get("box_2d")
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            y0, x0, y1, x1 = [float(v) / 1000.0 for v in box]
            xa, ya = max(0, int(min(x0, x1) * w)), max(0, int(min(y0, y1) * h))
            xb, yb = min(w, int(max(x0, x1) * w)), min(h, int(max(y0, y1) * h))
            if xb <= xa + 1 or yb <= ya + 1:
                continue
            local = (np.clip((dist[ya:yb, xa:xb] - 10) / 18, 0, 1) * 255).astype(np.uint8)
            full = np.zeros((h, w), dtype=np.uint8)
            full[ya:yb, xa:xb] = local
            union = np.maximum(union, full)
            part_masks.append(full)
        Image.fromarray(union).save(out / "union_mask.png")
        (_colored_parts(image, part_masks) if part_masks else _overlay(image.convert("RGB"), union)).save(
            out / "overlay.png"
        )
        (out / "labels.json").write_text(json.dumps(boxes, indent=2)[:50000])
        return _result("gemini_box_matte", time.time() - t0, _metrics(union, len(part_masks)))
    except Exception as exc:  # noqa: BLE001
        return _result("gemini_box_matte", time.time() - t0, {"parts": 0, "fg_frac": 0, "ok": False}, error=str(exc)[:300])


# ---------- Dismantle SAM (SageMaker) ----------
def _upload_presign(local_png: Path, key: str) -> str:
    env = _aws_env()
    subprocess.check_call(
        ["aws", "s3", "cp", str(local_png), f"s3://{BUCKET}/{key}"],
        stdout=subprocess.DEVNULL,
        env=env,
    )
    url = subprocess.check_output(
        ["aws", "s3", "presign", f"s3://{BUCKET}/{key}", "--expires-in", "3600"],
        text=True,
        env=env,
    ).strip()
    return url


def run_dismantle(image: Image.Image, out: Path, case_id: str) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    tmp = out / "_upload.png"
    image.save(tmp)
    try:
        key = f"{PREFIX}/{int(time.time())}_{case_id}.png"
        url = _upload_presign(tmp, key)
        payload_path = out / "sm_payload.json"
        payload_path.write_text(json.dumps({"instances": [{"imageUrl": url}]}))
        resp_path = out / "sm_resp.json"
        subprocess.check_call(
            [
                "aws",
                "sagemaker-runtime",
                "invoke-endpoint",
                "--endpoint-name",
                "dismantle-sam-endpoint",
                "--content-type",
                "application/json",
                "--accept",
                "application/json",
                "--body",
                f"fileb://{payload_path}",
                str(resp_path),
            ],
            stdout=subprocess.DEVNULL,
            env=_aws_env(),
        )
        data = json.loads(resp_path.read_text())
        pred = (data.get("predictions") or [None])[0]
        if not pred or pred.get("content_type") != "application/zip":
            raise RuntimeError(f"unexpected response: {str(data)[:300]}")
        zdir = out / "masks"
        if zdir.exists():
            shutil.rmtree(zdir)
        zdir.mkdir()
        zf = zipfile.ZipFile(BytesIO(base64.b64decode(pred["masks"])))
        zf.extractall(zdir)
        names = sorted(zf.namelist())
        masks: list[np.ndarray] = []
        union = np.zeros((image.height, image.width), dtype=np.uint8)
        for name in names:
            m = Image.open(zdir / name).convert("L")
            if m.size != image.size:
                m = m.resize(image.size, Image.Resampling.NEAREST)
            arr = np.array(m)
            masks.append(arr)
            union = np.maximum(union, arr)
        Image.fromarray(union).save(out / "union_mask.png")
        _colored_parts(image, masks).save(out / "overlay.png")
        return _result("dismantle_sam", time.time() - t0, _metrics(union, len(masks)))
    except Exception as exc:  # noqa: BLE001
        return _result("dismantle_sam", time.time() - t0, {"parts": 0, "fg_frac": 0, "ok": False}, error=str(exc)[:400])


DOCKER_IMAGE = os.environ.get("SAM_WORKER_IMAGE", "mtd-sam-worker:cpu")
SAM_METHODS = ("sam2", "mobile_sam", "fastsam", "yolo_seg")


def run_sam_docker(image: Image.Image, out: Path, method: str) -> dict:
    """Run SAM2 / MobileSAM / FastSAM / YOLO-seg via Docker (host torch hangs)."""
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    # Workdir layout for container mounts
    work = out / "_docker"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    in_path = work / "input.png"
    image.save(in_path)
    out_c = work / "out"
    out_c.mkdir()

    weights_mount = []
    weights_arg = []
    if method == "sam2":
        w = ROOT / "sam2_b.pt"
        if w.exists():
            weights_mount = ["-v", f"{w}:/weights/sam2_b.pt:ro"]
            weights_arg = ["--weights", "/weights/sam2_b.pt"]

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{work}:/data",
        *weights_mount,
        DOCKER_IMAGE,
        "--method",
        method,
        "--image",
        "/data/input.png",
        "--out",
        "/data/out",
        *weights_arg,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "docker failed")[-400:]
            return _result(method, time.time() - t0, {"parts": 0, "fg_frac": 0, "ok": False}, error=err)
        # Copy artifacts up
        for name in ("union_mask.png", "overlay.png", "result.json"):
            src = out_c / name
            if src.exists():
                shutil.copy2(src, out / name)
        meta = {"parts": 0, "fg_frac": 0.0, "ok": False}
        if (out / "result.json").exists():
            payload = json.loads((out / "result.json").read_text())
            meta = {
                "parts": int(payload.get("parts") or 0),
                "fg_frac": float(payload.get("fg_frac") or 0),
                "ok": bool(payload.get("ok")),
            }
            if payload.get("error"):
                return _result(method, time.time() - t0, meta, error=str(payload["error"])[:400])
        # Prefer wall time including docker overhead
        return _result(method, time.time() - t0, meta)
    except Exception as exc:  # noqa: BLE001
        return _result(method, time.time() - t0, {"parts": 0, "fg_frac": 0, "ok": False}, error=str(exc)[:400])


def _write_gallery(rows: list[dict], method_names: list[str]) -> None:
    rollup: dict[str, Any] = {}
    for name in method_names:
        stats = [r["methods"][name] for r in rows if name in r["methods"]]
        if not stats:
            continue
        ok = [s for s in stats if s["ok"]]
        rollup[name] = {
            "n": len(stats),
            "ok_rate": round(len(ok) / len(stats), 3),
            "latency_p50_s": round(float(np.median([s["sec"] for s in stats])), 2),
            "latency_mean_s": round(float(np.mean([s["sec"] for s in stats])), 2),
            "parts_mean": round(float(np.mean([s["parts"] for s in stats])), 1),
            "fg_mean": round(float(np.mean([s["fg_frac"] for s in stats])), 3),
            "errors": [s["error"] for s in stats if s.get("error")][:2],
        }
    (OUT / "rollup.json").write_text(json.dumps(rollup, indent=2))
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))

    cards: list[str] = []
    for row in rows:
        figs = [
            f'<figure><img src="{row["id"]}/original.png"/><figcaption>original<br/>{row["hint"]}</figcaption></figure>'
        ]
        for name in method_names:
            st = row["methods"].get(name)
            if not st:
                continue
            figs.append(
                f'<figure><img src="{row["id"]}/{name}/overlay.png" '
                f'onerror="this.src=\'{row["id"]}/{name}/union_mask.png\'"/>'
                f'<figcaption>{name}<br/>{st["sec"]}s · {st["parts"]} parts · fg={st["fg_frac"]:.2f}</figcaption></figure>'
            )
        n = max(3, len(figs))
        cards.append(
            f'<div class="card"><h3>{row["id"]}</h3>'
            f'<div class="row" style="grid-template-columns:repeat({min(n,5)},minmax(120px,1fr))">'
            f'{"".join(figs)}</div></div>'
        )

    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>Garment seg — SAM / SAM2 vs alternatives</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#f3f1ec;color:#1a1a1a}}
h1{{margin:0 0 8px}} .sub{{color:#666;max-width:980px;margin-bottom:16px}}
.card{{background:#fff;border-radius:12px;padding:16px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.row{{display:grid;gap:8px}}
img{{width:100%;border-radius:8px;background:#eee;aspect-ratio:1;object-fit:contain}}
figcaption{{font-size:11px;color:#555;margin-top:4px;line-height:1.35}}
pre{{background:#fff;padding:12px;border-radius:8px;overflow:auto}}
</style></head><body>
<h1>Garment segmentation grid</h1>
<p class="sub">Same garments × <b>dismantle SAM</b> (classic SAM, production parts) vs <b>SAM2 / MobileSAM / FastSAM / YOLO-seg</b>
vs GPT image 2.5 mask edits, Gemini box+matte, CV silhouette.</p>
<h2>Rollup</h2><pre>{json.dumps(rollup, indent=2)}</pre>
{''.join(cards)}
</body></html>"""
    (OUT / "compare.html").write_text(html)
    print("\nROLLUP", flush=True)
    print(json.dumps(rollup, indent=2), flush=True)
    print(f"\ngallery → {OUT / 'compare.html'}", flush=True)


def main() -> None:
    append_sam = "--append-sam" in sys.argv
    cases = [c for c in CASES if c["path"].exists()]
    if not cases:
        raise SystemExit("No garment fixtures found")
    if "--smoke" in sys.argv:
        cases = cases[:2]

    api_methods = [
        ("dismantle_sam", lambda im, o, cid: run_dismantle(im, o, cid)),
        ("cv_silhouette", lambda im, o, cid: run_cv(im, o)),
        ("gemini_box_matte", lambda im, o, cid: run_gemini(im, o)),
        ("gpt25_flare", lambda im, o, cid: run_gpt(im, o, "gpt25_flare", "gpt-image-2.5-flare")),
        ("gpt25_sunburst", lambda im, o, cid: run_gpt(im, o, "gpt25_sunburst", "gpt-image-2.5-sunburst")),
    ]
    sam_methods = [(m, lambda im, o, cid, _m=m: run_sam_docker(im, o, _m)) for m in SAM_METHODS]

    if append_sam:
        if not (OUT / "summary.json").exists():
            raise SystemExit("--append-sam requires an existing out/garment_seg_grid/summary.json")
        rows = json.loads((OUT / "summary.json").read_text())
        print(f"append-sam: refreshing {SAM_METHODS} on {len(rows)} cases", flush=True)
        for i, row in enumerate(rows, 1):
            case_dir = OUT / row["id"]
            orig = case_dir / "original.png"
            if not orig.exists():
                print(f"  skip {row['id']} (no original)", flush=True)
                continue
            flat = Image.open(orig).convert("RGB")
            print(f"\n=== [{i}/{len(rows)}] {row['id']} — SAM family ===", flush=True)
            for name, fn in sam_methods:
                st = fn(flat, case_dir / name, row["id"])
                row["methods"][name] = st
                status = "OK" if st["ok"] else "FAIL"
                print(
                    f"  {name:18} parts={st['parts']:3} fg={st['fg_frac']:.3f} {st['sec']:6.1f}s {status}"
                    + (f"  {st['error'][:100]}" if st.get("error") else ""),
                    flush=True,
                )
        method_names = list(dict.fromkeys([*api_methods[0:0], *rows[0]["methods"].keys()]))
        # stable order
        preferred = [
            "dismantle_sam",
            "sam2",
            "mobile_sam",
            "fastsam",
            "yolo_seg",
            "cv_silhouette",
            "gemini_box_matte",
            "gpt25_flare",
            "gpt25_sunburst",
        ]
        method_names = [m for m in preferred if m in rows[0]["methods"]] + [
            m for m in rows[0]["methods"] if m not in preferred
        ]
        _write_gallery(rows, method_names)
        return

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    methods = api_methods + sam_methods
    print(f"cases={len(cases)} out={OUT}", flush=True)
    rows: list[dict] = []

    for i, case in enumerate(cases, 1):
        im = _resize(Image.open(case["path"]).convert("RGBA"))
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
        print(f"\n=== [{i}/{len(cases)}] {case['id']} — {case['hint']} ===", flush=True)
        case_dir = OUT / case["id"]
        case_dir.mkdir(parents=True)
        flat.save(case_dir / "original.png")
        row: dict[str, Any] = {"id": case["id"], "hint": case["hint"], "methods": {}}

        for name, fn in methods:
            out = case_dir / name
            st = fn(flat, out, case["id"])
            row["methods"][name] = st
            status = "OK" if st["ok"] else "FAIL"
            print(
                f"  {name:18} parts={st['parts']:3} fg={st['fg_frac']:.3f} {st['sec']:6.1f}s {status}"
                + (f"  {st['error'][:100]}" if st.get("error") else ""),
                flush=True,
            )
        rows.append(row)

    _write_gallery(rows, [m for m, _ in methods])


if __name__ == "__main__":
    main()

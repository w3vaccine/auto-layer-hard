#!/usr/bin/env python3
"""Multi-model SAM-replacement bakeoff (API + local).

Compares candidates that could replace SAM2 for editor mask tasks:
  - cv                 OpenCV ink components (baseline local)
  - sam2               Ultralytics SAM2 refine on CV instances (if torch loads)
  - mobile_sam         Ultralytics MobileSAM refine (if available)
  - fastsam            FastSAM everything mode (if available)
  - yolo_seg           YOLO11n-seg (class instance, if available)
  - gemini_box_matte   Gemini 2.5 Flash boxes → ink matte
  - gpt25_flare_mask   gpt-image-2.5-flare images/edits → B/W mask
  - gpt25_sunburst_mask gpt-image-2.5-sunburst images/edits → B/W mask

Usage:
  .venv/bin/python scripts/bakeoff_sam_alternatives.py --quick
  .venv/bin/python scripts/bakeoff_sam_alternatives.py --models cv,gemini_box_matte,gpt25_flare_mask
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

for env_path in (
    ROOT / ".env",
    ROOT.parent / ".env",
    ROOT.parent / "scraper" / ".env",
    ROOT.parent / "consistency-eval-app" / ".env.local",
):
    load_dotenv(env_path)

from segment import (  # noqa: E402
    chroma_distance,
    discover_by_components,
    estimate_background_color,
    lab_distance,
    overlay_instances,
)

OUT = ROOT / "out" / "sam_alt_bakeoff"
MAX_SIDE = 1024

CASES = [
    ("showcase/s01_garden_roses.png", "floral_large", "motif"),
    ("showcase/s02_butterflies.png", "insects", "motif"),
    ("showcase/s03_geo_diamonds.png", "geometric", "motif"),
    ("showcase/s12_color_blobs.png", "abstract_blobs", "motif"),
    ("showcase/s14_citrus_slices.png", "fruit", "motif"),
    ("03-paisley.png", "interlocking", "motif"),
    ("hard/h06_thin_lines.png", "thin_lines", "motif"),
    ("hard/h14_one_giant.png", "one_giant", "subject"),
    ("06-seashells.png", "shells_multi", "subject"),
    ("showcase/s15_mixed_bouquet.png", "bouquet_parts", "parts"),
]

GPT_MASK_PROMPT = """Create a pure segmentation mask image of this print/photo.

Rules:
- WHITE (#FFFFFF) = every distinct foreground motif / subject / part to keep
- BLACK (#000000) = background only
- No gray, no colors, no labels, no borders, no shadows
- Preserve exact silhouette edges of each object
- Same framing and resolution as the input
- Output ONLY the mask image"""

GPT_SUBJECT_PROMPT = """Create a pure foreground cutout mask.

Rules:
- WHITE = main subject(s) only
- BLACK = background
- Soft edges OK but prefer crisp silhouettes
- No colors, labels, or decorative elements
- Same framing as input
- Output ONLY the black-and-white mask"""


def _resize(im: Image.Image) -> Image.Image:
    if max(im.size) <= MAX_SIDE:
        return im
    s = MAX_SIDE / max(im.size)
    return im.resize((int(im.width * s), int(im.height * s)), Image.Resampling.LANCZOS)


def _resolve(rel: str) -> Path:
    p = ROOT / "fixtures" / rel
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _ink(rgb: np.ndarray) -> np.ndarray:
    bg = estimate_background_color(rgb)
    lab = lab_distance(rgb, bg)
    chr_d = chroma_distance(rgb, bg)
    lab_frac = float((lab > 14).mean())
    chr_frac = float((chr_d > 8).mean())
    if lab_frac > 0.42 and chr_frac < lab_frac * 0.55:
        return chr_d > 8
    return lab > 14


def _coverage(union: np.ndarray, rgb: np.ndarray) -> float:
    ink = _ink(rgb)
    if not ink.any():
        return 0.0
    return float(((union > 20) & ink).sum() / ink.sum())


def _save_union(out: Path, union: np.ndarray, overlay: Image.Image | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(union.astype(np.uint8)).save(out / "union_mask.png")
    if overlay is not None:
        overlay.save(out / "seg_overlay.png")


def _overlay_from_union(image: Image.Image, union: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    tint = np.zeros((union.shape[0], union.shape[1], 4), dtype=np.uint8)
    tint[..., 0] = 255
    tint[..., 1] = 60
    tint[..., 2] = 60
    tint[..., 3] = (np.clip(union.astype(np.float32) * 0.45, 0, 255)).astype(np.uint8)
    return Image.alpha_composite(base, Image.fromarray(tint, "RGBA")).convert("RGB")


def _result(
    method: str,
    *,
    layers: int,
    metric: float,
    sec: float,
    ok: bool,
    notes: str = "",
    error: str | None = None,
) -> dict:
    return {
        "method": method,
        "layers": layers,
        "metric": round(float(metric), 4),
        "sec": round(float(sec), 2),
        "ok": bool(ok and error is None),
        "reasonable": bool(ok and metric > 0.02),
        "notes": notes,
        "error": error,
    }


# ---------- CV ----------
def run_cv(image: Image.Image, out: Path, task: str) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    if task == "motif":
        inst = discover_by_components(image, max_instances=200)
        for i, x in enumerate(inst, 1):
            x.id = f"m{i:03d}"
        overlay_instances(image, inst).save(out / "seg_overlay.png")
        union = np.zeros((image.height, image.width), dtype=np.uint8)
        for x in inst:
            union = np.maximum(union, x.mask)
        Image.fromarray(union).save(out / "union_mask.png")
        return _result(
            "cv",
            layers=len(inst),
            metric=_coverage(union, np.array(image)),
            sec=time.time() - t0,
            ok=len(inst) > 0,
            notes="ink CCs",
        )
    rgb = np.array(image)
    ink = (_ink(rgb).astype(np.uint8) * 255)
    n, lab, st, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    union = np.zeros_like(ink)
    layers = 0
    if n > 1:
        largest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        union = ((lab == largest) * 255).astype(np.uint8)
        layers = 1 if task == "subject" else min(8, n - 1)
        if task == "parts" and n > 2:
            order = np.argsort(-st[1:, cv2.CC_STAT_AREA])[:8]
            union = np.zeros_like(ink)
            for idx in order:
                union = np.maximum(union, ((lab == idx + 1) * 255).astype(np.uint8))
            layers = len(order)
    _save_union(out, union, _overlay_from_union(image, union))
    return _result(
        "cv",
        layers=layers,
        metric=float((union > 40).mean()) if task != "motif" else _coverage(union, rgb),
        sec=time.time() - t0,
        ok=layers > 0,
        notes="largest/top CCs",
    )


# ---------- Ultralytics family ----------
_TORCH_OK: bool | None = None


def _torch_ok() -> bool:
    global _TORCH_OK
    if _TORCH_OK is not None:
        return _TORCH_OK
    try:
        import torch  # noqa: F401

        _TORCH_OK = True
    except Exception:  # noqa: BLE001
        _TORCH_OK = False
    return _TORCH_OK


def _refine_with_sam_family(
    image: Image.Image,
    out: Path,
    *,
    method: str,
    model_name: str,
    task: str,
) -> dict:
    if not _torch_ok():
        return _result(method, layers=0, metric=0, sec=0, ok=False, error="torch unavailable")
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    try:
        from ultralytics import FastSAM, YOLO
    except Exception as exc:  # noqa: BLE001
        return _result(method, layers=0, metric=0, sec=0, ok=False, error=str(exc))

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]

    try:
        if method == "fastsam":
            model = FastSAM(model_name)
            results = model.predict(rgb, verbose=False, retina_masks=True, imgsz=max(h, w))
            union = np.zeros((h, w), dtype=np.uint8)
            layers = 0
            if results and results[0].masks is not None:
                data = results[0].masks.data
                arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
                for m in arr:
                    mm = (m > 0.5).astype(np.uint8) * 255
                    if mm.shape[:2] != (h, w):
                        mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                    union = np.maximum(union, mm)
                    layers += 1
            _save_union(out, union, _overlay_from_union(image, union))
            metric = _coverage(union, rgb) if task == "motif" else float((union > 40).mean())
            return _result(method, layers=layers, metric=metric, sec=time.time() - t0, ok=layers > 0, notes=model_name)

        if method == "yolo_seg":
            model = YOLO(model_name)
            results = model.predict(rgb, verbose=False, retina_masks=True)
            union = np.zeros((h, w), dtype=np.uint8)
            layers = 0
            if results and results[0].masks is not None:
                data = results[0].masks.data
                arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
                for m in arr:
                    mm = (m > 0.5).astype(np.uint8) * 255
                    if mm.shape[:2] != (h, w):
                        mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                    union = np.maximum(union, mm)
                    layers += 1
            _save_union(out, union, _overlay_from_union(image, union))
            metric = _coverage(union, rgb) if task == "motif" else float((union > 40).mean())
            return _result(
                method,
                layers=layers,
                metric=metric,
                sec=time.time() - t0,
                ok=layers > 0,
                notes=f"{model_name} (COCO classes — may miss print motifs)",
            )

        # SAM2 / MobileSAM: refine CV instances with bbox+point (same as product path)
        from sam_refine import refine_instances_sam

        inst = discover_by_components(image, max_instances=80)
        for i, x in enumerate(inst, 1):
            x.id = f"m{i:03d}"
        # MobileSAM weights name differs; Ultralytics SAM() accepts mobile_sam.pt
        inst = refine_instances_sam(image, inst, model_name=model_name, enabled=True)
        overlay_instances(image, inst).save(out / "seg_overlay.png")
        union = np.zeros((h, w), dtype=np.uint8)
        for x in inst:
            union = np.maximum(union, x.mask)
        Image.fromarray(union).save(out / "union_mask.png")
        metric = _coverage(union, rgb) if task == "motif" else float((union > 40).mean())
        return _result(method, layers=len(inst), metric=metric, sec=time.time() - t0, ok=len(inst) > 0, notes=model_name)
    except Exception as exc:  # noqa: BLE001
        return _result(method, layers=0, metric=0, sec=time.time() - t0, ok=False, error=str(exc)[:300])


def run_sam2(image: Image.Image, out: Path, task: str) -> dict:
    weights = ROOT / "sam2_b.pt"
    name = str(weights) if weights.exists() else "sam2_b.pt"
    return _refine_with_sam_family(image, out, method="sam2", model_name=name, task=task)


def run_mobile_sam(image: Image.Image, out: Path, task: str) -> dict:
    return _refine_with_sam_family(image, out, method="mobile_sam", model_name="mobile_sam.pt", task=task)


def run_fastsam(image: Image.Image, out: Path, task: str) -> dict:
    return _refine_with_sam_family(image, out, method="fastsam", model_name="FastSAM-s.pt", task=task)


def run_yolo_seg(image: Image.Image, out: Path, task: str) -> dict:
    return _refine_with_sam_family(image, out, method="yolo_seg", model_name="yolo11n-seg.pt", task=task)


# ---------- Gemini box + matte ----------
def _genai():
    from google import genai
    from google.genai import types

    return genai, types


def _gemini_boxes(image: Image.Image, api_key: str, task: str) -> list[dict]:
    genai, types = _genai()
    client = genai.Client(api_key=api_key)
    if task == "subject":
        prompt = (
            "Detect the main foreground subject(s). Return JSON list "
            '[{"box_2d":[ymin,xmin,ymax,xmax],"label":"..."}] coords 0-1000. At most 8.'
        )
    elif task == "parts":
        prompt = (
            "Detect major distinct parts/components. Return JSON list "
            '[{"box_2d":[ymin,xmin,ymax,xmax],"label":"..."}] coords 0-1000. At most 24.'
        )
    else:
        prompt = (
            "Detect distinct motif instances in this textile print (not background). "
            'Return JSON list [{"box_2d":[ymin,xmin,ymax,xmax],"label":"..."}] '
            "coords 0-1000. At most 40, prefer largest."
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
        data = data.get("boxes") or data.get("objects") or data.get("motifs") or []
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def _matte_boxes(image: Image.Image, boxes: list[dict]) -> tuple[np.ndarray, int]:
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    bg = estimate_background_color(rgb)
    union = np.zeros((h, w), dtype=np.uint8)
    n = 0
    for item in boxes:
        box = item.get("box_2d") or item.get("bbox_norm")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        a, b, c, d = [float(v) for v in box]
        if max(a, b, c, d) <= 1.5:
            # xywh or xyxy normalized
            if c <= 1 and d <= 1 and c < 1 and d < 1 and (c + a) <= 1.05:
                x0, y0, x1, y1 = a * w, b * h, (a + c) * w, (b + d) * h
            else:
                x0, y0, x1, y1 = a * w, b * h, c * w, d * h
        else:
            # Gemini [ymin,xmin,ymax,xmax] 0-1000
            y0, x0, y1, x1 = a / 1000 * h, b / 1000 * w, c / 1000 * h, d / 1000 * w
        xa, ya = max(0, int(min(x0, x1))), max(0, int(min(y0, y1)))
        xb, yb = min(w, int(max(x0, x1))), min(h, int(max(y0, y1)))
        if xb <= xa + 1 or yb <= ya + 1:
            continue
        crop = rgb[ya:yb, xa:xb]
        dist = lab_distance(crop, bg)
        alpha = (np.clip((dist - 8.0) / 20.0, 0, 1) * 255).astype(np.uint8)
        union[ya:yb, xa:xb] = np.maximum(union[ya:yb, xa:xb], alpha)
        n += 1
    return union, n


def run_gemini_box_matte(image: Image.Image, out: Path, task: str) -> dict:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return _result("gemini_box_matte", layers=0, metric=0, sec=0, ok=False, error="no GOOGLE_API_KEY")
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    try:
        boxes = _gemini_boxes(image, api_key, task)
        union, n = _matte_boxes(image, boxes)
        _save_union(out, union, _overlay_from_union(image, union))
        metric = _coverage(union, np.array(image)) if task == "motif" else float((union > 40).mean())
        return _result(
            "gemini_box_matte",
            layers=n,
            metric=metric,
            sec=time.time() - t0,
            ok=n > 0,
            notes="gemini-2.5-flash boxes + Lab matte",
        )
    except Exception as exc:  # noqa: BLE001
        return _result("gemini_box_matte", layers=0, metric=0, sec=time.time() - t0, ok=False, error=str(exc)[:300])


# ---------- OpenAI GPT image 2.5 → mask ----------
def _gpt_image_edit_mask(image: Image.Image, model: str, prompt: str) -> Image.Image:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    buf = BytesIO()
    # Keep under OpenAI input limits; square-ish PNG
    im = image.convert("RGBA")
    im.save(buf, format="PNG")
    files = {
        "image": ("input.png", buf.getvalue(), "image/png"),
    }
    data = {
        "model": model,
        "prompt": prompt,
        "size": "1024x1024",
        "quality": "high",
    }
    res = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {api_key}"},
        data=data,
        files=files,
        timeout=300,
    )
    text = res.text
    try:
        payload = res.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{model} non-JSON: {text[:300]}") from exc
    if not res.ok:
        raise RuntimeError(f"{model}: {payload.get('error', {}).get('message') or text[:300]}")
    b64 = payload["data"][0]["b64_json"]
    import base64

    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _rgb_to_mask(mask_rgb: Image.Image, size: tuple[int, int]) -> np.ndarray:
    m = mask_rgb.convert("L").resize(size, Image.Resampling.BILINEAR)
    arr = np.array(m)
    # Auto threshold: prefer Otsu-ish mid
    thr = 128
    if arr.std() > 5:
        thr = int(max(40, min(200, arr.mean())))
    return ((arr > thr).astype(np.uint8) * 255)


def _run_gpt_mask(image: Image.Image, out: Path, task: str, method: str, model: str) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    prompt = GPT_SUBJECT_PROMPT if task == "subject" else GPT_MASK_PROMPT
    try:
        gen = _gpt_image_edit_mask(image, model, prompt)
        gen.save(out / "gpt_raw.png")
        union = _rgb_to_mask(gen, image.size)
        # Count CCs as pseudo-layers
        n, _, st, _ = cv2.connectedComponentsWithStats((union > 40).astype(np.uint8), 8)
        layers = max(0, n - 1)
        _save_union(out, union, _overlay_from_union(image, union))
        metric = _coverage(union, np.array(image)) if task == "motif" else float((union > 40).mean())
        return _result(
            method,
            layers=layers,
            metric=metric,
            sec=time.time() - t0,
            ok=layers > 0 or metric > 0.01,
            notes=model,
        )
    except Exception as exc:  # noqa: BLE001
        return _result(method, layers=0, metric=0, sec=time.time() - t0, ok=False, error=str(exc)[:400])


def run_gpt25_flare_mask(image: Image.Image, out: Path, task: str) -> dict:
    return _run_gpt_mask(image, out, task, "gpt25_flare_mask", "gpt-image-2.5-flare")


def run_gpt25_sunburst_mask(image: Image.Image, out: Path, task: str) -> dict:
    return _run_gpt_mask(image, out, task, "gpt25_sunburst_mask", "gpt-image-2.5-sunburst")


METHODS: dict[str, Callable[[Image.Image, Path, str], dict]] = {
    "cv": run_cv,
    "sam2": run_sam2,
    "mobile_sam": run_mobile_sam,
    "fastsam": run_fastsam,
    "yolo_seg": run_yolo_seg,
    "gemini_box_matte": run_gemini_box_matte,
    "gpt25_flare_mask": run_gpt25_flare_mask,
    "gpt25_sunburst_mask": run_gpt25_sunburst_mask,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        default="cv,sam2,mobile_sam,fastsam,yolo_seg,gemini_box_matte,gpt25_flare_mask,gpt25_sunburst_mask",
    )
    p.add_argument("--quick", action="store_true", help="6 diverse cases")
    p.add_argument("--smoke", action="store_true", help="2 cases")
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in METHODS:
            raise SystemExit(f"unknown model {m}; choose from {list(METHODS)}")

    cases = CASES
    if args.smoke:
        cases = CASES[:2]
    elif args.quick:
        picks = {
            "floral_large",
            "insects",
            "fruit",
            "interlocking",
            "one_giant",
            "bouquet_parts",
        }
        cases = [c for c in CASES if c[1] in picks]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print(f"models={models} cases={len(cases)} out={OUT}", flush=True)
    rows: list[dict] = []
    cards: list[str] = []

    for i, (rel, category, task) in enumerate(cases, 1):
        image = _resize(Image.open(_resolve(rel)).convert("RGB"))
        stem = Path(rel).stem
        print(f"\n=== [{i}/{len(cases)}] {category}/{stem} task={task} ===", flush=True)
        case_row: dict[str, Any] = {"id": stem, "category": category, "task": task, "methods": {}}
        figs = [
            f'<figure><img src="{category}__{stem}/cv/original.png"/><figcaption>original</figcaption></figure>'
        ]
        # ensure original exists even if cv skipped
        (OUT / f"{category}__{stem}" / "_src").mkdir(parents=True, exist_ok=True)
        image.save(OUT / f"{category}__{stem}" / "_src" / "original.png")
        figs = [
            f'<figure><img src="{category}__{stem}/_src/original.png"/><figcaption>original</figcaption></figure>'
        ]

        for method in models:
            out = OUT / f"{category}__{stem}" / method
            st = METHODS[method](image, out, task)
            case_row["methods"][method] = st
            flag = "OK" if st["ok"] else "FAIL"
            print(
                f"  {method:22} L={st['layers']:3} m={st['metric']:.3f} {st['sec']:7.1f}s {flag}"
                + (f"  {st['error'][:80]}" if st.get("error") else ""),
                flush=True,
            )
            figs.append(
                f'<figure><img src="{category}__{stem}/{method}/union_mask.png" onerror="this.style.opacity=.2"/>'
                f"<figcaption>{method}<br/>{st['sec']}s · m={st['metric']:.2f}</figcaption></figure>"
            )

        rows.append(case_row)
        cards.append(
            f'<div class="card"><h3>{category} · {stem} <small>({task})</small></h3>'
            f'<div class="row">{"".join(figs)}</div></div>'
        )

    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))

    # Rollup per method
    rollup: dict[str, Any] = {}
    for method in models:
        stats = [r["methods"][method] for r in rows if method in r["methods"]]
        if not stats:
            continue
        ok = [s for s in stats if s["ok"]]
        rollup[method] = {
            "n": len(stats),
            "ok_rate": round(len(ok) / len(stats), 3),
            "reasonable_rate": round(sum(1 for s in stats if s.get("reasonable")) / len(stats), 3),
            "latency_p50_s": round(float(np.median([s["sec"] for s in stats])), 2),
            "latency_mean_s": round(float(np.mean([s["sec"] for s in stats])), 2),
            "metric_mean": round(float(np.mean([s["metric"] for s in stats])), 3),
            "metric_mean_ok": round(float(np.mean([s["metric"] for s in ok])), 3) if ok else 0.0,
            "layers_mean": round(float(np.mean([s["layers"] for s in stats])), 1),
            "errors": [s["error"] for s in stats if s.get("error")][:3],
        }
    (OUT / "rollup.json").write_text(json.dumps(rollup, indent=2))

    # Rank vs SAM2 when present
    ranking = []
    sam_m = rollup.get("sam2", {}).get("metric_mean")
    for method, r in rollup.items():
        ranking.append(
            {
                "method": method,
                "score": round(r["reasonable_rate"] * 0.4 + min(1.0, r["metric_mean"]) * 0.4 + (0.2 if r["latency_p50_s"] < 30 else 0.05 if r["latency_p50_s"] < 120 else 0.0), 3),
                "metric_mean": r["metric_mean"],
                "latency_p50_s": r["latency_p50_s"],
                "vs_sam2_metric": round(r["metric_mean"] - sam_m, 3) if sam_m is not None else None,
            }
        )
    ranking.sort(key=lambda x: -x["score"])
    (OUT / "ranking.json").write_text(json.dumps(ranking, indent=2))

    html = f"""<!doctype html><html><head><meta charset="utf-8"/><title>SAM alternatives bakeoff</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#f4f2ee}}
.card{{background:#fff;border-radius:12px;padding:16px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px}}
img{{width:100%;border-radius:8px;background:#111}}
figcaption{{font-size:11px;color:#555;margin-top:4px}}
pre{{background:#fff;padding:12px;border-radius:8px;overflow:auto}}
</style></head><body>
<h1>SAM2 alternatives bakeoff</h1>
<p>Includes GPT image 2.5 mask edits, Gemini box+matte, and local SAM-family when torch loads.</p>
<h2>Ranking</h2><pre>{json.dumps(ranking, indent=2)}</pre>
<h2>Rollup</h2><pre>{json.dumps(rollup, indent=2)}</pre>
{''.join(cards)}
</body></html>"""
    (OUT / "compare.html").write_text(html)
    print("\nRANKING", flush=True)
    print(json.dumps(ranking, indent=2), flush=True)
    print(f"\ngallery → {OUT / 'compare.html'}", flush=True)


if __name__ == "__main__":
    main()

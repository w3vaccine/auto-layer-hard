#!/usr/bin/env python3
"""E2E research: Gemini native AI segmentation as SAM-class replacement.

Covers editor segmentation *task types* (not just Magic Layer):
  1. motif_instance  — dense textile motifs (SAM2 refine use-case)
  2. subject_fg      — single-subject FG/BG cutout (PicWish / RMBG class)
  3. part_prompt     — conversational part masks (dismantle / knit zones class)

Methods compared:
  - cv_components   — OpenCV ink CC baseline (motifs only)
  - gemini_seg      — Gemini 2.5 Flash native masks (box_2d + mask)
  - vlm_tag_matte   — Gemini boxes → color-matte (no native mask)

No SAM2 weights required. Outputs latency + quality metrics + gallery.
"""

from __future__ import annotations

import ast
import base64
import json
import os
import re
import shutil
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")

from segment import (  # noqa: E402
    chroma_distance,
    discover_by_components,
    estimate_background_color,
    lab_distance,
    overlay_instances,
)

OUT = ROOT / "out" / "ai_seg_research"
MAX_SIDE = 1024
MODEL = os.environ.get("AI_SEG_MODEL", "gemini-2.5-flash")

# Diverse categories for "other than seashells/ditsy" validation
MOTIF_CASES = [
    ("showcase/s01_garden_roses.png", "floral_large", "motif_instance"),
    ("showcase/s02_butterflies.png", "insects", "motif_instance"),
    ("showcase/s03_geo_diamonds.png", "geometric", "motif_instance"),
    ("showcase/s07_wave_dots.png", "polka_dense", "motif_instance"),
    ("showcase/s12_color_blobs.png", "abstract_blobs", "motif_instance"),
    ("showcase/s14_citrus_slices.png", "fruit", "motif_instance"),
    ("showcase/s15_mixed_bouquet.png", "mixed_floral", "motif_instance"),
    ("01-ditsy-florals.png", "ditsy_dense", "motif_instance"),
    ("03-paisley.png", "interlocking", "motif_instance"),
    ("08-abstract-brush.png", "painterly", "motif_instance"),
    ("hard/h02_low_contrast.png", "low_contrast", "motif_instance"),
    ("hard/h06_thin_lines.png", "thin_lines", "motif_instance"),
    ("hard/h09_same_hue_family.png", "same_hue", "motif_instance"),
    ("hard/h18_outline_only.png", "outline_only", "motif_instance"),
    ("hard/h21_paisley_like.png", "paisley_like", "motif_instance"),
]

# Subject FG: treat one large motif print / brush as proxy when no garment pack present
SUBJECT_CASES = [
    ("06-seashells.png", "product_shells", "subject_fg"),
    ("hard/h14_one_giant.png", "one_giant_subject", "subject_fg"),
    ("showcase/s04_coastal_shells.png", "coastal_subject", "subject_fg"),
]

PART_CASES = [
    ("showcase/s15_mixed_bouquet.png", "bouquet_parts", "part_prompt"),
    ("02-tropical-leaves.png", "leaf_parts", "part_prompt"),
]


# Stay close to Google's recommended segmentation prompt (Flash + thinking off).
SEG_PROMPT_MOTIFS = """Give the segmentation masks for the distinct motif instances in this textile print
(flowers, leaves, shells, dots, shapes, brush marks, insects, fruit). Prefer the largest / most prominent
instances; return at most 40 masks. Do not merge separate instances. Background is not a motif.
Output a JSON list of segmentation masks where each entry contains the 2D
bounding box in the key "box_2d", the segmentation mask in key "mask", and
the text label in the key "label". Use descriptive labels."""

SEG_PROMPT_SUBJECT = """Give the segmentation masks for the main foreground subject only. Exclude background.
If many similar objects, return one mask per prominent object (at most 12).
Output a JSON list of segmentation masks where each entry contains the 2D
bounding box in the key "box_2d", the segmentation mask in key "mask", and
the text label in the key "label". Use descriptive labels."""

SEG_PROMPT_PARTS = """Give the segmentation masks for the major distinct parts / components
(flower heads vs leaves vs stems, or garment body/sleeves/collar). At most 20 masks.
Output a JSON list of segmentation masks where each entry contains the 2D
bounding box in the key "box_2d", the segmentation mask in key "mask", and
the text label in the key "label". Use descriptive labels."""

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


def _ink_mask(rgb: np.ndarray) -> np.ndarray:
    bg = estimate_background_color(rgb)
    lab = lab_distance(rgb, bg)
    chr_d = chroma_distance(rgb, bg)
    lab_frac = float((lab > 14).mean())
    chr_frac = float((chr_d > 8).mean())
    if lab_frac > 0.42 and chr_frac < lab_frac * 0.55:
        return chr_d > 8
    return lab > 14


def _coverage(union: np.ndarray, rgb: np.ndarray) -> float:
    ink = _ink_mask(rgb)
    if not ink.any():
        return 0.0
    return float(((union > 20) & ink).sum() / ink.sum())


def _genai():
    from google import genai
    from google.genai import types

    return genai, types


def _client(api_key: str):
    genai, _ = _genai()
    return genai.Client(api_key=api_key)


def _image_part(image: Image.Image):
    _, types = _genai()
    buf = BytesIO()
    image.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def _extract_json_list(text: str) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        for k in ("masks", "boxes", "items", "objects", "segmentation"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
        else:
            data = None
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    # Salvage truncated Gemini lists — keep boxes even when mask tokens break JSON
    found: list[dict[str, Any]] = []
    for m in re.finditer(
        r'\{\s*"box_2d"\s*:\s*\[([^\]]+)\]\s*,\s*"mask"\s*:\s*"(?:\\.|[^"\\])*"\s*,\s*"label"\s*:\s*"((?:\\.|[^"\\])*)"\s*\}',
        text,
    ):
        try:
            box = [float(x.strip()) for x in m.group(1).split(",")]
            label = m.group(2).encode("utf-8").decode("unicode_escape")
            if len(box) == 4:
                found.append({"box_2d": box, "mask": None, "label": label})
        except Exception:  # noqa: BLE001
            continue
    if found:
        return found
    # Alternate key order: box, label, mask
    for m in re.finditer(
        r'\{\s*"box_2d"\s*:\s*\[([^\]]+)\]\s*,\s*"label"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
    ):
        try:
            box = [float(x.strip()) for x in m.group(1).split(",")]
            label = m.group(2)
            if len(box) == 4:
                found.append({"box_2d": box, "mask": None, "label": label})
        except Exception:  # noqa: BLE001
            continue
    return found

def _parse_mask_field(mask_field: Any) -> Any:
    """Normalize mask field: stringified JSON polygon, base64 PNG, or list."""
    if isinstance(mask_field, list):
        return mask_field
    if not isinstance(mask_field, str) or not mask_field.strip():
        return None
    s = mask_field.strip()
    if s.startswith("data:image") or (len(s) > 80 and not s.startswith("[")):
        return s  # base64 path
    if s[0] in "[{":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(s)
            except Exception:  # noqa: BLE001
                return None
    return s


def _decode_mask_png(mask_field: Any, box_w: int, box_h: int) -> np.ndarray | None:
    if not isinstance(mask_field, str) or not mask_field:
        return None
    s = mask_field
    if s.startswith("data:image"):
        s = s.split(",", 1)[-1]
    if s.startswith("[") or len(s) < 32:
        return None
    try:
        raw = base64.b64decode(s, validate=False)
        im = Image.open(BytesIO(raw)).convert("L")
    except Exception:  # noqa: BLE001
        return None
    if im.size != (box_w, box_h):
        im = im.resize((box_w, box_h), Image.Resampling.BILINEAR)
    return np.array(im)


def _polygon_to_mask(poly: list, w: int, h: int) -> np.ndarray | None:
    """poly: list of [x,y] (or [y,x]) in 0..1000 image coords."""
    if not isinstance(poly, list) or len(poly) < 3:
        return None
    pts = []
    for p in poly:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append([float(p[0]), float(p[1])])
        else:
            return None
    arr = np.array(pts, dtype=np.float32)
    # Gemini docs: polygon [x,y] normalized to 0-1000
    if arr.max() <= 1000.0:
        xs = arr[:, 0] / 1000.0 * w
        ys = arr[:, 1] / 1000.0 * h
        arr = np.stack([xs, ys], axis=1)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [arr.astype(np.int32)], 255)
    return m


def _ink_matte_in_box(rgb: np.ndarray, xa: int, ya: int, xb: int, yb: int) -> np.ndarray:
    """Hybrid fallback: color-distance matte inside a Gemini box (replaces coarse SAM refine)."""
    crop = rgb[ya:yb, xa:xb]
    if crop.size == 0:
        return np.zeros((max(1, yb - ya), max(1, xb - xa)), dtype=np.uint8)
    bg = estimate_background_color(rgb)
    dist = lab_distance(crop, bg)
    alpha = np.clip((dist - 8.0) / 20.0, 0.0, 1.0)
    return (alpha * 255).astype(np.uint8)

def _compose_masks(
    image: Image.Image,
    items: list[dict[str, Any]],
    *,
    max_instances: int = 80,
) -> tuple[list[dict], np.ndarray, Image.Image]:
    """Return layer metas, union alpha, overlay RGB."""
    w, h = image.size
    union = np.zeros((h, w), dtype=np.uint8)
    overlay = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay)
    layers: list[dict] = []

    for i, item in enumerate(items[:max_instances]):
        box = item.get("box_2d") or item.get("bbox") or item.get("box")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        y0, x0, y1, x1 = [float(v) for v in box]
        # Heuristic: if values look like xyxy 0-1
        if max(y0, x0, y1, x1) <= 1.5:
            x0n, y0n, x1n, y1n = x0, y0, x1, y1
            x0, y0, x1, y1 = x0n * w, y0n * h, x1n * w, y1n * h
        else:
            # Gemini 0-1000: [ymin,xmin,ymax,xmax]
            x0, y0, x1, y1 = x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h
        xa, ya = max(0, int(min(x0, x1))), max(0, int(min(y0, y1)))
        xb, yb = min(w, int(max(x0, x1))), min(h, int(max(y0, y1)))
        if xb <= xa + 1 or yb <= ya + 1:
            continue
        bw, bh = xb - xa, yb - ya
        label = str(item.get("label") or f"obj{i+1}")
        mask_field = _parse_mask_field(item.get("mask"))
        local: np.ndarray | None = None
        fmt = "none"
        if isinstance(mask_field, str):
            local = _decode_mask_png(mask_field, bw, bh)
            fmt = "png" if local is not None else "none"
        elif isinstance(mask_field, list):
            full = _polygon_to_mask(mask_field, w, h)
            if full is not None:
                local = full[ya:yb, xa:xb]
                fmt = "polygon"
                # Coarse/rect polygons ≈ boxes — refine with ink matte
                if local is not None and float((local > 40).mean()) > 0.85:
                    local = _ink_matte_in_box(np.array(image.convert("RGB")), xa, ya, xb, yb)
                    fmt = "polygon+matte"
        if local is None:
            local = _ink_matte_in_box(np.array(image.convert("RGB")), xa, ya, xb, yb)
            fmt = "box+matte"

        alpha = local
        if alpha.dtype != np.uint8:
            alpha = np.clip(alpha, 0, 255).astype(np.uint8)
        # Place into full canvas
        region = union[ya:yb, xa:xb]
        union[ya:yb, xa:xb] = np.maximum(region, alpha)
        # Tint overlay
        tint = np.zeros((bh, bw, 4), dtype=np.uint8)
        color = [(255, 80, 80), (80, 180, 255), (80, 220, 120), (255, 200, 60)][i % 4]
        tint[..., 0] = color[0]
        tint[..., 1] = color[1]
        tint[..., 2] = color[2]
        tint[..., 3] = (alpha.astype(np.float32) * 0.45).astype(np.uint8)
        overlay.paste(Image.fromarray(tint, "RGBA"), (xa, ya), Image.fromarray(tint, "RGBA"))
        draw.rectangle([xa, ya, xb, yb], outline=(255, 0, 0, 220), width=max(1, w // 500))
        layers.append(
            {
                "id": f"g{i+1:03d}",
                "label": label,
                "bbox_px": [xa, ya, bw, bh],
                "mask_format": fmt,
                "area_px": int((alpha > 40).sum()),
            }
        )

    return layers, union, overlay.convert("RGB")


def run_gemini_seg(
    image: Image.Image,
    out: Path,
    *,
    api_key: str,
    task: str,
    max_instances: int = 60,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    prompt = {
        "motif_instance": SEG_PROMPT_MOTIFS,
        "subject_fg": SEG_PROMPT_SUBJECT,
        "part_prompt": SEG_PROMPT_PARTS,
    }[task]

    genai, types = _genai()
    client = genai.Client(api_key=api_key)
    t0 = time.time()
    # Best practice: Flash + thinking off
    err = None
    text = ""
    items: list[dict[str, Any]] = []
    # Try thinking off first (Google guidance); retry once without JSON mime if empty.
    attempts = [
        dict(temperature=0.0, response_mime_type="application/json", thinking_budget=0),
        dict(temperature=0.0, thinking_budget=0),
    ]
    for attempt_i, kw in enumerate(attempts):
        try:
            cfg: dict[str, Any] = {"temperature": kw.get("temperature", 0.0)}
            if kw.get("response_mime_type"):
                cfg["response_mime_type"] = kw["response_mime_type"]
            try:
                config = types.GenerateContentConfig(
                    **cfg,
                    thinking_config=types.ThinkingConfig(thinking_budget=kw.get("thinking_budget", 0)),
                )
            except Exception:  # noqa: BLE001
                config = types.GenerateContentConfig(**cfg)
            resp = client.models.generate_content(
                model=MODEL,
                contents=[prompt, _image_part(image)],
                config=config,
            )
            text = (resp.text or "").strip()
            items = _extract_json_list(text)
            if items:
                break
            print(f"  gemini empty parse attempt {attempt_i+1} (chars={len(text)})", flush=True)
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            print(f"  gemini attempt {attempt_i+1} err: {err[:160]}", flush=True)
    layers, union, overlay = _compose_masks(image, items, max_instances=max_instances)
    overlay.save(out / "seg_overlay.png")
    Image.fromarray(union).save(out / "union_mask.png")
    (out / "raw_items.json").write_text(json.dumps(items[:max_instances], indent=2)[:400000])

    rgb = np.array(image)
    cov = _coverage(union, rgb) if task == "motif_instance" else float((union > 40).mean())
    formats = sorted({L["mask_format"] for L in layers})
    sec = round(time.time() - t0, 2)
    return {
        "method": "gemini_seg",
        "model": MODEL,
        "task": task,
        "layers": len(layers),
        "coverage_or_fg_frac": round(cov, 4),
        "sec": sec,
        "mask_formats": formats,
        "labels": [L["label"] for L in layers[:12]],
        "error": err,
        "ok": bool(layers) and err is None,
        "reasonable": _reasonable(layers, union, task),
        "native_mask_usable": any(f in ("png", "polygon", "polygon+matte") for f in formats),
        "raw_chars": len(text),
    }


def run_cv(image: Image.Image, out: Path, *, task: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    t0 = time.time()
    if task != "motif_instance":
        # CV subject: largest ink component as FG
        rgb = np.array(image)
        ink = _ink_mask(rgb).astype(np.uint8) * 255
        n, lab, st, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
        union = np.zeros_like(ink)
        layers = 0
        if n > 1:
            areas = st[1:, cv2.CC_STAT_AREA]
            largest = 1 + int(np.argmax(areas))
            union = ((lab == largest) * 255).astype(np.uint8)
            layers = 1
        Image.fromarray(union).save(out / "union_mask.png")
        overlay_instances  # keep import used
        Image.fromarray(cv2.cvtColor(union, cv2.COLOR_GRAY2RGB)).save(out / "seg_overlay.png")
        return {
            "method": "cv_components",
            "task": task,
            "layers": layers,
            "coverage_or_fg_frac": round(float((union > 40).mean()), 4),
            "sec": round(time.time() - t0, 2),
            "mask_formats": ["cv"],
            "labels": ["largest_ink_cc"] if layers else [],
            "error": None,
            "ok": layers > 0,
            "reasonable": layers > 0 and float((union > 40).mean()) > 0.02,
        }

    inst = discover_by_components(image, max_instances=200)
    for i, x in enumerate(inst, 1):
        x.id = f"m{i:03d}"
    overlay_instances(image, inst).save(out / "seg_overlay.png")
    union = np.zeros((image.height, image.width), dtype=np.uint8)
    for x in inst:
        union = np.maximum(union, x.mask)
    Image.fromarray(union).save(out / "union_mask.png")
    rgb = np.array(image)
    cov = _coverage(union, rgb)
    return {
        "method": "cv_components",
        "task": task,
        "layers": len(inst),
        "coverage_or_fg_frac": round(cov, 4),
        "sec": round(time.time() - t0, 2),
        "mask_formats": ["cv"],
        "labels": [],
        "error": None,
        "ok": len(inst) > 0,
        "reasonable": _reasonable(
            [{"area_px": int((x.mask > 40).sum())} for x in inst],
            union,
            task,
        ),
    }


def _reasonable(layers: list[dict], union: np.ndarray, task: str) -> bool:
    if not layers:
        return False
    fg = float((union > 40).mean())
    if task == "subject_fg":
        return 0.02 < fg < 0.95 and len(layers) <= 8
    if task == "part_prompt":
        return len(layers) >= 2 and fg > 0.01
    # motif: need multiple instances or decent coverage
    areas = [L.get("area_px", 0) for L in layers]
    nonzero = sum(1 for a in areas if a > 30)
    return nonzero >= 1 and fg > 0.005


def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY required")

    cases = MOTIF_CASES + SUBJECT_CASES + PART_CASES
    if "--smoke" in sys.argv:
        cases = MOTIF_CASES[:2] + SUBJECT_CASES[:1] + PART_CASES[:1]
    elif "--quick" in sys.argv:
        # Diverse category subset for latency/quality research (~12 cases)
        picks = {
            "floral_large",
            "insects",
            "geometric",
            "polka_dense",
            "abstract_blobs",
            "fruit",
            "interlocking",
            "low_contrast",
            "thin_lines",
            "outline_only",
            "product_shells",
            "one_giant_subject",
            "bouquet_parts",
            "leaf_parts",
        }
        cases = [c for c in cases if c[1] in picks]
    if "--motifs-only" in sys.argv:
        cases = [c for c in cases if c[2] == "motif_instance"]
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    rows: list[dict] = []
    cards: list[str] = []
    print(f"model={MODEL} cases={len(cases)} out={OUT}", flush=True)

    for i, (rel, category, task) in enumerate(cases, 1):
        path = _resolve(rel)
        stem = Path(rel).stem
        image = _resize(Image.open(path).convert("RGB"))
        print(f"\n=== [{i}/{len(cases)}] {category}/{stem} task={task} ===", flush=True)
        case_dir = OUT / f"{category}__{stem}"

        cv_st = run_cv(image, case_dir / "cv", task=task)
        print(
            f"  CV     L={cv_st['layers']} metric={cv_st['coverage_or_fg_frac']:.3f} "
            f"{cv_st['sec']}s ok={cv_st['ok']} reasonable={cv_st['reasonable']}",
            flush=True,
        )

        g_st = run_gemini_seg(image, case_dir / "gemini_seg", api_key=api_key, task=task)
        print(
            f"  GEMINI L={g_st['layers']} metric={g_st['coverage_or_fg_frac']:.3f} "
            f"{g_st['sec']}s fmt={g_st['mask_formats']} ok={g_st['ok']} "
            f"reasonable={g_st['reasonable']} labels={g_st['labels'][:6]}",
            flush=True,
        )
        if g_st.get("error"):
            print(f"  ERR {g_st['error'][:200]}", flush=True)

        row = {
            "id": stem,
            "category": category,
            "task": task,
            "cv": cv_st,
            "gemini_seg": g_st,
            "latency_ratio_gemini_vs_cv": round(g_st["sec"] / max(0.01, cv_st["sec"]), 2),
        }
        rows.append(row)
        cards.append(
            f"""<div class="card">
            <h3>{category} · {stem}</h3>
            <div class="meta">task={task}
            &nbsp;|&nbsp; CV: {cv_st['layers']}L · {cv_st['coverage_or_fg_frac']:.3f} · {cv_st['sec']}s · ok={cv_st['reasonable']}
            &nbsp;|&nbsp; Gemini: {g_st['layers']}L · {g_st['coverage_or_fg_frac']:.3f} · {g_st['sec']}s · {','.join(g_st['mask_formats']) or '—'} · ok={g_st['reasonable']}
            </div>
            <div class="row">
              <figure><img src="{category}__{stem}/cv/original.png"/><figcaption>original</figcaption></figure>
              <figure><img src="{category}__{stem}/cv/seg_overlay.png"/><figcaption>CV</figcaption></figure>
              <figure><img src="{category}__{stem}/gemini_seg/seg_overlay.png"/><figcaption>Gemini seg</figcaption></figure>
              <figure><img src="{category}__{stem}/cv/union_mask.png"/><figcaption>CV mask</figcaption></figure>
              <figure><img src="{category}__{stem}/gemini_seg/union_mask.png"/><figcaption>Gemini mask</figcaption></figure>
            </div></div>"""
        )

    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))

    # Rollups
    by_task: dict[str, list] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    rollup = {}
    for task, rs in by_task.items():
        g = [r["gemini_seg"] for r in rs]
        c = [r["cv"] for r in rs]
        rollup[task] = {
            "n": len(rs),
            "gemini_ok_rate": round(sum(1 for x in g if x["ok"]) / len(g), 3),
            "gemini_reasonable_rate": round(sum(1 for x in g if x["reasonable"]) / len(g), 3),
            "cv_reasonable_rate": round(sum(1 for x in c if x["reasonable"]) / len(c), 3),
            "gemini_latency_p50_s": round(float(np.median([x["sec"] for x in g])), 2),
            "gemini_latency_mean_s": round(float(np.mean([x["sec"] for x in g])), 2),
            "cv_latency_mean_s": round(float(np.mean([x["sec"] for x in c])), 2),
            "gemini_layers_mean": round(float(np.mean([x["layers"] for x in g])), 1),
            "cv_layers_mean": round(float(np.mean([x["layers"] for x in c])), 1),
            "gemini_metric_mean": round(float(np.mean([x["coverage_or_fg_frac"] for x in g])), 3),
            "cv_metric_mean": round(float(np.mean([x["coverage_or_fg_frac"] for x in c])), 3),
            "categories": sorted({r["category"] for r in rs}),
        }
    (OUT / "rollup.json").write_text(json.dumps(rollup, indent=2))

    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>AI seg research — SAM replacement</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#f4f2ee;color:#1a1a1a}}
h1{{margin:0 0 6px}} .sub{{color:#666;margin-bottom:16px;max-width:900px}}
.card{{background:#fff;border-radius:12px;padding:16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.row{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}
img{{width:100%;border-radius:8px;background:#eee}}
.meta{{font-size:13px;color:#555;margin:6px 0 12px}}
pre{{background:#fff;padding:12px;border-radius:8px;overflow:auto}}
</style></head><body>
<h1>AI segmentation vs SAM-class tasks</h1>
<p class="sub">Model: {MODEL}. No SAM2. Tasks: motif_instance (Magic Layer), subject_fg (cutout), part_prompt (parts).
Gemini native masks vs CV baseline. Latency + reasonableness heuristics.</p>
<pre>{json.dumps(rollup, indent=2)}</pre>
{''.join(cards)}
</body></html>"""
    (OUT / "compare.html").write_text(html)
    print(f"\nrollup → {OUT / 'rollup.json'}", flush=True)
    print(f"gallery → {OUT / 'compare.html'}", flush=True)
    print(json.dumps(rollup, indent=2), flush=True)


if __name__ == "__main__":
    main()

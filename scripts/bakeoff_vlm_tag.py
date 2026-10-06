#!/usr/bin/env python3
"""Bakeoff: CV components vs VLM tag→map (Gemini boxes + ink matte).

Idea: ask a vision model to list every motif instance as labeled bboxes,
then map each box to a cutout via color-distance matte inside the box.
Same architecture can later swap Gemini for GPT / Astra once keyed.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discover import discover_motifs, overlay_boxes  # noqa: E402
from isolate import isolate_motifs, layers_to_dicts as iso_dicts  # noqa: E402
from isolate_v2 import isolate_instances, layers_to_dicts as cv_dicts  # noqa: E402
from inpaint_v2 import reconstruct_background_v2, save_mask  # noqa: E402
from scene import write_scene  # noqa: E402
from segment import (  # noqa: E402
    chroma_distance,
    discover_by_components,
    estimate_background_color,
    lab_distance,
    overlay_instances,
)

OUT = ROOT / "out" / "vlm_tag_bakeoff"
MAX_SIDE = 1280


def _resize(im: Image.Image) -> Image.Image:
    if max(im.size) <= MAX_SIDE:
        return im
    s = MAX_SIDE / max(im.size)
    return im.resize((int(im.width * s), int(im.height * s)), Image.Resampling.LANCZOS)


def _ink_cov(union: np.ndarray, rgb: np.ndarray, bg: np.ndarray) -> float:
    lab = lab_distance(rgb, bg)
    chr_d = chroma_distance(rgb, bg)
    lab_frac = float((lab > 14).mean())
    chr_frac = float((chr_d > 8).mean())
    ink = (chr_d > 8) if (lab_frac > 0.42 and chr_frac < lab_frac * 0.55) else (lab > 14)
    if not ink.any():
        return 0.0
    return float(((union > 20) & ink).sum() / ink.sum())


def run_cv(image: Image.Image, out: Path) -> dict:
    out.mkdir(parents=True)
    (out / "layers").mkdir()
    image.save(out / "original.png")
    t0 = time.time()
    inst = discover_by_components(image, max_instances=200)
    for i, x in enumerate(inst, 1):
        x.id = f"m{i:03d}"
    overlay_instances(image, inst).save(out / "discover_overlay.png")
    layers, union = isolate_instances(image, inst, out / "layers")
    save_mask(union, out / "union_mask.png")
    rgb = np.array(image)
    bg = estimate_background_color(rgb)
    cov = _ink_cov(union, rgb, bg)
    bg_img, meta = reconstruct_background_v2(image, union, use_gemini=False)
    bg_img.save(out / "background.png")
    write_scene(
        out,
        source_name="cv",
        width=image.width,
        height=image.height,
        layers=cv_dicts(layers),
        discover_count=len(inst),
        background_meta=meta,
        coverage=cov,
    )
    return {
        "method": "cv",
        "layers": len(layers),
        "discovered": len(inst),
        "coverage": cov,
        "sec": round(time.time() - t0, 2),
        "labels": [],
    }


def run_vlm_tag(
    image: Image.Image,
    out: Path,
    *,
    api_key: str,
    model: str,
    max_passes: int = 4,
    max_instances: int = 100,
) -> dict:
    out.mkdir(parents=True)
    (out / "layers").mkdir()
    image.save(out / "original.png")
    t0 = time.time()
    boxes = discover_motifs(
        image,
        api_key=api_key,
        model=model,
        max_passes=max_passes,
        max_instances=max_instances,
        min_area=0.0003,
        iou_thresh=0.4,
    )
    overlay_boxes(image, boxes).save(out / "discover_overlay.png")
    # Map boxes → cutouts via ink matte (no per-box Gemini extract — keep it fast)
    layers, union = isolate_motifs(
        image,
        boxes,
        out / "layers",
        api_key=api_key,
        use_gemini_fallback=False,
    )
    save_mask(union, out / "union_mask.png")
    rgb = np.array(image)
    bg = estimate_background_color(rgb)
    cov = _ink_cov(union, rgb, bg)
    bg_img, meta = reconstruct_background_v2(image, union, use_gemini=False)
    bg_img.save(out / "background.png")
    write_scene(
        out,
        source_name="vlm_tag",
        width=image.width,
        height=image.height,
        layers=iso_dicts(layers),
        discover_count=len(boxes),
        background_meta=meta,
        coverage=cov,
    )
    labels = sorted({b.label for b in boxes})
    return {
        "method": "vlm_tag",
        "model": model,
        "layers": len(layers),
        "discovered": len(boxes),
        "coverage": cov,
        "sec": round(time.time() - t0, 2),
        "labels": labels[:20],
    }


def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY required for VLM tag bakeoff")

    model = os.environ.get("VLM_TAG_MODEL", "gemini-2.5-flash")
    showcase = ROOT / "fixtures" / "showcase"
    # Diverse subset for speed; full set via --all
    picks = [
        "s01_garden_roses.png",
        "s02_butterflies.png",
        "s03_geo_diamonds.png",
        "s07_wave_dots.png",
        "s12_color_blobs.png",
        "s14_citrus_slices.png",
        "s15_mixed_bouquet.png",
    ]
    if "--all" in sys.argv:
        picks = [p.name for p in sorted(showcase.glob("s*.png"))]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    rows: list[dict] = []
    cards: list[str] = []
    print(f"VLM model: {model}", flush=True)
    print(f"prints: {len(picks)}", flush=True)

    for i, name in enumerate(picks, 1):
        path = showcase / name
        if not path.exists():
            print(f"skip missing {name}", flush=True)
            continue
        stem = path.stem
        image = _resize(Image.open(path).convert("RGB"))
        print(f"\n=== [{i}/{len(picks)}] {stem} ===", flush=True)

        cv_out = OUT / stem / "cv"
        vlm_out = OUT / stem / "vlm"
        cv_st = run_cv(image, cv_out)
        print(f"  CV   L={cv_st['layers']} cov={cv_st['coverage']:.1%} {cv_st['sec']}s", flush=True)
        vlm_st = run_vlm_tag(image, vlm_out, api_key=api_key, model=model)
        print(
            f"  VLM  L={vlm_st['layers']} (tagged {vlm_st['discovered']}) "
            f"cov={vlm_st['coverage']:.1%} {vlm_st['sec']}s labels={vlm_st['labels'][:8]}",
            flush=True,
        )
        row = {"id": stem, "cv": cv_st, "vlm": vlm_st}
        rows.append(row)
        cards.append(
            f"""<div class="card"><h3>{stem}</h3>
            <div class="meta">CV: {cv_st['layers']}L · {cv_st['coverage']*100:.1f}% · {cv_st['sec']}s
            &nbsp;|&nbsp; VLM: {vlm_st['layers']}L · {vlm_st['coverage']*100:.1f}% · {vlm_st['sec']}s
            · tags: {', '.join(vlm_st['labels'][:6]) or '—'}</div>
            <div class="row">
              <figure><img src="{stem}/cv/original.png"/><figcaption>original</figcaption></figure>
              <figure><img src="{stem}/cv/discover_overlay.png"/><figcaption>CV discover</figcaption></figure>
              <figure><img src="{stem}/vlm/discover_overlay.png"/><figcaption>VLM tags</figcaption></figure>
              <figure><img src="{stem}/cv/background.png"/><figcaption>CV bg</figcaption></figure>
              <figure><img src="{stem}/vlm/background.png"/><figcaption>VLM bg</figcaption></figure>
            </div></div>"""
        )

    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    html = f"""<!doctype html><html><head><meta charset="utf-8"/><title>CV vs VLM tag→map</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#f4f2ee;color:#1a1a1a}}
h1{{margin:0 0 6px}} .sub{{color:#666;margin-bottom:20px}}
.card{{background:#fff;border-radius:12px;padding:16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.row{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}
img{{width:100%;border-radius:8px;background:#eee}}
.meta{{font-size:13px;color:#555;margin:6px 0 12px}}
figure{{margin:0}} figcaption{{font-size:11px;color:#888;margin-top:4px}}
</style></head><body>
<h1>CV vs VLM tag→map bakeoff</h1>
<p class="sub">VLM = {model} multi-pass boxes → color-matte cutouts. No OpenAI/Astra key in env — Gemini only for this run.</p>
{''.join(cards)}
</body></html>"""
    (OUT / "compare.html").write_text(html)
    print(f"\ngallery → {OUT / 'compare.html'}", flush=True)
    print(f"summary → {OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

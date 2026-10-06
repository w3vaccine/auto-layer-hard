#!/usr/bin/env python3
"""Bakeoff: CV (route=off) vs hard route (VLM→SAM soft) on soft/camo/busy prints."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from print_type import classify_print, classify_print_routed  # noqa: E402
from run_auto_layer import run  # noqa: E402
from segment import estimate_background_color, ink_map  # noqa: E402

OUT = ROOT / "out" / "hard_route_bakeoff"

CASES = [
    ("camo", ROOT / "fixtures" / "hard" / "h17_camouflage.png", "camo"),
    ("soft", ROOT / "fixtures" / "hard" / "h12_soft_watercolor.png", "soft"),
    ("busy", ROOT / "fixtures" / "showcase" / "s08_ivy_trail.png", "busy"),
]


def _metrics(run_dir: Path) -> dict:
    scene = json.loads((run_dir / "scene.json").read_text())
    st = scene.get("stats") or {}
    rgb = np.array(Image.open(run_dir / "original.png").convert("RGB"))
    union = np.array(Image.open(run_dir / "union_mask.png").convert("L"))
    bg = estimate_background_color(rgb)
    ink = ink_map(rgb, bg)
    captured = (union > 20) & ink
    precision = float(captured.sum() / max(1, (union > 20).sum()))
    recall = float(captured.sum() / max(1, ink.sum()))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "layers": len(scene.get("layers") or []),
        "high": st.get("high_confidence"),
        "uncertain": st.get("uncertain"),
        "coverage": st.get("ink_coverage"),
        "residual_frac": st.get("residual_frac"),
        "print_type": st.get("print_type"),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "background_quality": st.get("background_quality"),
    }


def _side_by_side(cv_dir: Path, hard_dir: Path, out: Path) -> None:
    imgs = []
    for d, label in ((cv_dir, "CV"), (hard_dir, "HARD")):
        ov = d / "discover_overlay.png"
        if ov.is_file():
            im = Image.open(ov).convert("RGB")
            draw = ImageDraw.Draw(im)
            draw.rectangle([0, 0, 120, 28], fill=(0, 0, 0))
            draw.text((8, 6), label, fill=(255, 255, 255))
            imgs.append(im)
    if len(imgs) == 2:
        w = imgs[0].width + imgs[1].width
        h = max(imgs[0].height, imgs[1].height)
        canvas = Image.new("RGB", (w, h), (240, 240, 240))
        canvas.paste(imgs[0], (0, 0))
        canvas.paste(imgs[1], (imgs[0].width, 0))
        canvas.save(out)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows = []

    for label, path, force_type in CASES:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        print(f"\n=== {label}: {path.name} ===")
        im = Image.open(path).convert("RGB")
        if max(im.size) > 1280:
            s = 1280 / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.Resampling.LANCZOS)
        heur = classify_print(im)
        print(f"  heuristic → {heur.print_type} ({heur.reason})")

        case_dir = OUT / label
        cv_dir = case_dir / "cv"
        hard_dir = case_dir / "hard"

        t0 = time.time()
        run(
            path,
            cv_dir,
            use_sam=False,
            use_qa=False,
            use_tile_gapfill=False,
            use_gemini_inpaint=False,
            route="off",
        )
        cv_sec = time.time() - t0
        cv_m = _metrics(cv_dir)
        cv_m["sec"] = round(cv_sec, 2)

        t0 = time.time()
        run(
            path,
            hard_dir,
            use_sam=True,
            use_qa=False,
            use_tile_gapfill=False,
            use_gemini_inpaint=False,
            route=force_type,  # force the intended hard type for bakeoff
            use_vlm_classify=False,
        )
        hard_sec = time.time() - t0
        hard_m = _metrics(hard_dir)
        hard_m["sec"] = round(hard_sec, 2)

        _side_by_side(cv_dir, hard_dir, case_dir / "compare.png")
        row = {
            "id": label,
            "file": path.name,
            "heuristic_type": heur.print_type,
            "forced_type": force_type,
            "cv": cv_m,
            "hard": hard_m,
            "f1_delta": round(hard_m["f1"] - cv_m["f1"], 4),
            "coverage_delta": round((hard_m["coverage"] or 0) - (cv_m["coverage"] or 0), 4),
        }
        rows.append(row)
        print(
            f"  CV  f1={cv_m['f1']:.3f} cov={cv_m['coverage']:.3f} layers={cv_m['layers']} ({cv_m['sec']}s)"
        )
        print(
            f"  HARD f1={hard_m['f1']:.3f} cov={hard_m['coverage']:.3f} layers={hard_m['layers']} ({hard_m['sec']}s)"
            f"  Δf1={row['f1_delta']:+.3f}"
        )

    report = {"runs": rows}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))

    # Simple HTML gallery
    cards = []
    for r in rows:
        cards.append(
            f"""<div class="card">
            <h3>{r['id']} · {r['file']}</h3>
            <p>heuristic={r['heuristic_type']} forced={r['forced_type']}
               Δf1={r['f1_delta']:+.3f} Δcov={r['coverage_delta']:+.3f}</p>
            <p>CV: f1={r['cv']['f1']} layers={r['cv']['layers']} ·
               HARD: f1={r['hard']['f1']} layers={r['hard']['layers']}</p>
            <img src="{r['id']}/compare.png" style="width:100%;max-width:1100px"/>
            </div>"""
        )
    html = f"""<!doctype html><html><body style="font-family:system-ui;padding:24px;background:#f4f3f6">
    <h1>Hard route bakeoff · CV vs VLM→SAM soft</h1>
    {''.join(cards)}
    </body></html>"""
    (OUT / "compare.html").write_text(html)
    print(f"\nWrote {OUT / 'summary.json'}")
    print(f"Gallery {OUT / 'compare.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

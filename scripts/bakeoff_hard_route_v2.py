#!/usr/bin/env python3
"""Bakeoff v2: CV vs hybrid hard route (quality ship gate)."""

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

from print_type import classify_print  # noqa: E402
from run_auto_layer import run  # noqa: E402
from segment import estimate_background_color, ink_map  # noqa: E402

OUT = ROOT / "out" / "hard_route_bakeoff_v2"

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
    high = int(st.get("high_confidence") or 0)
    layers = len(scene.get("layers") or [])
    return {
        "layers": layers,
        "high": high,
        "uncertain": st.get("uncertain"),
        "high_ratio": round(high / max(1, layers), 3),
        "coverage": round(float(st.get("ink_coverage") or 0), 4),
        "residual_frac": round(float(st.get("residual_frac") or 0), 4),
        "print_type": st.get("print_type"),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "method": ((st.get("route") or {}).get("hard") or {}).get("method"),
    }


def _side_by_side(cv_dir: Path, hard_dir: Path, out: Path) -> None:
    imgs = []
    for d, label in ((cv_dir, "CV"), (hard_dir, "HARD")):
        ov = d / "discover_overlay.png"
        if ov.is_file():
            im = Image.open(ov).convert("RGB")
            draw = ImageDraw.Draw(im)
            draw.rectangle([0, 0, 140, 28], fill=(0, 0, 0))
            draw.text((8, 6), label, fill=(255, 255, 255))
            imgs.append(im)
    if len(imgs) == 2:
        w = imgs[0].width + imgs[1].width
        h = max(imgs[0].height, imgs[1].height)
        canvas = Image.new("RGB", (w, h), (240, 240, 240))
        canvas.paste(imgs[0], (0, 0))
        canvas.paste(imgs[1], (imgs[0].width, 0))
        canvas.save(out)


def _ship_score(cv: dict, hard: dict, kind: str) -> dict:
    wins: list[str] = []
    fails: list[str] = []
    if kind == "soft":
        ok = hard["f1"] >= cv["f1"] - 0.02 and hard["coverage"] >= 0.97
        if ok:
            wins.append("f1_parity")
        else:
            fails.append("f1_or_cov")
    elif kind == "busy":
        ok = hard["layers"] < cv["layers"] * 0.55 and (
            hard["coverage"] >= cv["coverage"] - 0.1 or hard["f1"] >= cv["f1"] - 0.1
        )
        if hard["layers"] < cv["layers"] * 0.55:
            wins.append("fewer_layers")
        if hard["coverage"] >= cv["coverage"] - 0.1:
            wins.append("coverage_close")
        if not ok:
            fails.append("busy_gate")
    else:  # camo
        ok = hard["coverage"] >= max(0.78, cv["coverage"] - 0.08) or (
            hard["f1"] >= cv["f1"] - 0.06 and hard["high"] >= cv["high"]
        )
        if ok:
            wins.append("camo_ok")
        else:
            fails.append("still_behind")
    return {"ship_ok": ok, "wins": wins, "fails": fails}


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows = []

    for label, path, force_type in CASES:
        if not path.is_file():
            continue
        print(f"\n=== {label}: {path.name} ===")
        im = Image.open(path).convert("RGB")
        if max(im.size) > 1280:
            s = 1280 / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.Resampling.LANCZOS)
        heur = classify_print(im)
        print(f"  heuristic → {heur.print_type}")

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
        cv_m = _metrics(cv_dir)
        cv_m["sec"] = round(time.time() - t0, 2)

        t0 = time.time()
        run(
            path,
            hard_dir,
            use_sam=True,
            use_qa=False,
            use_tile_gapfill=False,
            use_gemini_inpaint=False,
            route=force_type,
            use_vlm_classify=False,
            max_instances=220,
        )
        hard_m = _metrics(hard_dir)
        hard_m["sec"] = round(time.time() - t0, 2)

        _side_by_side(cv_dir, hard_dir, case_dir / "compare.png")
        gate = _ship_score(cv_m, hard_m, label)
        row = {
            "id": label,
            "file": path.name,
            "heuristic_type": heur.print_type,
            "cv": cv_m,
            "hard": hard_m,
            "f1_delta": round(hard_m["f1"] - cv_m["f1"], 4),
            "coverage_delta": round(hard_m["coverage"] - cv_m["coverage"], 4),
            "gate": gate,
        }
        rows.append(row)
        print(
            f"  CV   f1={cv_m['f1']:.3f} cov={cv_m['coverage']:.3f} "
            f"layers={cv_m['layers']} high={cv_m['high']}"
        )
        print(
            f"  HARD f1={hard_m['f1']:.3f} cov={hard_m['coverage']:.3f} "
            f"layers={hard_m['layers']} high={hard_m['high']}  "
            f"Δf1={row['f1_delta']:+.3f} ship_ok={gate['ship_ok']}"
        )

    report = {
        "runs": rows,
        "all_ship_ok": all(r["gate"]["ship_ok"] for r in rows),
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    cards = []
    for r in rows:
        g = r["gate"]
        color = "#2a6f5b" if g["ship_ok"] else "#a33"
        cards.append(
            f"""<div style="border-left:4px solid {color};padding:12px;margin:12px 0;background:#fff">
            <h3>{r['id']} · {r['file']} · {'SHIP OK' if g['ship_ok'] else 'NEEDS WORK'}</h3>
            <p>CV f1={r['cv']['f1']} cov={r['cv']['coverage']} L={r['cv']['layers']} H={r['cv']['high']}<br/>
               HARD f1={r['hard']['f1']} cov={r['hard']['coverage']} L={r['hard']['layers']} H={r['hard']['high']}
               Δf1={r['f1_delta']:+.3f} Δcov={r['coverage_delta']:+.3f}</p>
            <img src="{r['id']}/compare.png" style="width:100%;max-width:1100px"/>
            </div>"""
        )
    (OUT / "compare.html").write_text(
        "<!doctype html><html><body style='font-family:system-ui;background:#f4f3f6;padding:24px'>"
        "<h1>Hard route bakeoff v2 (hybrid)</h1>"
        f"<p>all_ship_ok={report['all_ship_ok']}</p>{''.join(cards)}</body></html>"
    )
    print(f"\nall_ship_ok={report['all_ship_ok']}")
    print(f"Wrote {OUT / 'summary.json'}")
    return 0 if report["all_ship_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

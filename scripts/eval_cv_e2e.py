#!/usr/bin/env python3
"""E2E CV reliability eval across good / showcase / hard fixtures."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segment import (  # noqa: E402
    discover_by_components,
    estimate_background_color,
    ink_map,
    overlay_instances,
)
from isolate_v2 import isolate_instances, layers_to_dicts  # noqa: E402
from inpaint_v2 import reconstruct_background_v2, save_mask  # noqa: E402
from scene import write_scene  # noqa: E402

OUT = ROOT / "out" / "cv_e2e"
MAX_SIDE = 1280


def metrics(union: np.ndarray, rgb: np.ndarray, bg: np.ndarray) -> dict:
    ink = ink_map(rgb, bg)
    u = union > 20
    ink_n = int(ink.sum()) or 1
    u_n = int(u.sum()) or 1
    hit = int((u & ink).sum())
    miss = int((ink & ~u).sum())
    extra = int((u & ~ink).sum())
    # edge precision: among mask boundary pixels, fraction that are ink
    k = np.ones((3, 3), np.uint8)
    edge = (cv2_dilate(u.astype(np.uint8)) > 0) & ~u
    # simpler: fraction of union that is ink
    precision = hit / u_n
    recall = hit / ink_n
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "coverage": hit / ink_n,  # recall alias
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "miss_frac": miss / ink_n,
        "extra_frac": extra / u_n,
        "ink_frac": ink_n / rgb.shape[0] / rgb.shape[1],
    }


def cv2_dilate(m: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)


def run_one(path: Path, out: Path, *, max_instances: int = 900) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "layers").mkdir()
    image = Image.open(path).convert("RGB")
    if max(image.size) > MAX_SIDE:
        s = MAX_SIDE / max(image.size)
        image = image.resize((int(image.width * s), int(image.height * s)), Image.Resampling.LANCZOS)
    image.save(out / "original.png")
    t0 = time.time()
    inst = discover_by_components(image, max_instances=max_instances)
    for i, x in enumerate(inst, 1):
        x.id = f"m{i:03d}"
    overlay_instances(image, inst).save(out / "discover_overlay.png")
    layers, union = isolate_instances(image, inst, out / "layers")
    save_mask(union, out / "union_mask.png")
    rgb = np.array(image)
    bg = estimate_background_color(rgb)
    m = metrics(union, rgb, bg)
    bg_img, meta = reconstruct_background_v2(image, union, use_gemini=False)
    bg_img.save(out / "background.png")
    write_scene(
        out,
        source_name=path.name,
        width=image.width,
        height=image.height,
        layers=layers_to_dicts(layers),
        discover_count=len(inst),
        background_meta=meta,
        coverage=m["coverage"],
    )
    # residual preview
    residual = (ink_map(rgb, bg) & ~(union > 20)).astype(np.uint8) * 255
    Image.fromarray(residual).save(out / "residual.png")
    return {
        "id": path.stem,
        "file": str(path.relative_to(ROOT)),
        "layers": len(layers),
        "discovered": len(inst),
        "sec": round(time.time() - t0, 2),
        **{k: round(float(v), 4) for k, v in m.items()},
        "ok": True,
    }


def collect_inputs() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    goods = [
        "06-seashells.png",
        "01-ditsy-florals.png",
        "03-paisley.png",
        "02-tropical-leaves.png",
        "04-gingham-floral.png",
        "08-abstract-brush.png",
    ]
    for name in goods:
        p = ROOT / "fixtures" / name
        if p.exists():
            items.append(("good", p))
    for p in sorted((ROOT / "fixtures" / "showcase").glob("s*.png")):
        items.append(("showcase", p))
    for p in sorted((ROOT / "fixtures" / "hard").glob("h*.png")):
        items.append(("hard", p))
    return items


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    out_root = OUT / tag
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    items = collect_inputs()
    print(f"e2e CV [{tag}]: {len(items)} prints", flush=True)
    rows: list[dict] = []
    for i, (group, path) in enumerate(items, 1):
        t0 = time.time()
        try:
            st = run_one(path, out_root / group / path.stem)
            st["group"] = group
            rows.append(st)
            print(
                f"[{i:02d}/{len(items)}] {group}/{path.stem}: "
                f"L={st['layers']:3d} F1={st['f1']:.3f} cov={st['coverage']:.1%} "
                f"prec={st['precision']:.1%} miss={st['miss_frac']:.1%} {st['sec']}s",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "id": path.stem,
                    "group": group,
                    "ok": False,
                    "error": str(e),
                    "layers": 0,
                    "f1": 0,
                    "coverage": 0,
                    "precision": 0,
                    "sec": round(time.time() - t0, 2),
                }
            )
            print(f"[{i:02d}/{len(items)}] FAIL {path.stem}: {e}", flush=True)

    (out_root / "summary.json").write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if r.get("ok")]
    by = {}
    for r in ok:
        by.setdefault(r["group"], []).append(r)

    print("\n=== SUMMARY ===", flush=True)
    for g, rs in by.items():
        print(
            f"  {g:9s} n={len(rs):2d}  avgF1={sum(r['f1'] for r in rs)/len(rs):.3f}  "
            f"avgCov={sum(r['coverage'] for r in rs)/len(rs):.1%}  "
            f"avgL={sum(r['layers'] for r in rs)/len(rs):.0f}  "
            f"avgSec={sum(r['sec'] for r in rs)/len(rs):.1f}",
            flush=True,
        )
    worst = sorted(ok, key=lambda r: r["f1"])[:12]
    print("\n=== WORST F1 ===", flush=True)
    for r in worst:
        print(
            f"  {r['f1']:.3f} cov={r['coverage']:.1%} prec={r['precision']:.1%} "
            f"L={r['layers']:3d}  {r['group']}/{r['id']}",
            flush=True,
        )

    # HTML gallery of worst 8
    cards = []
    for r in worst[:8]:
        rel = f"{r['group']}/{r['id']}"
        cards.append(
            f"""<div class="card"><h3>{rel}</h3>
            <div class="meta">F1 {r['f1']:.3f} · cov {r['coverage']*100:.1f}% · prec {r['precision']*100:.1f}% · L {r['layers']}</div>
            <div class="row">
              <figure><img src="{rel}/original.png"/><figcaption>original</figcaption></figure>
              <figure><img src="{rel}/discover_overlay.png"/><figcaption>layers</figcaption></figure>
              <figure><img src="{rel}/union_mask.png"/><figcaption>union</figcaption></figure>
              <figure><img src="{rel}/residual.png"/><figcaption>residual ink</figcaption></figure>
              <figure><img src="{rel}/background.png"/><figcaption>background</figcaption></figure>
            </div></div>"""
        )
    (out_root / "compare.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"/><title>CV e2e {tag}</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#f4f2ee}}
.card{{background:#fff;border-radius:12px;padding:14px;margin-bottom:14px}}
.row{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}
img{{width:100%;border-radius:8px;background:#eee}}
.meta{{font-size:13px;color:#555;margin:4px 0 10px}}
</style></head><body>
<h1>CV e2e — {tag}</h1>
<p>Worst F1 cases first</p>
{''.join(cards)}
</body></html>"""
    )
    print(f"\ngallery → {out_root / 'compare.html'}", flush=True)


if __name__ == "__main__":
    main()

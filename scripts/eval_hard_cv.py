#!/usr/bin/env python3
"""CV-only hard-fixture eval (no Gemini / SAM imports at runtime)."""

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
    chroma_distance,
    discover_by_components,
    estimate_background_color,
    lab_distance,
    overlay_instances,
)
from isolate_v2 import isolate_instances, layers_to_dicts  # noqa: E402
from inpaint_v2 import reconstruct_background_v2, save_mask  # noqa: E402
from scene import write_scene  # noqa: E402


def ink_cov(union: np.ndarray, rgb: np.ndarray, bg: np.ndarray) -> float:
    """Coverage vs the same ink definition used by build_ink_mask."""
    lab = lab_distance(rgb, bg)
    chr_d = chroma_distance(rgb, bg)
    lab_frac = float((lab > 14).mean())
    chr_frac = float((chr_d > 8).mean())
    ink = (chr_d > 8) if (lab_frac > 0.42 and chr_frac < lab_frac * 0.55) else (lab > 14)
    if not ink.any():
        return 0.0
    return float(((union > 20) & ink).sum() / ink.sum())


def run_cv(path: Path, out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    layers_dir = out / "layers"
    layers_dir.mkdir()

    image = Image.open(path).convert("RGB")
    if max(image.size) > 1280:
        s = 1280 / max(image.size)
        image = image.resize((int(image.width * s), int(image.height * s)))
    image.save(out / "original.png")

    inst = discover_by_components(image, max_instances=200)
    for i, x in enumerate(inst, 1):
        x.id = f"m{i:03d}"
    overlay_instances(image, inst).save(out / "discover_overlay.png")
    layers, union = isolate_instances(image, inst, layers_dir)
    save_mask(union, out / "union_mask.png")

    rgb = np.array(image)
    bg = estimate_background_color(rgb)
    cov = ink_cov(union, rgb, bg)
    background, meta = reconstruct_background_v2(image, union, use_gemini=False)
    background.save(out / "background.png")
    write_scene(
        out,
        source_name=path.name,
        width=image.width,
        height=image.height,
        layers=layers_to_dicts(layers),
        discover_count=len(inst),
        background_meta=meta,
        coverage=cov,
    )
    return {
        "layers": len(layers),
        "discovered": len(inst),
        "coverage": cov,
        "bg": meta.get("method"),
    }


def main() -> None:
    hard = sorted((ROOT / "fixtures" / "hard").glob("h*.png"))
    out_root = ROOT / "out" / "hard_eval"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print(f"hard fixtures: {len(hard)}", flush=True)
    for i, p in enumerate(hard, 1):
        t0 = time.time()
        try:
            st = run_cv(p, out_root / p.stem)
            st.update(id=p.stem, sec=round(time.time() - t0, 2), ok=True)
            rows.append(st)
            print(
                f"[{i:02d}/{len(hard)}] {p.stem}: layers={st['layers']} "
                f"cov={st['coverage']:.1%} {st['sec']}s",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "id": p.stem,
                    "ok": False,
                    "error": str(e),
                    "layers": 0,
                    "coverage": 0.0,
                    "sec": round(time.time() - t0, 2),
                }
            )
            print(f"[{i:02d}/{len(hard)}] FAIL {p.stem}: {e}", flush=True)

    (out_root / "summary.json").write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if r.get("ok")]
    ok.sort(key=lambda r: r["coverage"])
    print("\n=== WORST ===", flush=True)
    for r in ok[:12]:
        print(f"  {r['coverage']:.1%}  L={r['layers']:3d}  {r['id']}", flush=True)
    print("\n=== ZERO/FEW LAYERS ===", flush=True)
    for r in ok:
        if r["layers"] < 8:
            print(f"  L={r['layers']} cov={r['coverage']:.1%}  {r['id']}", flush=True)
    if ok:
        print(
            f"\navg cov {sum(r['coverage'] for r in ok)/len(ok):.1%}  "
            f"avg L {sum(r['layers'] for r in ok)/len(ok):.0f}",
            flush=True,
        )

    print("\n=== REGRESSION CHECK ===", flush=True)
    goods = [
        ("seashells", ROOT / "fixtures" / "06-seashells.png"),
        ("ditsy", ROOT / "fixtures" / "01-ditsy-florals.png"),
        ("paisley", ROOT / "fixtures" / "03-paisley.png"),
    ]
    for name, path in goods:
        if not path.exists():
            print(f"  skip missing {name}", flush=True)
            continue
        st = run_cv(path, out_root / f"_good_{name}")
        print(f"  {name}: L={st['layers']} cov={st['coverage']:.1%}", flush=True)


if __name__ == "__main__":
    main()

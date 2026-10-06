#!/usr/bin/env python3
"""QA a live auto-layer-hard job: download layers, score, write contact sheet."""
from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

MUSTARD = (214, 181, 103)


def qa_job(job_id: str, out: Path) -> dict:
    base = f"https://auto-layer-hard.onrender.com/jobs/{job_id}/cv"
    out.mkdir(parents=True, exist_ok=True)
    (out / "layers").mkdir(exist_ok=True)
    urllib.request.urlretrieve(f"{base}/scene.json", out / "scene.json")
    s = json.loads((out / "scene.json").read_text())
    hard = s["stats"]["background"]["route"]["hard"]
    rows = []
    for L in s["layers"]:
        path = out / "layers" / f"{L['id']}.png"
        urllib.request.urlretrieve(f"{base}/{L['src']}", path)
        a = np.array(Image.open(path).convert("RGBA"))
        mask = a[:, :, 3] > 128
        opaque = float(mask.mean())
        lab, n = ndimage.label(mask)
        sizes = (
            sorted(((lab == i).sum() for i in range(1, n + 1)), reverse=True)
            if n
            else [0]
        )
        main_frac = sizes[0] / max(1, int(mask.sum()))
        h, w = mask.shape
        pad = max(2, min(h, w) // 20)
        border = np.zeros_like(mask)
        border[:pad] = True
        border[-pad:] = True
        border[:, :pad] = True
        border[:, -pad:] = True
        border_opaque = float((mask & border).sum()) / max(1, int(border.sum()))
        # area_frac in scene.json is bbox area (w*h), not mask coverage
        area_frac = float(L.get("area_frac", 0))
        mask_canvas = area_frac * opaque
        flags = []
        if mask_canvas > 0.18 or (area_frac > 0.40 and opaque > 0.35):
            flags.append("huge_area")
        if n > 4 and main_frac < 0.8:
            flags.append("multi_cc")
        # Significant secondary blob = multi-motif glue (ignore tiny leaf shards/holes)
        if (
            n >= 2
            and len(sizes) >= 2
            and sizes[1] > 0.22 * max(1, sizes[0])
            and area_frac > 0.04
        ):
            flags.append("multi_motif")
        if border_opaque > 0.45 and area_frac > 0.12:
            flags.append("border_bleed")
        if opaque < 0.03 and area_frac < 0.015:
            flags.append("tiny")
        # Large bbox with low/medium fill = multi-motif glue (mustard m013)
        if area_frac > 0.17 and (opaque < 0.50 or n >= 3):
            flags.append("sparse_giant")
        serious_flags = {"huge_area", "sparse_giant", "border_bleed", "multi_motif"}
        if any(f in serious_flags for f in flags) and "multi_motif" in flags and area_frac < 0.03:
            flags = [f for f in flags if f != "multi_motif"]
        rows.append(
            {
                **L,
                "opaque": round(opaque, 3),
                "n_cc": int(n),
                "main_frac": round(float(main_frac), 3),
                "border_opaque": round(border_opaque, 3),
                "mask_canvas": round(mask_canvas, 3),
                "flags": flags,
                "path": str(path),
            }
        )

    flagged = [r for r in rows if r["flags"]]
    ok = [r for r in rows if not r["flags"]]
    giants = sum(1 for r in rows if "huge_area" in r["flags"])
    n_high = sum(1 for r in rows if r["confidence_tier"] == "high")
    # Mustard tropical pass bar — clean single motifs; residual sheet OK for leftovers
    residual = float(s["stats"].get("residual_frac") or 0)
    cov = float(s["stats"].get("ink_coverage") or 0)
    serious = sum(
        1
        for r in flagged
        if any(f in r["flags"] for f in ("huge_area", "sparse_giant", "border_bleed"))
    )
    pass_ok = (
        giants == 0
        and serious == 0
        and 8 <= len(rows) <= 24
        and n_high >= 8
        and (
            cov >= 0.85
            or (cov >= 0.55 and n_high >= 12 and residual <= 0.40)
        )
    )

    # contact sheet top high
    top = sorted(
        [r for r in rows if r["confidence_tier"] == "high"],
        key=lambda x: -x["area_frac"],
    )[:12]
    if len(top) < 8:
        top = sorted(rows, key=lambda x: -x["area_frac"])[:12]
    cell = 256
    cols = 4
    rn = (len(top) + cols - 1) // cols
    sheet = Image.new("RGB", (cell * cols, cell * rn), (30, 30, 30))
    dr = ImageDraw.Draw(sheet)
    for i, r in enumerate(top):
        a = np.array(Image.open(r["path"]).convert("RGBA")).astype(np.float32) / 255
        bg = np.zeros_like(a)
        bg[:, :, 0] = MUSTARD[0] / 255
        bg[:, :, 1] = MUSTARD[1] / 255
        bg[:, :, 2] = MUSTARD[2] / 255
        bg[:, :, 3] = 1
        al = a[:, :, 3:4]
        comp = np.clip((a[:, :, :3] * al + bg[:, :, :3] * (1 - al)) * 255, 0, 255).astype(
            np.uint8
        )
        im = Image.fromarray(comp)
        im.thumbnail((cell - 8, cell - 40))
        x = (i % cols) * cell
        y = (i // cols) * cell
        sheet.paste(im, (x + 4, y + 20))
        dr.text(
            (x + 4, y + 2),
            f"{r['id']} {','.join(r['flags']) or 'ok'}",
            fill=(255, 220, 0),
        )
    sheet.save(out / "contact_top.jpg", quality=90)

    summary = {
        "job": job_id,
        "pass": pass_ok,
        "n_layers": len(rows),
        "n_ok": len(ok),
        "n_flagged": len(flagged),
        "giants": giants,
        "high": n_high,
        "coverage": s["stats"].get("ink_coverage"),
        "residual": s["stats"].get("residual_frac"),
        "hard": hard,
        "flag_counts": dict(Counter(f for r in flagged for f in r["flags"])),
        "editor": f"https://auto-layer-hard.onrender.com/jobs/{job_id}/cv/editor.html",
        "layers": [{k: v for k, v in r.items() if k != "path"} for r in rows],
    }
    (out / "qa.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    job = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 else f"/tmp/mtd_qa_{job}")
    s = qa_job(job, out)
    print(json.dumps({k: s[k] for k in s if k != "layers"}, indent=2))
    print("PASS" if s["pass"] else "FAIL")

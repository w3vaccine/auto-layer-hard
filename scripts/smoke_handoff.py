#!/usr/bin/env python3
"""Handoff smoke test — seashells + showcase + hard.

Asserts scene contract, residual sheet, confidence ranking, and Layers Menu API.

Usage:
  cd auto-layer-poc
  .venv/bin/python scripts/smoke_handoff.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from layers_menu import (  # noqa: E402
    accept_motifs,
    build_motif_pack_zip,
    create_document_from_scene,
    layers_menu_view,
)
from run_auto_layer import run  # noqa: E402

OUT = ROOT / "out" / "smoke_handoff"
FIXTURES = [
    ("good", ROOT / "fixtures" / "06-seashells.png", {"min_high": 20, "max_residual": 0.05}),
    ("showcase", ROOT / "fixtures" / "showcase" / "s04_coastal_shells.png", {"min_high": 5, "max_residual": 0.12}),
    ("hard", ROOT / "fixtures" / "hard" / "h12_soft_watercolor.png", {"min_high": 0, "max_residual": 0.55, "allow_partial": True}),
]


def _assert(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)
        print(f"  ✗ {msg}")
    else:
        print(f"  ✓ {msg}")


def check_scene(scene: dict, rules: dict, errors: list[str]) -> None:
    st = scene.get("stats") or {}
    _assert(scene.get("kind") == "auto_layer_proposal", "kind=auto_layer_proposal", errors)
    _assert(scene.get("original") == "original.png", "original preserved", errors)
    _assert(scene.get("base") == "base.png", "base present", errors)
    _assert(isinstance(scene.get("residual"), dict), "residual sheet block", errors)
    res = scene["residual"]
    _assert(res.get("locked") is True, "residual locked", errors)
    _assert(res.get("src") == "residual_ink.png", "residual_ink.png src", errors)
    _assert("layers" in scene and len(scene["layers"]) > 0, "has layers", errors)

    high = [L for L in scene["layers"] if L.get("confidence_tier") == "high"]
    unc = [L for L in scene["layers"] if L.get("confidence_tier") == "uncertain"]
    _assert(st.get("high_confidence") == len(high), "stats.high_confidence matches", errors)
    _assert(st.get("uncertain") == len(unc), "stats.uncertain matches", errors)
    _assert(len(high) >= rules["min_high"], f"high>={rules['min_high']} (got {len(high)})", errors)
    residual = float(st.get("residual_frac") or 0)
    _assert(residual <= rules["max_residual"], f"residual<={rules['max_residual']} (got {residual:.3f})", errors)

    # Sorted high-first
    tiers = [L.get("confidence_tier") for L in scene["layers"]]
    if "high" in tiers and "uncertain" in tiers:
        first_unc = tiers.index("uncertain")
        _assert(all(t == "high" for t in tiers[:first_unc]), "layers sorted high-first", errors)

    for L in scene["layers"][:3]:
        _assert("rank_score" in L, f"{L.get('id')} has rank_score", errors)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    errors: list[str] = []
    summary = []

    for label, path, rules in FIXTURES:
        if not path.is_file():
            errors.append(f"missing fixture {path}")
            print(f"\n[{label}] MISSING {path}")
            continue
        print(f"\n[{label}] {path.name}")
        run_dir = OUT / label
        t0 = time.time()
        run(
            path,
            run_dir,
            use_sam=False,
            use_qa=False,
            use_tile_gapfill=False,
            use_gemini_inpaint=False,
            max_instances=900,
        )
        elapsed = time.time() - t0
        scene = json.loads((run_dir / "scene.json").read_text())
        print(f"  runtime {elapsed:.1f}s · layers={len(scene['layers'])} · "
              f"high={scene['stats'].get('high_confidence')} · "
              f"residual={scene['stats'].get('residual_frac', 0):.3f}")
        check_scene(scene, rules, errors)

        for name in ("original.png", "base.png", "background.png", "residual_ink.png", "editor.html"):
            _assert((run_dir / name).is_file(), f"artifact {name}", errors)

        # Layers Menu: create doc, accept all high, build pack
        doc = create_document_from_scene(ROOT / "out", run_dir, name=f"smoke-{label}", pipeline="cv")
        high_layers = [L for L in scene["layers"] if L.get("confidence_tier") == "high"]
        accept_list = high_layers if high_layers else scene["layers"][: min(5, len(scene["layers"]))]
        result = accept_motifs(
            ROOT / "out",
            doc["id"],
            accept_list,
            scene_dir=run_dir,
            proposal_id=scene.get("proposal_id"),
        )
        view = layers_menu_view(ROOT / "out", doc["id"])
        _assert(view["count"] == len(accept_list), f"layers menu count={len(accept_list)}", errors)
        pack = OUT / f"motif_pack_{label}.zip"
        build_motif_pack_zip(ROOT / "out", doc["id"], pack)
        _assert(pack.is_file() and pack.stat().st_size > 100, "motif pack zip written", errors)

        summary.append(
            {
                "label": label,
                "fixture": path.name,
                "layers": len(scene["layers"]),
                "high": scene["stats"].get("high_confidence"),
                "uncertain": scene["stats"].get("uncertain"),
                "residual_frac": scene["stats"].get("residual_frac"),
                "background_quality": scene["stats"].get("background_quality"),
                "document_id": doc["id"],
                "accepted": len(result["accepted_ids"]),
                "runtime_s": round(elapsed, 2),
                "partial": bool(rules.get("allow_partial")) or scene["stats"].get("partial"),
            }
        )

    report = {"ok": not errors, "errors": errors, "runs": summary}
    (OUT / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== SMOKE REPORT ===")
    print(json.dumps(report, indent=2))
    if errors:
        print(f"\nFAILED with {len(errors)} errors")
        return 1
    print("\nPASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build handoff/ folder for teammate: contracts + sample artifacts."""

from __future__ import annotations

import json
import shutil
import sys
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

HANDOFF = ROOT / "handoff"


CONTRACT = """# Auto Layer → Layers Menu contract (cheat sheet)

## Two objects

1. **Proposal** (`scene.json`) — ephemeral Auto Layer output. Review only.
2. **Document** (`document.json`) — durable Layers Menu. Accepted motifs live here.

Rerun Auto Layer → attach new proposal. Never wipe `document.layers` on rerun.

## Accept payload

```http
POST /api/documents/:id/accept
Content-Type: application/json

{
  "proposal_id": "…",
  "job_id": "…",          // optional; resolves layer PNG paths
  "pipeline": "cv",
  "motifs": [
    {
      "id": "m001",
      "label": "motif",
      "src": "layers/m001.png",
      "bbox_px": [10, 20, 100, 120],
      "transform": {"x": 10, "y": 20, "width": 100, "height": 120, "rotation": 0},
      "confidence_tier": "high"
    }
  ]
}
```

## Layers Menu view

```http
GET /api/documents/:id/layers-menu
```

Returns base/original URLs + accepted layers with `url` for each asset.

## Motif pack

```http
GET /api/documents/:id/motif-pack
→ ZIP { motifs.json, motifs/*.png, base.png? }
```
"""


def main() -> None:
    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    HANDOFF.mkdir()

    # Prefer smoke seashells scene; else rerank; else run quick
    candidates = [
        ROOT / "out" / "smoke_handoff" / "good",
        ROOT / "out" / "jobs" / "spec_check_seashells_rerank",
        ROOT / "out" / "jobs" / "spec_check_seashells",
    ]
    scene_dir = next((p for p in candidates if (p / "scene.json").is_file()), None)
    if scene_dir is None:
        from run_auto_layer import run

        scene_dir = HANDOFF / "_tmp_scene"
        run(
            ROOT / "fixtures" / "06-seashells.png",
            scene_dir,
            use_sam=False,
            use_qa=False,
            use_tile_gapfill=False,
            use_gemini_inpaint=False,
        )

    scene = json.loads((scene_dir / "scene.json").read_text())
    shutil.copy2(ROOT / "README.md", HANDOFF / "README.md")
    shutil.copy2(ROOT / "KNOWN_LIMITS.md", HANDOFF / "KNOWN_LIMITS.md")
    (HANDOFF / "CONTRACT.md").write_text(CONTRACT)
    (HANDOFF / "sample_scene.json").write_text(json.dumps(scene, indent=2))

    doc = create_document_from_scene(
        ROOT / "out", scene_dir, name="handoff-sample-seashells", pipeline="cv", doc_id="handoff_sample"
    )
    # Force overwrite if exists
    high = [L for L in scene["layers"] if L.get("confidence_tier") == "high"][:12]
    if not high:
        high = scene["layers"][:8]
    accept_motifs(ROOT / "out", doc["id"], high, scene_dir=scene_dir, proposal_id=scene.get("proposal_id"), replace=True)
    view = layers_menu_view(ROOT / "out", doc["id"])
    (HANDOFF / "sample_layers_menu.json").write_text(json.dumps(view, indent=2))

    pack_path = HANDOFF / "sample_motif_pack.zip"
    build_motif_pack_zip(ROOT / "out", doc["id"], pack_path)
    # Also extract motifs.json for easy reading
    import zipfile

    with zipfile.ZipFile(pack_path) as zf:
        (HANDOFF / "sample_motifs.json").write_bytes(zf.read("motifs.json"))

    # Copy a few preview assets
    previews = HANDOFF / "previews"
    previews.mkdir()
    for name in ("original.png", "base.png", "residual_ink.png", "discover_overlay.png"):
        src = scene_dir / name
        if src.is_file():
            shutil.copy2(src, previews / name)

    manifest = {
        "built_from": str(scene_dir),
        "document_id": doc["id"],
        "layers_accepted": len(high),
        "files": sorted(p.name for p in HANDOFF.iterdir() if p.is_file()),
    }
    (HANDOFF / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Handoff ready → {HANDOFF}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    # Allow re-create handoff_sample doc
    sample = ROOT / "out" / "documents" / "handoff_sample"
    if sample.exists():
        shutil.rmtree(sample)
    main()

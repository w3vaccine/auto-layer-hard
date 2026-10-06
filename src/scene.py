"""Write scene.json for Auto Layer review → Layers Menu handoff.

Scene is a *proposal*: user accepts/rejects/merges before items become
native Print Studio–style motif layers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _area_frac(item: dict[str, Any]) -> float:
    bbox = item.get("bbox_norm") or [0.0, 0.0, 0.0, 0.0]
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def _rank_score(item: dict[str, Any]) -> float:
    """Composite score: matte quality + discovery conf + size/coverage proxy."""
    matte = float(item.get("matte_score") or 0.0)
    conf = float(item.get("confidence") or 0.0)
    area = _area_frac(item)
    # Soft saturate size: motifs around ~2% of canvas are "full size" for ranking
    size = min(1.0, area / 0.02)
    # Coverage contribution ≈ how much solid ink this piece owns
    coverage_proxy = min(1.0, (area * max(matte, 0.05)) / 0.015)
    return 0.40 * matte + 0.30 * conf + 0.20 * coverage_proxy + 0.10 * size


def _confidence_tier(item: dict[str, Any], rank: float) -> str:
    matte = float(item.get("matte_score") or 0.0)
    conf = float(item.get("confidence") or 0.0)
    area = _area_frac(item)
    # Tiny fragments stay uncertain even if scores look ok
    if area < 0.00025 and matte < 0.45:
        return "uncertain"
    # Clear high: solid matte + decent conf + rank
    if matte >= 0.28 and conf >= 0.45 and rank >= 0.42:
        return "high"
    # Strong matte alone can promote mid-size pieces
    if matte >= 0.40 and conf >= 0.40 and area >= 0.001:
        return "high"
    return "uncertain"


def write_scene(
    out_dir: Path,
    *,
    source_name: str,
    width: int,
    height: int,
    layers: list[dict[str, Any]],
    discover_count: int,
    background_meta: dict[str, Any],
    coverage: float,
    proposal_id: str | None = None,
    residual_frac: float = 0.0,
    background_quality: str = "approximate",
    residual_src: str | None = "residual_ink.png",
) -> Path:
    """
    Schema notes (product model):
    - original.png always preserved (flattened fallback)
    - background.png = reconstructed ground (may be approximate)
    - base.png = working base (reconstructed bg + unextracted residual ink)
    - residual sheet = locked review item for leftover ink (never silently discarded)
    - layers[].status: proposed | accepted | rejected
    - layers[].confidence_tier: high | uncertain (sorted high-first)
    """
    enriched: list[dict[str, Any]] = []
    for L in layers:
        item = dict(L)
        rank = _rank_score(item)
        tier = _confidence_tier(item, rank)
        item["rank_score"] = round(rank, 4)
        item["area_frac"] = round(_area_frac(item), 6)
        # Always recompute tier from current calibration (don't trust stale values)
        item["confidence_tier"] = tier
        item.setdefault("status", "proposed")
        item.setdefault("review_note", None)
        enriched.append(item)

    # High first (best rank), then uncertain (best rank) — review UX default order
    enriched.sort(
        key=lambda x: (
            0 if x.get("confidence_tier") == "high" else 1,
            -float(x.get("rank_score") or 0.0),
        )
    )

    high_n = sum(1 for x in enriched if x.get("confidence_tier") == "high")
    unc_n = sum(1 for x in enriched if x.get("confidence_tier") == "uncertain")

    residual_block = None
    if residual_src:
        residual_block = {
            "id": "residual_sheet",
            "kind": "residual_sheet",
            "label": "Unextracted ink (kept in base)",
            "src": residual_src,
            "locked": True,
            "status": "kept_in_base",
            "residual_frac": residual_frac,
            "empty": residual_frac < 1e-6,
        }

    scene = {
        "version": 2,
        "kind": "auto_layer_proposal",
        "proposal_id": proposal_id or out_dir.name,
        "source": source_name,
        "width": width,
        "height": height,
        # Flattened source — never destroyed; restore / compare / fallback
        "original": "original.png",
        # Reconstructed ground only (may show holes if quality unreliable)
        "background": "background.png",
        # Working canvas: bg + residual unextracted ink (partial success)
        "base": "base.png",
        # Locked residual sheet for review (same pixels kept in base)
        "residual": residual_block,
        "layers": enriched,
        "stats": {
            "discovered": discover_count,
            "isolated": len(enriched),
            "high_confidence": high_n,
            "uncertain": unc_n,
            "ink_coverage": coverage,
            "residual_frac": residual_frac,
            "background": background_meta,
            "background_quality": background_quality,
            "partial": residual_frac > 0.02 or unc_n > 0 or background_quality != "clean",
        },
        "review": {
            "accept_high_by_default": False,
            "allow_merge": True,
            "allow_reject": True,
            "preserve_original": True,
            "rerun_policy": "new_proposal",
            "uncertain_collapsed_by_default": True,
        },
        "handoff": {
            "motif_pack": {
                "kind": "motif_pack",
                "assets_dir": "motifs/",
                "manifest": "motifs.json",
            }
        },
    }
    path = out_dir / "scene.json"
    path.write_text(json.dumps(scene, indent=2))
    return path

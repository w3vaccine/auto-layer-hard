"""MTD-2396 Auto Layer POC v3 — SAM2 refine + residual QA gate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from discover import boxes_to_dicts  # noqa: E402
from hard_extract import discover_hard_motifs  # noqa: E402
from inpaint_v2 import reconstruct_background_v2, save_mask  # noqa: E402
from isolate_v2 import isolate_instances, layers_to_dicts  # noqa: E402
from print_type import classify_print_routed  # noqa: E402
from qa_gate import qa_gapfill_loop  # noqa: E402
from sam_refine import refine_instances_sam, sam_available  # noqa: E402
from scene import write_scene  # noqa: E402
from segment import (  # noqa: E402
    discover_motifs_v2,
    estimate_background_color,
    ink_map,
    instances_to_boxes,
    overlay_instances,
)


def _load_api_key() -> str | None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / ".env")
    import os

    return os.environ.get("GOOGLE_API_KEY") or None


def _ink_coverage(union_alpha: np.ndarray, rgb: np.ndarray, bg: np.ndarray) -> float:
    ink = ink_map(rgb, bg)
    if not ink.any():
        return 0.0
    captured = (union_alpha > 20) & ink
    return float(captured.sum() / ink.sum())


def _background_quality(bg_meta: dict, residual_frac: float) -> str:
    ghost = float(bg_meta.get("ghost_ink_frac") or 0.0)
    if ghost > 0.12 or residual_frac > 0.15:
        return "unreliable"
    if ghost > 0.04 or residual_frac > 0.04 or not bg_meta.get("gemini_used"):
        # Flat fill is usually fine for textile flats but not pixel-perfect.
        if ghost <= 0.04 and residual_frac <= 0.02:
            return "clean"
        return "approximate"
    return "clean"


def _compose_base(
    original: Image.Image,
    background: Image.Image,
    union_alpha: np.ndarray,
    rgb: np.ndarray,
    bg: np.ndarray,
) -> tuple[Image.Image, float]:
    """Working base = reconstructed bg + unextracted ink (never silently discard)."""
    ink = ink_map(rgb, bg)
    residual = ink & ~(union_alpha > 20)
    residual_frac = float(residual.mean())
    base = np.array(background.convert("RGB")).copy()
    orig = np.array(original.convert("RGB"))
    if residual.any():
        base[residual] = orig[residual]
    return Image.fromarray(base), residual_frac


def _copy_editor(out_dir: Path) -> None:
    web = ROOT / "web"
    for name in ("editor.html", "editor.js"):
        src = web / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def run(
    input_path: Path,
    out_dir: Path,
    *,
    max_instances: int = 900,
    use_tile_gapfill: bool = True,
    use_gemini_inpaint: bool = True,
    use_sam: bool = True,
    use_qa: bool = True,
    target_coverage: float = 0.995,
    label: bool = False,
    model: str = "gemini-2.5-flash",
    sam_model: str = "sam2_b.pt",
    route: str = "auto",  # auto | clean | soft | camo | busy | off
    use_vlm_classify: bool = True,
) -> Path:
    api_key = _load_api_key()
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not set (tile gapfill / gemini inpaint / hard VLM disabled).")

    out_dir.mkdir(parents=True, exist_ok=True)
    layers_dir = out_dir / "layers"
    if layers_dir.exists():
        shutil.rmtree(layers_dir)
    layers_dir.mkdir()

    image = Image.open(input_path).convert("RGB")
    max_side = 1280
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        w, h = image.size
        print(f"  resized to {w}x{h}")

    image.save(out_dir / "original.png")

    # --- Print-type routing ---
    route_meta: dict = {"route": route, "print_type": "clean", "hard_path": False}
    print_type = "clean"
    if route != "off":
        if route in ("clean", "soft", "camo", "busy"):
            print_type = route
            route_meta["classifier"] = "forced"
        else:
            classified = classify_print_routed(
                image, api_key=api_key, model=model, use_vlm=use_vlm_classify and bool(api_key)
            )
            print_type = classified.print_type
            route_meta["classifier"] = classified.to_dict()
        route_meta["print_type"] = print_type
        print(f"  print_type={print_type} (route={route})")

    hard_path = print_type in ("soft", "camo", "busy") and route != "off"
    route_meta["hard_path"] = hard_path

    qa_meta = {"passed": None, "history": []}
    hard_meta: dict = {}

    if hard_path:
        print(f"Discovering motifs (hard path: VLM→SAM soft · {print_type})…")
        instances, hard_meta = discover_hard_motifs(
            image,
            print_type,
            api_key=api_key,
            model=model,
            max_instances=min(max_instances, 220 if print_type == "busy" else 200),
            sam_model=sam_model,
        )
        route_meta["hard"] = hard_meta
        # Skip CV QA for camo (ink_map is wrong); light QA only for soft/busy if desired
        if use_qa and print_type in ("soft", "busy") and api_key:
            print("QA residual gate (hard path, light)…")
            instances, qa_meta = qa_gapfill_loop(
                image,
                instances,
                api_key=api_key,
                model=model,
                target_coverage=min(target_coverage, 0.98),
                max_residual_ink_frac=1.0 - min(target_coverage, 0.98),
                max_rounds=2,
                use_gemini_tiles=False,
            )
    else:
        print("Discovering motifs (CV components + optional tile gapfill)…")
        instances = discover_motifs_v2(
            image,
            api_key=api_key,
            model=model,
            max_instances=max_instances,
            use_tile_gapfill=use_tile_gapfill and bool(api_key),
            label=False,
        )

        if use_sam:
            if sam_available():
                instances = refine_instances_sam(
                    image, instances, model_name=sam_model, enabled=True
                )
            else:
                print("  SAM not installed; continuing with CV masks")

        if use_qa:
            print("QA residual gate…")
            instances, qa_meta = qa_gapfill_loop(
                image,
                instances,
                api_key=api_key,
                model=model,
                target_coverage=target_coverage,
                max_residual_ink_frac=1.0 - target_coverage,
                max_rounds=3,
                use_gemini_tiles=use_tile_gapfill and bool(api_key),
            )
            if use_sam and sam_available():
                newcomers = [i for i in instances if getattr(i, "pass_index", 0) >= 50]
                if newcomers:
                    refine_instances_sam(image, newcomers, model_name=sam_model, enabled=True)

    if label and api_key:
        from segment import label_instances

        label_instances(image, instances, api_key=api_key, model=model)

    boxes = instances_to_boxes(instances)
    (out_dir / "discover.json").write_text(json.dumps(boxes_to_dicts(boxes), indent=2))
    overlay_instances(image, instances).save(out_dir / "discover_overlay.png")
    (out_dir / "route.json").write_text(json.dumps(route_meta, indent=2))

    print("Isolating layers (exact masks)…")
    layers, union = isolate_instances(image, instances, layers_dir)
    save_mask(union, out_dir / "union_mask.png")

    rgb = np.array(image)
    bg = estimate_background_color(rgb)
    coverage = _ink_coverage(union, rgb, bg)
    print(f"  ink coverage {coverage:.1%}")

    print("Reconstructing background…")
    background, bg_meta = reconstruct_background_v2(
        image,
        union,
        api_key=api_key,
        use_gemini=use_gemini_inpaint and bool(api_key),
    )
    background.save(out_dir / "background.png")
    print(
        f"  background via {bg_meta['method']} "
        f"(ghost_ink={bg_meta.get('ghost_ink_frac', 0):.2f})"
    )

    base_img, residual_frac = _compose_base(image, background, union, rgb, bg)
    base_img.save(out_dir / "base.png")
    residual_vis = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rem = ink_map(rgb, bg) & ~(union > 20)
    residual_vis[rem, :3] = rgb[rem]
    residual_vis[rem, 3] = 255
    Image.fromarray(residual_vis).save(out_dir / "residual_ink.png")
    bg_quality = _background_quality(bg_meta, residual_frac)
    print(f"  residual ink {residual_frac:.1%} · bg quality={bg_quality}")

    scene_path = write_scene(
        out_dir,
        source_name=input_path.name,
        width=w,
        height=h,
        layers=layers_to_dicts(layers),
        discover_count=len(instances),
        background_meta={**bg_meta, "qa": qa_meta, "route": route_meta},
        coverage=coverage,
        proposal_id=out_dir.name,
        residual_frac=residual_frac,
        background_quality=bg_quality,
    )
    scene = json.loads(scene_path.read_text())
    scene["stats"]["qa_passed"] = qa_meta.get("passed")
    scene["stats"]["qa"] = {k: v for k, v in qa_meta.items() if k != "history"} | {
        "history": qa_meta.get("history", [])
    }
    scene["stats"]["print_type"] = print_type
    scene["stats"]["route"] = route_meta
    scene_path.write_text(json.dumps(scene, indent=2))

    _copy_editor(out_dir)
    print(f"Done → {scene_path} (qa_passed={qa_meta.get('passed')} print_type={print_type})")
    return scene_path


def main() -> None:
    p = argparse.ArgumentParser(description="MTD-2396 Auto Layer POC v3 (SAM2 + QA + hard route)")
    p.add_argument("--input", "-i", required=True, type=Path)
    p.add_argument("--out", "-o", required=True, type=Path)
    p.add_argument("--max-instances", type=int, default=900)
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--sam-model", default="sam2_b.pt")
    p.add_argument("--target-coverage", type=float, default=0.995)
    p.add_argument("--no-tile-gapfill", action="store_true")
    p.add_argument("--no-gemini-inpaint", action="store_true")
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--no-qa", action="store_true")
    p.add_argument("--label", action="store_true")
    p.add_argument(
        "--route",
        default="auto",
        choices=["auto", "clean", "soft", "camo", "busy", "off"],
        help="Print-type route: auto classifies; off forces legacy CV",
    )
    p.add_argument("--no-vlm-classify", action="store_true")
    p.add_argument("--max-passes", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--no-gemini-fallback", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--gemini-inpaint", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    run(
        args.input.resolve(),
        args.out.resolve(),
        max_instances=args.max_instances,
        use_tile_gapfill=not args.no_tile_gapfill,
        use_gemini_inpaint=not args.no_gemini_inpaint,
        use_sam=not args.no_sam,
        use_qa=not args.no_qa,
        target_coverage=args.target_coverage,
        label=args.label,
        model=args.model,
        sam_model=args.sam_model,
        route=args.route,
        use_vlm_classify=not args.no_vlm_classify,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run both Auto Layer pipelines: CV baseline (v2) and SAM2+QA (v3).

Outputs per print:
  out/<stem>/cv/     — initial CV-only work (no SAM, no QA)
  out/<stem>/sam2/   — SAM2 refine + residual QA gate
  out/ab_compare.html — side-by-side gallery

Example:
  .venv/bin/python run_both.py \\
    --inputs fixtures/06-seashells.png fixtures/01-ditsy-florals.png fixtures/03-paisley.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Prefer src/run_auto_layer (has run()); build_compare lives at repo root
from build_compare import build as build_compare  # noqa: E402
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "auto_layer_runner", SRC / "run_auto_layer.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
run = _mod.run


def run_pair(
    input_path: Path,
    out_root: Path,
    *,
    max_instances: int = 200,
    model: str = "gemini-2.5-flash",
    sam_model: str = "sam2_b.pt",
    target_coverage: float = 0.995,
    use_tile_gapfill: bool = False,
    use_gemini_inpaint: bool = False,
) -> tuple[Path, Path]:
    stem = input_path.stem.replace(" ", "_")
    # Normalize fixture names like 06-seashells → seashells
    for prefix in ("01-", "02-", "03-", "04-", "05-", "06-", "07-", "08-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break

    cv_dir = out_root / stem / "cv"
    sam_dir = out_root / stem / "sam2"

    print("\n" + "=" * 60)
    print(f"CV baseline → {cv_dir}")
    print("=" * 60)
    run(
        input_path,
        cv_dir,
        max_instances=max_instances,
        use_tile_gapfill=use_tile_gapfill,
        use_gemini_inpaint=use_gemini_inpaint,
        use_sam=False,
        use_qa=False,
        target_coverage=target_coverage,
        model=model,
        sam_model=sam_model,
    )
    # Tag pipeline in scene.json
    _tag_scene(cv_dir, "cv", "CV components + exact mask (v2 baseline)")

    print("\n" + "=" * 60)
    print(f"SAM2 + QA → {sam_dir}")
    print("=" * 60)
    run(
        input_path,
        sam_dir,
        max_instances=max_instances,
        use_tile_gapfill=use_tile_gapfill,
        use_gemini_inpaint=use_gemini_inpaint,
        use_sam=True,
        use_qa=True,
        target_coverage=target_coverage,
        model=model,
        sam_model=sam_model,
    )
    _tag_scene(sam_dir, "sam2", "SAM2 refine + residual QA gate (v3)")

    return cv_dir, sam_dir


def _tag_scene(run_dir: Path, pipeline: str, label: str) -> None:
    scene_path = run_dir / "scene.json"
    if not scene_path.exists():
        return
    data = json.loads(scene_path.read_text())
    data["pipeline"] = pipeline
    data["pipeline_label"] = label
    stats = data.setdefault("stats", {})
    stats["pipeline"] = pipeline
    stats["pipeline_label"] = label
    scene_path.write_text(json.dumps(data, indent=2))


def build_ab_html(pairs: list[tuple[str, Path, Path]], out_path: Path) -> Path:
    """Paired CV vs SAM2 gallery."""
    import html as html_mod

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = out_path.parent

    def rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            return p.resolve().as_posix()

    def stats(run: Path) -> dict:
        scene = run / "scene.json"
        if not scene.exists():
            return {}
        d = json.loads(scene.read_text())
        st = d.get("stats") or {}
        return {
            "label": d.get("pipeline_label") or st.get("pipeline_label") or run.name,
            "pipeline": d.get("pipeline") or st.get("pipeline") or "?",
            "isolated": st.get("isolated"),
            "coverage": st.get("ink_coverage"),
            "qa": st.get("qa_passed"),
            "bg": (st.get("background") or {}).get("method"),
            "source": d.get("source"),
        }

    def thumbs(run: Path, n: int = 16) -> str:
        layers = sorted((run / "layers").glob("*.png"))[:n]
        if not layers:
            return "<p class='muted'>No layers</p>"
        bits = [
            f'<img src="{html_mod.escape(rel(p))}" alt="" title="{html_mod.escape(p.stem)}" />'
            for p in layers
        ]
        extra = len(list((run / "layers").glob("*.png"))) - len(layers)
        if extra > 0:
            bits.append(f"<span class='muted'>+{extra}</span>")
        return '<div class="thumbs">' + "".join(bits) + "</div>"

    def panel(title: str, run: Path, badge: str) -> str:
        st = stats(run)
        cov = st.get("coverage")
        cov_s = f"{cov:.1%}" if isinstance(cov, (int, float)) else "—"
        qa = st.get("qa")
        qa_s = "pass" if qa is True else ("fail" if qa is False else "—")
        editor = rel(run / "editor.html") if (run / "editor.html").exists() else "#"
        return f"""
<div class="panel">
  <div class="badge {html_mod.escape(badge)}">{html_mod.escape(title)}</div>
  <p class="muted">layers {st.get("isolated")} · coverage {cov_s} · QA {qa_s} · bg {html_mod.escape(str(st.get("bg") or "—"))}</p>
  <div class="grid2">
    <figure>
      <img src="{html_mod.escape(rel(run / "background.png"))}" alt="bg" />
      <figcaption>Background</figcaption>
    </figure>
    <figure>
      <img src="{html_mod.escape(rel(run / "discover_overlay.png"))}" alt="boxes" />
      <figcaption>Discovery</figcaption>
    </figure>
  </div>
  <h4>Layers</h4>
  {thumbs(run)}
  <p><a class="btn" href="{html_mod.escape(editor)}" target="_blank" rel="noopener">Open editor</a></p>
</div>
"""

    sections = []
    for name, cv_dir, sam_dir in pairs:
        orig = cv_dir / "original.png"
        if not orig.exists():
            orig = sam_dir / "original.png"
        sections.append(
            f"""
<section class="card">
  <header>
    <h2>{html_mod.escape(name)}</h2>
    <p class="muted">CV baseline (initial POC) vs SAM2 + QA</p>
  </header>
  <figure class="orig">
    <img src="{html_mod.escape(rel(orig))}" alt="original" />
    <figcaption>Original print</figcaption>
  </figure>
  <div class="ab">
    {panel("A · CV baseline", cv_dir, "cv")}
    {panel("B · SAM2 + QA", sam_dir, "sam")}
  </div>
</section>
"""
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Auto Layer A/B — CV vs SAM2</title>
  <style>
    body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 0; padding: 24px;
      background: #f4f3f6; color: #1a1a1e; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    h2 {{ margin: 0; font-size: 18px; }}
    h4 {{ margin: 12px 0 6px; font-size: 13px; }}
    .muted {{ color: #6e6e76; font-size: 13px; }}
    .card {{ background: #fff; border-radius: 12px; padding: 20px; margin: 24px 0;
      box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
    .orig {{ margin: 12px 0 16px; max-width: 320px; }}
    .orig img {{ width: 100%; border-radius: 8px; background: #eee; }}
    .ab {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    @media (max-width: 900px) {{ .ab {{ grid-template-columns: 1fr; }} }}
    .panel {{ border: 1px solid #e7e7ea; border-radius: 10px; padding: 14px; }}
    .badge {{ display: inline-block; font-size: 12px; font-weight: 650; padding: 4px 10px;
      border-radius: 999px; margin-bottom: 8px; }}
    .badge.cv {{ background: #eef2ff; color: #3730a3; }}
    .badge.sam {{ background: #ecfdf5; color: #065f46; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; display: block; border-radius: 6px; background: #eee; }}
    figcaption {{ font-size: 11px; color: #6e6e76; margin-top: 4px; }}
    .thumbs {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
    .thumbs img {{ width: 52px; height: 52px; object-fit: contain; background: #f6f6f8;
      border-radius: 4px; border: 1px solid #e7e7ea; }}
    .btn {{ display: inline-block; padding: 8px 12px; background: #2a6f5b; color: #fff;
      text-decoration: none; border-radius: 8px; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>MTD-2396 Auto Layer — A/B POC</h1>
  <p class="muted">
    <strong>A · CV baseline</strong> = initial ink-mask / connected-components work.&nbsp;
    <strong>B · SAM2 + QA</strong> = SAM2 mask refine + residual coverage gate.
    Open each editor to move / recolor / delete layers.
  </p>
  {"".join(sections)}
</body>
</html>
"""
    out_path.write_text(doc)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Run CV baseline + SAM2 Auto Layer side by side")
    p.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            ROOT / "fixtures" / "06-seashells.png",
            ROOT / "fixtures" / "01-ditsy-florals.png",
            ROOT / "fixtures" / "03-paisley.png",
        ],
    )
    p.add_argument("--out-root", type=Path, default=ROOT / "out")
    p.add_argument("--max-instances", type=int, default=200)
    p.add_argument("--sam-model", default="sam2_b.pt")
    p.add_argument("--target-coverage", type=float, default=0.995)
    p.add_argument("--tile-gapfill", action="store_true")
    p.add_argument("--gemini-inpaint", action="store_true")
    p.add_argument("--skip-cv", action="store_true", help="Only run SAM2 path")
    p.add_argument("--skip-sam", action="store_true", help="Only run CV path")
    args = p.parse_args()

    pairs: list[tuple[str, Path, Path]] = []
    for inp in args.inputs:
        inp = inp.resolve()
        if not inp.exists():
            print(f"SKIP missing {inp}")
            continue
        stem = inp.stem
        for prefix in ("01-", "02-", "03-", "04-", "05-", "06-", "07-", "08-"):
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
                break

        if args.skip_cv and args.skip_sam:
            raise SystemExit("Cannot skip both pipelines")

        if args.skip_sam:
            cv_dir = args.out_root / stem / "cv"
            print(f"\nCV only → {cv_dir}")
            run(
                inp,
                cv_dir,
                max_instances=args.max_instances,
                use_tile_gapfill=args.tile_gapfill,
                use_gemini_inpaint=args.gemini_inpaint,
                use_sam=False,
                use_qa=False,
            )
            _tag_scene(cv_dir, "cv", "CV components + exact mask (v2 baseline)")
            pairs.append((stem, cv_dir, cv_dir))
            continue

        if args.skip_cv:
            sam_dir = args.out_root / stem / "sam2"
            print(f"\nSAM2 only → {sam_dir}")
            run(
                inp,
                sam_dir,
                max_instances=args.max_instances,
                use_tile_gapfill=args.tile_gapfill,
                use_gemini_inpaint=args.gemini_inpaint,
                use_sam=True,
                use_qa=True,
                target_coverage=args.target_coverage,
                sam_model=args.sam_model,
            )
            _tag_scene(sam_dir, "sam2", "SAM2 refine + residual QA gate (v3)")
            pairs.append((stem, sam_dir, sam_dir))
            continue

        cv_dir, sam_dir = run_pair(
            inp,
            args.out_root,
            max_instances=args.max_instances,
            sam_model=args.sam_model,
            target_coverage=args.target_coverage,
            use_tile_gapfill=args.tile_gapfill,
            use_gemini_inpaint=args.gemini_inpaint,
        )
        pairs.append((stem, cv_dir, sam_dir))

    ab = build_ab_html(pairs, args.out_root / "ab_compare.html")
    # Also flat list for legacy compare
    flat = []
    for _name, cv_dir, sam_dir in pairs:
        if cv_dir != sam_dir:
            flat.extend([cv_dir, sam_dir])
        else:
            flat.append(cv_dir)
    if flat:
        build_compare(flat, args.out_root / "compare.html")

    summary = {
        "pairs": [
            {
                "name": name,
                "cv": str(cv),
                "sam2": str(sam),
            }
            for name, cv, sam in pairs
        ],
        "ab_compare": str(ab),
    }
    (args.out_root / "ab_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(f"A/B gallery → {ab}")
    print("Open that file (or serve out/ with python -m http.server).")
    print("=" * 60)


if __name__ == "__main__":
    main()

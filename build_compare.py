#!/usr/bin/env python3
"""Build a side-by-side compare gallery for Auto Layer runs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _stats(run_dir: Path) -> dict:
    scene = run_dir / "scene.json"
    if not scene.exists():
        return {"error": "missing scene.json"}
    data = json.loads(scene.read_text())
    st = data.get("stats") or {}
    return {
        "source": data.get("source"),
        "discovered": st.get("discovered"),
        "isolated": st.get("isolated"),
        "ink_coverage": st.get("ink_coverage"),
        "bg_method": (st.get("background") or {}).get("method"),
        "width": data.get("width"),
        "height": data.get("height"),
        "layer_count": len(data.get("layers") or []),
    }


def _layer_thumbs(run_dir: Path, base: Path, limit: int = 24) -> str:
    layers = sorted((run_dir / "layers").glob("*.png"))[:limit]
    if not layers:
        return "<p class='muted'>No layers</p>"
    bits = []
    for p in layers:
        bits.append(
            f'<img src="{html.escape(_rel(p, base))}" alt="{html.escape(p.stem)}" title="{html.escape(p.stem)}" />'
        )
    extra = len(list((run_dir / "layers").glob("*.png"))) - len(layers)
    if extra > 0:
        bits.append(f"<span class='muted'>+{extra} more</span>")
    return '<div class="thumbs">' + "".join(bits) + "</div>"


def build(runs: list[Path], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = out_path.parent

    cards = []
    for run in runs:
        run = run.resolve()
        st = _stats(run)
        cov = st.get("ink_coverage")
        cov_s = f"{cov:.1%}" if isinstance(cov, (int, float)) else "—"
        editor = run / "editor.html"
        editor_href = _rel(editor, base) if editor.exists() else "#"
        cards.append(
            f"""
<section class="card">
  <header>
    <h2>{html.escape(run.name)}</h2>
    <p class="muted">{html.escape(str(st.get("source") or ""))} ·
      discovered {st.get("discovered")} · isolated {st.get("isolated")} ·
      ink coverage {cov_s} · bg {html.escape(str(st.get("bg_method") or "—"))}</p>
  </header>
  <div class="grid">
    <figure>
      <img src="{html.escape(_rel(run / "original.png", base))}" alt="original" />
      <figcaption>Original</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(_rel(run / "background.png", base))}" alt="background" />
      <figcaption>Background fill</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(_rel(run / "discover_overlay.png", base))}" alt="overlay" />
      <figcaption>Discovery boxes</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(_rel(run / "union_mask.png", base))}" alt="mask" />
      <figcaption>Union mask</figcaption>
    </figure>
  </div>
  <h3>Layers</h3>
  {_layer_thumbs(run, base)}
  <p><a class="btn" href="{html.escape(editor_href)}" target="_blank" rel="noopener">Open editor</a></p>
</section>
"""
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Auto Layer Compare</title>
  <style>
    body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 0; padding: 24px;
      background: #f4f3f6; color: #1a1a1e; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    h2 {{ margin: 0; font-size: 18px; }}
    h3 {{ margin: 16px 0 8px; font-size: 14px; }}
    .muted {{ color: #6e6e76; font-size: 13px; }}
    .card {{ background: #fff; border-radius: 12px; padding: 20px; margin: 20px 0;
      box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; height: auto; display: block; border-radius: 8px;
      background: #eee; }}
    figcaption {{ font-size: 12px; color: #6e6e76; margin-top: 6px; }}
    .thumbs {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .thumbs img {{ width: 64px; height: 64px; object-fit: contain; background: #f6f6f8;
      border-radius: 6px; border: 1px solid #e7e7ea; }}
    .btn {{ display: inline-block; padding: 8px 14px; background: #2a6f5b; color: #fff;
      text-decoration: none; border-radius: 8px; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>MTD-2396 Auto Layer — compare</h1>
  <p class="muted">Original vs background fill vs discovery overlay; open the editor to move / recolor / delete layers.</p>
  {"".join(cards)}
</body>
</html>
"""
    out_path.write_text(doc)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("out/compare.html"))
    args = p.parse_args()
    path = build(args.runs, args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

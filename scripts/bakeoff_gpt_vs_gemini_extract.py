#!/usr/bin/env python3
"""Bake off white-bg motif cutouts: Gemini flash-image vs GPT image edits.

stdlib + numpy/PIL/scipy only (no google.genai / requests — those hang in some venvs).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/tmp/extract_bakeoff")
CROPS_JSON = Path("/tmp/prompt_bake/crops.json")

PROMPT = (
    "Cut out only the centered {label} on pure white #FFFFFF. "
    "Delete all other leaves even if they overlap the subject. "
    "Holes white. No checkerboard. Keep original colors."
)

GPT_MODELS = [
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
]
GEMINI_MODEL = os.environ.get("GEMINI_EXTRACT_MODEL", "gemini-2.5-flash-image")


def _load_env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    for p in (
        ROOT / ".env",
        ROOT / "sketch-fabric-experiments/scripts/.env",
        ROOT / "scraper/.env",
        ROOT / "gpt6-astra-research/.env.local",
        ROOT / "consistency-eval-app/.env.local",
    ):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.startswith(f"{name}="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    os.environ[name] = v
                    return v
    return None


def _alpha_from_white_bg(rgb: np.ndarray) -> np.ndarray:
    d = np.sqrt(((rgb.astype(np.float32) - 255.0) ** 2).sum(axis=2))
    return np.clip((d - 8.0) / 18.0, 0, 1)


def _looks_like_checkerboard(rgba: np.ndarray) -> bool:
    rgb = rgba[:, :, :3].astype(np.float32)
    h, w = rgb.shape[:2]
    if h < 16 or w < 16:
        return False
    small = np.array(
        Image.fromarray(rgb.astype(np.uint8)).resize((64, 64), Image.Resampling.NEAREST)
    ).astype(np.float32)
    a = small[0::2, 0::2]
    b = small[0::2, 1::2]
    c = small[1::2, 0::2]
    d = small[1::2, 1::2]
    n = min(a.shape[0], b.shape[0], c.shape[0], d.shape[0])
    m = min(a.shape[1], b.shape[1], c.shape[1], d.shape[1])
    if n < 4 or m < 4:
        return False
    alt = np.abs(a[:n, :m] - b[:n, :m]).mean() + np.abs(c[:n, :m] - d[:n, :m]).mean()
    return float(alt) > 40


def cutout_to_rgba(
    raw: Image.Image, crop_rgb: Image.Image, *, prefer_model_rgb: bool = False
) -> np.ndarray | None:
    img = raw.convert("RGBA")
    if img.size != crop_rgb.size:
        img = img.resize(crop_rgb.size, Image.Resampling.LANCZOS)
    out = np.array(img)
    rgb = out[:, :, :3].astype(np.float32)
    a = out[:, :, 3].astype(np.float32)
    white_frac = float((rgb.min(axis=2) > 245).mean())
    # White-bg path: skip checker reject (fenestrated leaves trip the detector)
    if white_frac >= 0.15 or float(a.mean()) >= 240:
        alpha = _alpha_from_white_bg(rgb)
        if prefer_model_rgb:
            # GPT often redraws cleanly; keep its RGB under the matte
            out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            src = np.array(crop_rgb.convert("RGB"))
            out[:, :, :3] = src
        out[:, :, 3] = (alpha * 255).astype(np.uint8)
    elif _looks_like_checkerboard(out):
        return None
    opaque = float((out[:, :, 3] > 128).mean())
    if opaque < 0.02 or opaque > 0.88:
        return None
    return out


def score(rgba: np.ndarray | None) -> dict:
    if rgba is None:
        return {"ok": False, "opaque": 0, "n_cc": 0, "main_frac": 0, "border": 1}
    mask = rgba[:, :, 3] > 128
    opaque = float(mask.mean())
    lab, n = ndimage.label(mask)
    sizes = (
        sorted(((lab == i).sum() for i in range(1, n + 1)), reverse=True) if n else [0]
    )
    main = sizes[0] / max(1, int(mask.sum()))
    h, w = mask.shape
    pad = max(2, min(h, w) // 20)
    border = np.zeros_like(mask)
    border[:pad] = True
    border[-pad:] = True
    border[:, :pad] = True
    border[:, -pad:] = True
    border_o = float((mask & border).sum()) / max(1, int(border.sum()))
    good = opaque >= 0.08 and opaque <= 0.72 and main >= 0.75 and border_o < 0.55
    return {
        "ok": good,
        "opaque": round(opaque, 3),
        "n_cc": int(n),
        "main_frac": round(float(main), 3),
        "border": round(border_o, 3),
    }


def gemini_extract(crop: Image.Image, label: str, key: str) -> Image.Image:
    buf = BytesIO()
    crop.save(buf, format="PNG")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
        f":generateContent?key={key}"
    )
    body = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT.format(label=label)},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(buf.getvalue()).decode(),
                        }
                    },
                ]
            }
        ]
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    for cand in data.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return Image.open(BytesIO(base64.b64decode(inline["data"]))).convert(
                    "RGBA"
                )
    raise RuntimeError(f"no_image:{json.dumps(data)[:240]}")


def gpt_extract(crop: Image.Image, label: str, model: str, key: str) -> Image.Image:
    w, h = crop.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    ox, oy = (side - w) // 2, (side - h) // 2
    canvas.paste(crop.convert("RGB"), (ox, oy))
    # API wants reasonable size
    send = canvas
    if side > 1024:
        send = canvas.resize((1024, 1024), Image.Resampling.LANCZOS)
    buf = BytesIO()
    send.save(buf, format="PNG")
    png = buf.getvalue()

    boundary = "----MTDBoundary7MA4YWxkTrZu0gW"
    prompt = PROMPT.format(label=label)
    parts = []
    for name, val in (
        ("model", model),
        ("prompt", prompt),
        ("size", "1024x1024"),
        ("quality", "high"),
    ):
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{val}\r\n"
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="in.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    )
    body = b"".join(p.encode() for p in parts) + png + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/images/edits",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {e.code}: {err}") from e
    b64 = payload["data"][0]["b64_json"]
    full = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    # Map back to original crop rect
    full_sq = full.resize((side, side), Image.Resampling.LANCZOS)
    return full_sq.crop((ox, oy, ox + w, oy + h))


def comp_on_mustard(rgba: np.ndarray) -> Image.Image:
    mustard = np.array([214, 181, 103], dtype=np.float32)
    a = rgba.astype(np.float32) / 255.0
    al = a[:, :, 3:4]
    out = a[:, :, :3] * al + (mustard / 255.0) * (1 - al)
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


def main() -> int:
    gkey = _load_env_key("GOOGLE_API_KEY")
    okey = _load_env_key("OPENAI_API_KEY")
    if not gkey:
        print("GOOGLE_API_KEY missing", file=sys.stderr)
        return 1
    if not okey:
        print("OPENAI_API_KEY missing", file=sys.stderr)
        return 1
    crops = json.loads(CROPS_JSON.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    methods = [("gemini", GEMINI_MODEL)] + [("gpt", m) for m in GPT_MODELS]

    for crop_meta in crops:
        name = crop_meta["name"]
        label = crop_meta["label"]
        crop = Image.open(crop_meta["path"]).convert("RGB")
        print(f"\n=== {name} ({label}) {crop.size} ===", flush=True)
        for kind, model in methods:
            mid = f"{kind}__{model.replace('/', '_').replace('.', '_')}"
            odir = OUT / name / mid
            odir.mkdir(parents=True, exist_ok=True)
            crop.save(odir / "input.png")
            t0 = time.time()
            err = None
            rgba = None
            try:
                if kind == "gemini":
                    raw = gemini_extract(crop, label, gkey)
                else:
                    raw = gpt_extract(crop, label, model, okey)
                raw.save(odir / "raw.png")
                rgba = cutout_to_rgba(raw, crop)
                if rgba is None:
                    raise RuntimeError("alpha rebuild rejected")
                Image.fromarray(rgba).save(odir / "cutout.png")
                comp_on_mustard(rgba).save(odir / "preview.jpg", quality=90)
            except Exception as exc:  # noqa: BLE001
                err = str(exc)[:280]
                print(f"  {mid} FAIL {err}", flush=True)
            sec = round(time.time() - t0, 2)
            sc = score(rgba)
            row = {
                "crop": name,
                "label": label,
                "method": mid,
                "kind": kind,
                "model": model,
                "sec": sec,
                "error": err,
                **sc,
                "preview": str(odir / "preview.jpg") if rgba is not None else None,
            }
            rows.append(row)
            print(
                f"  {mid}: ok={sc['ok']} opaque={sc['opaque']} "
                f"main={sc['main_frac']} {sec}s",
                flush=True,
            )

    (OUT / "results.json").write_text(json.dumps(rows, indent=2))

    by_crop: dict[str, list] = {}
    for r in rows:
        by_crop.setdefault(r["crop"], []).append(r)
    html = [
        "<!doctype html><meta charset=utf-8><title>GPT vs Gemini extract</title>",
        "<style>body{font:14px/1.4 system-ui;background:#1a1a1a;color:#eee;padding:16px}"
        "h2{margin-top:28px}.row{display:flex;gap:12px;flex-wrap:wrap}"
        ".card{width:280px;background:#2a2a2a;padding:8px;border-radius:8px}"
        "img{width:100%;background:#d6b567}.ok{color:#8f8}.bad{color:#f88}</style>",
        "<h1>White-bg cutout bakeoff — Gemini vs GPT image</h1>",
    ]
    wins: dict[str, int] = {"gemini": 0, "gpt": 0}
    for crop, rs in by_crop.items():
        html.append(f"<h2>{crop}</h2><div class=row>")
        best = None
        for r in rs:
            if r["ok"]:
                score_v = r["main_frac"] - abs(r["opaque"] - 0.35) * 0.3 - r["border"] * 0.2
                if best is None or score_v > best[0]:
                    best = (score_v, r)
        if best:
            wins[best[1]["kind"]] = wins.get(best[1]["kind"], 0) + 1
        for r in rs:
            cls = "ok" if r["ok"] else "bad"
            mark = " ★" if best and r is best[1] else ""
            img = (
                f"<img src='file://{r['preview']}'>"
                if r["preview"]
                else "<div style='height:200px;background:#333'></div>"
            )
            html.append(
                f"<div class=card><b class={cls}>{r['method']}{mark}</b>"
                f"<div>opaque={r['opaque']} main={r['main_frac']} cc={r['n_cc']} "
                f"border={r['border']} {r['sec']}s</div>"
                f"<div class=bad>{r['error'] or ''}</div>{img}</div>"
            )
        html.append("</div>")
    html.append(f"<h2>Crop wins</h2><pre>{json.dumps(wins, indent=2)}</pre>")
    (OUT / "compare.html").write_text("\n".join(html))
    print("\nWINS", wins, flush=True)
    print("Wrote", OUT / "compare.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

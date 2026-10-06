#!/usr/bin/env python3
"""Bake off Gemini extract prompts on hard tropical crops.

Uses REST (avoids google.genai import hangs). Scores alpha quality + bleed.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / ".env"
OUT = Path("/tmp/prompt_bake")
CROPS = OUT / "crops.json"
MODEL = os.environ.get("GEMINI_EXTRACT_MODEL", "gemini-2.5-flash-image")


def load_key() -> str:
    if os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]
    for line in ENV.read_text().splitlines():
        if line.startswith("GOOGLE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("GOOGLE_API_KEY missing")


PROMPTS: dict[str, str] = {
    "A_current": (
        "You are cutting a single textile motif for a layers menu.\n"
        "Subject: {label} nearest the center of this crop.\n\n"
        "Output requirements:\n"
        "- PNG with a REAL alpha channel (alpha=0 = transparent). "
        "Never draw a checkerboard, grid, or fake transparency pattern.\n"
        "- Keep ONLY that one complete motif. Exclude every neighboring leaf/frond/flower "
        "even if it overlaps or sits behind the subject.\n"
        "- Natural holes (e.g. monstera fenestrations) must be transparent, not filled.\n"
        "- Preserve original print colors and edges. No shadows, outlines, or new pixels.\n"
        "- If unsure which motif is primary, pick the largest one touching the crop center."
    ),
    "B_short": (
        "Extract ONE motif only: {label} at the crop center.\n"
        "Return PNG with true transparency (alpha channel). "
        "No checkerboard. No other leaves. Keep fenestration holes transparent. "
        "Do not redraw — cut the original pixels."
    ),
    "C_matte": (
        "Create a cutout matte of a single print motif.\n"
        "Target: {label} touching the image center.\n"
        "Return an RGBA PNG where:\n"
        "- alpha=255 only on that one motif\n"
        "- alpha=0 on background AND on every other overlapping motif\n"
        "- alpha=0 inside natural holes in the motif\n"
        "Never paint a checkerboard or gray/white grid. Never invent colors."
    ),
    "D_removebg": (
        "Act like a professional background-removal tool for apparel print layers.\n"
        "Isolate only the single {label} centered in the frame.\n"
        "Hard requirements:\n"
        "1) Real alpha transparency — never a checkerboard preview\n"
        "2) Exactly one motif instance — delete neighbors even when overlapping\n"
        "3) Preserve original artwork colors/edges 1:1\n"
        "4) Keep leaf holes fully transparent\n"
        "Output: RGBA PNG cutout."
    ),
    "E_negative": (
        "Task: single-instance motif extraction for {label}.\n"
        "DO:\n"
        "- Output RGBA PNG with real alpha\n"
        "- Keep the complete centered motif\n"
        "- Make background and holes transparent\n"
        "DO NOT:\n"
        "- Include any second leaf, frond, stem, or flower\n"
        "- Draw checkerboards, grids, drop shadows, or new outlines\n"
        "- Fill holes with color\n"
        "- Soft-blend neighbors into the subject — cut them out completely"
    ),
    "F_white_bg": (
        "Extract the single centered motif ({label}) onto a pure white background "
        "(RGB 255,255,255). Keep only that one motif; erase all neighboring leaves. "
        "Do not use checkerboard. Do not add shadows. Preserve original motif colors. "
        "Fenestration holes must be pure white like the background."
    ),
    "G_binary_mask_hint": (
        "Return an RGBA PNG cutout of ONLY the {label} at center.\n"
        "Think in two steps: (1) decide the binary mask of that one motif, "
        "(2) keep original pixels inside the mask and set alpha=0 outside.\n"
        "Neighbors that touch the subject must be outside the mask.\n"
        "No checkerboard. No recoloring. Holes in the leaf are outside the mask."
    ),
}


def call_gemini(api_key: str, prompt: str, crop_png: bytes) -> tuple[Image.Image | None, str]:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}"
        f":generateContent?key={api_key}"
    )
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(crop_png).decode(),
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
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:400]
        return None, f"HTTP {e.code}: {err}"
    except Exception as e:  # noqa: BLE001
        return None, str(e)

    for cand in data.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                raw = base64.b64decode(inline["data"])
                return Image.open(BytesIO(raw)).convert("RGBA"), "ok"
    return None, f"no_image:{json.dumps(data)[:300]}"


def looks_like_checkerboard(rgba: np.ndarray) -> bool:
    rgb = rgba[:, :, :3].astype(np.float32)
    h, w = rgb.shape[:2]
    if h < 16 or w < 16:
        return False
    # nearest resize preserves fine checkers
    small = np.array(
        Image.fromarray(rgb.astype(np.uint8)).resize((64, 64), Image.Resampling.NEAREST)
    ).astype(np.float32)
    a = small[0::2, 0::2]
    b = small[0::2, 1::2]
    c = small[1::2, 0::2]
    d = small[1::2, 1::2]
    n = min(a.shape[0], b.shape[0], c.shape[0], d.shape[0])
    m = min(a.shape[1], b.shape[1], c.shape[1], d.shape[1])
    a, b, c, d = a[:n, :m], b[:n, :m], c[:n, :m], d[:n, :m]
    adj = (
        np.mean(np.abs(a - b), axis=2)
        + np.mean(np.abs(a - c), axis=2)
        + np.mean(np.abs(d - b), axis=2)
        + np.mean(np.abs(d - c), axis=2)
    ) / 4.0
    diag = (np.mean(np.abs(a - d), axis=2) + np.mean(np.abs(b - c), axis=2)) / 2.0
    score = adj - diag
    lum = small.mean(axis=2)
    hi = float((lum > 220).mean())
    if float((score > 25).mean()) > 0.12 and hi > 0.08:
        return True
    if float(np.percentile(score, 85)) > 40 and hi > 0.05:
        return True
    return False


def rebuild_alpha_from_white(rgba: np.ndarray) -> np.ndarray:
    """For white-bg prompts: build alpha from distance to white."""
    out = rgba.copy()
    rgb = out[:, :, :3].astype(np.float32)
    d = np.sqrt(((rgb - 255.0) ** 2).sum(axis=2))
    alpha = np.clip((d - 8.0) / 18.0, 0, 1)
    out[:, :, 3] = (alpha * 255).astype(np.uint8)
    return out


def score_cutout(rgba: np.ndarray, crop_rgb: np.ndarray, *, white_bg: bool) -> dict:
    arr = rebuild_alpha_from_white(rgba) if white_bg else rgba.copy()
    # If nearly opaque, rebuild from crop border like production
    a = arr[:, :, 3].astype(np.float32)
    if float(a.mean()) >= 240 and not white_bg:
        rgb = arr[:, :, :3].astype(np.float32)
        h, w = arr.shape[:2]
        b = max(2, min(6, h // 16, w // 16))
        border = np.concatenate(
            [
                crop_rgb[:b].reshape(-1, 3),
                crop_rgb[-b:].reshape(-1, 3),
                crop_rgb[:, :b].reshape(-1, 3),
                crop_rgb[:, -b:].reshape(-1, 3),
            ],
            axis=0,
        ).astype(np.float32)
        bg = np.median(border, axis=0)
        d = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
        white = rgb.mean(axis=2)
        alpha = np.clip((d - 12.0) / 20.0, 0, 1)
        alpha[white > 245] = 0
        arr[:, :, 3] = (alpha * 255).astype(np.uint8)
        a = arr[:, :, 3].astype(np.float32)

    checker = looks_like_checkerboard(arr)
    mask = a > 128
    opaque = float(mask.mean())
    h, w = mask.shape
    cy, cx = h // 2, w // 2
    center_ok = bool(mask[cy, cx])
    # border opaque = often neighbor bleed / incomplete isolation
    pad = max(2, min(h, w) // 20)
    border_mask = np.zeros_like(mask)
    border_mask[:pad, :] = True
    border_mask[-pad:, :] = True
    border_mask[:, :pad] = True
    border_mask[:, -pad:] = True
    border_opaque = float((mask & border_mask).sum()) / max(1, int(border_mask.sum()))

    # connected components (simple flood via scipy if available else approx)
    n_cc = 0
    try:
        from scipy import ndimage  # type: ignore

        lab, n_cc = ndimage.label(mask)
        if n_cc > 0:
            sizes = [(lab == i).sum() for i in range(1, n_cc + 1)]
            sizes.sort(reverse=True)
            main_frac = sizes[0] / max(1, int(mask.sum()))
        else:
            main_frac = 0.0
    except Exception:
        main_frac = 1.0 if opaque > 0 else 0.0
        n_cc = 1 if opaque > 0 else 0

    near_white_opaque = float(
        (((arr[:, :, :3] > 230).all(axis=2)) & mask).mean()
    )

    # Prefer mid opaque (single leaf), center covered, one CC, no checker, low border
    score = 0.0
    if checker:
        score -= 50
    if not center_ok:
        score -= 20
    # ideal opaque ~0.15–0.45 for these crops
    if 0.12 <= opaque <= 0.50:
        score += 25
    elif 0.08 <= opaque <= 0.60:
        score += 10
    else:
        score -= 15
    if main_frac >= 0.85:
        score += 15
    elif main_frac >= 0.65:
        score += 5
    else:
        score -= 10
    if n_cc <= 2:
        score += 10
    elif n_cc <= 5:
        score += 0
    else:
        score -= 10
    if border_opaque < 0.08:
        score += 15
    elif border_opaque < 0.18:
        score += 5
    else:
        score -= 15
    if near_white_opaque > 0.02:
        score -= 10

    return {
        "score": round(score, 1),
        "checker": checker,
        "opaque": round(opaque, 3),
        "center_ok": center_ok,
        "n_cc": int(n_cc),
        "main_frac": round(float(main_frac), 3),
        "border_opaque": round(border_opaque, 3),
        "near_white_opaque": round(near_white_opaque, 3),
        "alpha_mean": round(float(a.mean()), 1),
        "arr": arr,
    }


def composite_on(rgba: np.ndarray, bg_rgb: tuple[int, int, int]) -> Image.Image:
    a = rgba.astype(np.float32) / 255.0
    bg = np.zeros_like(rgba)
    bg[:, :, 0] = bg_rgb[0]
    bg[:, :, 1] = bg_rgb[1]
    bg[:, :, 2] = bg_rgb[2]
    bg[:, :, 3] = 255
    b = bg.astype(np.float32) / 255.0
    al = a[:, :, 3:4]
    comp = a[:, :, :3] * al + b[:, :, :3] * (1 - al)
    return Image.fromarray(np.clip(comp * 255, 0, 255).astype(np.uint8))


def main() -> None:
    api_key = load_key()
    crops = json.loads(CROPS.read_text())
    # Use first 3 hard crops to keep runtime reasonable
    crops = crops[:3]
    out_dir = OUT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for crop in crops:
        crop_img = Image.open(crop["path"]).convert("RGB")
        crop_rgb = np.array(crop_img)
        buf = BytesIO()
        crop_img.save(buf, format="PNG")
        crop_png = buf.getvalue()
        label = crop["label"]
        print(f"\n=== crop {crop['name']} ({label}) ===", flush=True)

        for pid, tmpl in PROMPTS.items():
            prompt = tmpl.format(label=label)
            t0 = time.time()
            img, status = call_gemini(api_key, prompt, crop_png)
            sec = round(time.time() - t0, 1)
            row = {
                "crop": crop["name"],
                "prompt": pid,
                "sec": sec,
                "status": status,
            }
            if img is None:
                print(f"  {pid}: FAIL {status[:80]} ({sec}s)", flush=True)
                row["score"] = -100
                results.append(row)
                continue
            # resize to crop
            if img.size != crop_img.size:
                img = img.resize(crop_img.size, Image.Resampling.LANCZOS)
            rgba = np.array(img)
            sc = score_cutout(rgba, crop_rgb, white_bg=(pid == "F_white_bg"))
            arr = sc.pop("arr")
            row.update(sc)
            stem = f"{crop['name']}__{pid}"
            Image.fromarray(arr).save(out_dir / f"{stem}.png")
            composite_on(arr, (214, 181, 103)).save(out_dir / f"{stem}_mustard.png")
            print(
                f"  {pid}: score={row['score']} opaque={row['opaque']} "
                f"cc={row['n_cc']} border={row['border_opaque']} "
                f"checker={row['checker']} ({sec}s)",
                flush=True,
            )
            results.append(row)

    Path(OUT / "results.json").write_text(json.dumps(results, indent=2))

    # Aggregate by prompt
    by: dict[str, list[float]] = {}
    for r in results:
        by.setdefault(r["prompt"], []).append(float(r.get("score", -100)))
    ranking = sorted(
        ((pid, float(np.mean(scores)), float(np.median(scores)), scores) for pid, scores in by.items()),
        key=lambda x: (-x[1], -x[2]),
    )
    print("\n=== RANKING (mean score) ===", flush=True)
    for pid, mean, med, scores in ranking:
        print(f"  {pid}: mean={mean:.1f} median={med:.1f} scores={scores}", flush=True)

    best = ranking[0][0] if ranking else None
    summary = {
        "best_prompt": best,
        "ranking": [
            {"prompt": pid, "mean": mean, "median": med, "scores": scores}
            for pid, mean, med, scores in ranking
        ],
        "prompt_text": PROMPTS.get(best or "", ""),
        "results": results,
    }
    Path(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Contact sheet: rows=crops, cols=prompts
    pids = list(PROMPTS.keys())
    cnames = [c["name"] for c in crops]
    cell = 220
    sheet = Image.new("RGB", (cell * (len(pids) + 1), cell * (len(cnames) + 1)), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    for j, pid in enumerate(pids):
        draw.text((cell * (j + 1) + 8, 8), pid, fill=(240, 240, 240))
    for i, cname in enumerate(cnames):
        draw.text((8, cell * (i + 1) + 8), cname.replace("_", "\n"), fill=(240, 240, 240))
        # original crop thumb
        thumb = Image.open(next(c["path"] for c in crops if c["name"] == cname)).convert("RGB")
        thumb.thumbnail((cell - 8, cell - 8))
        sheet.paste(thumb, (4, cell * (i + 1) + 4))
        for j, pid in enumerate(pids):
            p = out_dir / f"{cname}__{pid}_mustard.png"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell - 8, cell - 8))
            sheet.paste(im, (cell * (j + 1) + 4, cell * (i + 1) + 4))
            sc = next(
                (r["score"] for r in results if r["crop"] == cname and r["prompt"] == pid),
                None,
            )
            if sc is not None:
                draw.text(
                    (cell * (j + 1) + 8, cell * (i + 1) + cell - 18),
                    f"{sc}",
                    fill=(255, 255, 0),
                )
    sheet_path = OUT / "compare_sheet.jpg"
    sheet.save(sheet_path, quality=90)
    print(f"\nBEST={best}", flush=True)
    print(f"sheet={sheet_path}", flush=True)
    print(f"summary={OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

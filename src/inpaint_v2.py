"""Background reconstruction — flat-ground fill first, Gemini for hard cases."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from segment import estimate_background_color, lab_distance


def _texture_noise(h: int, w: int, strength: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, strength, size=(h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.2)
    return np.stack([noise, noise, noise], axis=2)


def reconstruct_background_v2(
    image: Image.Image,
    union_alpha: np.ndarray,
    *,
    dilate: int = 4,
    api_key: str | None = None,
    use_gemini: bool = True,
    ghost_threshold: float = 18.0,
) -> tuple[Image.Image, dict]:
    """
    1) Dilate motif holes
    2) Fill with estimated ground color + subtle matched texture (best for textile flats)
    3) OpenCV Telea as secondary blend on edges
    4) Optional Gemini if residual structure remains high
    """
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if union_alpha.shape[:2] != (h, w):
        union_alpha = cv2.resize(union_alpha, (w, h), interpolation=cv2.INTER_NEAREST)

    # Harder threshold so soft fringe is fully cleared from the background
    mask = (union_alpha > 8).astype(np.uint8) * 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1,) * 2)
        mask = cv2.dilate(mask, k, iterations=1)

    bg = estimate_background_color(rgb)
    # Sample non-mask pixels for local texture variance
    inv = mask == 0
    if inv.any():
        local_std = float(np.std(rgb[inv]))
    else:
        local_std = 4.0

    filled = rgb.astype(np.float32).copy()
    holes = mask > 0
    noise = _texture_noise(h, w, strength=max(1.5, min(6.0, local_std * 0.25)))
    fill_color = bg.reshape(1, 1, 3) + noise
    filled[holes] = np.clip(fill_color[holes], 0, 255)

    # Soft-edge Telea only near hole boundary to avoid motif smear into holes
    edge = mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner = cv2.erode(mask, k, iterations=2)
    ring = cv2.subtract(edge, inner)
    bgr = cv2.cvtColor(filled.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if ring.any():
        telea = cv2.inpaint(bgr, ring, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        # Blend telea only on ring
        ring_b = ring > 0
        bgr[ring_b] = telea[ring_b]

    result = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    method = "flat_fill+edge_telea"

    # Second pass: hard-clear dilated hole neighborhood (kills soft watermark ghosts)
    scrub = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), 1)
    noise2 = _texture_noise(h, w, strength=max(1.5, min(6.0, local_std * 0.25)), seed=1)
    fill2 = bg.reshape(1, 1, 3) + noise2
    result = result.astype(np.float32)
    result[scrub > 0] = np.clip(fill2[scrub > 0], 0, 255)

    # Third pass: remove any ink-like leftovers anywhere (pale motifs missed by mask)
    dist_all = lab_distance(result.astype(np.uint8), bg)
    leftover = dist_all > 12.0
    if leftover.any():
        result[leftover] = np.clip(fill2[leftover], 0, 255)
    result = result.astype(np.uint8)
    method = "flat_fill+scrub"
    holes = mask > 0
    residual = float(np.std(result[holes])) if holes.any() else 0.0
    dist = lab_distance(result, bg)
    ghost_ink = float((dist[holes] > 22).mean()) if holes.any() else 0.0

    meta = {
        "method": method,
        "mask_coverage": float(holes.mean()),
        "residual_std": residual,
        "ghost_ink_frac": ghost_ink,
        "gemini_used": False,
        "bg_color": [float(x) for x in bg.tolist()],
    }

    needs_gemini = use_gemini and (ghost_ink > 0.08 or residual > ghost_threshold)
    if needs_gemini:
        gem = _gemini_inpaint(Image.fromarray(result), mask, api_key)
        if gem is not None:
            result = gem
            meta["method"] = "flat_fill+gemini"
            meta["gemini_used"] = True
            dist2 = lab_distance(result, bg)
            meta["ghost_ink_frac"] = float((dist2[holes] > 22).mean()) if holes.any() else 0.0
            meta["residual_std"] = float(np.std(result[holes])) if holes.any() else 0.0

    return Image.fromarray(result.astype(np.uint8)), meta


def _gemini_inpaint(
    image: Image.Image,
    mask: np.ndarray,
    api_key: str | None,
    model: str = "gemini-2.5-flash-image",
) -> np.ndarray | None:
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        vis = np.array(image.convert("RGB"))
        holes = mask > 0
        vis[holes] = (255, 0, 255)
        buf = BytesIO()
        Image.fromarray(vis).save(buf, format="PNG")
        prompt = (
            "Textile print with MAGENTA holes where motifs were removed. "
            "Fill ONLY magenta with continuous plain ground/background matching surrounding "
            "color and subtle texture. Do NOT invent motifs, flowers, shells, or pattern. "
            "Same dimensions. No magenta left. Photoreal flat textile ground only."
        )
        resp = client.models.generate_content(
            model=model,
            contents=[
                prompt,
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            ],
        )
        for cand in getattr(resp, "candidates", None) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    out = Image.open(BytesIO(inline.data)).convert("RGB")
                    if out.size != image.size:
                        out = out.resize(image.size, Image.Resampling.LANCZOS)
                    arr = np.array(out)
                    # Keep non-hole pixels from original to avoid global rewrite
                    keep = mask == 0
                    base = np.array(image.convert("RGB"))
                    arr[keep] = base[keep]
                    return arr
    except Exception as exc:  # noqa: BLE001
        print(f"  gemini inpaint failed: {exc}")
    return None


def save_mask(union_alpha: np.ndarray, path: Path) -> None:
    Image.fromarray(union_alpha).save(path)

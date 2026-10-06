"""Background reconstruction after motif removal."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def reconstruct_background(
    image: Image.Image,
    union_alpha: np.ndarray,
    *,
    dilate: int = 3,
    api_key: str | None = None,
    use_gemini: bool = False,
    ghost_threshold: float = 12.0,
) -> tuple[Image.Image, dict]:
    """
    Fill holes where motifs were removed.
    Primary: OpenCV Telea inpaint. Optional Gemini pass if ghosts remain.
    """
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if union_alpha.shape[:2] != (h, w):
        union_alpha = cv2.resize(union_alpha, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = (union_alpha > 20).astype(np.uint8) * 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        mask = cv2.dilate(mask, k, iterations=1)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    method = "opencv_telea"

    # Ghost check: high residual structure inside mask vs border bg
    residual = float(np.std(result[mask > 0])) if (mask > 0).any() else 0.0
    meta = {
        "method": method,
        "mask_coverage": float((mask > 0).mean()),
        "residual_std": residual,
        "gemini_used": False,
    }

    if use_gemini and residual > ghost_threshold:
        gem = _gemini_inpaint(Image.fromarray(result), mask, api_key)
        if gem is not None:
            result = gem
            meta["method"] = "opencv_telea+gemini"
            meta["gemini_used"] = True
            meta["residual_std"] = float(np.std(result[mask > 0])) if (mask > 0).any() else 0.0

    return Image.fromarray(result), meta


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
        # Visualize holes as magenta so the model knows what to fill
        vis = image.convert("RGBA")
        arr = np.array(vis)
        holes = mask > 0
        arr[holes, 0] = 255
        arr[holes, 1] = 0
        arr[holes, 2] = 255
        arr[holes, 3] = 255
        vis = Image.fromarray(arr)

        buf = BytesIO()
        vis.save(buf, format="PNG")
        prompt = (
            "This textile print has magenta regions marking holes where motifs were removed. "
            "Fill ONLY the magenta regions with continuous background/ground color and texture "
            "that matches surrounding areas. Do NOT invent new motifs, flowers, or decorative elements. "
            "Return the completed print as a PNG, same size, no magenta remaining."
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
                    return np.array(out)
    except Exception as exc:  # noqa: BLE001
        print(f"  gemini inpaint failed: {exc}")
    return None


def save_mask(union_alpha: np.ndarray, path: Path) -> None:
    Image.fromarray(union_alpha).save(path)

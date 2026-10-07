"""Per-motif isolation via color-distance matte (+ optional Gemini fallback)."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from discover import MotifBox


@dataclass
class IsolatedLayer:
    id: str
    label: str
    path: str
    bbox_px: list[int]  # [x, y, w, h] on full image for the saved crop
    bbox_norm: list[float]
    confidence: float
    matte_score: float
    method: str


def estimate_background_color(rgb: np.ndarray, border: int = 8) -> np.ndarray:
    """Estimate ground color from image border strips."""
    h, w = rgb.shape[:2]
    b = max(2, min(border, h // 8, w // 8))
    strips = [
        rgb[:b, :, :],
        rgb[-b:, :, :],
        rgb[:, :b, :],
        rgb[:, -b:, :],
    ]
    samples = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)
    # Robust: median of border pixels
    return np.median(samples, axis=0).astype(np.float32)


def _lab_distance(rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(
        np.float32
    )
    diff = lab - bg_lab
    return np.sqrt(np.sum(diff * diff, axis=2))


def color_matte(
    crop_rgb: np.ndarray,
    bg_color: np.ndarray,
    *,
    soft_lo: float = 8.0,
    soft_hi: float = 28.0,
) -> tuple[np.ndarray, float]:
    """Return alpha uint8 and a matte confidence score in [0,1]."""
    dist = _lab_distance(crop_rgb, bg_color)
    alpha = np.clip((dist - soft_lo) / max(1e-6, soft_hi - soft_lo), 0.0, 1.0)
    alpha_u8 = (alpha * 255.0).astype(np.uint8)

    # Keep largest connected component(s) that are "ink"
    binary = (alpha_u8 > 40).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return alpha_u8, 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    # Keep components until we cover most ink area (handles multi-part motifs)
    order = np.argsort(-areas)
    keep = np.zeros_like(binary)
    total = float(areas.sum()) or 1.0
    covered = 0.0
    for idx in order:
        label = int(idx) + 1
        keep[labels == label] = 1
        covered += float(areas[idx])
        if covered / total >= 0.92:
            break

    # Soft mask: feather keep region
    keep_f = keep.astype(np.float32)
    keep_f = cv2.GaussianBlur(keep_f, (0, 0), sigmaX=1.2)
    alpha_f = (alpha_u8.astype(np.float32) / 255.0) * keep_f
    alpha_out = np.clip(alpha_f * 255.0, 0, 255).astype(np.uint8)

    ink_frac = float((alpha_out > 20).mean())
    bg_clean = float((alpha_out[binary == 0] < 10).mean()) if (binary == 0).any() else 0.5
    score = float(np.clip(0.35 * ink_frac / 0.35 + 0.65 * bg_clean, 0.0, 1.0))
    # Prefer middling ink coverage in the crop
    if ink_frac < 0.02:
        score *= 0.3
    if ink_frac > 0.95:
        score *= 0.5
    return alpha_out, score


def _bbox_to_pixels(bbox_norm: list[float], w: int, h: int, pad: float = 0.04) -> tuple[int, int, int, int]:
    x, y, bw, bh = bbox_norm
    pad_x = bw * pad
    pad_y = bh * pad
    x0 = max(0, int((x - pad_x) * w))
    y0 = max(0, int((y - pad_y) * h))
    x1 = min(w, int((x + bw + pad_x) * w))
    y1 = min(h, int((y + bh + pad_y) * h))
    if x1 <= x0 or y1 <= y0:
        x0, y0 = max(0, int(x * w)), max(0, int(y * h))
        x1, y1 = min(w, int((x + bw) * w)), min(h, int((y + bh) * h))
    return x0, y0, x1, y1


def _trim_alpha(rgba: np.ndarray, pad: int = 2) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return rgba, (0, 0, rgba.shape[1], rgba.shape[0])
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(rgba.shape[1], x1 + pad)
    y1 = min(rgba.shape[0], y1 + pad)
    return rgba[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


# Bakeoff: short white-bg cutout >> asking for real alpha.
# GPT image edits beat Gemini on whole single motifs (mustard tropical 2026-10);
# Gemini stays as fast fallback. Same prompt for both.
_EXTRACT_PROMPT = (
    "Cut out only the centered {label} on pure white #FFFFFF. "
    "Delete all other leaves even if they overlap the subject. "
    "Holes white. No checkerboard. Keep original colors."
)
_GEMINI_EXTRACT_PROMPT = _EXTRACT_PROMPT  # alias


def _alpha_from_white_bg(rgb: np.ndarray) -> np.ndarray:
    """Build soft alpha from distance to pure white (white-bg cutout path)."""
    d = np.sqrt(((rgb.astype(np.float32) - 255.0) ** 2).sum(axis=2))
    # Near-white → transparent; ink stays opaque
    return np.clip((d - 8.0) / 18.0, 0, 1)


def _ensure_cutout_alpha(
    rgba: np.ndarray,
    crop_rgb: np.ndarray,
    *,
    keep_model_rgb: bool = False,
) -> np.ndarray | None:
    """Normalize white-bg cutout alpha; optionally keep model RGB or original print."""
    out = rgba.copy()
    a = out[:, :, 3].astype(np.float32)
    rgb = out[:, :, :3].astype(np.float32)

    white_frac = float((rgb.min(axis=2) > 245).mean())
    # White-bg path first — fenestrated leaves false-trigger checker detection
    if white_frac >= 0.15 or float(a.mean()) >= 240:
        alpha = _alpha_from_white_bg(rgb)
        if float((alpha > 0.5).mean()) < 0.02:
            h, w = out.shape[:2]
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
            alpha = np.clip((d - 12.0) / 20.0, 0, 1)
            alpha[rgb.mean(axis=2) > 245] = 0
        if keep_model_rgb:
            out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            # Print fidelity: original crop pixels under the model's matte
            out[:, :, :3] = crop_rgb
        out[:, :, 3] = (alpha * 255).astype(np.uint8)
        if float((out[:, :, 3] > 128).mean()) > 0.85:
            print("    extract reject: cutout still full-frame")
            return None
        return out

    # Non-white path: reject painted checkers
    if _looks_like_checkerboard(out):
        print("    extract reject: checkerboard/fake transparency")
        return None
    return out


def _gemini_extract_crop(
    crop_rgb: Image.Image,
    label: str,
    api_key: str | None,
    model: str = "gemini-2.5-flash-image",
) -> np.ndarray | None:
    """Ask Gemini for a white-bg single-motif cutout; rebuild alpha locally."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        buf = BytesIO()
        crop_rgb.save(buf, format="PNG")
        prompt = _EXTRACT_PROMPT.format(label=label or "primary motif")
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
                    img = Image.open(BytesIO(inline.data)).convert("RGBA")
                    arr = np.array(img)
                    arr = _ensure_cutout_alpha(arr, np.array(crop_rgb.convert("RGB")))
                    if arr is None:
                        return None
                    return arr
    except Exception as exc:  # noqa: BLE001
        print(f"    gemini extract fallback failed: {exc}")
    return None


def _gpt_extract_crop(
    crop_rgb: Image.Image,
    label: str,
    api_key: str | None = None,
    model: str = "gpt-image-2.5-flare",
) -> np.ndarray | None:
    """OpenAI images/edits white-bg cutout; alpha from white, RGB from original print."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        w, h = crop_rgb.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        ox, oy = (side - w) // 2, (side - h) // 2
        canvas.paste(crop_rgb.convert("RGB"), (ox, oy))
        send = canvas
        if side > 1024:
            send = canvas.resize((1024, 1024), Image.Resampling.LANCZOS)
        buf = BytesIO()
        send.save(buf, format="PNG")
        png = buf.getvalue()
        prompt = _EXTRACT_PROMPT.format(label=label or "primary motif")
        boundary = "----MTDExtractBoundary"
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
        body = (
            b"".join(p.encode() for p in parts)
            + png
            + f"\r\n--{boundary}--\r\n".encode()
        )
        req = urllib.request.Request(
            "https://api.openai.com/v1/images/edits",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.load(resp)
        b64 = payload["data"][0]["b64_json"]
        full = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        full_sq = full.resize((side, side), Image.Resampling.LANCZOS)
        cut = full_sq.crop((ox, oy, ox + w, oy + h))
        rgba = np.dstack(
            [np.array(cut), np.full((h, w), 255, dtype=np.uint8)]
        )
        # GPT silhouette is the quality win; keep original print colors under it
        return _ensure_cutout_alpha(
            rgba, np.array(crop_rgb.convert("RGB")), keep_model_rgb=False
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    gpt extract failed: {exc}")
    return None


def _extract_crop(
    crop_rgb: Image.Image,
    label: str,
    *,
    google_key: str | None = None,
    openai_key: str | None = None,
    backend: str | None = None,
    gemini_model: str = "gemini-2.5-flash-image",
    gpt_model: str = "gpt-image-2.5-flare",
) -> tuple[np.ndarray | None, str]:
    """Dispatch motif cutout. backend: gpt | gemini | auto (gpt then gemini)."""
    mode = (backend or os.environ.get("HARD_BUSY_EXTRACT", "auto")).lower().strip()
    if mode in ("gpt", "auto"):
        arr = _gpt_extract_crop(crop_rgb, label, openai_key, model=gpt_model)
        if arr is not None:
            return arr, "gpt"
        if mode == "gpt":
            return None, "gpt"
    arr = _gemini_extract_crop(crop_rgb, label, google_key, model=gemini_model)
    if arr is not None:
        return arr, "gemini"
    return None, mode


def _looks_like_checkerboard(rgba: np.ndarray) -> bool:
    """Detect common model failure: drawing a checker instead of alpha.

    Use NEAREST downsample — AREA averaging kills fine checker signal.
    Also flag alpha that dither-flips at high frequency over a large region.
    """
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3].astype(np.float32)
    h, w = rgb.shape[:2]
    if h < 16 or w < 16:
        return False

    # --- RGB painted checkers (green/white, gray/white, etc.) ---
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_NEAREST)
    a = small[0::2, 0::2]
    b = small[0::2, 1::2]
    c = small[1::2, 0::2]
    d = small[1::2, 1::2]
    n = min(a.shape[0], b.shape[0], c.shape[0], d.shape[0])
    m = min(a.shape[1], b.shape[1], c.shape[1], d.shape[1])
    if n < 4 or m < 4:
        return False
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
    # Strong checker: adjacent cells differ, diagonals match
    if float((score > 25).mean()) > 0.12 and hi > 0.08:
        return True
    if float(np.percentile(score, 85)) > 40 and hi > 0.05:
        return True

    # --- Alpha dither checkers (opaque/transparent flip as fake transparency) ---
    if float(alpha.mean()) < 250:
        sa = cv2.resize(alpha, (64, 64), interpolation=cv2.INTER_NEAREST)
        aa = sa[0::2, 0::2][:n, :m]
        bb = sa[0::2, 1::2][:n, :m]
        cc = sa[1::2, 0::2][:n, :m]
        dd = sa[1::2, 1::2][:n, :m]
        a_adj = (np.abs(aa - bb) + np.abs(aa - cc) + np.abs(dd - bb) + np.abs(dd - cc)) / 4.0
        a_diag = (np.abs(aa - dd) + np.abs(bb - cc)) / 2.0
        a_score = a_adj - a_diag
        if float((a_score > 80).mean()) > 0.10:
            return True
    return False


def isolate_motifs(
    image: Image.Image,
    boxes: list[MotifBox],
    layers_dir: Path,
    *,
    api_key: str | None = None,
    matte_threshold: float = 0.35,
    use_gemini_fallback: bool = True,
) -> tuple[list[IsolatedLayer], np.ndarray]:
    """
    Isolate each motif. Returns layers metadata and a full-size union alpha mask (uint8).
    """
    layers_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    bg = estimate_background_color(rgb)
    union = np.zeros((h, w), dtype=np.uint8)
    results: list[IsolatedLayer] = []

    for box in boxes:
        x0, y0, x1, y1 = _bbox_to_pixels(box.bbox_norm, w, h)
        crop = rgb[y0:y1, x0:x1].copy()
        if crop.size == 0:
            continue

        alpha, score = color_matte(crop, bg)
        method = "color_matte"
        rgba = np.dstack([crop, alpha])

        if score < matte_threshold and use_gemini_fallback:
            gem = _gemini_extract_crop(Image.fromarray(crop), box.label, api_key)
            if gem is not None:
                # Resize gemini result to crop if needed
                if gem.shape[0] != crop.shape[0] or gem.shape[1] != crop.shape[1]:
                    gem_img = Image.fromarray(gem).resize((crop.shape[1], crop.shape[0]), Image.Resampling.LANCZOS)
                    gem = np.array(gem_img)
                rgba = gem
                method = "gemini_extract"
                score = max(score, 0.55)

        trimmed, (tx, ty, tw, th) = _trim_alpha(rgba)
        if trimmed[:, :, 3].max() < 8:
            print(f"  skip empty layer {box.id} ({box.label})")
            continue

        out_name = f"{box.id}.png"
        out_path = layers_dir / out_name
        Image.fromarray(trimmed).save(out_path)

        # Stamp union mask in full-image coords
        full_alpha = np.zeros((h, w), dtype=np.uint8)
        # Map trimmed alpha back into crop then full image
        local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        local[ty : ty + th, tx : tx + tw] = trimmed[:, :, 3]
        full_alpha[y0:y1, x0:x1] = np.maximum(full_alpha[y0:y1, x0:x1], local)
        union = np.maximum(union, full_alpha)

        abs_x = x0 + tx
        abs_y = y0 + ty
        results.append(
            IsolatedLayer(
                id=box.id,
                label=box.label,
                path=f"layers/{out_name}",
                bbox_px=[abs_x, abs_y, tw, th],
                bbox_norm=[abs_x / w, abs_y / h, tw / w, th / h],
                confidence=box.confidence,
                matte_score=float(score),
                method=method,
            )
        )
        print(f"  isolate {box.id} ({box.label}) score={score:.2f} via {method}")

    return results, union


def layers_to_dicts(layers: list[IsolatedLayer]) -> list[dict[str, Any]]:
    return [
        {
            "id": L.id,
            "label": L.label,
            "src": L.path,
            "bbox_px": L.bbox_px,
            "bbox_norm": L.bbox_norm,
            "confidence": L.confidence,
            "matte_score": L.matte_score,
            "method": L.method,
        }
        for L in layers
    ]

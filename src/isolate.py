"""Per-motif isolation via color-distance matte (+ optional Gemini fallback)."""

from __future__ import annotations

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


def _gemini_extract_crop(
    crop_rgb: Image.Image,
    label: str,
    api_key: str | None,
    model: str = "gemini-2.5-flash-image",
) -> np.ndarray | None:
    """Ask Gemini to return a transparent cutout of ONE motif; best-effort."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        buf = BytesIO()
        crop_rgb.save(buf, format="PNG")
        prompt = (
            f"You are cutting a single textile motif for a layers menu.\n"
            f"Subject: {label or 'the primary motif'} nearest the center of this crop.\n\n"
            "Output requirements:\n"
            "- PNG with a REAL alpha channel (alpha=0 = transparent). "
            "Never draw a checkerboard, grid, or fake transparency pattern.\n"
            "- Keep ONLY that one complete motif. Exclude every neighboring leaf/frond/flower "
            "even if it overlaps or sits behind the subject.\n"
            "- Natural holes (e.g. monstera fenestrations) must be transparent, not filled.\n"
            "- Preserve original print colors and edges. No shadows, outlines, or new pixels.\n"
            "- If unsure which motif is primary, pick the largest one touching the crop center."
        )
        resp = client.models.generate_content(
            model=model,
            contents=[
                prompt,
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            ],
        )
        # Pull first inline image if present
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


def _ensure_cutout_alpha(rgba: np.ndarray, crop_rgb: np.ndarray) -> np.ndarray | None:
    """Normalize Gemini cutout alpha; return None if the model faked transparency."""
    out = rgba.copy()
    a = out[:, :, 3].astype(np.float32)
    rgb = out[:, :, :3].astype(np.float32)

    # Reject obvious checkerboard / grid "transparency" renders
    if _looks_like_checkerboard(out):
        print("    gemini reject: checkerboard/fake transparency")
        return None

    if float(a.mean()) >= 240:
        # Opaque output — rebuild alpha from crop border color + kill near-white
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
        white = rgb.mean(axis=2)
        alpha = np.clip((d - 12.0) / 20.0, 0, 1)
        alpha[white > 245] = 0
        out[:, :, 3] = (alpha * 255).astype(np.uint8)
    return out


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

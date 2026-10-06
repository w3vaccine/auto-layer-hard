"""Isolate motifs using exact instance masks + edge defringe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from segment import MotifInstance, estimate_background_color


@dataclass
class IsolatedLayer:
    id: str
    label: str
    path: str
    bbox_px: list[int]
    bbox_norm: list[float]
    confidence: float
    matte_score: float
    method: str


def defringe(rgba: np.ndarray, bg: np.ndarray, erode: int = 2) -> np.ndarray:
    """Pull edge colors away from background contamination; tighten near-bg fringe."""
    out = rgba.copy()
    alpha = out[:, :, 3]
    rgb = out[:, :, :3].astype(np.float32)
    bgv = bg.astype(np.float32).reshape(1, 1, 3)

    solid = (alpha > 180).astype(np.uint8)
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1,) * 2)
        core = cv2.erode(solid, k, iterations=1)
    else:
        core = solid
    edge = ((alpha > 15) & (core == 0)).astype(bool)
    if edge.any():
        diff = rgb - bgv
        # Stronger unmix toward pure motif color
        boosted = bgv + diff * 1.65
        rgb[edge] = np.clip(boosted[edge], 0, 255)
        # Drop near-bg fringe pixels
        dist = np.sqrt(np.sum((rgb - bgv) ** 2, axis=2))
        near_bg = edge & (dist < 16)
        alpha = alpha.astype(np.float32)
        alpha[near_bg] *= 0.08
        # Mid-fringe: soft attenuate
        mid = edge & (dist >= 16) & (dist < 28)
        alpha[mid] *= 0.55

    a = alpha.astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), sigmaX=0.45)
    a[(out[:, :, 3] < 8)] = 0
    out[:, :, :3] = rgb.astype(np.uint8)
    out[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
    return out


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


def isolate_instances(
    image: Image.Image,
    instances: list[MotifInstance],
    layers_dir: Path,
) -> tuple[list[IsolatedLayer], np.ndarray]:
    """Cut each instance with its exact mask. Returns layers + union alpha."""
    layers_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    bg = estimate_background_color(rgb)
    union = np.zeros((h, w), dtype=np.uint8)
    results: list[IsolatedLayer] = []

    for inst in instances:
        mask = inst.mask
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Accumulate coverage from the discovery mask before any defringe dropouts
        union = np.maximum(union, mask)

        rgba = np.dstack([rgb, mask])
        rgba = defringe(rgba, bg)
        trimmed, (tx, ty, tw, th) = _trim_alpha(rgba)
        if trimmed[:, :, 3].max() < 8:
            # Defringe can wipe camo/low-contrast edges — fall back to raw cutout
            rgba = np.dstack([rgb, mask])
            trimmed, (tx, ty, tw, th) = _trim_alpha(rgba)
            if trimmed[:, :, 3].max() < 8:
                continue

        ys, xs = np.where(mask > 8)
        if len(xs) == 0:
            continue
        abs_x, abs_y = tx, ty

        out_name = f"{inst.id}.png"
        Image.fromarray(trimmed).save(layers_dir / out_name)

        score = float(np.clip((mask > 40).sum() / max(1, tw * th), 0, 1))
        results.append(
            IsolatedLayer(
                id=inst.id,
                label=inst.label,
                path=f"layers/{out_name}",
                bbox_px=[abs_x, abs_y, tw, th],
                bbox_norm=[abs_x / w, abs_y / h, tw / w, th / h],
                confidence=inst.confidence,
                matte_score=score,
                method="exact_mask",
            )
        )

    print(f"  isolated {len(results)} layers (exact masks)")
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

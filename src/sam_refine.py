"""SAM2 (via Ultralytics) mask refinement for motif instances."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from segment import MotifInstance

_SAM = None
_SAM_ERROR: str | None = None


def sam_available() -> bool:
    global _SAM_ERROR
    try:
        from ultralytics import SAM  # noqa: F401

        return True
    except Exception as exc:  # noqa: BLE001
        _SAM_ERROR = str(exc)
        return False


def _get_sam(model_name: str = "sam2_b.pt"):
    """Lazy-load SAM2 weights (downloads once)."""
    global _SAM, _SAM_ERROR
    if _SAM is not None:
        return _SAM
    try:
        from ultralytics import SAM

        _SAM = SAM(model_name)
        return _SAM
    except Exception as exc:  # noqa: BLE001
        _SAM_ERROR = str(exc)
        print(f"  SAM unavailable: {exc}")
        return None


def _bbox_xyxy(bbox_norm: list[float], w: int, h: int, pad: float = 0.02) -> list[int]:
    x, y, bw, bh = bbox_norm
    pad_x, pad_y = bw * pad, bh * pad
    x0 = max(0, int((x - pad_x) * w))
    y0 = max(0, int((y - pad_y) * h))
    x1 = min(w, int((x + bw + pad_x) * w))
    y1 = min(h, int((y + bh + pad_y) * h))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        x0, y0 = max(0, int(x * w)), max(0, int(y * h))
        x1, y1 = min(w, int((x + bw) * w)), min(h, int((y + bh) * h))
    return [x0, y0, x1, y1]


def _center_point(mask: np.ndarray, bbox_norm: list[float], w: int, h: int) -> list[int]:
    ys, xs = np.where(mask > 40)
    if len(xs) > 0:
        return [int(xs.mean()), int(ys.mean())]
    x, y, bw, bh = bbox_norm
    return [int((x + bw / 2) * w), int((y + bh / 2) * h)]


def _mask_from_sam_result(result, h: int, w: int) -> np.ndarray | None:
    if result is None:
        return None
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    data = getattr(masks, "data", None)
    if data is None:
        return None
    arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    if arr.ndim == 3:
        # Take highest-area mask if multiple
        areas = arr.reshape(arr.shape[0], -1).sum(axis=1)
        arr = arr[int(np.argmax(areas))]
    m = (arr > 0.5).astype(np.uint8) * 255
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def refine_instance_sam(
    image_rgb: np.ndarray,
    instance: "MotifInstance",
    sam_model,
) -> np.ndarray:
    """Refine one instance mask with SAM2 using bbox + center point prompts."""
    h, w = image_rgb.shape[:2]
    box = _bbox_xyxy(instance.bbox_norm, w, h)
    point = _center_point(instance.mask, instance.bbox_norm, w, h)

    try:
        results = sam_model.predict(
            image_rgb,
            bboxes=[box],
            points=[point],
            labels=[1],
            verbose=False,
        )
        if not results:
            return instance.mask
        m = _mask_from_sam_result(results[0], h, w)
        if m is None or m.sum() < 20:
            return instance.mask

        # Intersect lightly with dilated CV mask to avoid SAM grabbing neighbors
        cv_dil = cv2.dilate(
            (instance.mask > 20).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            1,
        )
        combined = m.copy()
        combined[cv_dil == 0] = (combined[cv_dil == 0] * 0.15).astype(np.uint8)
        # Prefer SAM interior where confident
        alpha = np.maximum(combined, np.minimum(instance.mask, combined))
        # Soft feather
        alpha_f = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), sigmaX=0.6)
        alpha = np.clip(alpha_f, 0, 255).astype(np.uint8)
        # Keep only largest CC
        n, lab, st, _ = cv2.connectedComponentsWithStats((alpha > 40).astype(np.uint8), 8)
        if n > 1:
            largest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            keep = (lab == largest).astype(np.uint8)
            alpha = (alpha * keep).astype(np.uint8)
        return alpha
    except Exception as exc:  # noqa: BLE001
        print(f"  SAM refine failed for {instance.id}: {exc}")
        return instance.mask


def refine_instances_sam(
    image: Image.Image,
    instances: list,
    *,
    model_name: str = "sam2_b.pt",
    enabled: bool = True,
) -> list:
    """Refine all instance masks in-place with SAM2. Returns same list."""
    if not enabled or not instances:
        return instances
    sam = _get_sam(model_name)
    if sam is None:
        print("  skipping SAM refine (model not loaded)")
        return instances

    rgb = np.array(image.convert("RGB"))
    print(f"  SAM2 refining {len(instances)} masks…")
    improved = 0
    for inst in instances:
        before = int((inst.mask > 40).sum())
        new_mask = refine_instance_sam(rgb, inst, sam)
        after = int((new_mask > 40).sum())
        # Keep SAM if reasonably sized vs original
        if after >= max(20, int(before * 0.35)) and after <= int(before * 3.5 + 500):
            if abs(after - before) > before * 0.05 or after > before:
                improved += 1
            inst.mask = new_mask
            # Refresh bbox from mask
            ys, xs = np.where(new_mask > 20)
            if len(xs):
                h, w = new_mask.shape
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                inst.bbox_norm = [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]
    print(f"  SAM2 updated {improved}/{len(instances)} masks")
    return instances

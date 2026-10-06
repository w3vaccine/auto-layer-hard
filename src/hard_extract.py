"""Hard-print extraction: VLM seeds → SAM2 → soft alpha.

Used when print_type is soft | camo | busy. Clean prints stay on CV.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from discover import MotifBox  # noqa: E402
from sam_refine import _bbox_xyxy, _get_sam, _mask_from_sam_result, sam_available
from segment import MotifInstance

if TYPE_CHECKING:
    pass


SOFT_PROMPT = """You are analyzing a soft / watercolor textile print.
Find EVERY distinct motif instance (flowers, leaves, blobs, strokes).

Hard rules:
- Include the soft bled halo in each box — do not crop to the darkest core only.
- Every distinct instance is its own entry (do not merge repeats).
- Prefer slightly padded boxes around soft edges.
- Background wash that is continuous ground is NOT a motif.

Return ONLY JSON:
{"motifs":[{"id":"m01","label":"short","bbox_norm":[x,y,w,h],"confidence":0.0_to_1.0}]}
bbox_norm = [x,y,w,h] normalized 0..1, top-left + size."""

CAMO_PROMPT = """You are analyzing a camouflage / low figure-ground textile print.
Motifs blend into neighboring colors. Still find each distinct shape-island.

Hard rules:
- Segment by shape/region, not by a single background color.
- Every separate blob / organic shape is its own entry.
- Boxes should be tight around each region.
- Do not invent motifs; do not return one giant full-image box.

Return ONLY JSON:
{"motifs":[{"id":"m01","label":"camo region","bbox_norm":[x,y,w,h],"confidence":0.0_to_1.0}]}
bbox_norm = [x,y,w,h] normalized 0..1."""

BUSY_PROMPT = """You are analyzing a dense textile print (leaves, florals, paisley, interlocking motifs).
Find EACH complete primary motif as its OWN tight box — one full leaf, one full flower, one full palm frond, one paisley.

Hard rules:
- WHOLE motif only: the box must contain the entire silhouette (all lobes, tip, stem). Never cut through the middle of a leaf.
- ONE motif per box. Overlapping leaves each get their own box (monstera ≠ palm behind it).
- Tight crop: minimize empty ground and pieces of neighboring motifs inside the box.
- Typical prints have 8–40 primary motifs. List every complete one you can see.
- Skip ultra-tiny noise (< ~0.2% of image) and continuous background/ground.
- NEVER return a box covering more than ~28% of the image, and NEVER a full-image box.

Return ONLY JSON:
{"motifs":[{"id":"m01","label":"leaf|flower|frond|paisley|motif","bbox_norm":[x,y,w,h],"confidence":0.0_to_1.0}]}
bbox_norm = [x,y,w,h] normalized 0..1."""


def _prompt_for(print_type: str) -> str:
    if print_type == "camo":
        return CAMO_PROMPT
    if print_type == "soft":
        return SOFT_PROMPT
    if print_type == "busy":
        return BUSY_PROMPT
    return SOFT_PROMPT


def _filter_max_area(boxes: list[MotifBox], max_area: float = 0.32) -> list[MotifBox]:
    """Drop full-image / giant boxes that ruin SAM (they swallow every motif)."""
    kept = [b for b in boxes if b.area() <= max_area]
    dropped = len(boxes) - len(kept)
    if dropped:
        print(f"  dropped {dropped} giant boxes (area>{max_area})")
    return kept


def _discover_boxes_typed(
    image: Image.Image,
    print_type: str,
    *,
    api_key: str | None,
    model: str,
    max_instances: int,
) -> list[MotifBox]:
    """Use type-specific first pass, then generic gapfill passes."""
    from discover import (
        PASS_N_PROMPT,
        _call_gemini,
        _dedupe,
        _filter_min_area,
        _load_client,
        _parse_motifs,
        overlay_boxes as ov,
    )

    if not api_key:
        return []

    client = _load_client(api_key)
    rgb = image.convert("RGB")
    prompt = _prompt_for(print_type)
    payload = _call_gemini(client, model, prompt, [rgb])
    boxes = _parse_motifs(payload, pass_index=1, id_offset=0)
    min_area = 0.0008 if print_type == "busy" else 0.00035
    if print_type == "camo":
        min_area = 0.0005
    boxes = _filter_min_area(boxes, min_area=min_area)
    if print_type == "busy":
        boxes = _filter_max_area(boxes, max_area=0.32)
    boxes = _dedupe(boxes, iou_thresh=0.4)
    print(f"  hard discover pass1 ({print_type}): {len(boxes)}")

    # Busy quality > raw speed: keep gapfilling until we have a real motif set.
    # Latency stays OK because SAM is batched @512 and box count is capped.
    if print_type == "busy":
        try:
            max_passes = max(1, int(os.environ.get("HARD_BUSY_GEMINI_PASSES", "3")))
        except ValueError:
            max_passes = 3
        min_boxes = min(12, max_instances)
        early_stop = max_instances
    elif print_type == "camo":
        max_passes = 3
        min_boxes = 0
        early_stop = max_instances
    else:
        max_passes = 2
        min_boxes = 0
        early_stop = max_instances

    for pass_index in range(2, max_passes + 1):
        if len(boxes) >= max_instances or len(boxes) >= early_stop:
            break
        # Only early-exit busy once we have enough whole motifs
        if print_type == "busy" and len(boxes) >= max(min_boxes, max_instances // 2):
            break
        overlay = ov(rgb, boxes)
        payload = _call_gemini(client, model, PASS_N_PROMPT, [rgb, overlay])
        new_boxes = _parse_motifs(payload, pass_index, id_offset=len(boxes))
        new_boxes = _filter_min_area(new_boxes, min_area=min_area * (0.7 if print_type == "camo" else 1.0))
        if print_type == "busy":
            new_boxes = _filter_max_area(new_boxes, max_area=0.32)
        before = len(boxes)
        boxes = _dedupe(boxes + new_boxes, iou_thresh=0.35 if print_type == "camo" else 0.4)[
            :max_instances
        ]
        gained = len(boxes) - before
        print(f"  hard discover pass {pass_index}: +{gained} (total {len(boxes)})")
        if gained == 0 and len(boxes) >= min_boxes:
            break
        if gained == 0 and print_type != "busy":
            break

    for i, b in enumerate(boxes, start=1):
        b.id = f"m{i:03d}"
    return boxes


def _slic_fallback_boxes(image: Image.Image, print_type: str, max_instances: int) -> list[MotifBox]:
    """No-API fallback: SLIC superpixels merged by color (esp. camo)."""
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    try:
        from cv2 import ximgproc

        slic = ximgproc.createSuperpixelSLIC(rgb, region_size=max(20, min(h, w) // 40), ruler=20.0)
        slic.iterate(8)
        labels = slic.getLabels()
        n = int(labels.max()) + 1
    except Exception:
        # Fallback: grid-ish watershed on quantized colors
        small = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        q = (small // 24) * 24
        gray = cv2.cvtColor(q, cv2.COLOR_LAB2RGB)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n_cc, labels = cv2.connectedComponents(bw)
        n = n_cc

    boxes: list[MotifBox] = []
    img_a = float(h * w)
    min_a = 0.0006 * img_a
    max_a = 0.35 * img_a
    for i in range(0 if labels.min() == 0 else 1, n):
        ys, xs = np.where(labels == i)
        if len(xs) < min_a or len(xs) > max_a:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bw_ = (x1 - x0) / w
        bh_ = (y1 - y0) / h
        if bw_ * bh_ < 0.0005:
            continue
        boxes.append(
            MotifBox(
                id=f"s{i:03d}",
                label="region",
                bbox_norm=[x0 / w, y0 / h, bw_, bh_],
                confidence=0.45,
                pass_index=0,
            )
        )
    boxes.sort(key=lambda b: -b.area())
    return boxes[:max_instances]


def soft_alpha_from_sam(
    rgb: np.ndarray,
    sam_mask: np.ndarray,
    bbox_xyxy: list[int],
    *,
    mode: str = "soft",
) -> np.ndarray:
    """Build soft alpha from SAM binary mask + local color affinity."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    binary = sam_mask > 127

    # Hard clip SAM to the prompt box (prevents neighbor grab)
    box_mask = np.zeros((h, w), dtype=bool)
    box_mask[y0:y1, x0:x1] = True
    binary = binary & box_mask

    if not binary.any():
        return np.zeros((h, w), dtype=np.uint8)

    if mode == "camo":
        feather = 2.0
        dilate_px = 2
    elif mode == "busy":
        feather = 1.2
        dilate_px = 1
    else:  # soft
        feather = 4.0
        dilate_px = 5

    core = binary.astype(np.uint8) * 255
    dist_in = cv2.distanceTransform((core > 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha_core = np.clip(dist_in / max(feather, 1e-3), 0, 1)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    dil = cv2.dilate((core > 0).astype(np.uint8), k, 1)
    er = cv2.erode((core > 0).astype(np.uint8), k, 1)
    band = (dil > 0) & (er == 0) & box_mask

    roi = rgb[y0:y1, x0:x1]
    roi_bin = binary[y0:y1, x0:x1]
    if roi_bin.any() and (~roi_bin).any():
        fg = roi[roi_bin].astype(np.float32).mean(axis=0)
        bg = roi[~roi_bin].astype(np.float32).mean(axis=0)
    else:
        fg = rgb[binary].astype(np.float32).mean(axis=0)
        bg = np.array([255, 255, 255], dtype=np.float32)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_lab = cv2.cvtColor(fg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(
        np.float32
    )
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(
        np.float32
    )
    d_fg = np.sqrt(((lab - fg_lab) ** 2).sum(axis=2))
    d_bg = np.sqrt(((lab - bg_lab) ** 2).sum(axis=2))
    aff = np.clip(d_bg / (d_fg + d_bg + 1e-3), 0, 1)

    alpha = alpha_core.copy()
    if mode == "soft":
        alpha[band] = np.maximum(alpha[band], np.where(aff[band] > 0.55, aff[band] * 0.75, 0))
    elif mode == "camo":
        alpha[band] = np.maximum(alpha[band], np.where(aff[band] > 0.58, aff[band] * 0.5, 0))
    else:
        alpha[band] = np.maximum(alpha[band], np.where(aff[band] > 0.6, aff[band] * 0.35, 0))

    alpha *= box_mask.astype(np.float32)
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    sigma = 1.0 if mode == "soft" else 0.5
    alpha_u8 = cv2.GaussianBlur(alpha_u8, (0, 0), sigmaX=sigma)
    alpha_u8 = (alpha_u8.astype(np.float32) * box_mask.astype(np.float32)).astype(np.uint8)
    return alpha_u8


def _sam_max_side() -> int:
    """CPU SAM at full res is ~10s+/box; 512 keeps quality usable and is ~4× faster."""
    try:
        return max(256, int(os.environ.get("HARD_SAM_MAX_SIDE", "512")))
    except ValueError:
        return 512


def _prepare_sam_rgb(rgb: np.ndarray, max_side: int | None = None) -> tuple[np.ndarray, float, int, int]:
    """Downscale for SAM inference. Returns (rgb_small, scale, full_h, full_w)."""
    h, w = rgb.shape[:2]
    side = _sam_max_side() if max_side is None else max_side
    m = max(h, w)
    if m <= side:
        return rgb, 1.0, h, w
    scale = side / float(m)
    small = cv2.resize(
        rgb,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return small, scale, h, w


def sam_mask_from_box(
    rgb: np.ndarray,
    bbox_norm: list[float],
    sam_model,
) -> np.ndarray | None:
    masks = sam_masks_from_boxes(rgb, [bbox_norm], sam_model)
    return masks[0] if masks else None


def sam_masks_from_boxes(
    rgb: np.ndarray,
    bbox_norms: list[list[float]],
    sam_model,
    *,
    batch_size: int = 16,
) -> list[np.ndarray | None]:
    """Batched SAM2 prompts on a downscaled image; upsample masks to full res."""
    if not bbox_norms:
        return []
    small, scale, h, w = _prepare_sam_rgb(rgb)
    sh, sw = small.shape[:2]
    out: list[np.ndarray | None] = [None] * len(bbox_norms)

    def _one(idx: int, bbox_norm: list[float]) -> np.ndarray | None:
        box = _bbox_xyxy(bbox_norm, sw, sh, pad=0.03)
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        try:
            results = sam_model.predict(
                small,
                bboxes=[box],
                points=[[cx, cy]],
                labels=[1],
                verbose=False,
            )
            if not results:
                return None
            return _mask_from_sam_result(results[0], h, w)
        except Exception as exc:  # noqa: BLE001
            print(f"  SAM box predict failed: {exc}")
            return None

    # Prefer multi-box predict (one encoder pass); fall back to sequential.
    for start in range(0, len(bbox_norms), max(1, batch_size)):
        chunk = bbox_norms[start : start + batch_size]
        boxes = [_bbox_xyxy(b, sw, sh, pad=0.03) for b in chunk]
        points = [[int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)] for b in boxes]
        labels = [1] * len(boxes)
        batched_ok = False
        try:
            results = sam_model.predict(
                small,
                bboxes=boxes,
                points=points,
                labels=labels,
                verbose=False,
            )
            if results and getattr(results[0], "masks", None) is not None:
                data = results[0].masks.data
                arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
                if arr.ndim == 3 and arr.shape[0] == len(chunk):
                    for i, m in enumerate(arr):
                        mm = (m > 0.5).astype(np.uint8) * 255
                        if mm.shape[:2] != (h, w):
                            mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                        out[start + i] = mm
                    batched_ok = True
                elif arr.ndim == 3 and arr.shape[0] == 1 and len(chunk) == 1:
                    mm = (arr[0] > 0.5).astype(np.uint8) * 255
                    if mm.shape[:2] != (h, w):
                        mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                    out[start] = mm
                    batched_ok = True
        except Exception:  # noqa: BLE001
            batched_ok = False
        if not batched_ok:
            for i, bn in enumerate(chunk):
                out[start + i] = _one(start + i, bn)
    return out


def _color_matte_in_box(rgb: np.ndarray, bbox_norm: list[float], *, soft: bool) -> np.ndarray:
    """Fallback when SAM unavailable: local FG/BG matte inside box."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = _bbox_xyxy(bbox_norm, w, h, pad=0.02)
    alpha = np.zeros((h, w), dtype=np.uint8)
    roi = rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return alpha
    # Assume center patch is FG, border of ROI is BG
    rh, rw = roi.shape[:2]
    cy0, cy1 = rh // 3, 2 * rh // 3
    cx0, cx1 = rw // 3, 2 * rw // 3
    fg = roi[cy0:cy1, cx0:cx1].reshape(-1, 3).astype(np.float32).mean(axis=0)
    border = np.concatenate(
        [
            roi[:2].reshape(-1, 3),
            roi[-2:].reshape(-1, 3),
            roi[:, :2].reshape(-1, 3),
            roi[:, -2:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg = border.mean(axis=0)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_l = cv2.cvtColor(fg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(
        np.float32
    )
    bg_l = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(
        np.float32
    )
    d_fg = np.sqrt(((lab - fg_l) ** 2).sum(axis=2))
    d_bg = np.sqrt(((lab - bg_l) ** 2).sum(axis=2))
    aff = d_bg / (d_fg + d_bg + 1e-3)
    thr = 0.42 if soft else 0.55
    a = np.clip((aff - thr) / (1.0 - thr + 1e-3), 0, 1)
    if soft:
        a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.5)
    alpha[y0:y1, x0:x1] = (a * 255).astype(np.uint8)
    return alpha


def seeded_graphic_matte(
    rgb: np.ndarray,
    bbox_norm: list[float],
    bg: np.ndarray,
    *,
    competitor_xy: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Cut one flat-graphic motif from a VLM box.

    Watershed from the box-center seed (all ink tones of that leaf) with
    ground + other motif centers as background markers. Leaf holes = transparent.
    """
    from segment import lab_distance

    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = _bbox_xyxy(bbox_norm, w, h, pad=0.02)
    alpha = np.zeros((h, w), dtype=np.uint8)
    roi = rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return alpha
    rh, rw = roi.shape[:2]
    dist_bg = lab_distance(roi, bg)
    ink = dist_bg > 11.0
    if not ink.any():
        return alpha

    cy, cx = rh // 2, rw // 2
    if not ink[cy, cx]:
        ys, xs = np.where(ink)
        i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        cy, cx = int(ys[i]), int(xs[i])

    # Multi-marker watershed on the ROI (OpenCV needs a 3-channel image)
    markers = np.zeros((rh, rw), dtype=np.int32)
    # BG: ground + ROI border
    markers[dist_bg < 9.0] = 1
    markers[0, :] = 1
    markers[-1, :] = 1
    markers[:, 0] = 1
    markers[:, -1] = 1
    # Competing motif centers inside this ROI → BG (prevents neighbor swallow)
    for px, py in competitor_xy or []:
        lx, ly = px - x0, py - y0
        if 0 <= lx < rw and 0 <= ly < rh:
            cv2.circle(markers, (lx, ly), max(4, min(rh, rw) // 25), 1, -1)
    # FG seed
    seed_r = max(4, min(rh, rw) // 18)
    cv2.circle(markers, (cx, cy), seed_r, 2, -1)
    # Don't put FG on clear ground
    markers[(markers == 2) & (dist_bg < 9.0)] = 1

    try:
        ws = roi.copy()
        cv2.watershed(ws, markers)
        keep = markers == 2
    except Exception:  # noqa: BLE001
        # Fallback: same-hue CC from seed
        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32)
        seed_rgb = roi[cy, cx].astype(np.float32)
        seed_lab = cv2.cvtColor(seed_rgb.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[
            0, 0
        ].astype(np.float32)
        d_seed = np.sqrt(((lab - seed_lab) ** 2).sum(axis=2))
        same = (d_seed < 32.0) & ink
        n, labels = cv2.connectedComponents(same.astype(np.uint8), 8)
        sid = int(labels[cy, cx]) if n > 1 else 0
        keep = labels == sid if sid > 0 else ink

    # Restrict to ink so bg holes stay empty
    keep = keep & ink
    if not keep.any():
        return alpha
    # Reattach to seed CC only
    n2, lab2 = cv2.connectedComponents(keep.astype(np.uint8), 8)
    if n2 > 1:
        sid = int(lab2[cy, cx])
        if sid == 0:
            ys, xs = np.where(lab2 > 0)
            if len(xs):
                i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
                sid = int(lab2[ys[i], xs[i]])
        if sid > 0:
            keep = lab2 == sid

    a = keep.astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 0.5)
    a = np.clip(a, 0, 1)
    a[dist_bg < 7.5] = 0.0
    alpha[y0:y1, x0:x1] = (a * 255.0).astype(np.uint8)
    return alpha


def _exclusive_alphas(instances: list[MotifInstance]) -> list[MotifInstance]:
    """Larger motifs claim pixels first so crops don't include neighbors."""
    if len(instances) < 2:
        return instances
    order = sorted(
        range(len(instances)),
        key=lambda i: -int((instances[i].mask > 40).sum()),
    )
    claimed = np.zeros_like(instances[0].mask, dtype=bool)
    for i in order:
        m = instances[i].mask
        solid = m > 40
        # Strip pixels already owned by a larger motif
        strip = solid & claimed
        if strip.any():
            m = m.copy()
            m[strip] = 0
            instances[i].mask = m
            ys, xs = np.where(m > 20)
            if len(xs) == 0:
                continue
            h, w = m.shape
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            instances[i].bbox_norm = [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]
        claimed |= instances[i].mask > 40
    # Drop emptied
    return [inst for inst in instances if int((inst.mask > 20).sum()) >= 20]


def boxes_to_instances_graphic(
    image: Image.Image,
    boxes: list[MotifBox],
) -> list[MotifInstance]:
    """Busy/sharp graphics: multi-seed watershed cutouts (VLM box = seed only).

    SAM soft-alpha chops leaves and fills holes. Here each VLM center is a
    watershed marker on the full canvas so motifs grow to their natural
    silhouette (including past a clipped box) and neighbors compete.
    """
    from segment import estimate_background_color, lab_distance

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    bg = estimate_background_color(rgb)
    dist_bg = lab_distance(rgb, bg)
    ink = dist_bg > 11.0

    centers: list[tuple[int, int]] = []
    for b in boxes:
        x, y, bw, bh = b.bbox_norm
        cx, cy = int((x + bw / 2) * w), int((y + bh / 2) * h)
        cx, cy = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
        # Snap seed onto ink inside the box if center landed in a hole/ground
        if not ink[cy, cx]:
            x0, y0, x1, y1 = _bbox_xyxy(b.bbox_norm, w, h, pad=0.0)
            roi_ink = ink[y0:y1, x0:x1]
            if roi_ink.any():
                ys, xs = np.where(roi_ink)
                i = int(np.argmin((ys + y0 - cy) ** 2 + (xs + x0 - cx) ** 2))
                cy, cx = int(ys[i] + y0), int(xs[i] + x0)
        centers.append((cx, cy))

    print(f"  graphic matte {len(boxes)} seeds (full-image watershed)…")
    t0 = time.time()
    markers = np.zeros((h, w), dtype=np.int32)
    markers[~ink] = 1  # ground / holes = background
    # Border as bg so edge-cropped motifs don't flood the frame
    markers[0, :] = 1
    markers[-1, :] = 1
    markers[:, 0] = 1
    markers[:, -1] = 1
    seed_r = max(5, min(h, w) // 80)
    for i, (cx, cy) in enumerate(centers):
        cv2.circle(markers, (cx, cy), seed_r, i + 2, -1)
        # Never plant FG on clear ground
        if dist_bg[cy, cx] < 9.0:
            markers[cy, cx] = 1

    try:
        cv2.watershed(rgb.copy(), markers)
    except Exception as exc:  # noqa: BLE001
        print(f"  watershed failed ({exc}) — per-box fallback")
        return _boxes_to_instances_graphic_fallback(rgb, boxes, bg)

    instances: list[MotifInstance] = []
    img_px = float(h * w)
    for i, b in enumerate(boxes):
        keep = (markers == (i + 2)) & ink
        if not keep.any():
            continue
        # Keep largest CC (drop speckles from watershed bleed)
        n, lab = cv2.connectedComponents(keep.astype(np.uint8), 8)
        if n > 1:
            areas = [(int((lab == j).sum()), j) for j in range(1, n)]
            areas.sort(reverse=True)
            keep = lab == areas[0][1]
            # If seed not in largest, prefer CC containing seed
            cx, cy = centers[i]
            if lab[cy, cx] > 0:
                keep = lab == lab[cy, cx]

        a = keep.astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), 0.45)
        a = np.clip(a, 0, 1)
        a[dist_bg < 7.5] = 0.0
        alpha = (a * 255.0).astype(np.uint8)
        if int((alpha > 20).sum()) < 40:
            continue
        if float((alpha > 20).sum()) / img_px > 0.4:
            print(f"  skip {b.id}: basin covers >40% of image")
            continue
        ys, xs = np.where(alpha > 20)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        instances.append(
            MotifInstance(
                id=b.id,
                label=b.label or "motif",
                bbox_norm=[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
                confidence=float(b.confidence),
                mask=alpha,
                pass_index=b.pass_index,
            )
        )

    before = len(instances)
    instances = _exclusive_alphas(instances)
    # Drop near-duplicate basins (two seeds on one leaf)
    instances = _dedupe_mask_instances(instances, iou_thr=0.55)
    print(
        f"  graphic matte done in {time.time() - t0:.1f}s → {len(instances)} "
        f"(seeds {len(boxes)}, pre-dedupe {before})"
    )
    return instances


def _dedupe_mask_instances(
    instances: list[MotifInstance], *, iou_thr: float = 0.55
) -> list[MotifInstance]:
    if len(instances) < 2:
        return instances
    order = sorted(instances, key=lambda i: -int((i.mask > 40).sum()))
    kept: list[MotifInstance] = []
    for inst in order:
        aa = inst.mask > 40
        drop = False
        for k in kept:
            bb = k.mask > 40
            inter = float((aa & bb).sum())
            if inter <= 0:
                continue
            union = float((aa | bb).sum()) or 1.0
            if inter / union >= iou_thr:
                drop = True
                break
        if not drop:
            kept.append(inst)
    return kept


def _boxes_to_instances_graphic_fallback(
    rgb: np.ndarray, boxes: list[MotifBox], bg: np.ndarray
) -> list[MotifInstance]:
    h, w = rgb.shape[:2]
    instances: list[MotifInstance] = []
    for b in boxes:
        alpha = seeded_graphic_matte(rgb, b.bbox_norm, bg, competitor_xy=None)
        if int((alpha > 20).sum()) < 30:
            continue
        ys, xs = np.where(alpha > 20)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        instances.append(
            MotifInstance(
                id=b.id,
                label=b.label or "motif",
                bbox_norm=[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
                confidence=float(b.confidence),
                mask=alpha,
                pass_index=b.pass_index,
            )
        )
    return _exclusive_alphas(instances)


def boxes_to_instances_sam(
    image: Image.Image,
    boxes: list[MotifBox],
    *,
    print_type: str,
    sam_model_name: str = "sam2_b.pt",
) -> list[MotifInstance]:
    if print_type == "busy":
        return boxes_to_instances_graphic(image, boxes)

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    mode = "camo" if print_type == "camo" else "soft"
    sam = _get_sam(sam_model_name) if sam_available() else None
    if sam is None:
        print("  SAM unavailable — using color matte fallback inside boxes")
        sam_masks: list[np.ndarray | None] = [None] * len(boxes)
    else:
        t0 = time.time()
        print(f"  SAM2 {len(boxes)} boxes @max{_sam_max_side()} (batched)…")
        sam_masks = sam_masks_from_boxes(rgb, [b.bbox_norm for b in boxes], sam)
        print(f"  SAM2 done in {time.time() - t0:.1f}s")

    instances: list[MotifInstance] = []
    img_px = float(h * w)
    for b, m in zip(boxes, sam_masks):
        box = _bbox_xyxy(b.bbox_norm, w, h, pad=0.03)
        # Reject SAM masks that blew past the prompt box (whole-print grab)
        if m is not None and int(m.sum()) >= 30:
            mask_frac = float((m > 127).sum()) / img_px
            box_frac = max(b.area() * 1.8, 0.02)
            if mask_frac > min(0.45, max(0.12, box_frac * 4)):
                print(f"  SAM reject {b.id}: mask {mask_frac:.1%} >> box {b.area():.1%}")
                m = None
        if m is not None and int(m.sum()) >= 30:
            # IMPORTANT: no CV-ink intersection for camo/soft
            alpha = soft_alpha_from_sam(rgb, m, box, mode=mode)
        else:
            alpha = _color_matte_in_box(rgb, b.bbox_norm, soft=(mode == "soft"))

        if int((alpha > 20).sum()) < 20:
            continue
        # Also reject color-matte / alpha that covers most of the canvas
        if float((alpha > 20).sum()) / img_px > 0.5:
            print(f"  skip {b.id}: alpha covers >50% of image")
            continue
        ys, xs = np.where(alpha > 20)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        instances.append(
            MotifInstance(
                id=b.id,
                label=b.label or "motif",
                bbox_norm=[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
                confidence=float(b.confidence),
                mask=alpha,
                pass_index=b.pass_index,
            )
        )
    return _exclusive_alphas(instances)


def _merge_overlapping(instances: list[MotifInstance], iou_thr: float = 0.55) -> list[MotifInstance]:
    """Greedy merge for busy prints — keep larger mask."""
    if len(instances) < 2:
        return instances

    def iou(a: MotifInstance, b: MotifInstance) -> float:
        aa = a.mask > 40
        bb = b.mask > 40
        inter = float((aa & bb).sum())
        if inter <= 0:
            return 0.0
        union = float((aa | bb).sum())
        return inter / union if union else 0.0

    kept: list[MotifInstance] = []
    order = sorted(instances, key=lambda i: -int((i.mask > 40).sum()))
    suppressed = set()
    for i, inst in enumerate(order):
        if i in suppressed:
            continue
        acc = inst.mask.copy()
        for j in range(i + 1, len(order)):
            if j in suppressed:
                continue
            if iou(inst, order[j]) >= iou_thr:
                acc = np.maximum(acc, order[j].mask)
                suppressed.add(j)
        inst.mask = acc
        ys, xs = np.where(acc > 20)
        if len(xs):
            h, w = acc.shape
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            inst.bbox_norm = [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]
        kept.append(inst)
    return kept


def _merge_box_lists(a: list[MotifBox], b: list[MotifBox], *, iou_thresh: float = 0.4) -> list[MotifBox]:
    from discover import _dedupe

    return _dedupe(a + b, iou_thresh=iou_thresh)


def _camo_contrast_boxes(
    rgb: np.ndarray,
    union: np.ndarray,
    *,
    max_new: int = 50,
) -> list[MotifBox]:
    """Seed boxes on uncovered high local-contrast regions (no bg assumption)."""
    h, w = rgb.shape[:2]
    med = cv2.medianBlur(rgb, 25)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    med_lab = cv2.cvtColor(med, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = np.sqrt(((lab - med_lab) ** 2).sum(axis=2))
    # Adaptive threshold
    thr = float(np.percentile(diff, 70))
    thr = max(thr, 8.0)
    uncovered = (diff > thr) & (union <= 20)
    u8 = uncovered.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k, iterations=1)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((u8 > 0).astype(np.uint8), 8)
    img_a = float(h * w)
    boxes: list[MotifBox] = []
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) if n > 1 else []
    for idx in order:
        i = int(idx) + 1
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 0.0004 * img_a or area > 0.25 * img_a:
            continue
        x, y, bw, bh = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        # Pad slightly
        pad = 4
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        boxes.append(
            MotifBox(
                id=f"c{len(boxes)+1:03d}",
                label="camo region",
                bbox_norm=[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
                confidence=0.55,
                pass_index=90,
            )
        )
        if len(boxes) >= max_new:
            break
    return boxes


def _union_alpha(instances: list[MotifInstance], h: int, w: int) -> np.ndarray:
    union = np.zeros((h, w), dtype=np.uint8)
    for inst in instances:
        if inst.mask is not None:
            union = np.maximum(union, inst.mask)
    return union


def _merge_touching_small(
    instances: list[MotifInstance],
    *,
    min_keep_area: int,
) -> list[MotifInstance]:
    """Merge tiny residual layers into a touching larger neighbor (whole-leaf preference)."""
    if len(instances) < 2:
        return instances
    order = sorted(range(len(instances)), key=lambda i: -int((instances[i].mask > 40).sum()))
    alive = [True] * len(instances)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for i in order:
        if not alive[i]:
            continue
        area_i = int((instances[i].mask > 40).sum())
        if area_i >= min_keep_area:
            continue
        touch = cv2.dilate((instances[i].mask > 40).astype(np.uint8), k, 1)
        best_j, best_ov, best_area = None, 0, 0
        for j in order:
            if i == j or not alive[j]:
                continue
            area_j = int((instances[j].mask > 40).sum())
            if area_j < area_i:
                continue
            ov = int((touch & (instances[j].mask > 40)).sum())
            if ov > best_ov or (ov == best_ov and area_j > best_area):
                best_ov, best_j, best_area = ov, j, area_j
        if best_j is not None and best_ov > 0:
            instances[best_j].mask = np.maximum(instances[best_j].mask, instances[i].mask)
            h, w = instances[best_j].mask.shape
            ys, xs = np.where(instances[best_j].mask > 20)
            if len(xs):
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                instances[best_j].bbox_norm = [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]
            alive[i] = False
    kept = [instances[i] for i in range(len(instances)) if alive[i]]
    for n, inst in enumerate(kept, start=1):
        inst.id = f"m{n:03d}"
    print(f"  merge small residuals: {len(instances)} → {len(kept)}")
    return kept


def _hybrid_residual_fill(
    image: Image.Image,
    instances: list[MotifInstance],
    print_type: str,
    *,
    max_instances: int,
    sam_model_name: str,
) -> tuple[list[MotifInstance], dict]:
    """Push coverage up after primary SAM motifs without exploding layer count."""
    from segment import _cv_residual_gapfill, estimate_background_color, ink_distance

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    meta: dict = {"residual_fill": print_type}
    before = len(instances)

    if print_type == "camo":
        # Extra contrast seeds on uncovered areas → SAM (capped for CPU latency)
        union = _union_alpha(instances, h, w)
        camo_sam_cap = min(24, max(0, max_instances - before))
        extra_boxes = _camo_contrast_boxes(rgb, union, max_new=camo_sam_cap)
        meta["camo_extra_boxes"] = len(extra_boxes)
        if extra_boxes:
            print(f"  camo residual seeds: {len(extra_boxes)}")
            extra = boxes_to_instances_sam(
                image, extra_boxes, print_type="camo", sam_model_name=sam_model_name
            )
            # Prefer absorb into existing when heavy overlap
            for ex in extra:
                best = None
                best_ov = 0
                ea = ex.mask > 40
                for inst in instances:
                    ov = int((ea & (inst.mask > 40)).sum())
                    if ov > best_ov:
                        best_ov = ov
                        best = inst
                if best is not None and best_ov > 0.35 * int(ea.sum()):
                    best.mask = np.maximum(best.mask, ex.mask)
                    best.confidence = max(best.confidence, ex.confidence)
                else:
                    ex.confidence = min(ex.confidence, 0.5)
                    instances.append(ex)
        # Second contrast pass stays CV-only (no more SAM) to keep camo under SLA
        meta["after_camo_fill"] = len(instances)
    else:
        # busy / soft: CV ink residual absorb + limited new uncertain pieces
        bg = estimate_background_color(rgb)
        _dist, _mode = ink_distance(rgb, bg)
        soft = np.clip((_dist - 8.0) / 16.0, 0, 1)
        if print_type == "busy":
            # Prefer NEW whole-motif layers over gluing leftovers into the few SAM seeds.
            # absorb_any_touch=True was collapsing tropical/fox prints into 1–3 blobs.
            target = 0.97
            min_area = max(60, int(0.0006 * h * w))
            filled = _cv_residual_gapfill(
                rgb,
                soft,
                instances,
                bg=bg,
                max_instances=max_instances,
                min_area=min_area,
                max_rounds=4,
                target_coverage=target,
                absorb_dilate=9,
                absorb_any_touch=False,
            )
            filled = _merge_touching_small(filled, min_keep_area=int(0.0008 * h * w))
        else:
            target = 0.985
            filled = _cv_residual_gapfill(
                rgb,
                soft,
                instances,
                bg=bg,
                max_instances=max_instances,
                min_area=max(8, int(0.00008 * h * w)),
                max_rounds=3,
                target_coverage=target,
            )
        # New residual instances → slightly lower confidence (uncertain tier)
        primary_ids = {inst.id for inst in instances}
        for inst in filled:
            if inst.id not in primary_ids and getattr(inst, "pass_index", 0) >= 0:
                if inst.confidence >= 0.6 and int((inst.mask > 40).sum()) < 0.01 * h * w:
                    inst.confidence = min(inst.confidence, 0.48)
        instances = filled
        meta["after_cv_fill"] = len(instances)
        meta["target_coverage"] = target

    meta["added"] = len(instances) - before
    print(f"  hybrid residual fill: {before} → {len(instances)} (+{meta['added']})")
    return instances, meta


def soft_feather_mask(mask: np.ndarray, *, feather: float = 3.5) -> np.ndarray:
    """Soften edges inside the mask only — never expand outside (keeps precision)."""
    binary = (mask > 40).astype(np.uint8)
    if not binary.any():
        return mask
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # Full opacity in the interior; ramp only near the boundary (still inside)
    alpha = np.ones(mask.shape, dtype=np.float32)
    near = dist < feather
    alpha[near] = np.clip(dist[near] / max(feather, 1e-3), 0.35, 1.0)
    alpha *= binary.astype(np.float32)
    # Keep at least original mask strength on solid pixels
    orig = mask.astype(np.float32) / 255.0
    out = np.maximum(alpha, orig) * binary.astype(np.float32)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def discover_hard_motifs(
    image: Image.Image,
    print_type: str,
    *,
    api_key: str | None,
    model: str = "gemini-2.5-flash",
    max_instances: int = 120,
    sam_model: str = "sam2_b.pt",
) -> tuple[list[MotifInstance], dict]:
    """Full hard path → MotifInstance list + meta."""
    assert print_type in ("soft", "camo", "busy")
    meta: dict = {"print_type": print_type, "method": "vlm_sam_hybrid"}

    # Soft-on-print: CV components + feathered alpha (keeps coverage, soft edges)
    if print_type == "soft":
        from segment import discover_by_components

        print("  soft path: CV components + edge feather")
        instances = discover_by_components(
            image, max_instances=min(max_instances, 200), residual_gapfill=True
        )
        for inst in instances:
            inst.mask = soft_feather_mask(inst.mask, feather=4.0)
            inst.confidence = max(float(inst.confidence), 0.6)
        instances, fill_meta = _hybrid_residual_fill(
            image,
            instances,
            "soft",
            max_instances=min(max_instances, 220),
            sam_model_name=sam_model,
        )
        meta["method"] = "cv_soft_feather_hybrid"
        meta["fill"] = fill_meta
        for i, inst in enumerate(instances, start=1):
            inst.id = f"m{i:03d}"
        meta["instances"] = len(instances)
        print(f"  hard path (soft): {len(instances)} feathered instances")
        return instances, meta

    # Camo: CV primary (coverage) + contrast SAM densify (missed regions)
    if print_type == "camo":
        from segment import discover_by_components

        print("  camo path: CV primary + contrast densify")
        instances = discover_by_components(
            image, max_instances=min(max_instances, 120), residual_gapfill=True
        )
        for inst in instances:
            inst.confidence = max(float(inst.confidence), 0.58)
            # Keep CV masks as-is for camo (feather was dropping isolate coverage)
        # Densify with contrast seeds → SAM (adds regions ink_map misses)
        instances, fill_meta = _hybrid_residual_fill(
            image,
            instances,
            "camo",
            max_instances=min(max_instances, 200),
            sam_model_name=sam_model,
        )
        meta["method"] = "cv_camo_contrast_hybrid"
        meta["fill"] = fill_meta
        for i, inst in enumerate(instances, start=1):
            inst.id = f"m{i:03d}"
        meta["instances"] = len(instances)
        print(f"  hard path (camo): {len(instances)} instances (CV+contrast)")
        return instances, meta

    # Busy: VLM+SAM primary motifs + CV residual fill
    # More SAM seeds = separable overlapping leaves; 512px batch keeps CPU under ~90s.
    try:
        busy_box_cap = max(12, int(os.environ.get("HARD_BUSY_BOX_CAP", "28")))
    except ValueError:
        busy_box_cap = 28
    primary_cap = min(max_instances, busy_box_cap)

    t_hard = time.time()
    boxes = _discover_boxes_typed(
        image,
        print_type,
        api_key=api_key,
        model=model,
        max_instances=primary_cap,
    )
    # Sparse / failed VLM → SLIC seeds (better than 1 giant residual blob)
    if len(boxes) < 6:
        from discover import _dedupe as _dedupe_boxes

        print(f"  hard path: only {len(boxes)} VLM boxes — merging SLIC fallback")
        slic = _slic_fallback_boxes(image, print_type, primary_cap)
        boxes = _dedupe_boxes(boxes + slic, iou_thresh=0.4)[:primary_cap]
        meta["method"] = "vlm_slic_sam_hybrid"
    if not boxes:
        print("  hard path: no VLM boxes — SLIC fallback")
        boxes = _slic_fallback_boxes(image, print_type, primary_cap)
        meta["method"] = "slic_sam_hybrid"

    # Prefer larger motifs when over cap
    boxes = sorted(boxes, key=lambda b: -b.area())[:primary_cap]
    for i, b in enumerate(boxes, start=1):
        b.id = f"m{i:03d}"

    meta["boxes"] = len(boxes)
    print(f"  hard discover done in {time.time() - t_hard:.1f}s → {len(boxes)} boxes")
    t_sam = time.time()
    instances = boxes_to_instances_sam(
        image, boxes, print_type=print_type, sam_model_name=sam_model
    )
    meta["sam_seconds"] = round(time.time() - t_sam, 2)
    for inst in instances:
        inst.confidence = max(float(inst.confidence), 0.62)

    before = len(instances)
    instances = _merge_overlapping(instances, iou_thr=0.45)
    meta["merged_from"] = before
    meta["merged_to"] = len(instances)

    # Hybrid residual fill — the coverage win
    instances, fill_meta = _hybrid_residual_fill(
        image,
        instances,
        print_type,
        max_instances=max_instances,
        sam_model_name=sam_model,
    )
    meta["fill"] = fill_meta
    meta["hard_seconds"] = round(time.time() - t_hard, 2)

    for i, inst in enumerate(instances, start=1):
        inst.id = f"m{i:03d}"
    meta["instances"] = len(instances)
    print(
        f"  hard path ({print_type}): {len(boxes)} boxes → {len(instances)} instances "
        f"(hybrid, {meta['hard_seconds']}s)"
    )
    return instances, meta

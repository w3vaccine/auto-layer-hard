"""Hard-print extraction: VLM seeds → SAM2 → soft alpha.

Used when print_type is soft | camo | busy. Clean prints stay on CV.
"""

from __future__ import annotations

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
Find each COMPLETE primary motif — one full leaf, one full flower, one full paisley.

Hard rules:
- WHOLE motif: box must cover the entire leaf/flower from tip to stem base (or petal edge to center). Never box only a tip, vein fragment, hole, or edge nick.
- Prefer fewer complete motifs over many tiny fragments.
- Separate overlapping leaves into distinct boxes (each full leaf gets its own box).
- Skip ultra-tiny noise (< ~0.2% of image) and continuous background.
- Do not return one giant full-image box.

Return ONLY JSON:
{"motifs":[{"id":"m01","label":"leaf|flower|paisley|motif","bbox_norm":[x,y,w,h],"confidence":0.0_to_1.0}]}
bbox_norm = [x,y,w,h] normalized 0..1."""


def _prompt_for(print_type: str) -> str:
    if print_type == "camo":
        return CAMO_PROMPT
    if print_type == "soft":
        return SOFT_PROMPT
    if print_type == "busy":
        return BUSY_PROMPT
    return SOFT_PROMPT


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
    boxes = _dedupe(boxes, iou_thresh=0.4)
    print(f"  hard discover pass1 ({print_type}): {len(boxes)}")

    max_passes = 4 if print_type == "camo" else (3 if print_type == "soft" else 4)
    for pass_index in range(2, max_passes + 1):
        if len(boxes) >= max_instances:
            break
        # Busy: stop early once we have a decent whole-motif set (avoid over-asking)
        if print_type == "busy" and len(boxes) >= 40:
            break
        overlay = ov(rgb, boxes)
        payload = _call_gemini(client, model, PASS_N_PROMPT, [rgb, overlay])
        new_boxes = _parse_motifs(payload, pass_index, id_offset=len(boxes))
        new_boxes = _filter_min_area(new_boxes, min_area=min_area * (0.7 if print_type == "camo" else 1.0))
        before = len(boxes)
        boxes = _dedupe(boxes + new_boxes, iou_thresh=0.35 if print_type == "camo" else 0.4)[
            :max_instances
        ]
        gained = len(boxes) - before
        print(f"  hard discover pass {pass_index}: +{gained} (total {len(boxes)})")
        if gained == 0:
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


def sam_mask_from_box(
    rgb: np.ndarray,
    bbox_norm: list[float],
    sam_model,
) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    box = _bbox_xyxy(bbox_norm, w, h, pad=0.03)
    cx = int((box[0] + box[2]) / 2)
    cy = int((box[1] + box[3]) / 2)
    try:
        results = sam_model.predict(
            rgb,
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


def boxes_to_instances_sam(
    image: Image.Image,
    boxes: list[MotifBox],
    *,
    print_type: str,
    sam_model_name: str = "sam2_b.pt",
) -> list[MotifInstance]:
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    mode = "camo" if print_type == "camo" else ("busy" if print_type == "busy" else "soft")
    sam = _get_sam(sam_model_name) if sam_available() else None
    if sam is None:
        print("  SAM unavailable — using color matte fallback inside boxes")

    instances: list[MotifInstance] = []
    for b in boxes:
        box = _bbox_xyxy(b.bbox_norm, w, h, pad=0.03)
        if sam is not None:
            m = sam_mask_from_box(rgb, b.bbox_norm, sam)
            if m is None or m.sum() < 30:
                alpha = _color_matte_in_box(rgb, b.bbox_norm, soft=(mode == "soft"))
            else:
                # IMPORTANT: no CV-ink intersection for camo/soft
                alpha = soft_alpha_from_sam(rgb, m, box, mode=mode)
        else:
            alpha = _color_matte_in_box(rgb, b.bbox_norm, soft=(mode == "soft"))

        if int((alpha > 20).sum()) < 20:
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
    return instances


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
        # Extra contrast seeds on uncovered areas → SAM
        union = _union_alpha(instances, h, w)
        extra_boxes = _camo_contrast_boxes(rgb, union, max_new=min(60, max_instances - before))
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
        # Second contrast pass (lighter)
        union = _union_alpha(instances, h, w)
        extra2 = _camo_contrast_boxes(rgb, union, max_new=30)
        if extra2:
            extra_i = boxes_to_instances_sam(
                image, extra2, print_type="camo", sam_model_name=sam_model_name
            )
            for ex in extra_i:
                ea = ex.mask > 40
                absorbed = False
                for inst in instances:
                    if int((ea & (inst.mask > 40)).sum()) > 0.3 * max(1, int(ea.sum())):
                        inst.mask = np.maximum(inst.mask, ex.mask)
                        absorbed = True
                        break
                if not absorbed and len(instances) < max_instances:
                    ex.confidence = 0.45
                    instances.append(ex)
        meta["after_camo_fill"] = len(instances)
    else:
        # busy / soft: CV ink residual absorb + limited new uncertain pieces
        bg = estimate_background_color(rgb)
        _dist, _mode = ink_distance(rgb, bg)
        soft = np.clip((_dist - 8.0) / 16.0, 0, 1)
        if print_type == "busy":
            # Whole-motif priority: absorb aggressively; only large leftovers become layers
            target = 0.90
            min_area = max(80, int(0.0012 * h * w))
            filled = _cv_residual_gapfill(
                rgb,
                soft,
                instances,
                bg=bg,
                max_instances=max_instances,
                min_area=min_area,
                max_rounds=3,
                target_coverage=target,
                absorb_dilate=21,
                absorb_any_touch=True,
            )
            filled = _merge_touching_small(filled, min_keep_area=int(0.002 * h * w))
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
    primary_cap = min(max_instances, 90)

    boxes = _discover_boxes_typed(
        image,
        print_type,
        api_key=api_key,
        model=model,
        max_instances=primary_cap,
    )
    if not boxes:
        print("  hard path: no VLM boxes — SLIC fallback")
        boxes = _slic_fallback_boxes(image, print_type, primary_cap)
        meta["method"] = "slic_sam_hybrid"

    meta["boxes"] = len(boxes)
    instances = boxes_to_instances_sam(
        image, boxes, print_type=print_type, sam_model_name=sam_model
    )
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

    for i, inst in enumerate(instances, start=1):
        inst.id = f"m{i:03d}"
    meta["instances"] = len(instances)
    print(f"  hard path ({print_type}): {len(boxes)} boxes → {len(instances)} instances (hybrid)")
    return instances, meta

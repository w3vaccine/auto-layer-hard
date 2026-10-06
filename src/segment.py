"""CV-first motif segmentation for flat textile prints.

Primary path (Canva-like): estimate ground → ink mask → connected components →
tight instance masks. Gemini is optional for labeling / gap fill only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _discover():
    """Lazy import — keep CV path free of google.genai import cost/hangs."""
    from discover import MotifBox, _call_gemini, _dedupe, _filter_min_area, _load_client, _parse_motifs

    return MotifBox, _call_gemini, _dedupe, _filter_min_area, _load_client, _parse_motifs


@dataclass
class MotifInstance:
    id: str
    label: str
    bbox_norm: list[float]
    confidence: float
    mask: np.ndarray  # full-image uint8 alpha 0..255
    pass_index: int = 0


def lab_distance(rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(
        np.float32
    )
    diff = lab - bg_lab
    return np.sqrt(np.sum(diff * diff, axis=2))


def chroma_distance(rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """LAB distance with luminance down-weighted — ignores achromatic patterned grounds."""
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(
        np.float32
    )
    d = lab - bg_lab
    return np.sqrt((0.25 * d[:, :, 0]) ** 2 + d[:, :, 1] ** 2 + d[:, :, 2] ** 2)


def ink_distance(rgb: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, str]:
    """Choose LAB vs chroma distance for ink separation."""
    lab = lab_distance(rgb, bg)
    chr_d = chroma_distance(rgb, bg)
    lab_frac = float((lab > 14).mean())
    chr_frac = float((chr_d > 8).mean())
    # Patterned achromatic grounds: chroma isolates motifs better.
    if lab_frac > 0.42 and chr_frac < lab_frac * 0.55:
        return chr_d, "chroma"
    return lab, "lab"


def ink_map(rgb: np.ndarray, bg: np.ndarray | None = None, *, thresh: float | None = None) -> np.ndarray:
    """Boolean ink map using the same distance mode as build_ink_mask."""
    if bg is None:
        bg = estimate_background_color(rgb)
    dist, mode = ink_distance(rgb, bg)
    t = 8.0 if mode == "chroma" else 14.0
    if thresh is not None:
        t = thresh
    return dist > t


def estimate_background_color(rgb: np.ndarray, border: int = 10) -> np.ndarray:
    """Pick ground color among k-means centers + border median by ink-selectivity."""
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    k = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    # Deterministic centers — critical for stable ink_map / coverage across pipeline stages
    cv2.setRNGSeed(42)
    _c, labels, centers = cv2.kmeans(pixels, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)

    bb = max(2, min(border, h // 10, w // 10))
    strips = [rgb[:bb], rgb[-bb:], rgb[:, :bb], rgb[:, -bb:]]
    samples = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)
    border_med = np.median(samples, axis=0).astype(np.float32)

    candidates = [centers[i].astype(np.float32) for i in np.argsort(-counts)[:3]]
    candidates.append(border_med)

    def ink_frac(bg: np.ndarray) -> float:
        return float((lab_distance(rgb, bg) > 14).mean())

    def score(f: float) -> float:
        if f < 0.04 or f > 0.85:
            return -10.0
        return 1.0 - abs(f - 0.40)

    return max(candidates, key=lambda c: score(ink_frac(c)))


def build_ink_mask(
    rgb: np.ndarray,
    bg: np.ndarray | None = None,
    *,
    soft_lo: float = 8.0,
    soft_hi: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary ink mask uint8 0/255, soft alpha float 0..1)."""
    if bg is None:
        bg = estimate_background_color(rgb)
    dist, mode = ink_distance(rgb, bg)
    h, w = dist.shape
    bb = max(2, min(10, h // 12, w // 12))
    border = np.concatenate(
        [dist[:bb].ravel(), dist[-bb:].ravel(), dist[:, :bb].ravel(), dist[:, -bb:].ravel()]
    )
    noise_floor = float(np.median(border))

    if mode == "chroma":
        soft_lo, soft_hi = 4.0, 14.0
    else:
        # Adaptive, but never below ground noise floor (kills low-contrast false ink).
        sample = dist[::4, ::4].ravel()
        p55, p80 = np.percentile(sample, [55, 80])
        soft_lo = float(np.clip(max(soft_lo, noise_floor + 3.0, p55 * 0.6), 5.5, 16.0))
        soft_hi = float(np.clip(max(soft_hi * 0.9, soft_lo + 8.0, p80 * 0.7), soft_lo + 8.0, 36.0))

    soft = np.clip((dist - soft_lo) / max(1e-6, soft_hi - soft_lo), 0.0, 1.0)
    # Tight binary — residual gapfill recovers misses; fringe OR destroyed precision.
    binary = (soft > 0.30).astype(np.uint8) * 255

    # Light close only — heavy open/close erases hairlines and merges touching motifs.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=1)
    return binary, soft


def _mask_from_comp(comp: np.ndarray, soft: np.ndarray) -> np.ndarray:
    # Soft-weighted alpha — do NOT force solid opacity on whole CC (kills precision).
    alpha = (soft * 255.0 * comp).astype(np.float32)
    core = (comp > 0) & (soft > 0.45)
    alpha[core] = np.maximum(alpha[core], 230.0)
    mid = (comp > 0) & (soft > 0.28) & (soft <= 0.45)
    alpha[mid] = np.maximum(alpha[mid], 180.0)
    alpha[(comp > 0) & (soft < 0.14)] = 0
    alpha_f = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.55)
    alpha_u8 = np.clip(alpha_f, 0, 255).astype(np.uint8)
    alpha_u8[comp == 0] = 0
    return alpha_u8


def _alphas_from_label_map(
    label_map: np.ndarray,
    soft: np.ndarray,
    *,
    min_area: int,
    id_offset: int = 2,
) -> list[np.ndarray]:
    parts: list[np.ndarray] = []
    for mid in range(id_offset, int(label_map.max()) + 1):
        part = (label_map == mid).astype(np.uint8)
        if int(part.sum()) < min_area:
            continue
        parts.append(_mask_from_comp(part, soft))
    return parts


def _component_instances(
    binary: np.ndarray,
    soft: np.ndarray,
    rgb: np.ndarray,
    *,
    min_area_frac: float = 0.00012,
    max_area_frac: float = 0.92,
) -> list[tuple[np.ndarray, list[float], float]]:
    """Return list of (mask_uint8, bbox_norm, confidence)."""
    h, w = binary.shape
    img_area = float(h * w)
    min_a = max(6, int(min_area_frac * img_area))
    max_a = int(max_area_frac * img_area)
    split_a = int(0.02 * img_area)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    out: list[tuple[np.ndarray, list[float], float]] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_a or area > max_a:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw < 2 or bh < 2:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 22 and area < min_a * 10:
            continue

        comp = (labels == i).astype(np.uint8)
        masks = [_mask_from_comp(comp, soft)]
        complexity = _color_complexity(rgb, masks[0])
        should_split = (
            area >= split_a
            or (aspect >= 2.5 and area >= min_a * 3)
            or (area >= int(0.05 * img_area) and complexity >= 2.0)
        )
        if should_split:
            masks = _watershed_split(rgb, masks[0], soft, min_area=min_a)
            if len(masks) == 1:
                # Elongated chains OR compact multi-color overlaps (flowers/leaves glued).
                if aspect >= 2.8 or complexity >= 2.2 or area >= int(0.08 * img_area):
                    masks = _color_split(rgb, masks[0], soft, min_area=min_a)

        for alpha in masks:
            ys, xs = np.where(alpha > 20)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            bbw, bbh = x1 - x0, y1 - y0
            bbox = [x0 / w, y0 / h, bbw / w, bbh / h]
            fill = float((alpha[y0:y1, x0:x1] > 20).mean())
            conf = float(np.clip(0.45 + 0.55 * fill, 0.0, 1.0))
            out.append((alpha, bbox, conf))

    out.sort(key=lambda t: -int((t[0] > 0).sum()))
    return out


def _watershed_split(
    rgb: np.ndarray,
    mask: np.ndarray,
    soft: np.ndarray,
    *,
    min_area: int,
) -> list[np.ndarray]:
    """Split a large blob into sub-motifs when distance peaks exist."""
    binary = (mask > 40).astype(np.uint8)
    if binary.sum() < min_area * 2:
        return [mask]

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if float(dist.max()) < 1.2:
        return [mask]

    thr = max(1.2, float(dist.max()) * 0.28)
    sure_fg = (dist > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_fg = cv2.erode(sure_fg, k, iterations=1)
    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 2:
        thr2 = max(1.0, float(dist.max()) * 0.18)
        sure_fg = cv2.erode((dist > thr2).astype(np.uint8), k, iterations=1)
        n_markers, markers = cv2.connectedComponents(sure_fg)
        if n_markers <= 2:
            return [mask]

    markers = markers + 1
    unknown = binary.copy()
    unknown[sure_fg > 0] = 0
    markers[unknown == 0] = 0
    ws = cv2.watershed(rgb.copy(), markers.astype(np.int32))
    parts = _alphas_from_label_map(ws, soft, min_area=min_area, id_offset=2)
    return parts if len(parts) >= 2 else [mask]


def _color_complexity(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Rough color diversity inside a blob (higher → likely merged multi-motif)."""
    ys, xs = np.where(mask > 40)
    if len(xs) < 40:
        return 0.0
    # Subsample
    step = max(1, len(xs) // 800)
    pix = rgb[ys[::step], xs[::step]].astype(np.float32)
    if len(pix) < 20:
        return 0.0
    lab = cv2.cvtColor(pix.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return float(lab.std(axis=0).mean() / 12.0)


def _color_split(
    rgb: np.ndarray,
    mask: np.ndarray,
    soft: np.ndarray,
    *,
    min_area: int,
) -> list[np.ndarray]:
    """Split merged multi-color blobs (chains + overlapping florals)."""
    binary = (mask > 40).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) < min_area * 2:
        return [mask]

    pixels = rgb[ys, xs].astype(np.float32)
    # Cap k so we don't shatter soft painterly motifs
    k = int(np.clip(round(len(xs) / max(min_area * 10, 1)), 2, 16))
    complexity = _color_complexity(rgb, mask)
    if complexity < 2.0:
        k = min(k, 4)
    elif complexity >= 3.0:
        k = max(k, 8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _c, labels, _centers = cv2.kmeans(pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)

    label_map = np.zeros(binary.shape, dtype=np.int32)
    label_map[ys, xs] = labels.flatten() + 1

    parts: list[np.ndarray] = []
    for lab in range(1, k + 1):
        region = (label_map == lab).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(region, 8)
        for j in range(1, n):
            if int(st[j, cv2.CC_STAT_AREA]) < min_area:
                continue
            part = (cc == j).astype(np.uint8)
            parts.append(_mask_from_comp(part, soft))
    return parts if len(parts) >= 2 else [mask]


def _union_masks(instances: list[MotifInstance], h: int, w: int) -> np.ndarray:
    union = np.zeros((h, w), dtype=np.uint8)
    for inst in instances:
        m = inst.mask
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            inst.mask = m
        union = np.maximum(union, m)
    return union


def _cv_residual_gapfill(
    rgb: np.ndarray,
    soft: np.ndarray,
    instances: list[MotifInstance],
    *,
    bg: np.ndarray,
    max_instances: int,
    min_area: int,
    max_rounds: int = 4,
    target_coverage: float = 0.992,
    absorb_dilate: int = 7,
    absorb_any_touch: bool = False,
) -> list[MotifInstance]:
    """Add missed ink CCs / absorb fringe into neighbors until coverage plateaus."""
    h, w = rgb.shape[:2]
    dil_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, absorb_dilate), max(3, absorb_dilate))
    )
    for round_i in range(1, max_rounds + 1):
        if len(instances) >= max_instances:
            break
        union = _union_masks(instances, h, w)
        dist, mode = ink_distance(rgb, bg)
        ink = ink_map(rgb, bg)
        ink_n = int(ink.sum()) or 1
        covered = union > 20
        cov = float((covered & ink).sum() / ink_n)
        # If stuck under coverage, widen residual ink slightly (soft painterly edges).
        if cov < 0.92 and round_i >= 2:
            widen = 6.0 if mode == "chroma" else 11.0
            ink_wide = dist > widen
            residual = (ink_wide & ~covered).astype(np.uint8) * 255
        else:
            residual = (ink & ~covered).astype(np.uint8) * 255
        # Keep soft fringe but drop single-pixel noise; larger close merges leaf nicks
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, k, iterations=1)
        close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7) if absorb_any_touch else (3, 3)
        )
        residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, close_k, iterations=1)
        res_n = int((residual > 0).sum())
        print(f"  cv residual round {round_i}: cov={cov:.1%} residual_px={res_n}")
        if cov >= target_coverage or res_n < 120:
            break

        touch = cv2.dilate((union > 20).astype(np.uint8), dil_k, 1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats((residual > 0).astype(np.uint8), 8)
        if n <= 1:
            break
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])
        before = len(instances)
        absorbed = 0
        added = 0
        for idx in order:
            i = int(idx) + 1
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < max(4, min_area // 2):
                continue
            comp = (labels == i).astype(np.uint8)
            # Absorb fringe (or any touching blob when absorb_any_touch) into neighbors
            can_absorb = absorb_any_touch or area < min_area * 3
            if can_absorb and (comp & touch).any() and instances:
                best = None
                best_ov = 0
                for inst in instances:
                    ov = int(
                        (
                            comp
                            & cv2.dilate((inst.mask > 20).astype(np.uint8), dil_k, 1)
                        ).sum()
                    )
                    if ov > best_ov:
                        best_ov = ov
                        best = inst
                ov_need = 1 if absorb_any_touch else max(3, area // 4)
                if best is not None and best_ov > ov_need:
                    add = (soft * 255 * comp).astype(np.uint8)
                    add = np.maximum(add, (comp * 230).astype(np.uint8))
                    best.mask = np.maximum(best.mask, add)
                    ys, xs = np.where(best.mask > 20)
                    if len(xs):
                        x0, x1 = int(xs.min()), int(xs.max()) + 1
                        y0, y1 = int(ys.min()), int(ys.max()) + 1
                        best.bbox_norm = [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]
                    absorbed += 1
                    continue

            if len(instances) >= max_instances:
                break
            # Only spawn new layers for sizable leftovers (busy: whole motifs)
            if area < min_area:
                continue
            # Strict alpha: require soft support so residual doesn't paint ground.
            alpha = (soft * 255 * comp).astype(np.uint8)
            alpha = np.maximum(alpha, ((comp > 0) & (soft > 0.25)).astype(np.uint8) * 220)
            alpha[soft < 0.12] = 0

            ys, xs = np.where(alpha > 20)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            instances.append(
                MotifInstance(
                    id=f"r{round_i}{added+1:03d}",
                    label="motif",
                    bbox_norm=[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
                    confidence=0.72,
                    mask=alpha,
                    pass_index=20 + round_i,
                )
            )
            added += 1
        print(f"  cv residual: +{added} new, absorbed {absorbed}")
        if added == 0 and absorbed == 0:
            break
        if len(instances) == before and absorbed == 0:
            break

    for i, inst in enumerate(instances, start=1):
        inst.id = f"m{i:03d}"
    return instances


def discover_by_components(
    image: Image.Image,
    *,
    max_instances: int = 900,
    min_area_frac: float = 0.0001,
    split_large: bool = True,
    residual_gapfill: bool = True,
) -> list[MotifInstance]:
    rgb = np.array(image.convert("RGB"))
    bg = estimate_background_color(rgb)
    binary, soft = build_ink_mask(rgb, bg)
    h, w = binary.shape
    _ = split_large  # splitting handled inside _component_instances
    img_area = float(h * w)
    min_a = max(6, int(min_area_frac * img_area))

    raw = _component_instances(binary, soft, rgb, min_area_frac=min_area_frac)
    instances: list[MotifInstance] = []

    for alpha, bbox, conf in raw:
        ys, xs = np.where(alpha > 20)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bw, bh = x1 - x0, y1 - y0
        bbox_n = [x0 / w, y0 / h, bw / w, bh / h]
        instances.append(
            MotifInstance(
                id=f"m{len(instances)+1:03d}",
                label="motif",
                bbox_norm=bbox_n,
                confidence=conf,
                mask=alpha,
                pass_index=0,
            )
        )
        if len(instances) >= max_instances:
            break

    print(f"  cv components: {len(instances)} instances")
    if residual_gapfill and len(instances) < max_instances:
        instances = _cv_residual_gapfill(
            rgb,
            soft,
            instances,
            bg=bg,
            max_instances=max_instances,
            min_area=min_a,
        )
        print(f"  cv after residual: {len(instances)} instances")
    return instances


TILE_PROMPT = """This is a CROP of a textile print (not the full image).
Find EVERY distinct motif instance visible in THIS crop only.
Tight bbox_norm relative to THIS crop (x,y,w,h in 0..1).
Separate every repeat. Skip background.
Return ONLY JSON: {"motifs":[{"id":"t1","label":"short","bbox_norm":[x,y,w,h],"confidence":0.8}]}
If none: {"motifs":[]}"""


def discover_tile_gapfill(
    image: Image.Image,
    existing: list[MotifInstance],
    *,
    api_key: str | None,
    model: str = "gemini-2.5-flash",
    grid: int = 3,
    max_new: int = 40,
) -> list[MotifInstance]:
    """Ask Gemini on tiles for motifs missed by CV (textured / low-contrast)."""
    if not api_key or max_new <= 0:
        return []

    MotifBox, _call_gemini, _dedupe, _filter_min_area, _load_client, _parse_motifs = _discover()
    _ = MotifBox
    client = _load_client(api_key)
    rgb = image.convert("RGB")
    w, h = rgb.size
    covered = np.zeros((h, w), dtype=np.uint8)
    for inst in existing:
        covered = np.maximum(covered, (inst.mask > 20).astype(np.uint8))

    new_boxes = []
    tw, th = w // grid, h // grid
    for gy in range(grid):
        for gx in range(grid):
            x0, y0 = gx * tw, gy * th
            x1 = w if gx == grid - 1 else (gx + 1) * tw
            y1 = h if gy == grid - 1 else (gy + 1) * th
            tile = rgb.crop((x0, y0, x1, y1))
            # Skip tiles that are already well covered
            region = covered[y0:y1, x0:x1]
            if region.size and region.mean() > 0.55:
                continue
            try:
                payload = _call_gemini(client, model, TILE_PROMPT, [tile])
            except Exception as exc:  # noqa: BLE001
                print(f"  tile {gx},{gy} failed: {exc}")
                continue
            local = _parse_motifs(payload, pass_index=10 + gy * grid + gx, id_offset=0)
            # Map to full-image coords
            for b in local:
                lx, ly, lw, lh = b.bbox_norm
                tw_px, th_px = x1 - x0, y1 - y0
                fx = (x0 + lx * tw_px) / w
                fy = (y0 + ly * th_px) / h
                fw = (lw * tw_px) / w
                fh = (lh * th_px) / h
                b.bbox_norm = [
                    max(0, min(1, fx)),
                    max(0, min(1, fy)),
                    max(0, min(1 - fx, fw)),
                    max(0, min(1 - fy, fh)),
                ]
                new_boxes.append(b)

    new_boxes = _filter_min_area(_dedupe(new_boxes, iou_thresh=0.4), min_area=0.0003)
    # Drop boxes that heavily overlap existing masks
    added: list[MotifInstance] = []
    bg = estimate_background_color(np.array(rgb))
    soft = lab_distance(np.array(rgb), bg)
    soft_a = np.clip((soft - 10) / 16.0, 0, 1)

    for b in new_boxes:
        x, y, bw, bh = b.bbox_norm
        x0, y0 = int(x * w), int(y * h)
        x1, y1 = int((x + bw) * w), int((y + bh) * h)
        if x1 <= x0 or y1 <= y0:
            continue
        crop_cover = covered[y0:y1, x0:x1]
        if crop_cover.size and crop_cover.mean() > 0.5:
            continue
        # Build mask from ink inside box
        local = np.zeros((h, w), dtype=np.uint8)
        ink = (soft_a[y0:y1, x0:x1] > 0.35).astype(np.uint8)
        if ink.mean() < 0.02:
            continue
        local[y0:y1, x0:x1] = (ink * 255)
        # Keep largest CC in box
        n, lab, st, _ = cv2.connectedComponentsWithStats((local > 0).astype(np.uint8), 8)
        if n <= 1:
            continue
        largest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        mask = ((lab == largest).astype(np.uint8) * 230)
        if mask.sum() < 20:
            continue
        ys, xs = np.where(mask > 0)
        bx0, bx1 = int(xs.min()), int(xs.max()) + 1
        by0, by1 = int(ys.min()), int(ys.max()) + 1
        added.append(
            MotifInstance(
                id=f"g{len(added)+1:03d}",
                label=b.label,
                bbox_norm=[bx0 / w, by0 / h, (bx1 - bx0) / w, (by1 - by0) / h],
                confidence=b.confidence,
                mask=mask,
                pass_index=b.pass_index,
            )
        )
        covered = np.maximum(covered, (mask > 0).astype(np.uint8))
        if len(added) >= max_new:
            break

    print(f"  gemini tile gapfill: +{len(added)}")
    return added


def label_instances(
    image: Image.Image,
    instances: list[MotifInstance],
    *,
    api_key: str | None,
    model: str = "gemini-2.5-flash",
    sample: int = 24,
) -> None:
    """Optional: label a sample of instances via Gemini (mutates labels)."""
    if not api_key or not instances:
        return
    # Cheap heuristic labels from average color if no API budget
    # Full labeling can be slow; only label up to `sample`
    _MotifBox, _call_gemini, _d, _f, _load_client, _p = _discover()
    client = _load_client(api_key)
    rgb = image.convert("RGB")
    subset = instances[:sample]
    # Build a contact sheet of crops
    crops = []
    for inst in subset:
        x, y, bw, bh = inst.bbox_norm
        W, H = rgb.size
        box = (
            max(0, int(x * W) - 2),
            max(0, int(y * H) - 2),
            min(W, int((x + bw) * W) + 2),
            min(H, int((y + bh) * H) + 2),
        )
        crops.append(rgb.crop(box).resize((96, 96), Image.Resampling.LANCZOS))

    sheet_w = min(6, len(crops))
    sheet_h = (len(crops) + sheet_w - 1) // sheet_w
    sheet = Image.new("RGB", (sheet_w * 96, sheet_h * 96), (240, 240, 240))
    for i, c in enumerate(crops):
        sheet.paste(c, ((i % sheet_w) * 96, (i // sheet_w) * 96))

    prompt = (
        f"Contact sheet of {len(crops)} textile motif crops, left-to-right, top-to-bottom, "
        f"{sheet_w} per row. Return JSON only: "
        '{"labels":["short label", ...]} with exactly '
        f"{len(crops)} labels."
    )
    try:
        payload = _call_gemini(client, model, prompt, [sheet])
        labels = payload.get("labels") or []
        for inst, lab in zip(subset, labels):
            if isinstance(lab, str) and lab.strip():
                inst.label = lab.strip()[:40]
    except Exception as exc:  # noqa: BLE001
        print(f"  labeling skipped: {exc}")


def instances_to_boxes(instances: list[MotifInstance]) -> list:
    MotifBox, *_rest = _discover()
    return [
        MotifBox(
            id=i.id,
            label=i.label,
            bbox_norm=i.bbox_norm,
            confidence=i.confidence,
            pass_index=i.pass_index,
        )
        for i in instances
    ]


def overlay_instances(image: Image.Image, instances: list[MotifInstance]) -> Image.Image:
    overlay = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    # Tint masks lightly
    tint = np.array(overlay)
    for inst in instances:
        m = inst.mask > 40
        tint[m, 0] = np.clip(tint[m, 0].astype(np.int16) + 40, 0, 255).astype(np.uint8)
        tint[m, 2] = np.clip(tint[m, 2].astype(np.int16) - 20, 0, 255).astype(np.uint8)
    overlay = Image.fromarray(tint)
    draw = ImageDraw.Draw(overlay)
    for inst in instances:
        x, y, bw, bh = inst.bbox_norm
        draw.rectangle(
            [int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)],
            outline=(255, 40, 40, 220),
            width=max(1, w // 500),
        )
    return overlay.convert("RGB")


def discover_motifs_v2(
    image: Image.Image,
    *,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    max_instances: int = 900,
    use_tile_gapfill: bool = True,
    label: bool = False,
) -> list[MotifInstance]:
    instances = discover_by_components(image, max_instances=max_instances)
    # Always gap-fill when sparse, or force when CV finds almost nothing (busy prints)
    need_tiles = use_tile_gapfill and bool(api_key) and (
        len(instances) < max(12, max_instances // 10) or len(instances) < max_instances
    )
    if need_tiles and len(instances) < max_instances:
        # denser grid when CV failed hard
        grid = 4 if len(instances) < 8 else 3
        extra = discover_tile_gapfill(
            image,
            instances,
            api_key=api_key,
            model=model,
            grid=grid,
            max_new=min(120 if len(instances) < 8 else 40, max_instances - len(instances)),
        )
        instances.extend(extra)
    for i, inst in enumerate(instances, start=1):
        inst.id = f"m{i:03d}"
    if label:
        label_instances(image, instances, api_key=api_key, model=model)
    print(f"  discover v2 total: {len(instances)}")
    return instances

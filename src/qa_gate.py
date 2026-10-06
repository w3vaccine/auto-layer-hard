"""Residual ink QA gate — keep extracting until coverage / residual pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from segment import (
    MotifInstance,
    discover_tile_gapfill,
    estimate_background_color,
    ink_map,
    lab_distance,
)

if TYPE_CHECKING:
    pass


def residual_ink_map(
    rgb: np.ndarray,
    union_alpha: np.ndarray,
    bg: np.ndarray | None = None,
    *,
    ink_thresh: float = 14.0,
) -> np.ndarray:
    """Binary map of ink pixels not yet captured by union mask."""
    if bg is None:
        bg = estimate_background_color(rgb)
    _ = ink_thresh  # kept for API compat; ink_map picks LAB/chroma mode
    ink = ink_map(rgb, bg)
    covered = union_alpha > 20
    residual = ink & ~covered
    # Drop tiny speckles
    residual_u8 = residual.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_OPEN, k, iterations=1)
    return residual_u8


def coverage_and_residual(
    rgb: np.ndarray,
    union_alpha: np.ndarray,
    bg: np.ndarray | None = None,
) -> dict:
    if bg is None:
        bg = estimate_background_color(rgb)
    ink = ink_map(rgb, bg)
    ink_count = int(ink.sum()) or 1
    captured = (union_alpha > 20) & ink
    residual = residual_ink_map(rgb, union_alpha, bg)
    return {
        "ink_coverage": float(captured.sum() / ink_count),
        "residual_frac": float((residual > 0).mean()),
        "residual_ink_frac": float((residual > 0).sum() / ink_count),
        "residual_pixels": int((residual > 0).sum()),
        "bg": bg,
    }


def residual_to_instances(
    residual: np.ndarray,
    soft_dist: np.ndarray,
    existing: list[MotifInstance] | None = None,
    *,
    min_area: int = 12,
    max_new: int = 40,
) -> list[MotifInstance]:
    """Turn residual ink blobs into new MotifInstances.

    Residual that touches an existing mask is absorbed into that mask instead of
    spawning a new filigree layer (critical for paisley-style prints).
    """
    h, w = residual.shape
    soft = np.clip((soft_dist - 8.0) / 16.0, 0, 1)
    existing = existing or []

    # Dilated union of existing for absorb test
    union = np.zeros((h, w), dtype=np.uint8)
    for inst in existing:
        m = inst.mask
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            inst.mask = m
        union = np.maximum(union, (m > 20).astype(np.uint8))
    touch = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((residual > 0).astype(np.uint8), 8)
    out: list[MotifInstance] = []
    absorbed = 0
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) if n > 1 else []
    for idx in order:
        i = int(idx) + 1
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        # Absorb into nearest existing instance if touching
        if existing and (comp & touch).any():
            best = None
            best_ov = 0
            for inst in existing:
                ov = int((comp & (inst.mask > 20).astype(np.uint8)).sum())
                # also count dilated touch
                ov2 = int((comp & cv2.dilate((inst.mask > 20).astype(np.uint8), np.ones((9, 9), np.uint8), 1)).sum())
                score = max(ov, ov2)
                if score > best_ov:
                    best_ov = score
                    best = inst
            if best is not None and best_ov > 0:
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

        alpha = (soft * 255 * comp).astype(np.uint8)
        alpha = np.maximum(alpha, (comp * 230).astype(np.uint8))
        ys, xs = np.where(alpha > 20)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        out.append(
            MotifInstance(
                id=f"r{len(out)+1:03d}",
                label="motif",
                bbox_norm=[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
                confidence=0.75,
                mask=alpha,
                pass_index=50,
            )
        )
        if len(out) >= max_new:
            break
    if absorbed:
        print(f"  QA absorbed {absorbed} residual blobs into existing layers")
    return out


def qa_gapfill_loop(
    image: Image.Image,
    instances: list[MotifInstance],
    *,
    api_key: str | None,
    model: str = "gemini-2.5-flash",
    target_coverage: float = 0.995,
    max_residual_ink_frac: float = 0.005,
    max_rounds: int = 3,
    use_gemini_tiles: bool = True,
) -> tuple[list[MotifInstance], dict]:
    """
    Until coverage/residual pass (or rounds exhausted):
      1) add residual CC instances
      2) optional Gemini tile gapfill on leftover regions
    """
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    bg = estimate_background_color(rgb)
    history = []

    for round_i in range(1, max_rounds + 1):
        union = np.zeros((h, w), dtype=np.uint8)
        for inst in instances:
            m = inst.mask
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            union = np.maximum(union, m)

        stats = coverage_and_residual(rgb, union, bg)
        history.append({"round": round_i, **{k: v for k, v in stats.items() if k != "bg"}})
        print(
            f"  QA round {round_i}: coverage={stats['ink_coverage']:.2%} "
            f"residual_ink={stats['residual_ink_frac']:.2%} "
            f"({stats['residual_pixels']} px)"
        )
        if stats["residual_ink_frac"] <= max_residual_ink_frac and (
            stats["ink_coverage"] >= target_coverage or stats["residual_pixels"] < 150
        ):
            print("  QA gate PASSED")
            return instances, {"passed": True, "history": history, "final": history[-1]}
        # Soft pass: very low residual even if coverage slightly under target
        if stats["residual_ink_frac"] <= max_residual_ink_frac * 0.5 and stats["ink_coverage"] >= 0.98:
            print("  QA gate PASSED (soft)")
            return instances, {"passed": True, "history": history, "final": history[-1]}

        residual = residual_ink_map(rgb, union, bg)
        dist = lab_distance(rgb, bg)
        before = len(instances)
        new_cc = residual_to_instances(residual, dist, existing=instances, max_new=50)
        instances.extend(new_cc)
        print(f"  QA +{len(new_cc)} residual components")

        if use_gemini_tiles and api_key and stats["residual_ink_frac"] > max_residual_ink_frac:
            extra = discover_tile_gapfill(
                image,
                instances,
                api_key=api_key,
                model=model,
                grid=4,
                max_new=30,
            )
            instances.extend(extra)

        # Renumber
        for i, inst in enumerate(instances, start=1):
            inst.id = f"m{i:03d}"

        if len(instances) == before:
            print("  QA: no new instances; stopping")
            break

    union = np.zeros((h, w), dtype=np.uint8)
    for inst in instances:
        union = np.maximum(union, inst.mask)
    stats = coverage_and_residual(rgb, union, bg)
    history.append({"round": "final", **{k: v for k, v in stats.items() if k != "bg"}})
    passed = stats["residual_ink_frac"] <= max_residual_ink_frac and (
        stats["ink_coverage"] >= target_coverage or stats["residual_pixels"] < 150
    )
    print(f"  QA gate {'PASSED' if passed else 'FAILED'} after loop")
    return instances, {"passed": passed, "history": history, "final": history[-1]}

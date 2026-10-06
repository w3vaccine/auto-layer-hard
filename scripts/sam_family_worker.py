#!/usr/bin/env python3
"""SAM-family worker for garment seg bakeoff (runs inside Docker with working torch).

Usage:
  python sam_family_worker.py --method sam2 --image in.png --out out_dir
  methods: sam2 | mobile_sam | fastsam | yolo_seg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _overlay(image: Image.Image, union: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    tint = np.zeros((*union.shape, 4), dtype=np.uint8)
    tint[..., 0] = 255
    tint[..., 1] = 60
    tint[..., 2] = 60
    tint[..., 3] = (np.clip(union.astype(np.float32) * 0.5, 0, 180)).astype(np.uint8)
    return Image.alpha_composite(base, Image.fromarray(tint, "RGBA")).convert("RGB")


def _colored_parts(image: Image.Image, masks: list[np.ndarray]) -> Image.Image:
    base = np.array(image.convert("RGB")).astype(np.float32)
    palette = [
        (255, 70, 70),
        (70, 140, 255),
        (60, 200, 120),
        (255, 200, 60),
        (200, 80, 220),
        (80, 220, 220),
        (255, 140, 60),
        (160, 160, 255),
    ]
    out = base.copy()
    for i, m in enumerate(masks):
        if m.shape[:2] != out.shape[:2]:
            m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = palette[i % len(palette)]
        alpha = (m.astype(np.float32) / 255.0)[..., None] * 0.55
        color = np.array(c, dtype=np.float32).reshape(1, 1, 3)
        out = out * (1 - alpha) + color * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _metrics(union: np.ndarray, n_parts: int) -> dict:
    fg = float((union > 40).mean())
    return {"parts": n_parts, "fg_frac": round(fg, 4), "ok": n_parts > 0 and fg > 0.005}


def run_fastsam(rgb: np.ndarray, out: Path) -> dict:
    from ultralytics import FastSAM

    t0 = time.time()
    model = FastSAM("FastSAM-s.pt")
    h, w = rgb.shape[:2]
    results = model.predict(rgb, verbose=False, retina_masks=True, imgsz=max(h, w))
    union = np.zeros((h, w), dtype=np.uint8)
    masks: list[np.ndarray] = []
    if results and results[0].masks is not None:
        data = results[0].masks.data
        arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        for m in arr[:40]:
            mm = (m > 0.5).astype(np.uint8) * 255
            if mm.shape[:2] != (h, w):
                mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(mm)
            union = np.maximum(union, mm)
    Image.fromarray(union).save(out / "union_mask.png")
    _colored_parts(Image.fromarray(rgb), masks).save(out / "overlay.png")
    meta = _metrics(union, len(masks))
    return {"method": "fastsam", "sec": round(time.time() - t0, 2), **meta, "error": None}


def run_yolo_seg(rgb: np.ndarray, out: Path) -> dict:
    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO("yolo11n-seg.pt")
    h, w = rgb.shape[:2]
    results = model.predict(rgb, verbose=False, retina_masks=True)
    union = np.zeros((h, w), dtype=np.uint8)
    masks: list[np.ndarray] = []
    if results and results[0].masks is not None:
        data = results[0].masks.data
        arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        for m in arr[:40]:
            mm = (m > 0.5).astype(np.uint8) * 255
            if mm.shape[:2] != (h, w):
                mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(mm)
            union = np.maximum(union, mm)
    Image.fromarray(union).save(out / "union_mask.png")
    _colored_parts(Image.fromarray(rgb), masks).save(out / "overlay.png")
    meta = _metrics(union, len(masks))
    return {
        "method": "yolo_seg",
        "sec": round(time.time() - t0, 2),
        **meta,
        "notes": "COCO-class seg — may miss garments",
        "error": None,
    }


def _silhouette_box(rgb: np.ndarray) -> tuple[list[int], list[int]]:
    """BBox + center from non-white silhouette (for SAM prompts)."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Non-near-white
    ink = gray < 245
    # also chroma
    diff = np.abs(rgb.astype(np.int16) - 255).sum(axis=2) > 30
    ink = ink | diff
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)], [w // 2, h // 2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = 4
    box = [max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)]
    point = [int(xs.mean()), int(ys.mean())]
    return box, point


def run_sam_prompted(rgb: np.ndarray, out: Path, *, method: str, weights: str) -> dict:
    from ultralytics import SAM

    t0 = time.time()
    model = SAM(weights)
    h, w = rgb.shape[:2]
    box, point = _silhouette_box(rgb)
    results = model.predict(
        rgb,
        bboxes=[box],
        points=[point],
        labels=[1],
        verbose=False,
    )
    union = np.zeros((h, w), dtype=np.uint8)
    masks: list[np.ndarray] = []
    if results and results[0].masks is not None:
        data = results[0].masks.data
        arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        if arr.ndim == 3:
            for m in arr:
                mm = (m > 0.5).astype(np.uint8) * 255
                if mm.shape[:2] != (h, w):
                    mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                masks.append(mm)
                union = np.maximum(union, mm)
        else:
            mm = (arr > 0.5).astype(np.uint8) * 255
            if mm.shape[:2] != (h, w):
                mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(mm)
            union = mm
    Image.fromarray(union).save(out / "union_mask.png")
    (_colored_parts(Image.fromarray(rgb), masks) if masks else _overlay(Image.fromarray(rgb), union)).save(
        out / "overlay.png"
    )
    meta = _metrics(union, max(1, len(masks)) if union.any() else 0)
    return {"method": method, "sec": round(time.time() - t0, 2), **meta, "weights": weights, "error": None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["sam2", "mobile_sam", "fastsam", "yolo_seg"])
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default="")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    image.save(out / "original.png")
    rgb = np.array(image)

    try:
        if args.method == "fastsam":
            result = run_fastsam(rgb, out)
        elif args.method == "yolo_seg":
            result = run_yolo_seg(rgb, out)
        elif args.method == "sam2":
            weights = args.weights or "sam2_b.pt"
            result = run_sam_prompted(rgb, out, method="sam2", weights=weights)
        else:
            weights = args.weights or "mobile_sam.pt"
            result = run_sam_prompted(rgb, out, method="mobile_sam", weights=weights)
    except Exception as exc:  # noqa: BLE001
        result = {
            "method": args.method,
            "sec": 0,
            "parts": 0,
            "fg_frac": 0.0,
            "ok": False,
            "error": str(exc)[:400],
        }

    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

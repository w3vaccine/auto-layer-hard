"""Multi-pass Gemini motif discovery for flat textile prints."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _genai():
    """Lazy import — google.genai can hang / block at import time on some envs."""
    from google import genai
    from google.genai import types

    return genai, types


@dataclass
class MotifBox:
    id: str
    label: str
    bbox_norm: list[float]  # [x, y, w, h] in 0..1
    confidence: float
    pass_index: int = 0

    def area(self) -> float:
        return max(0.0, self.bbox_norm[2]) * max(0.0, self.bbox_norm[3])

    def iou(self, other: "MotifBox") -> float:
        ax, ay, aw, ah = self.bbox_norm
        bx, by, bw, bh = other.bbox_norm
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


PASS1_PROMPT = """You are analyzing a flat textile/print design image.
Find EVERY distinct motif instance as a separate object (flowers, leaves, shells, paisleys, animals, brush marks, etc.).

Hard rules:
- Every distinct instance is its own entry — do NOT merge repeats into one box.
- Include small filler motifs, edge-cropped motifs, and overlapping motifs.
- Prefer tight boxes around each motif (little empty background inside the box).
- Do not invent motifs that are not visible.
- Background/ground color is NOT a motif.

Return ONLY valid JSON (no markdown) with this shape:
{"motifs":[{"id":"m01","label":"short label","bbox_norm":[x,y,w,h],"confidence":0.0_to_1.0}]}

bbox_norm uses normalized coordinates where x,y are top-left and w,h are size, all in [0,1].
Aim for as complete a coverage of ink motifs as possible."""

PASS_N_PROMPT = """Same textile print. Boxes already found are drawn in RED on the overlay image.
Return ONLY motifs that were MISSED — small, overlapping, edge-cropped, or repeated instances not covered by red boxes.

Hard rules:
- Do NOT re-list motifs already covered by red boxes.
- Every distinct missed instance is its own entry.
- Prefer tight boxes. Background is not a motif.

Return ONLY valid JSON (no markdown):
{"motifs":[{"id":"mXX","label":"short label","bbox_norm":[x,y,w,h],"confidence":0.0_to_1.0}]}

If nothing was missed, return {"motifs":[]}."""


def _load_client(api_key: str | None = None):
    genai, _ = _genai()
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            # Truncated motif lists — salvage complete objects
            motifs = re.findall(
                r'\{\s*"id"\s*:\s*"[^"]*"\s*,\s*"label"\s*:\s*"[^"]*"\s*,'
                r'\s*"bbox_norm"\s*:\s*\[[^\]]+\]\s*,\s*"confidence"\s*:\s*[0-9.]+',
                m.group(0),
            )
            out = []
            for raw in motifs:
                try:
                    out.append(json.loads(raw + "}"))
                except json.JSONDecodeError:
                    continue
            if out:
                return {"motifs": out}
            raise


def _clamp_box(bbox: list[float]) -> list[float] | None:
    if len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def _parse_motifs(payload: dict[str, Any], pass_index: int, id_offset: int) -> list[MotifBox]:
    raw = payload.get("motifs") or []
    out: list[MotifBox] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        bbox = _clamp_box(item.get("bbox_norm") or item.get("bbox") or [])
        if bbox is None:
            continue
        label = str(item.get("label") or "motif").strip() or "motif"
        conf = float(item.get("confidence") or 0.7)
        mid = str(item.get("id") or f"m{id_offset + i + 1:02d}")
        out.append(
            MotifBox(
                id=mid,
                label=label,
                bbox_norm=bbox,
                confidence=conf,
                pass_index=pass_index,
            )
        )
    return out


def _dedupe(boxes: list[MotifBox], iou_thresh: float = 0.45) -> list[MotifBox]:
    kept: list[MotifBox] = []
    for box in sorted(boxes, key=lambda b: (-b.confidence, -b.area())):
        if any(box.iou(k) >= iou_thresh for k in kept):
            continue
        kept.append(box)
    return kept


def _filter_min_area(boxes: list[MotifBox], min_area: float = 0.0005) -> list[MotifBox]:
    return [b for b in boxes if b.area() >= min_area]


def overlay_boxes(image: Image.Image, boxes: list[MotifBox]) -> Image.Image:
    overlay = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    for b in boxes:
        x, y, bw, bh = b.bbox_norm
        x0, y0 = int(x * w), int(y * h)
        x1, y1 = int((x + bw) * w), int((y + bh) * h)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 220), width=max(2, w // 400))
    return overlay.convert("RGB")


def _image_part(image: Image.Image, mime: str = "image/png"):
    from io import BytesIO

    _, types = _genai()
    buf = BytesIO()
    image.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type=mime)


def _call_gemini(
    client: Any,
    model: str,
    prompt: str,
    images: list[Image.Image],
) -> dict[str, Any]:
    _, types = _genai()
    contents: list[Any] = [prompt]
    for img in images:
        contents.append(_image_part(img))
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.15 if attempt == 0 else 0.05,
                    response_mime_type="application/json",
                ),
            )
            text = (resp.text or "").strip()
            if not text:
                return {"motifs": []}
            return _extract_json(text)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  gemini parse/retry {attempt + 1}: {exc}")
    print(f"  gemini failed after retries: {last_err}")
    return {"motifs": []}


def discover_motifs(
    image: Image.Image,
    *,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    max_passes: int = 4,
    max_instances: int = 80,
    min_area: float = 0.0005,
    iou_thresh: float = 0.45,
) -> list[MotifBox]:
    """Multi-pass motif bbox discovery."""
    client = _load_client(api_key)
    rgb = image.convert("RGB")
    all_boxes: list[MotifBox] = []

    for pass_index in range(1, max_passes + 1):
        if len(all_boxes) >= max_instances:
            break
        if pass_index == 1:
            payload = _call_gemini(client, model, PASS1_PROMPT, [rgb])
        else:
            overlay = overlay_boxes(rgb, all_boxes)
            payload = _call_gemini(client, model, PASS_N_PROMPT, [rgb, overlay])

        new_boxes = _parse_motifs(payload, pass_index, id_offset=len(all_boxes))
        new_boxes = _filter_min_area(new_boxes, min_area=min_area)
        before = len(all_boxes)
        all_boxes = _dedupe(all_boxes + new_boxes, iou_thresh=iou_thresh)
        all_boxes = all_boxes[:max_instances]
        gained = len(all_boxes) - before
        print(f"  discover pass {pass_index}: +{gained} (total {len(all_boxes)})")
        if pass_index > 1 and gained == 0:
            break

    # Stable renumber ids
    for i, b in enumerate(all_boxes, start=1):
        b.id = f"m{i:02d}"
    return all_boxes


def boxes_to_dicts(boxes: list[MotifBox]) -> list[dict[str, Any]]:
    return [asdict(b) for b in boxes]


def save_debug_overlay(image: Image.Image, boxes: list[MotifBox], path: Path) -> None:
    overlay_boxes(image, boxes).save(path)

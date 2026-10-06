"""Print-type classifier for Auto Layer routing.

Types:
  clean  — flat ground, distinct motifs → CV path
  soft   — watercolor / bled edges → VLM + SAM + soft matte
  camo   — motif ≈ ground → VLM + SAM (no bg ink_map)
  busy   — dense interlocking → hierarchical VLM + SAM + merge
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from segment import estimate_background_color, ink_map, lab_distance


@dataclass
class PrintTypeResult:
    print_type: str  # clean | soft | camo | busy
    confidence: float
    scores: dict[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "print_type": self.print_type,
            "confidence": self.confidence,
            "scores": self.scores,
            "reason": self.reason,
        }


def _edge_softness(rgb: np.ndarray) -> float:
    """High when ink boundaries are soft (watercolor): strong low-freq gradients, weak Canny."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    soft_grad = cv2.Laplacian(blur, cv2.CV_32F)
    soft_energy = float(np.mean(np.abs(soft_grad)))
    canny = cv2.Canny(gray, 80, 160)
    hard_frac = float((canny > 0).mean())
    # Soft prints: elevated soft_energy relative to hard edges
    if hard_frac < 1e-6:
        return 1.0
    return float(np.clip(soft_energy / (hard_frac * 800.0 + 1e-3), 0, 3) / 3.0)


def _camo_score(rgb: np.ndarray, bg: np.ndarray) -> float:
    """High when no clear ground: ink_frac mid-high and colors interleave."""
    ink = ink_map(rgb, bg)
    ink_frac = float(ink.mean())
    # Cluster entropy — camo has many similarly sized color regions
    small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    pix = small.reshape(-1, 3).astype(np.float32)
    k = 6
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _centers = cv2.kmeans(pix, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float64)
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p + 1e-9)).sum() / np.log(k))  # 0..1
    # Bg selectivity: if many pixels are close to *several* centers, camo-like
    dist = lab_distance(rgb, bg)
    near_bg = float((dist < 18).mean())
    # Camo: ink_frac not tiny, entropy high, near_bg not dominant
    score = 0.0
    if 0.28 <= ink_frac <= 0.78:
        score += 0.35
    score += 0.40 * entropy
    score += 0.25 * (1.0 - abs(near_bg - 0.35) / 0.35) if near_bg < 0.7 else 0.0
    # Penalize clear flat grounds (near_bg high + ink_frac moderate-low)
    if near_bg > 0.55 and ink_frac < 0.35:
        score *= 0.35
    return float(np.clip(score, 0, 1))


def _busy_score(rgb: np.ndarray, bg: np.ndarray) -> float:
    ink = ink_map(rgb, bg).astype(np.uint8) * 255
    n, _lab, st, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    if n <= 1:
        return 0.0
    areas = st[1:, cv2.CC_STAT_AREA].astype(np.float64)
    img_a = float(rgb.shape[0] * rgb.shape[1])
    small = areas[areas < 0.01 * img_a]
    count_score = float(np.clip((n - 1) / 120.0, 0, 1))
    small_frac = float(len(small) / max(1, len(areas)))
    return float(np.clip(0.55 * count_score + 0.45 * small_frac, 0, 1))


def _clean_score(rgb: np.ndarray, bg: np.ndarray) -> float:
    ink = ink_map(rgb, bg)
    ink_frac = float(ink.mean())
    dist = lab_distance(rgb, bg)
    # Clear separation: bimodal-ish distances
    near = float((dist < 12).mean())
    far = float((dist > 22).mean())
    mid = 1.0 - near - far
    sep = near * far * (1.0 - mid)
    if 0.05 <= ink_frac <= 0.45 and near > 0.35:
        return float(np.clip(0.4 + 2.0 * sep, 0, 1))
    return float(np.clip(2.0 * sep, 0, 1))


def classify_print(image: Image.Image) -> PrintTypeResult:
    """Heuristic print-type classifier (no API required)."""
    rgb = np.array(image.convert("RGB"))
    # Mild denoise for stats only
    rgb_s = cv2.bilateralFilter(rgb, d=5, sigmaColor=40, sigmaSpace=5)
    bg = estimate_background_color(rgb_s)

    soft = _edge_softness(rgb)
    camo = _camo_score(rgb_s, bg)
    busy = _busy_score(rgb_s, bg)
    clean = _clean_score(rgb_s, bg)

    scores = {"clean": clean, "soft": soft, "camo": camo, "busy": busy}

    ink_frac = float(ink_map(rgb_s, bg).mean())

    # Decision tree tuned for known failure modes (not pure argmax)
    print_type = "clean"
    conf = clean
    reason_tag = "default_clean"

    if camo >= 0.55 and clean < 0.45 and ink_frac >= 0.35:
        # Camo before soft — BUT not when busy CCs scream "many distinct motifs"
        # (tropical leaves / florals on dark ground get false camo scores)
        if busy >= 0.85 and ink_frac >= 0.45:
            print_type, conf, reason_tag = "busy", busy, "busy_over_false_camo"
        else:
            print_type, conf, reason_tag = "camo", camo, "camo_over_soft"
    elif soft >= 0.55 and clean < 0.75:
        # Soft path only when ground is NOT already clear (else CV wins)
        print_type, conf, reason_tag = "soft", soft, "soft_edges"
        if camo >= 0.72 and clean < 0.35:
            print_type, conf, reason_tag = "camo", camo, "camo_beats_soft"
    elif soft >= 0.55 and clean >= 0.75:
        # Watercolor-on-clear-ground: keep CV (measured better ink F1)
        print_type, conf, reason_tag = "clean", clean, "soft_but_clear_ground_use_cv"
    elif busy >= 0.75 and clean < 0.75 and ink_frac >= 0.2:
        # Busy only when clean is not clearly winning
        print_type, conf, reason_tag = "busy", busy, "dense_ccs"
    elif clean >= 0.55:
        print_type, conf, reason_tag = "clean", clean, "clear_ground"
    else:
        # Fallback argmax
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        print_type, conf = ordered[0]
        reason_tag = "argmax"

    # Seashells-like: many CCs but clear ground → clean
    if busy >= 0.8 and clean >= 0.55 and ink_frac <= 0.35 and camo < 0.4:
        print_type, conf, reason_tag = "clean", clean, "clear_ground_over_busy"

    second = sorted(scores.values(), reverse=True)[1]
    confidence = float(np.clip(conf - 0.35 * second + 0.25, 0, 1))
    reason = (
        f"{reason_tag} | ink_frac={ink_frac:.2f} soft={soft:.2f} camo={camo:.2f} "
        f"busy={busy:.2f} clean={clean:.2f}"
    )
    return PrintTypeResult(print_type=print_type, confidence=confidence, scores=scores, reason=reason)


def classify_print_vlm(
    image: Image.Image,
    *,
    api_key: str | None,
    model: str = "gemini-2.5-flash",
) -> PrintTypeResult | None:
    """Optional Gemini confirmation. Returns None if unavailable."""
    if not api_key:
        return None
    try:
        from discover import _call_gemini, _load_client
    except Exception:
        return None

    prompt = """Classify this flat textile/print image into ONE type:
- clean: clear flat ground, distinct separate motifs
- soft: watercolor, bled, fuzzy, soft edges
- camo: camouflage / motif colors blend into ground / low figure-ground
- busy: dense interlocking motifs, paisley-like, lots of touching pieces

Return ONLY JSON: {"print_type":"clean|soft|camo|busy","confidence":0.0_to_1.0,"reason":"short"}"""
    try:
        client = _load_client(api_key)
        payload = _call_gemini(client, model, prompt, [image.convert("RGB")])
        pt = str(payload.get("print_type") or "").strip().lower()
        if pt not in ("clean", "soft", "camo", "busy"):
            return None
        conf = float(payload.get("confidence") or 0.7)
        return PrintTypeResult(
            print_type=pt,
            confidence=conf,
            scores={pt: conf},
            reason=str(payload.get("reason") or "vlm"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  print-type VLM failed: {exc}")
        return None


def classify_print_routed(
    image: Image.Image,
    *,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    use_vlm: bool = True,
) -> PrintTypeResult:
    """Heuristic first; VLM can override when confident and disagrees on hard types."""
    base = classify_print(image)
    if not use_vlm or not api_key:
        return base
    # Skip the VLM round-trip when the heuristic is already decisive (saves ~10–20s).
    if base.confidence >= 0.7 and base.print_type in ("clean", "busy"):
        return base
    if base.confidence >= 0.85:
        return base
    vlm = classify_print_vlm(image, api_key=api_key, model=model)
    if vlm is None:
        return base
    # Trust VLM for camo/soft when it is confident
    if vlm.print_type in ("camo", "soft") and vlm.confidence >= 0.55:
        return PrintTypeResult(
            print_type=vlm.print_type,
            confidence=vlm.confidence,
            scores={**base.scores, "vlm": vlm.confidence},
            reason=f"vlm:{vlm.reason} | heur:{base.reason}",
        )
    if vlm.print_type == base.print_type:
        return PrintTypeResult(
            print_type=base.print_type,
            confidence=max(base.confidence, vlm.confidence),
            scores={**base.scores, "vlm": vlm.confidence},
            reason=f"agree:{base.reason}",
        )
    # Otherwise keep heuristic (cleaner default)
    return base

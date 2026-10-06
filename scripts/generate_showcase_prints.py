#!/usr/bin/env python3
"""Generate 15 high-resolution, visually distinct motif prints for Auto Layer QA."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "showcase"
SIZE = 2048  # high-res for proper visual check


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _blank(rgb: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (SIZE, SIZE), rgb)


def flower(draw, cx, cy, r, color, petals=6, center=None):
    for i in range(petals):
        a = 2 * math.pi * i / petals
        px = cx + int(math.cos(a) * r * 0.55)
        py = cy + int(math.sin(a) * r * 0.55)
        draw.ellipse([px - r // 2, py - r // 2, px + r // 2, py + r // 2], fill=color)
    c = center or (max(0, color[0] // 2), max(0, color[1] // 2), max(0, color[2] // 2))
    draw.ellipse([cx - r // 3, cy - r // 3, cx + r // 3, cy + r // 3], fill=c)


def leaf(draw, cx, cy, r, color, angle=0.0):
    pts = []
    for t in range(0, 360, 10):
        rad = math.radians(t)
        x = r * math.cos(rad)
        y = r * 0.42 * math.sin(rad)
        ca, sa = math.cos(angle), math.sin(angle)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    draw.polygon(pts, fill=color)


def star(draw, cx, cy, r, color, n=5):
    pts = []
    for i in range(n * 2):
        ang = -math.pi / 2 + i * math.pi / n
        rad = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    draw.polygon(pts, fill=color)


def shell(draw, cx, cy, r, color):
    draw.pieslice([cx - r, cy - r, cx + r, cy + r], 200, 340, fill=color)
    rim = (max(0, color[0] - 45), max(0, color[1] - 45), max(0, color[2] - 45))
    for i in range(6):
        draw.arc(
            [cx - r + i * 4, cy - r + i * 4, cx + r - i * 4, cy + r - i * 4],
            200,
            340,
            fill=rim,
            width=2,
        )


def butterfly(draw, cx, cy, r, color):
    # wings
    draw.ellipse([cx - r, cy - r // 2, cx - 2, cy + r // 2], fill=color)
    draw.ellipse([cx + 2, cy - r // 2, cx + r, cy + r // 2], fill=color)
    wing2 = (min(255, color[0] + 30), min(255, color[1] + 20), min(255, color[2] + 40))
    draw.ellipse([cx - r // 1.2, cy - r, cx - 4, cy], fill=wing2)
    draw.ellipse([cx + 4, cy - r, cx + r // 1.2, cy], fill=wing2)
    body = (40, 35, 30)
    draw.ellipse([cx - 4, cy - r // 2, cx + 4, cy + r // 2], fill=body)


def berry(draw, cx, cy, r, color):
    for i in range(7):
        a = 2 * math.pi * i / 7
        px = cx + int(math.cos(a) * r * 0.45)
        py = cy + int(math.sin(a) * r * 0.45)
        draw.ellipse([px - r // 3, py - r // 3, px + r // 3, py + r // 3], fill=color)
    draw.ellipse([cx - r // 4, cy - r // 4, cx + r // 4, cy + r // 4], fill=(min(255, color[0] + 40),) * 3 if color[0] == color[1] else color)


def fan(draw, cx, cy, r, color, n=7):
    for i in range(n):
        a0 = -110 + i * (140 / max(1, n - 1))
        a1 = a0 + 140 / max(1, n)
        draw.pieslice([cx - r, cy - r, cx + r, cy + r], a0, a1, fill=color if i % 2 == 0 else (min(255, color[0] + 35), min(255, color[1] + 25), min(255, color[2] + 20)))
    draw.ellipse([cx - r // 6, cy - r // 6, cx + r // 6, cy + r // 6], fill=(60, 50, 40))


def citrus(draw, cx, cy, r, rind, flesh):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rind)
    draw.ellipse([cx - int(r * 0.82), cy - int(r * 0.82), cx + int(r * 0.82), cy + int(r * 0.82)], fill=flesh)
    for i in range(8):
        a = 2 * math.pi * i / 8
        draw.line(
            [cx, cy, cx + int(math.cos(a) * r * 0.75), cy + int(math.sin(a) * r * 0.75)],
            fill=rind,
            width=max(2, r // 20),
        )
    draw.ellipse([cx - r // 10, cy - r // 10, cx + r // 10, cy + r // 10], fill=rind)


def medallion(draw, cx, cy, r, color):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=max(3, r // 18))
    draw.ellipse([cx - int(r * 0.7), cy - int(r * 0.7), cx + int(r * 0.7), cy + int(r * 0.7)], outline=color, width=max(2, r // 24))
    star(draw, cx, cy, int(r * 0.45), color, 8)
    draw.ellipse([cx - r // 8, cy - r // 8, cx + r // 8, cy + r // 8], fill=color)


def bird(draw, cx, cy, r, color):
    # simple folk bird silhouette
    draw.ellipse([cx - r, cy - r // 2, cx + r // 2, cy + r // 2], fill=color)
    beak = (color[0], max(0, color[1] - 40), 40)
    draw.polygon([(cx + r // 2, cy), (cx + r, cy - r // 5), (cx + r // 2, cy + r // 6)], fill=beak)
    draw.ellipse([cx - r // 3, cy - r, cx + r // 4, cy - r // 6], fill=(min(255, color[0] + 40), min(255, color[1] + 30), min(255, color[2] + 20)))
    draw.ellipse([cx - r // 5, cy - r // 8, cx - r // 12, cy + r // 20], fill=(20, 20, 20))


def scatter(rng, margin, n):
    for _ in range(n):
        yield rng.randint(margin, SIZE - margin), rng.randint(margin, SIZE - margin)


CASES: list[tuple[str, str, str, callable]] = []


def case(id_: str, label: str, hint: str):
    def deco(fn):
        CASES.append((id_, label, hint, fn))
        return fn

    return deco


@case("s01_garden_roses", "Garden roses", "large florals + leaves")
def _(seed=101):
    rng = _rng(seed)
    img = _blank((246, 240, 232))
    d = ImageDraw.Draw(img)
    for x, y in scatter(rng, 90, 28):
        leaf(d, x + rng.randint(-40, 40), y + rng.randint(-40, 40), rng.randint(28, 55), (55, rng.randint(110, 150), 70), rng.random() * 6.28)
    for x, y in scatter(rng, 110, 22):
        flower(d, x, y, rng.randint(55, 95), (rng.randint(180, 230), rng.randint(50, 100), rng.randint(90, 140)), 7, (240, 200, 80))
    return img


@case("s02_butterflies", "Butterflies", "scattered insects")
def _(seed=102):
    rng = _rng(seed)
    img = _blank((232, 242, 250))
    d = ImageDraw.Draw(img)
    palette = [(220, 90, 70), (70, 120, 210), (240, 180, 50), (160, 80, 180), (50, 170, 140)]
    for x, y in scatter(rng, 100, 36):
        butterfly(d, x, y, rng.randint(40, 85), palette[rng.randint(0, len(palette) - 1)])
    return img


@case("s03_geo_diamonds", "Geo diamonds", "clean geometric")
def _(seed=103):
    rng = _rng(seed)
    img = _blank((250, 248, 245))
    d = ImageDraw.Draw(img)
    colors = [(40, 70, 110), (180, 70, 80), (220, 170, 70), (60, 130, 120)]
    for x, y in scatter(rng, 80, 48):
        r = rng.randint(35, 70)
        c = colors[rng.randint(0, 3)]
        pts = [(x, y - r), (x + r * 0.7, y), (x, y + r), (x - r * 0.7, y)]
        d.polygon(pts, fill=c)
        if rng.random() > 0.45:
            d.polygon([(x, y - r // 2), (x + r * 0.35, y), (x, y + r // 2), (x - r * 0.35, y)], fill=(250, 248, 245))
    return img


@case("s04_coastal_shells", "Coastal shells", "marine motifs")
def _(seed=104):
    rng = _rng(seed)
    img = _blank((225, 238, 242))
    d = ImageDraw.Draw(img)
    palette = [(232, 210, 180), (210, 170, 150), (190, 200, 210), (240, 220, 200)]
    for x, y in scatter(rng, 100, 40):
        shell(d, x, y, rng.randint(45, 90), palette[rng.randint(0, 3)])
    for x, y in scatter(rng, 60, 55):
        # small starfish fillers
        star(d, x, y, rng.randint(12, 22), (200, 120, 100), 5)
    return img


@case("s05_berry_sprigs", "Berry sprigs", "fruit clusters + stems")
def _(seed=105):
    rng = _rng(seed)
    img = _blank((245, 242, 235))
    d = ImageDraw.Draw(img)
    for x, y in scatter(rng, 100, 30):
        leaf(d, x + 30, y - 20, 32, (70, 130, 80), rng.random() * 3)
        leaf(d, x - 25, y + 15, 28, (60, 120, 70), rng.random() * 3)
        berry(d, x, y, rng.randint(28, 48), (170, 40, 70) if rng.random() > 0.35 else (80, 50, 140))
    return img


@case("s06_art_deco_fans", "Art deco fans", "symmetric fans")
def _(seed=106):
    rng = _rng(seed)
    img = _blank((28, 32, 48))
    d = ImageDraw.Draw(img)
    palette = [(220, 180, 90), (200, 200, 210), (180, 120, 80), (140, 170, 200)]
    cols, rows = 5, 5
    for iy in range(rows):
        for ix in range(cols):
            x = int((ix + 0.5) * SIZE / cols) + rng.randint(-20, 20)
            y = int((iy + 0.5) * SIZE / rows) + rng.randint(-20, 20)
            fan(d, x, y, rng.randint(110, 160), palette[(ix + iy) % 4], n=7)
    return img


@case("s07_wave_dots", "Wave dots", "organic polka field")
def _(seed=107):
    rng = _rng(seed)
    img = _blank((240, 236, 228))
    d = ImageDraw.Draw(img)
    for i in range(220):
        x = int((i * 137.5) % (SIZE - 80)) + 40
        y = int((i * 97.3) % (SIZE - 80)) + 40
        y += int(28 * math.sin(x / 90))
        r = 14 + (i % 7) * 5
        c = (60 + (i * 17) % 140, 90 + (i * 11) % 100, 140 + (i * 13) % 80)
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
    return img


@case("s08_ivy_trail", "Ivy trail", "vine + leaves")
def _(seed=108):
    rng = _rng(seed)
    img = _blank((248, 250, 245))
    d = ImageDraw.Draw(img)
    for trail in range(7):
        x, y = rng.randint(100, SIZE - 100), rng.randint(80, 200)
        for step in range(55):
            x += rng.randint(-15, 25)
            y += rng.randint(18, 38)
            if not (40 < x < SIZE - 40 and 40 < y < SIZE - 40):
                break
            leaf(d, x, y, rng.randint(22, 48), (40, rng.randint(100, 150), 55), math.atan2(20, 10) + rng.random())
            if step % 4 == 0:
                d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(150, 50, 70))
    return img


@case("s09_starfish_marine", "Starfish marine", "bold sea stars")
def _(seed=109):
    rng = _rng(seed)
    img = _blank((20, 55, 75))
    d = ImageDraw.Draw(img)
    palette = [(240, 140, 70), (230, 200, 90), (220, 110, 100), (180, 210, 200)]
    for x, y in scatter(rng, 120, 32):
        star(d, x, y, rng.randint(55, 110), palette[rng.randint(0, 3)], 5)
    for x, y in scatter(rng, 50, 40):
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(120, 200, 210))
    return img


@case("s10_folk_birds", "Folk birds", "bird silhouettes")
def _(seed=110):
    rng = _rng(seed)
    img = _blank((250, 245, 235))
    d = ImageDraw.Draw(img)
    palette = [(40, 70, 100), (160, 60, 50), (70, 110, 80), (100, 70, 120)]
    for x, y in scatter(rng, 130, 28):
        bird(d, x, y, rng.randint(50, 95), palette[rng.randint(0, 3)])
        leaf(d, x + rng.randint(-60, 60), y + 50, 30, (80, 130, 70), rng.random())
    return img


@case("s11_tulip_grid", "Tulip grid", "ordered florals")
def _(seed=111):
    rng = _rng(seed)
    img = _blank((245, 248, 250))
    d = ImageDraw.Draw(img)
    cols, rows = 6, 6
    colors = [(210, 70, 90), (230, 150, 60), (180, 80, 160), (70, 130, 180)]
    for iy in range(rows):
        for ix in range(cols):
            x = int((ix + 0.5) * SIZE / cols) + rng.randint(-12, 12)
            y = int((iy + 0.5) * SIZE / rows) + rng.randint(-12, 12)
            # stem
            d.line([(x, y + 40), (x, y + 120)], fill=(60, 120, 70), width=6)
            leaf(d, x - 25, y + 90, 28, (60, 130, 75), -0.8)
            leaf(d, x + 25, y + 90, 28, (60, 130, 75), 0.8)
            # tulip head = 3 ellipses
            c = colors[(ix + iy) % 4]
            for dx in (-22, 0, 22):
                d.ellipse([x + dx - 28, y - 50, x + dx + 28, y + 20], fill=c)
            d.ellipse([x - 35, y - 10, x + 35, y + 45], fill=c)
    return img


@case("s12_color_blobs", "Color blobs", "soft abstract shapes")
def _(seed=112):
    rng = _rng(seed)
    img = _blank((250, 248, 245))
    base = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    palette = [(230, 100, 90, 210), (90, 150, 220, 200), (240, 190, 70, 200), (120, 190, 140, 200), (180, 120, 200, 200)]
    for _ in range(55):
        x, y = rng.randint(80, SIZE - 80), rng.randint(80, SIZE - 80)
        r = rng.randint(60, 140)
        c = palette[rng.randint(0, 4)]
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        if rng.random() > 0.5:
            d.ellipse([x - r // 2, y - r // 3, x + r // 2, y + r // 2], fill=(c[0], c[1], c[2], 160))
    soft = base.filter(ImageFilter.GaussianBlur(2.5))
    return Image.alpha_composite(img.convert("RGBA"), soft).convert("RGB")


@case("s13_lace_medallions", "Lace medallions", "outline ornaments")
def _(seed=113):
    rng = _rng(seed)
    img = _blank((255, 252, 248))
    d = ImageDraw.Draw(img)
    ink = (45, 50, 70)
    for x, y in scatter(rng, 140, 24):
        medallion(d, x, y, rng.randint(70, 120), ink)
    for x, y in scatter(rng, 60, 50):
        star(d, x, y, rng.randint(10, 18), ink, 6)
    return img


@case("s14_citrus_slices", "Citrus slices", "fruit cross-sections")
def _(seed=114):
    rng = _rng(seed)
    img = _blank((250, 245, 235))
    d = ImageDraw.Draw(img)
    pairs = [
        ((230, 140, 40), (255, 220, 120)),
        ((220, 70, 60), (255, 180, 160)),
        ((80, 160, 70), (200, 230, 140)),
        ((240, 160, 50), (255, 230, 150)),
    ]
    for x, y in scatter(rng, 130, 28):
        rind, flesh = pairs[rng.randint(0, 3)]
        citrus(d, x, y, rng.randint(70, 120), rind, flesh)
    return img


@case("s15_mixed_bouquet", "Mixed bouquet", "flowers + leaves + fillers")
def _(seed=115):
    rng = _rng(seed)
    img = _blank((242, 238, 230))
    d = ImageDraw.Draw(img)
    for x, y in scatter(rng, 70, 40):
        leaf(d, x, y, rng.randint(20, 45), (50, rng.randint(100, 150), 65), rng.random() * 6)
    for x, y in scatter(rng, 120, 18):
        flower(d, x, y, rng.randint(50, 85), (rng.randint(180, 230), rng.randint(60, 110), rng.randint(100, 150)), rng.randint(5, 8))
    for x, y in scatter(rng, 100, 14):
        flower(d, x, y, rng.randint(35, 55), (240, 200, 70), 6, (180, 120, 40))
    for x, y in scatter(rng, 50, 60):
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(90, 140, 180))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = []
    for id_, label, hint, fn in CASES:
        img = fn()
        # light sharpen for crisp edges at 2k
        path = OUT / f"{id_}.png"
        img.save(path, optimize=True)
        catalog.append({"id": f"showcase/{id_}.png", "label": label, "hint": hint, "file": path.name, "size": SIZE})
        print(f"wrote {path.name} — {label} ({SIZE}×{SIZE})")
    (OUT / "catalog.json").write_text(json.dumps(catalog, indent=2))
    print(f"\n{len(catalog)} showcase prints → {OUT}")


if __name__ == "__main__":
    main()

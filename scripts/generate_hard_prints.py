#!/usr/bin/env python3
"""Generate ~30 hard Auto Layer stress-test prints into fixtures/hard/."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "hard"
SIZE = 768


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _blank(rgb: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (SIZE, SIZE), rgb)


def _noise(img: Image.Image, amp: float = 6.0, seed: int = 0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    rng = np.random.default_rng(seed)
    arr += rng.normal(0, amp, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def flower(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color, petals: int = 6):
    for i in range(petals):
        a = 2 * math.pi * i / petals
        px = cx + int(math.cos(a) * r * 0.55)
        py = cy + int(math.sin(a) * r * 0.55)
        draw.ellipse([px - r // 2, py - r // 2, px + r // 2, py + r // 2], fill=color)
    draw.ellipse([cx - r // 3, cy - r // 3, cx + r // 3, cy + r // 3], fill=(color[0] // 2, color[1] // 2, color[2] // 2))


def leaf(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color, angle: float = 0):
    # Approximate leaf as rotated ellipse via polygon
    pts = []
    for t in range(0, 360, 12):
        rad = math.radians(t)
        x = r * math.cos(rad)
        y = r * 0.45 * math.sin(rad)
        ca, sa = math.cos(angle), math.sin(angle)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    draw.polygon(pts, fill=color)


def star(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color, n: int = 5):
    pts = []
    for i in range(n * 2):
        ang = -math.pi / 2 + i * math.pi / n
        rad = r if i % 2 == 0 else r * 0.4
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    draw.polygon(pts, fill=color)


def shell(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color):
    draw.pieslice([cx - r, cy - r, cx + r, cy + r], 200, 340, fill=color)
    for i in range(5):
        draw.arc([cx - r + i * 3, cy - r + i * 3, cx + r - i * 3, cy + r - i * 3], 200, 340, fill=(max(0, color[0] - 40),) * 3 if isinstance(color[0], int) else color)


CASES: list[tuple[str, str, callable]] = []


def case(id_: str, label: str, hint: str):
    def deco(fn):
        CASES.append((id_, label, hint, fn))
        return fn

    return deco


@case("h01_overlap_flowers", "Overlap flowers", "heavy overlaps")
def _(seed=1):
    rng = _rng(seed)
    img = _blank((250, 245, 235))
    d = ImageDraw.Draw(img)
    colors = [(220, 90, 110), (90, 160, 100), (230, 170, 70), (120, 100, 180)]
    for _ in range(45):
        x, y = rng.randint(40, SIZE - 40), rng.randint(40, SIZE - 40)
        r = rng.randint(28, 70)
        flower(d, x, y, r, colors[rng.randrange(len(colors))], petals=rng.randint(5, 8))
    return _noise(img, 4, seed)


@case("h02_low_contrast", "Low contrast", "motifs ≈ background")
def _(seed=2):
    rng = _rng(seed)
    bg = (210, 215, 208)
    img = _blank(bg)
    d = ImageDraw.Draw(img)
    for _ in range(50):
        x, y = rng.randint(30, SIZE - 30), rng.randint(30, SIZE - 30)
        r = rng.randint(20, 55)
        c = (bg[0] + rng.randint(-18, 18), bg[1] + rng.randint(-14, 14), bg[2] + rng.randint(-16, 16))
        flower(d, x, y, r, c, petals=5)
    return _noise(img, 3, seed)


@case("h03_micro_ditsy", "Micro ditsy", "200+ tiny motifs")
def _(seed=3):
    rng = _rng(seed)
    img = _blank((255, 250, 245))
    d = ImageDraw.Draw(img)
    for _ in range(220):
        x, y = rng.randint(8, SIZE - 8), rng.randint(8, SIZE - 8)
        r = rng.randint(4, 12)
        c = (rng.randint(180, 255), rng.randint(80, 160), rng.randint(100, 180))
        if rng.random() < 0.5:
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        else:
            leaf(d, x, y, r + 4, (80, rng.randint(120, 170), 90), rng.random() * 6)
    return img


@case("h04_touching_grid", "Touching grid", "kissing edges")
def _(seed=4):
    img = _blank((240, 248, 255))
    d = ImageDraw.Draw(img)
    colors = [(40, 120, 180), (220, 100, 60), (60, 160, 120)]
    step = 70
    i = 0
    for y in range(40, SIZE - 40, step):
        for x in range(40, SIZE - 40, step):
            r = 34
            star(d, x, y, r, colors[i % 3], n=5)
            i += 1
    return img


@case("h05_nested_rings", "Nested rings", "holes / donuts")
def _(seed=5):
    rng = _rng(seed)
    img = _blank((30, 30, 35))
    d = ImageDraw.Draw(img)
    for _ in range(35):
        x, y = rng.randint(60, SIZE - 60), rng.randint(60, SIZE - 60)
        r = rng.randint(25, 55)
        c = (rng.randint(180, 255), rng.randint(100, 200), rng.randint(60, 120))
        d.ellipse([x - r, y - r, x + r, y + r], outline=c, width=6)
        d.ellipse([x - r // 2, y - r // 2, x + r // 2, y + r // 2], outline=c, width=4)
    return img


@case("h06_thin_lines", "Thin line motifs", "hairline strokes")
def _(seed=6):
    rng = _rng(seed)
    img = _blank((252, 250, 245))
    d = ImageDraw.Draw(img)
    for _ in range(60):
        x, y = rng.randint(50, SIZE - 50), rng.randint(50, SIZE - 50)
        r = rng.randint(20, 60)
        c = (rng.randint(40, 100),) * 3
        for a in range(0, 360, 30):
            rad = math.radians(a)
            d.line([(x, y), (x + r * math.cos(rad), y + r * math.sin(rad))], fill=c, width=1)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c)
    return img


@case("h07_noisy_ground", "Noisy ground", "busy textured bg")
def _(seed=7):
    rng = _rng(seed)
    arr = np.zeros((SIZE, SIZE, 3), np.uint8)
    arr[:, :] = (200, 190, 175)
    noise = np.random.default_rng(seed).integers(-40, 40, (SIZE, SIZE, 3))
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.randint(40, SIZE - 40), rng.randint(40, SIZE - 40)
        flower(d, x, y, rng.randint(22, 40), (180, 60, 80), 6)
    return img


@case("h08_gradient_ground", "Gradient ground", "non-flat bg")
def _(seed=8):
    rng = _rng(seed)
    arr = np.zeros((SIZE, SIZE, 3), np.float32)
    for y in range(SIZE):
        t = y / SIZE
        arr[y, :] = (240 - 40 * t, 230 - 20 * t, 220 + 20 * t)
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.randint(40, SIZE - 40), rng.randint(40, SIZE - 40)
        leaf(d, x, y, rng.randint(25, 50), (40, 120, 70), rng.random() * 6)
    return img


@case("h09_same_hue_family", "Same-hue family", "all pink tones")
def _(seed=9):
    rng = _rng(seed)
    img = _blank((255, 240, 245))
    d = ImageDraw.Draw(img)
    for _ in range(55):
        x, y = rng.randint(30, SIZE - 30), rng.randint(30, SIZE - 30)
        r = rng.randint(18, 48)
        c = (rng.randint(200, 255), rng.randint(100, 180), rng.randint(130, 190))
        flower(d, x, y, r, c, petals=rng.randint(5, 7))
    return img


@case("h10_mixed_scale", "Mixed scale", "tiny + huge")
def _(seed=10):
    rng = _rng(seed)
    img = _blank((245, 250, 255))
    d = ImageDraw.Draw(img)
    for _ in range(8):
        x, y = rng.randint(80, SIZE - 80), rng.randint(80, SIZE - 80)
        star(d, x, y, rng.randint(70, 110), (50, 90, 160), 6)
    for _ in range(80):
        x, y = rng.randint(10, SIZE - 10), rng.randint(10, SIZE - 10)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(220, 140, 60))
    return img


@case("h11_edge_cropped", "Edge cropped", "motifs cut by frame")
def _(seed=11):
    rng = _rng(seed)
    img = _blank((235, 245, 240))
    d = ImageDraw.Draw(img)
    positions = [(0, y) for y in range(0, SIZE, 90)] + [(SIZE, y) for y in range(0, SIZE, 90)]
    positions += [(x, 0) for x in range(0, SIZE, 90)] + [(x, SIZE) for x in range(0, SIZE, 90)]
    for x, y in positions:
        flower(d, x, y, rng.randint(40, 70), (200, 80, 90), 6)
    for _ in range(20):
        flower(d, rng.randint(80, SIZE - 80), rng.randint(80, SIZE - 80), 35, (80, 140, 100), 5)
    return img


@case("h12_soft_watercolor", "Soft watercolor", "blurred edges")
def _(seed=12):
    rng = _rng(seed)
    img = _blank((250, 248, 242))
    overlay = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for _ in range(35):
        x, y = rng.randint(50, SIZE - 50), rng.randint(50, SIZE - 50)
        r = rng.randint(30, 70)
        c = (*[rng.randint(100, 220) for _ in range(3)], 160)
        flower(d, x, y, r, c, 6)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3.5))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img


@case("h13_checker_plus", "Checker + motifs", "patterned ground")
def _(seed=13):
    rng = _rng(seed)
    img = _blank((255, 255, 255))
    d = ImageDraw.Draw(img)
    s = 48
    for y in range(0, SIZE, s):
        for x in range(0, SIZE, s):
            if ((x // s) + (y // s)) % 2 == 0:
                d.rectangle([x, y, x + s, y + s], fill=(230, 230, 235))
    for _ in range(40):
        x, y = rng.randint(30, SIZE - 30), rng.randint(30, SIZE - 30)
        star(d, x, y, rng.randint(18, 32), (200, 60, 80), 5)
    return img


@case("h14_one_giant", "One giant motif", "dominant object")
def _(seed=14):
    img = _blank((245, 240, 230))
    d = ImageDraw.Draw(img)
    flower(d, SIZE // 2, SIZE // 2, 280, (210, 90, 110), 8)
    for i in range(12):
        a = 2 * math.pi * i / 12
        leaf(d, int(SIZE / 2 + 200 * math.cos(a)), int(SIZE / 2 + 200 * math.sin(a)), 40, (70, 140, 80), a)
    return img


@case("h15_fillers_between", "Fillers between", "large + micro fillers")
def _(seed=15):
    rng = _rng(seed)
    img = _blank((248, 245, 240))
    d = ImageDraw.Draw(img)
    for _ in range(18):
        flower(d, rng.randint(60, SIZE - 60), rng.randint(60, SIZE - 60), rng.randint(40, 60), (180, 70, 100), 6)
    for _ in range(150):
        x, y = rng.randint(5, SIZE - 5), rng.randint(5, SIZE - 5)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(120, 160, 90))
    return img


@case("h16_stripes_break", "Broken stripes", "elongated motifs")
def _(seed=16):
    rng = _rng(seed)
    img = _blank((30, 40, 50))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x = rng.randint(20, SIZE - 20)
        y1, y2 = rng.randint(0, SIZE // 3), rng.randint(2 * SIZE // 3, SIZE)
        c = (rng.randint(180, 255), rng.randint(140, 200), rng.randint(60, 120))
        d.rectangle([x - 6, y1, x + 6, y2], fill=c)
    return img


@case("h17_camouflage", "Camouflage blobs", "similar multi-tone")
def _(seed=17):
    rng = _rng(seed)
    img = _blank((90, 100, 70))
    d = ImageDraw.Draw(img)
    for _ in range(70):
        x, y = rng.randint(0, SIZE), rng.randint(0, SIZE)
        r = rng.randint(30, 90)
        c = (rng.randint(60, 120), rng.randint(70, 130), rng.randint(40, 90))
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
    return img.filter(ImageFilter.GaussianBlur(1.2))


@case("h18_outline_only", "Outline only", "no fill shapes")
def _(seed=18):
    rng = _rng(seed)
    img = _blank((255, 255, 255))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.randint(50, SIZE - 50), rng.randint(50, SIZE - 50)
        r = rng.randint(25, 55)
        c = (40, 40, 40)
        # outline flower
        for i in range(6):
            a = 2 * math.pi * i / 6
            px = x + int(math.cos(a) * r * 0.55)
            py = y + int(math.sin(a) * r * 0.55)
            d.ellipse([px - r // 2, py - r // 2, px + r // 2, py + r // 2], outline=c, width=2)
        d.ellipse([x - r // 3, y - r // 3, x + r // 3, y + r // 3], outline=c, width=2)
    return img


@case("h19_stacked_same", "Stacked identical", "exact repeats overlap")
def _(seed=19):
    img = _blank((240, 235, 245))
    d = ImageDraw.Draw(img)
    base = [(200, 200), (400, 280), (550, 500), (250, 520), (480, 150)]
    for x, y in base:
        for dx, dy in [(0, 0), (12, 8), (-10, 14)]:
            flower(d, x + dx, y + dy, 50, (160, 80, 170), 6)
    return img


@case("h20_confetti", "Confetti scatter", "many shapes/types")
def _(seed=20):
    rng = _rng(seed)
    img = _blank((20, 20, 24))
    d = ImageDraw.Draw(img)
    for _ in range(120):
        x, y = rng.randint(10, SIZE - 10), rng.randint(10, SIZE - 10)
        r = rng.randint(6, 28)
        c = (rng.randint(80, 255), rng.randint(80, 255), rng.randint(80, 255))
        choice = rng.randrange(4)
        if choice == 0:
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        elif choice == 1:
            d.rectangle([x - r, y - r // 2, x + r, y + r // 2], fill=c)
        elif choice == 2:
            star(d, x, y, r, c, 4)
        else:
            leaf(d, x, y, r, c, rng.random() * 6)
    return img


@case("h21_paisley_like", "Paisley-like", "complex teardrops")
def _(seed=21):
    rng = _rng(seed)
    img = _blank((252, 245, 230))
    d = ImageDraw.Draw(img)
    for _ in range(28):
        x, y = rng.randint(60, SIZE - 60), rng.randint(60, SIZE - 60)
        r = rng.randint(35, 70)
        ang = rng.random() * 6.28
        c = (rng.randint(120, 180), 40, rng.randint(80, 140))
        # teardrop approx
        pts = []
        for t in range(0, 360, 10):
            rad = math.radians(t)
            rr = r * (0.55 + 0.45 * (1 + math.cos(rad)) / 2)
            px = rr * math.cos(rad)
            py = rr * math.sin(rad) * 0.75
            ca, sa = math.cos(ang), math.sin(ang)
            pts.append((x + px * ca - py * sa, y + px * sa + py * ca))
        d.polygon(pts, fill=c)
        # inner filigree dots
        for __ in range(8):
            ox, oy = x + rng.randint(-r // 3, r // 3), y + rng.randint(-r // 3, r // 3)
            d.ellipse([ox - 2, oy - 2, ox + 2, oy + 2], fill=(230, 200, 80))
    return img


@case("h22_near_white_on_white", "Near-white on white", "barely visible")
def _(seed=22):
    rng = _rng(seed)
    img = _blank((255, 255, 255))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.randint(40, SIZE - 40), rng.randint(40, SIZE - 40)
        flower(d, x, y, rng.randint(25, 50), (245, 245, 248), 6)
    return _noise(img, 2, seed)


@case("h23_black_on_black", "Black on charcoal", "dark-on-dark")
def _(seed=23):
    rng = _rng(seed)
    img = _blank((28, 28, 30))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.randint(40, SIZE - 40), rng.randint(40, SIZE - 40)
        star(d, x, y, rng.randint(20, 45), (10, 10, 12), 5)
    return _noise(img, 2, seed)


@case("h24_connected_chain", "Connected chain", "merged CC risk")
def _(seed=24):
    img = _blank((245, 250, 255))
    d = ImageDraw.Draw(img)
    x, y = 80, SIZE // 2
    for i in range(18):
        c = (60 + i * 8, 100, 180 - i * 5)
        d.ellipse([x - 28, y - 28, x + 28, y + 28], fill=c)
        x += 36
        y += int(20 * math.sin(i))
    return img


@case("h25_halos", "Glow halos", "soft fringe")
def _(seed=25):
    rng = _rng(seed)
    img = _blank((20, 24, 40))
    base = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    for _ in range(30):
        x, y = rng.randint(50, SIZE - 50), rng.randint(50, SIZE - 50)
        r = rng.randint(20, 40)
        d.ellipse([x - r * 2, y - r * 2, x + r * 2, y + r * 2], fill=(255, 200, 80, 40))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 220, 120, 220))
    glow = base.filter(ImageFilter.GaussianBlur(6))
    return Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")


@case("h26_dense_leaves", "Dense leaves", "similar shapes packed")
def _(seed=26):
    rng = _rng(seed)
    img = _blank((245, 250, 240))
    d = ImageDraw.Draw(img)
    for _ in range(100):
        x, y = rng.randint(20, SIZE - 20), rng.randint(20, SIZE - 20)
        leaf(d, x, y, rng.randint(18, 40), (40, rng.randint(100, 160), 50), rng.random() * 6.28)
    return img


@case("h27_two_tone_split", "Two-tone split", "bg changes mid-image")
def _(seed=27):
    rng = _rng(seed)
    img = _blank((240, 230, 220))
    d = ImageDraw.Draw(img)
    d.rectangle([SIZE // 2, 0, SIZE, SIZE], fill=(40, 50, 70))
    for _ in range(25):
        flower(d, rng.randint(40, SIZE // 2 - 40), rng.randint(40, SIZE - 40), 30, (200, 80, 90), 5)
    for _ in range(25):
        star(d, rng.randint(SIZE // 2 + 40, SIZE - 40), rng.randint(40, SIZE - 40), 28, (220, 200, 100), 5)
    return img


@case("h28_spiral_scatter", "Spiral scatter", "structured density")
def _(seed=28):
    img = _blank((250, 248, 245))
    d = ImageDraw.Draw(img)
    cx = cy = SIZE // 2
    for i in range(90):
        a = i * 0.35
        rad = 20 + i * 3.2
        x = int(cx + rad * math.cos(a))
        y = int(cy + rad * math.sin(a))
        if 10 < x < SIZE - 10 and 10 < y < SIZE - 10:
            flower(d, x, y, 12 + (i % 5), (180, 70 + i % 80, 110), 5)
    return img


@case("h29_ink_blots", "Ink blots", "irregular organic")
def _(seed=29):
    rng = _rng(seed)
    img = _blank((250, 245, 235))
    d = ImageDraw.Draw(img)
    for _ in range(35):
        x, y = rng.randint(60, SIZE - 60), rng.randint(60, SIZE - 60)
        for __ in range(rng.randint(5, 14)):
            rr = rng.randint(8, 35)
            ox = x + rng.randint(-25, 25)
            oy = y + rng.randint(-25, 25)
            d.ellipse([ox - rr, oy - rr, ox + rr, oy + rr], fill=(20, 20, 30))
    return img.filter(ImageFilter.GaussianBlur(0.8))


@case("h30_chaos_mix", "Chaos mix", "kitchen-sink hard mode")
def _(seed=30):
    rng = _rng(seed)
    img = _noise(_blank((230, 225, 235)), 8, seed)
    d = ImageDraw.Draw(img)
    for _ in range(25):
        flower(d, rng.randint(40, SIZE - 40), rng.randint(40, SIZE - 40), rng.randint(20, 55),
               (rng.randint(100, 255), rng.randint(40, 120), rng.randint(80, 180)), rng.randint(5, 8))
    for _ in range(25):
        star(d, rng.randint(40, SIZE - 40), rng.randint(40, SIZE - 40), rng.randint(15, 40),
             (rng.randint(40, 100), rng.randint(100, 200), rng.randint(150, 255)), 5)
    for _ in range(30):
        leaf(d, rng.randint(20, SIZE - 20), rng.randint(20, SIZE - 20), rng.randint(15, 35),
             (40, rng.randint(90, 160), 50), rng.random() * 6)
    # overlaps forced
    for x, y in [(200, 200), (210, 205), (400, 400), (405, 410), (600, 300)]:
        flower(d, x, y, 45, (220, 90, 100), 6)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = []
    for id_, label, hint, fn in CASES:
        img = fn()
        path = OUT / f"{id_}.png"
        img.save(path, optimize=True)
        catalog.append({"id": f"hard/{id_}.png", "label": label, "hint": hint, "file": path.name})
        print(f"wrote {path.name} — {label}")
    (OUT / "catalog.json").write_text(json.dumps(catalog, indent=2))
    print(f"\n{len(catalog)} hard prints → {OUT}")


if __name__ == "__main__":
    main()

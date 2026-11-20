#!/usr/bin/env python3
"""Christmas-lights Cybertruck wraps for Tesla Paint Shop.

Reads cybertruck/template.png and paints three static twinkle states:

    example/Christmas_Lights_1.png  all bulbs lit
    example/Christmas_Lights_2.png  every other bulb dim
    example/Christmas_Lights_3.png  the opposite phase

The layout is shared. Only bulb brightness and sparkle positions change, so
switching wraps in Paint Shop reads as a twinkle. Paint Shop images cannot
animate.

Panel map (measured from the 1024x768 template and a calibrated top/front
view of the same UV):

    Bottom island (y > ~512)  driver side.  +x is rearward, +y is down.
    Top island    (y < ~256)  passenger side, the vertical mirror of the
                              driver side (y -> 767 - y).
    Curved island (x ~127-265, y ~210-558)
                              hood / frunk. +x points at the windshield
                              (up, when you look at the hood from the front).
                              +y points at the driver's side.
    Left bar      (x ~26-90)  front fascia. +x is up, +y is driver's side.
    Right bar     (x ~919-1016) tailgate. +x is down, +y is driver's side.

The family photo is drawn upright, then rotated 90 degrees clockwise so the
heads point at the windshield and the group reads left-to-right from the front.

Requires Pillow and numpy. Run from anywhere:

    python3 cybertruck/generate_christmas_lights.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "template.png"
FAMILY = ROOT / "assets" / "family_photo.jpg"
# Tight crop of the four people on the 1600x1067 source. Every head stays in frame.
FAMILY_CROP = (160, 0, 1520, 780)
OUT_DIR = ROOT / "example"
PREVIEW_DIR = ROOT / "preview"

W, H = 1024, 768

# Night evergreen. Vehicle-up is a little lighter, like moonlight on the panels.
NIGHT_UP = np.array([16, 42, 48], np.float32)
NIGHT_DOWN = np.array([5, 16, 13], np.float32)

BULB_COLORS = (
    (235, 32, 40),     # red
    (36, 186, 72),     # green
    (245, 188, 48),    # gold
    (48, 116, 245),    # blue
    (250, 248, 236),   # white
)
WIRE = (14, 58, 32)
WIRE_HI = (36, 92, 48)
SOCKET = (18, 28, 20)

# Hood portrait. place_hood() recenters this on the hood island.
# Upright size: width runs across the family, height runs heads-to-feet.
# A clockwise quarter-turn maps height onto template +x (toward the windshield)
# and width onto template +y (across the hood).
HOOD_CENTER = (186, 383)
PHOTO_H = 100
PHOTO_W = 174
WREATH_PAD = 6
HOOD_RING_RX = PHOTO_H / 2 + WREATH_PAD - 1
HOOD_RING_RY = PHOTO_W / 2 + WREATH_PAD - 1

BULB_SPACING = 28
RING_SPACING = 17
BULB_RX, BULB_RY = 4.2, 5.6


def load_mask(template: Image.Image) -> np.ndarray:
    """Paintable islands: near-white texels, including the faint guide grid."""
    rgb = np.asarray(template.convert("RGB"), dtype=np.float32)
    lum = rgb.mean(axis=2)
    return lum > 190


def panel_masks(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Split the five islands by connected component and where each sits."""
    labels, count = ndimage.label(mask)
    found = {}
    for i in range(1, count + 1):
        sel = labels == i
        ys, xs = np.where(sel)
        if len(ys) < 1000:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        if cy < 256 and sel.sum() > 80000:
            found["passenger"] = sel
        elif cy > 512 and sel.sum() > 80000:
            found["driver"] = sel
        elif cx < 120:
            found["front"] = sel
        elif cx > 880:
            found["tail"] = sel
        else:
            found["hood"] = sel
    missing = {"passenger", "driver", "front", "hood", "tail"} - found.keys()
    if missing:
        raise SystemExit(f"template islands not found: {sorted(missing)}")
    return found


def _ellipse_inside(hood: np.ndarray, cx: int, cy: int, rx: float, ry: float) -> bool:
    if rx < 2 or ry < 2:
        return False
    yy, xx = np.ogrid[: hood.shape[0], : hood.shape[1]]
    ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    return bool(ell.any() and np.all(hood[ell]))


def place_hood(hood: np.ndarray) -> None:
    """Center the portrait on the hood island and size it to the panel.

    Vertical center is the middle of the island. Horizontal center is the
    middle of the hood's depth at that row, so the oval sits in the panel
    rather than on the inscribed-circle point the single avatar used.
    """
    global HOOD_CENTER, PHOTO_H, PHOTO_W, HOOD_RING_RX, HOOD_RING_RY
    rows = np.where(hood.any(axis=1))[0]
    cy = int((int(rows.min()) + int(rows.max())) // 2)
    cols = np.where(hood[cy])[0]
    cx = int((int(cols.min()) + int(cols.max())) // 2)
    HOOD_CENTER = (cx, cy)

    crop = Image.open(FAMILY).convert("RGB").crop(FAMILY_CROP)
    aspect = crop.size[0] / crop.size[1]
    # Grow until the light ellipse, plus a couple of pixels for the bulb glass,
    # would leave the hood. Then step back so the frame stays on the panel.
    height = 80
    while height < 130:
        width = int(round(height * aspect))
        rx = height / 2 + WREATH_PAD + 2
        ry = width / 2 + WREATH_PAD + 2
        if not _ellipse_inside(hood, cx, cy, rx, ry):
            break
        height += 2
    height -= 2
    PHOTO_H = height
    PHOTO_W = int(round(height * aspect))
    HOOD_RING_RX = PHOTO_H / 2 + WREATH_PAD - 1
    HOOD_RING_RY = PHOTO_W / 2 + WREATH_PAD - 1
    print(
        f"  hood portrait center {HOOD_CENTER}  "
        f"photo {PHOTO_W}x{PHOTO_H}  light ellipse rx={HOOD_RING_RX:.0f} ry={HOOD_RING_RY:.0f}"
    )


def _runs(ys: np.ndarray) -> list[np.ndarray]:
    if len(ys) == 0:
        return []
    cuts = np.where(np.diff(ys) > 1)[0]
    parts, start = [], 0
    for c in cuts:
        parts.append(ys[start : c + 1])
        start = c + 1
    parts.append(ys[start:])
    return parts


def trace_side(mask: np.ndarray) -> dict[str, list[tuple[float, float]]]:
    """Shoulder, beltline, and rocker polylines on the driver-side island.

    +y is down on the truck, so the shoulder is the upper silhouette and the
    rocker follows the lower edge, including the wheel-arch cutouts.
    """
    shoulder, belt, rocker = [], [], []
    height, width = mask.shape
    for x in range(width):
        runs = _runs(np.where(mask[:, x])[0])
        if not runs:
            continue
        top = runs[0]
        if len(top) >= 4:
            inset = min(6, len(top) // 2)
            shoulder.append((x, float(top[0] + inset)))
        if len(runs) >= 2:
            door = runs[-1]
            yb = int(door[0] + 9)
            if yb < door[-1] - 6:
                belt.append((float(x), float(yb)))
        bot = runs[-1]
        if len(bot) < 16:
            continue
        yr = int(bot[-1] - 15)
        if bot[0] + 8 < yr < height - 2:
            rocker.append((float(x), float(yr)))
    return {"shoulder": shoulder, "belt": belt, "rocker": rocker}


def quad(p0, p1, p2, n=48) -> list[tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 2
        b = 2 * (1 - t) * t
        c = t ** 2
        pts.append((a * p0[0] + b * p1[0] + c * p2[0], a * p0[1] + b * p1[1] + c * p2[1]))
    return pts


def edge_path(mask: np.ndarray, side: str, inset: float) -> list[tuple[float, float]]:
    """Walk the left or right silhouette of an island (hood nose / cowl)."""
    pts = []
    height = mask.shape[0]
    for y in range(height):
        xs = np.where(mask[y])[0]
        if len(xs) < 8:
            continue
        x = (xs[0] + inset) if side == "left" else (xs[-1] - inset)
        if mask[y, int(np.clip(x, 0, mask.shape[1] - 1))]:
            pts.append((float(x), float(y)))
    return pts


def vertical_span(mask: np.ndarray, x: int, y0: int, y1: int) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for y in range(y0, y1) if mask[y, x]]


def horizontal_span(mask: np.ndarray, y: int, x0: int, x1: int) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x in range(x0, x1) if mask[y, x]]


def mirror_y(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Passenger island is the driver island flipped over the canvas midline."""
    return [(x, (H - 1) - y) for x, y in points]


def resample(points, spacing: float) -> np.ndarray:
    pts = np.asarray(points, np.float64)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < spacing:
        return pts[:1]
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.arange(spacing * 0.5, total, spacing)
    out = []
    j = 0
    for t in targets:
        while j < len(seg) - 1 and dist[j + 1] < t:
            j += 1
        span = seg[j] if seg[j] > 1e-6 else 1.0
        u = (t - dist[j]) / span
        out.append(pts[j] * (1 - u) + pts[j + 1] * u)
    return np.asarray(out, np.float64)


def keep_inside(points: np.ndarray, mask: np.ndarray, margin: int = 3) -> np.ndarray:
    if len(points) == 0:
        return points
    kept = []
    h, w = mask.shape
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        if not (margin <= ix < w - margin and margin <= iy < h - margin):
            continue
        if mask[iy - margin : iy + margin + 1, ix - margin : ix + margin + 1].all():
            kept.append((x, y))
    return np.asarray(kept, np.float64) if kept else np.zeros((0, 2))


def ellipse_points(cx, cy, rx, ry, spacing) -> np.ndarray:
    """Bulbs around the oval portrait. Angle 0 is template +x, the windshield side."""
    h = ((rx - ry) / (rx + ry)) ** 2
    circ = math.pi * (rx + ry) * (1 + 3 * h / (10 + math.sqrt(max(0.0, 4 - 3 * h))))
    n = max(12, int(round(circ / spacing)))
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(ang), cy + ry * np.sin(ang)], axis=1)


def outside_portrait(x: float, y: float, margin: float = 8) -> bool:
    cx, cy = HOOD_CENTER
    rx = HOOD_RING_RX + margin
    ry = HOOD_RING_RY + margin
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 > 1.0


class Strand:
    def __init__(self, points, down, phase, wire="green", spacing=BULB_SPACING, radius=None):
        self.points = np.asarray(points, np.float64)
        self.down = down
        self.phase = phase
        self.wire = wire
        self.spacing = spacing
        self.radius = radius  # (rx, ry) or None for the default teardrop


def build_strands(panels: dict[str, np.ndarray]) -> list[Strand]:
    driver = panels["driver"]
    side = trace_side(driver)
    swag = quad((210, 648), (430, 724), (630, 662))

    strands = [
        Strand(side["shoulder"], (0, 1), phase=0),
        Strand(side["belt"], (0, 1), phase=2),
        Strand(side["rocker"], (0, 1), phase=4),
        Strand(swag, (0, 1), phase=1, wire="cane"),
    ]
    # Passenger side: same draping, hung the other way so bulbs point down.
    for s in list(strands):
        strands.append(Strand(mirror_y(s.points), (0, -1), phase=s.phase + 1, wire=s.wire, spacing=s.spacing))

    hood = panels["hood"]
    # Nose (front edge) and cowl (windshield edge), skipping the medallion.
    for side_name, phase in (("left", 0), ("right", 3)):
        raw = edge_path(hood, side_name, inset=8)
        trimmed = [(x, y) for x, y in raw if outside_portrait(x, y, margin=6)]
        strands.append(Strand(trimmed, (0, 1), phase=phase, spacing=22))

    cx, cy = HOOD_CENTER
    ring = ellipse_points(cx, cy, HOOD_RING_RX, HOOD_RING_RY, RING_SPACING)
    # Drawn after the portrait so the bulbs sit on the wreath, not under it.
    strands.append(Strand(ring, (0, 0), phase=0, wire="ring", spacing=RING_SPACING, radius=(3.6, 3.6)))

    front = panels["front"]
    # Two strands across the fascia. +x is up, so the bulbs hang toward -x.
    for x, phase in ((72, 1), (48, 4)):
        strands.append(Strand(vertical_span(front, x, 230, 545), (-1, 0), phase=phase, spacing=24))

    tail = panels["tail"]
    # Two strands across the tailgate. +x is down.
    for x, phase in ((942, 2), (988, 5)):
        strands.append(Strand(vertical_span(tail, x, 236, 536), (1, 0), phase=phase, spacing=24))
    return strands


def paint_background(panels: dict[str, np.ndarray], mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((H, W, 3), np.float32)
    upness = np.zeros((H, W), np.float32)

    def ramp_rows(sel, y0, y1, flip=False):
        ys = np.arange(H, dtype=np.float32)
        t = np.clip((ys - y0) / max(1, y1 - y0), 0, 1)
        if not flip:
            t = 1 - t
        row = np.broadcast_to(t[:, None], (H, W))
        upness[sel] = row[sel]

    ramp_rows(panels["driver"], 512, 760, flip=False)
    ramp_rows(panels["passenger"], 8, 256, flip=True)
    # Hood: windshield (larger x) catches a little more sky.
    xs = np.arange(W, dtype=np.float32)

    def ramp_cols(sel, t):
        col = np.broadcast_to(t[None, :], (H, W))
        upness[sel] = col[sel]

    ramp_cols(panels["hood"], np.clip((xs - 130) / 120, 0, 1))
    # Front fascia: larger x is up. Tailgate: smaller x is up.
    ramp_cols(panels["front"], np.clip((xs - 30) / 55, 0, 1))
    ramp_cols(panels["tail"], np.clip((1010 - xs) / 80, 0, 1))

    color = NIGHT_DOWN * (1 - upness[..., None]) + NIGHT_UP * upness[..., None]
    # Fine grain so the panels are not flat plastic. Kept tiny for PNG size.
    grain = rng.normal(0, 1.1, size=(H, W, 1)).astype(np.float32)
    img = color + grain
    img *= mask[..., None]
    return np.clip(img, 0, 255)


def paint_snow(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    """Sparse flakes. Same in every variant so only the bulbs twinkle."""
    h, w = mask.shape
    stamp_cache = {}

    def stamp(radius: int) -> np.ndarray:
        if radius not in stamp_cache:
            yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
            s = np.exp(-(xx ** 2 + yy ** 2) / (2 * (radius * 0.55) ** 2)).astype(np.float32)
            stamp_cache[radius] = s
        return stamp_cache[radius]

    specs = [(520, 0, 0.55), (90, 1, 0.40), (28, 2, 0.30)]
    for count, radius, alpha in specs:
        xs = rng.integers(2, w - 2, count)
        ys = rng.integers(2, h - 2, count)
        gains = rng.uniform(0.45, 1.0, count)
        for x, y, g in zip(xs, ys, gains):
            if not mask[y, x]:
                continue
            if radius == 0:
                img[y, x] = np.maximum(img[y, x], 255 * (alpha * g))
                continue
            s = stamp(radius) * (alpha * g)
            y0, x0 = y - radius, x - radius
            y1, x1 = y + radius + 1, x + radius + 1
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                continue
            tile = img[y0:y1, x0:x1]
            m = mask[y0:y1, x0:x1, None]
            tile[:] = np.maximum(tile, (180 + 75 * g) * s[..., None]) * m + tile * (1 - m)


def draw_wire(draw: ImageDraw.ImageDraw, points, kind: str) -> None:
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 2:
        return
    if kind == "green":
        draw.line(pts, fill=WIRE_HI, width=3)
        draw.line(pts, fill=WIRE, width=1)
    elif kind == "cane":
        # White rope with red spiral bands, like a candy-cane cord.
        draw.line(pts, fill=(236, 230, 220), width=4)
        arr = np.asarray(pts, np.float64)
        seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        dist = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(dist[-1])
        band, gap = 9.0, 8.0
        t = 0.0
        j = 0
        while t < total:
            t1 = min(total, t + band)
            chunk = []
            u = t
            while True:
                while j < len(seg) - 1 and dist[j + 1] < u:
                    j += 1
                span = seg[j] if seg[j] else 1.0
                f = (u - dist[j]) / span
                chunk.append(tuple(arr[j] * (1 - f) + arr[j + 1] * f))
                if u >= t1 - 1e-6:
                    break
                u = t1
            if len(chunk) >= 2:
                draw.line(chunk, fill=(196, 28, 36), width=4)
            t += band + gap
    # "none" (the hood ring) has no wire; the wreath is the cord.


def bulb_gain(index: int, variant: int) -> float:
    """Brightness of bulb `index` in this twinkle state. 0.08 reads as off."""
    if variant == 1:
        return 1.0
    if variant == 2:
        return (1.0, 0.08, 0.62, 0.10)[index % 4]
    return (0.10, 0.85, 0.08, 1.0)[index % 4]


def draw_bulb(draw: ImageDraw.ImageDraw, x, y, down, color, gain, radius=None) -> None:
    rx, ry = radius if radius else (BULB_RX, BULB_RY)
    dx, dy = down
    # Teardrop stretched along the hang direction. Axis-aligned is enough:
    # every body strand hangs on a template axis.
    if abs(dx) > abs(dy):
        rx, ry = ry, rx
    shade = tuple(int(c * (0.22 + 0.78 * gain)) for c in color)
    box = [x - rx, y - ry, x + rx, y + ry]
    draw.ellipse(box, fill=shade)
    if gain > 0.45:
        # Hot glass center.
        hr = max(1.2, rx * 0.45)
        draw.ellipse([x - hr, y - hr * 0.8, x + hr * 0.3, y + hr * 0.2], fill=(255, 252, 230))
    # Socket toward the wire (opposite the hang).
    if dx or dy:
        sx = x - dx * (rx + 0.5)
        sy = y - dy * (ry + 0.5)
        draw.ellipse([sx - 1.6, sy - 1.6, sx + 1.6, sy + 1.6], fill=SOCKET)


def draw_sparkle(draw: ImageDraw.ImageDraw, x, y, length: float) -> None:
    """A short four-point glint, not a full star, so the bulbs stay the subject."""
    L = length
    white = (255, 252, 240)
    draw.line((x - L, y, x + L, y), fill=white, width=1)
    draw.line((x, y - L, x, y + L), fill=white, width=1)
    draw.point((x, y), fill=(255, 255, 255))


def render_lights(base: np.ndarray, strands: list[Strand], mask: np.ndarray, variant: int) -> np.ndarray:
    """Wires, bloom, bulb glass, and sparkles. Returns uint8 RGB."""
    wires = Image.new("RGB", (W, H), (0, 0, 0))
    wd = ImageDraw.Draw(wires)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    bulbs = Image.new("RGB", (W, H), (0, 0, 0))
    bd = ImageDraw.Draw(bulbs)
    sparks = Image.new("RGB", (W, H), (0, 0, 0))
    sd = ImageDraw.Draw(sparks)

    for strand in strands:
        if strand.wire not in ("none", "ring") and len(strand.points) >= 2:
            draw_wire(wd, strand.points, strand.wire)
        samples = resample(strand.points, strand.spacing)
        samples = keep_inside(samples, mask, margin=3)
        for n, (x, y) in enumerate(samples):
            color = BULB_COLORS[(n + strand.phase) % len(BULB_COLORS)]
            gain = bulb_gain(n + strand.phase * 3, variant)
            dx, dy = strand.down
            if dx or dy:
                # Hang the glass just off the wire.
                bx = x + dx * 5.0
                by = y + dy * 5.0
            else:
                # Ring: sit on the circle, already placed.
                bx, by = x, y
            if not mask[int(by), int(bx)]:
                continue
            draw_bulb(bd, bx, by, strand.down if (dx or dy) else (0, 1), color, gain, strand.radius)
            if gain > 0.35:
                gr = 3.0 + 4.0 * gain
                gc = tuple(int(c * gain) for c in color)
                gd.ellipse([bx - gr, by - gr, bx + gr, by + gr], fill=gc)
            # A few bulbs carry a small glint. Most of the twinkle is the glow.
            spark = gain > 0.8 and ((n + strand.phase * 2 + variant) % 6 == 0)
            if spark:
                length = 3.4 if (n + variant) % 2 == 0 else 2.4
                ox = 3.2 if (n % 2 == 0) else -2.8
                draw_sparkle(sd, bx + ox, by - 2.4, length)

    bloom = glow.filter(ImageFilter.GaussianBlur(radius=7))
    tight = glow.filter(ImageFilter.GaussianBlur(radius=2.2))

    out = base.copy()
    wire_a = np.asarray(wires, np.float32)
    # Wires only darken where they were drawn.
    painted = wire_a.max(axis=2, keepdims=True) > 0
    out = np.where(painted, wire_a * 0.85 + out * 0.15, out)
    out += np.asarray(bloom, np.float32) * 0.95
    out += np.asarray(tight, np.float32) * 0.55
    bulb_a = np.asarray(bulbs, np.float32)
    bulb_m = bulb_a.max(axis=2, keepdims=True) > 0
    out = np.where(bulb_m, np.maximum(out, bulb_a), out)
    out += np.asarray(sparks, np.float32)
    return np.clip(out, 0, 255)


def make_medallion() -> Image.Image:
    """Oval family portrait with a wreath. Rotated so heads point at the windshield.

    No Santa hat: the crop is four people, and a hat would cover the back row.
    """
    scale = 3
    pad = WREATH_PAD * scale
    photo = Image.open(FAMILY).convert("RGB").crop(FAMILY_CROP)
    face = photo.resize((PHOTO_W * scale, PHOTO_H * scale), Image.Resampling.LANCZOS)
    fw, fh = face.size
    canvas = Image.new("RGBA", (fw + pad * 2, fh + pad * 2), (0, 0, 0, 0))
    ox, oy = pad, pad

    # Oval alpha, feathered so the photo meets the wreath cleanly.
    alpha = Image.new("L", face.size, 0)
    ImageDraw.Draw(alpha).ellipse((2, 2, fw - 3, fh - 3), fill=255)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.4))
    face_rgba = face.convert("RGBA")
    face_rgba.putalpha(alpha)

    # Wreath band just outside the oval.
    wreath = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    wd = ImageDraw.Draw(wreath)
    rng = np.random.default_rng(12)
    cx, cy = canvas.size[0] / 2, canvas.size[1] / 2
    inner_rx, inner_ry = fw / 2 - 1, fh / 2 - 1
    outer_rx, outer_ry = canvas.size[0] / 2 - 2, canvas.size[1] / 2 - 2
    for i in range(360):
        ang = i / 360 * 2 * math.pi
        t = 0.35 + 0.55 * ((i % 3) / 2)
        rx = inner_rx + (outer_rx - inner_rx) * t
        ry = inner_ry + (outer_ry - inner_ry) * t
        x = cx + rx * math.cos(ang)
        y = cy + ry * math.sin(ang)
        shade = (14, 72, 34, 255) if i % 2 == 0 else (22, 96, 44, 255)
        nr = 3.2 + (i % 4)
        wd.ellipse([x - nr, y - nr * 0.7, x + nr, y + nr * 0.7], fill=shade)
    wreath = wreath.filter(ImageFilter.GaussianBlur(radius=1.2))
    wd = ImageDraw.Draw(wreath)
    for i in range(70):
        ang = rng.uniform(0, 2 * math.pi)
        t = rng.uniform(0.25, 0.95)
        rx = inner_rx + (outer_rx - inner_rx) * t
        ry = inner_ry + (outer_ry - inner_ry) * t
        x = cx + rx * math.cos(ang)
        y = cy + ry * math.sin(ang)
        nr = float(rng.uniform(1.6, 3.4))
        shade = (52, 130, 64, 220) if i % 2 == 0 else (8, 46, 22, 220)
        wd.ellipse([x - nr, y - nr * 0.55, x + nr, y + nr * 0.55], fill=shade)
    canvas.alpha_composite(wreath)
    canvas.paste(face_rgba, (ox, oy), face_rgba)

    draw = ImageDraw.Draw(canvas)
    draw.ellipse(
        [ox + 1, oy + 1, ox + fw - 2, oy + fh - 2],
        outline=(212, 168, 64, 255),
        width=max(2, scale),
    )
    # A few berries on the wreath, clear of the faces.
    for ang in (0.15, 1.15, 2.2, 3.3, 4.4, 5.4):
        rx = (inner_rx + outer_rx) * 0.5
        ry = (inner_ry + outer_ry) * 0.5
        bx = cx + rx * math.cos(ang)
        by = cy + ry * math.sin(ang)
        draw.ellipse([bx - 3.2, by - 3.2, bx + 3.2, by + 3.2], fill=(186, 28, 38, 255))

    display = canvas.resize((canvas.size[0] // scale, canvas.size[1] // scale), Image.Resampling.LANCZOS)
    # Clockwise: heads (top of the upright photo) land on template +x, the windshield.
    return display.transpose(Image.Transpose.ROTATE_270)


def paste_medallion(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    medal = make_medallion()
    mw, mh = medal.size
    cx, cy = HOOD_CENTER
    x0, y0 = int(cx - mw / 2), int(cy - mh / 2)
    sprite = np.asarray(medal, np.float32)
    # Clip to the canvas, then to the hood so the hat cannot spill into a gap.
    sy0, sx0 = max(0, -y0), max(0, -x0)
    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(H, y0 + mh), min(W, x0 + mw)
    sy1, sx1 = sy0 + (y1c - y0c), sx0 + (x1c - x0c)
    if y1c <= y0c or x1c <= x0c:
        return rgb
    patch = sprite[sy0:sy1, sx0:sx1]
    alpha = (patch[..., 3:4] / 255.0) * mask[y0c:y1c, x0c:x1c, None]
    dst = rgb[y0c:y1c, x0c:x1c]
    rgb[y0c:y1c, x0c:x1c] = dst * (1 - alpha) + patch[..., :3] * alpha
    return rgb


def compose(template_rgb_alpha: np.ndarray, color: np.ndarray, mask: np.ndarray) -> Image.Image:
    """RGB in the paintable islands, transparent everywhere else."""
    out = np.zeros((H, W, 4), np.uint8)
    painted = np.clip(color, 0, 255).astype(np.uint8)
    out[..., :3] = np.where(mask[..., None], painted, 0)
    # Soften the island edge by one texel using the template's own alpha,
    # but never open a pixel the mask says is a gap.
    tmpl_a = template_rgb_alpha[..., 3]
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    # Keep template antialiasing on the fringe just outside the hard mask.
    fringe = (tmpl_a > 0) & (~mask)
    # Only where the template texel is actually a light edge, not black-with-alpha.
    lum = template_rgb_alpha[..., :3].mean(axis=2)
    fringe &= lum > 40
    alpha = np.where(fringe, tmpl_a, alpha).astype(np.uint8)
    out[..., 3] = alpha
    # Fringe RGB: blend a little of the nearest painted color by leaving the
    # template edge dark-green rather than white.
    out[..., :3][fringe] = (6, 20, 16)
    return Image.fromarray(out, "RGBA")


def build_variant(bg: np.ndarray, strands, mask, template, variant: int) -> Image.Image:
    body = [s for s in strands if s.wire != "ring"]
    ring = [s for s in strands if s.wire == "ring"]
    lit = render_lights(bg, body, mask, variant)
    lit = paste_medallion(lit, mask)
    if ring:
        lit = render_lights(lit, ring, mask, variant)
    lit = np.where(mask[..., None], lit, 0)
    return compose(template, lit, mask)


def verify(path: Path) -> dict:
    im = Image.open(path)
    raw = path.read_bytes()
    name = path.name
    # Tesla allows letters, digits, underscore, dash, and space, plus the .png suffix.
    ok_name = bool(re.fullmatch(r"[A-Za-z0-9_\- ]{1,26}\.png", name)) and len(name) <= 30
    info = {
        "path": str(path.relative_to(ROOT.parent)) if path.is_relative_to(ROOT.parent) else str(path),
        "format": im.format,
        "size": im.size,
        "mode": im.mode,
        "bytes": len(raw),
        "name_ok": ok_name,
        "dims_ok": im.size == (W, H),
        "under_1mb": len(raw) <= 1_000_000,
        "png": im.format == "PNG",
    }
    info["ok"] = info["png"] and info["dims_ok"] and info["under_1mb"] and info["name_ok"]
    return info


def coverage_report(wrap: Image.Image, mask: np.ndarray) -> str:
    arr = np.asarray(wrap)
    rgb = arr[..., :3]
    painted = rgb.max(axis=2) > 8
    inside = painted[mask].mean()
    spill = painted[~mask].mean()
    return f"paintable filled {inside:.1%}, spill outside mask {spill:.1%}"


def save_previews(template: Image.Image, wraps: list[Image.Image], mask: np.ndarray) -> list[Path]:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    labels = [
        "template",
        "Christmas_Lights_1  (all lit)",
        "Christmas_Lights_2  (phase A)",
        "Christmas_Lights_3  (phase B)",
    ]
    images = [template.convert("RGBA")] + wraps

    cell_w, cell_h = 512, 384
    label_h = 22
    sheet = Image.new("RGB", (cell_w * 2, (cell_h + label_h) * 2), (12, 14, 16))
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(zip(images, labels)):
        col, row = i % 2, i // 2
        x = col * cell_w
        y = row * (cell_h + label_h)
        draw.text((x + 8, y + 4), label, fill=(240, 236, 220))
        thumb = im.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y + label_h))
    comparison = PREVIEW_DIR / "Christmas_Lights_comparison.png"
    sheet.save(comparison, optimize=True)

    # Overlay: wrap color with the template island outline, to confirm the
    # design stays on the panels.
    overlay = wraps[0].convert("RGB")
    edge = mask ^ _erode(mask)
    ov = np.asarray(overlay).copy()
    ov[edge] = (255, 64, 96)
    overlay_path = PREVIEW_DIR / "Christmas_Lights_overlay.png"
    Image.fromarray(ov, "RGB").save(overlay_path, optimize=True)

    # Hood, rotated so template +x (windshield) is up: the portrait should
    # read right-side up, the way it does from in front of the truck.
    hx0, hy0, hx1, hy1 = 110, 200, 280, 570
    hood = wraps[0].crop((hx0, hy0, hx1, hy1)).transpose(Image.Transpose.ROTATE_90)
    hood = hood.resize((hood.width * 2, hood.height * 2), Image.Resampling.NEAREST)
    hood_path = PREVIEW_DIR / "Christmas_Lights_hood.png"
    hood.convert("RGB").save(hood_path, optimize=True)
    return [comparison, overlay_path, hood_path]


def _erode(mask: np.ndarray) -> np.ndarray:
    # 4-neighbor erosion, no scipy.
    m = mask
    return m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)


def main() -> None:
    template_im = Image.open(TEMPLATE).convert("RGBA")
    if template_im.size != (W, H):
        raise SystemExit(f"template is {template_im.size}, expected {(W, H)}")
    template = np.asarray(template_im)
    mask = load_mask(template_im)
    panels = panel_masks(mask)
    # Anything paintable must belong to a panel. Pixels the gates missed stay
    # in `mask` so they still get background instead of leftover white.
    owned = np.zeros_like(mask)
    for name, sel in panels.items():
        owned |= sel
        print(f"  panel {name:10} {int(sel.sum()):6d} px")
    missed = int((mask & ~owned).sum())
    if missed:
        print(f"  unassigned paintable pixels: {missed}")
    place_hood(panels["hood"])

    rng = np.random.default_rng(20251225)
    bg = paint_background(panels, mask, rng)
    paint_snow(bg, mask, rng)
    strands = build_strands(panels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wraps = []
    reports = []
    for variant, name in ((1, "Christmas_Lights_1.png"), (2, "Christmas_Lights_2.png"), (3, "Christmas_Lights_3.png")):
        image = build_variant(bg, strands, mask, template, variant)
        path = OUT_DIR / name
        image.save(path, format="PNG", optimize=True, compress_level=9)
        info = verify(path)
        reports.append(info)
        wraps.append(image)
        print(f"{name}: {info['bytes']/1024:.1f} KB  {image.size} {image.mode}  {coverage_report(image, mask)}  ok={info['ok']}")

    previews = save_previews(template_im, wraps, mask)
    for p in previews:
        print(f"preview {p.relative_to(ROOT)}  {p.stat().st_size/1024:.1f} KB")

    if not all(r["ok"] for r in reports):
        raise SystemExit("a wrap failed the Paint Shop checks")


if __name__ == "__main__":
    main()

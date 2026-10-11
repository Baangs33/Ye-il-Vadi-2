"""Prosedürel dokular (numpy + PIL) ve malzeme kayıtları.

Hiçbir görsel dosya kullanılmaz; tüm dokular açılışta üretilir ve önbelleğe ('onbellek/') yazılır,
böylece sonraki açılışlar hızlıdır.
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from panda3d.core import (Texture, SamplerState, RenderState, TextureAttrib, CullFaceAttrib, PTA_uchar, CPTA_uchar)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'onbellek')
SIZE = 512


# =========================================================================== periyodik gürültü
def pnoise(size, cx, cy=None, seed=0):
    cy = cx if cy is None else cy
    r = np.random.default_rng(seed).random((cy, cx))
    x = np.arange(size) * cx / size
    y = np.arange(size) * cy / size
    ix0 = np.floor(x).astype(int); fx = x - ix0; fx = fx * fx * (3 - 2 * fx); ix1 = (ix0 + 1) % cx
    iy0 = np.floor(y).astype(int); fy = y - iy0; fy = fy * fy * (3 - 2 * fy); iy1 = (iy0 + 1) % cy
    r0, r1 = r[iy0], r[iy1]
    a = r0[:, ix0] * (1 - fx) + r0[:, ix1] * fx
    b = r1[:, ix0] * (1 - fx) + r1[:, ix1] * fx
    return a * (1 - fy[:, None]) + b * fy[:, None]


def pfbm(size, cx, cy=None, octaves=5, seed=0, gain=0.5):
    cy = cx if cy is None else cy
    tot, amp, norm = 0, 1.0, 0
    for o in range(octaves):
        tot = tot + pnoise(size, cx * 2 ** o, cy * 2 ** o, seed + o * 13) * amp
        norm += amp
        amp *= gain
    return tot / norm


def pridged(size, cx, cy=None, octaves=4, seed=0):
    cy = cx if cy is None else cy
    tot, amp, norm = 0, 1.0, 0
    for o in range(octaves):
        n = 1 - np.abs(pnoise(size, cx * 2 ** o, cy * 2 ** o, seed + o * 7) * 2 - 1)
        tot = tot + n ** 2 * amp
        norm += amp
        amp *= 0.5
    return tot / norm


def lerp3(a, b, t):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return a * (1 - t[..., None]) + b * t[..., None]


def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def to_img(rgb, alpha=None):
    rgb = np.clip(rgb, 0, 1)
    if alpha is None:
        alpha = np.ones(rgb.shape[:2])
    arr = np.dstack([rgb, np.clip(alpha, 0, 1)])
    return Image.fromarray((arr * 255).astype(np.uint8), 'RGBA')


# =========================================================================== dokular
def t_grass():
    s = SIZE
    n1 = pfbm(s, 4, seed=1)
    n2 = pfbm(s, 16, seed=2)
    n3 = pnoise(s, 128, seed=3)
    base = lerp3((0.17, 0.25, 0.09), (0.27, 0.34, 0.13), n1)
    base = lerp3(base, (0.36, 0.36, 0.18), ss(0.6, 0.8, n2) * 0.6)
    base *= (0.8 + 0.4 * n3)[..., None]
    img = to_img(base)
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(4)
    for _ in range(9000):
        x, y = r.uniform(0, s), r.uniform(0, s)
        L = r.uniform(6, 18)
        a = r.uniform(-0.5, 0.5) - math.pi / 2
        g = r.uniform(0.85, 1.1)
        c = (int(52 * g), int(72 * g), int(28 * g), 255) if r.random() < 0.85 else (int(92 * g), int(88 * g), int(48 * g), 255)
        for ox in (-s, 0, s):
            for oy in (-s, 0, s):
                d.line([(x + ox, y + oy), (x + ox + math.cos(a) * L, y + oy + math.sin(a) * L)], fill=c, width=2)
    return img.filter(ImageFilter.GaussianBlur(1.0))


def t_dirt():
    s = SIZE
    n1 = pfbm(s, 6, seed=11)
    n2 = pnoise(s, 96, seed=12)
    base = lerp3((0.30, 0.22, 0.15), (0.45, 0.35, 0.24), n1) * (0.85 + 0.3 * n2)[..., None]
    img = to_img(base)
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(13)
    for _ in range(900):
        x, y = r.uniform(0, s), r.uniform(0, s)
        rad = r.uniform(1.5, 6)
        g = r.uniform(0.25, 0.45)
        for ox in (-s, 0, s):
            for oy in (-s, 0, s):
                d.ellipse([x + ox - rad, y + oy - rad * 0.8, x + ox + rad, y + oy + rad * 0.8],
                          fill=(int(255 * g), int(240 * g * 0.95), int(220 * g * 0.88), 255))
                d.ellipse([x + ox - rad * 0.5, y + oy - rad * 0.6, x + ox + rad * 0.2, y + oy - rad * 0.1],
                          fill=(int(255 * min(1, g + 0.1)),) * 3 + (255,))
    return img.filter(ImageFilter.GaussianBlur(0.7))


def t_rock():
    s = SIZE
    n1 = pfbm(s, 4, seed=21)
    cr = pridged(s, 6, seed=22)
    n3 = pnoise(s, 64, seed=23)
    strata = (np.sin(np.linspace(0, 2 * np.pi * 9, s))[:, None] * 0.5 + 0.5) * 0.08
    base = lerp3((0.28, 0.27, 0.25), (0.47, 0.45, 0.42), n1)
    base *= (1 - ss(0.75, 0.95, cr) * 0.55)[..., None]
    base *= (0.85 + 0.3 * n3 + strata)[..., None]
    lich = ss(0.68, 0.8, pfbm(s, 8, seed=24))
    base = lerp3(base, (0.45, 0.48, 0.30), lich * 0.5)
    return to_img(base)


def t_sand():
    s = SIZE
    n1 = pfbm(s, 5, seed=31)
    n2 = pnoise(s, 200, seed=32)
    ripple = np.sin((np.arange(s)[:, None] * 0.12 + pfbm(s, 3, seed=33) * 20)) * 0.04
    base = lerp3((0.50, 0.45, 0.34), (0.64, 0.58, 0.45), n1) * (0.9 + 0.2 * n2 + ripple)[..., None]
    return to_img(base)


def t_bark():
    s = SIZE
    v = pfbm(s, 20, 3, seed=41)
    cr = pridged(s, 14, 2, seed=42)
    fine = pnoise(s, 160, 40, seed=43)
    base = lerp3((0.22, 0.16, 0.11), (0.42, 0.33, 0.24), v)
    base *= (1 - ss(0.55, 0.9, cr) * 0.65)[..., None]
    base *= (0.85 + 0.3 * fine)[..., None]
    moss = ss(0.65, 0.85, pfbm(s, 6, seed=44))
    base = lerp3(base, (0.25, 0.32, 0.12), moss * 0.45)
    return to_img(base)


def t_birch():
    s = SIZE
    base = np.ones((s, s, 3)) * np.array([0.88, 0.86, 0.80])
    base *= (0.9 + 0.1 * pnoise(s, 30, 6, seed=51))[..., None]
    img = to_img(base)
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(52)
    for _ in range(60):
        x, y = r.uniform(0, s), r.uniform(0, s)
        w, h = r.uniform(10, 50), r.uniform(2, 7)
        for ox in (-s, 0, s):
            for oy in (-s, 0, s):
                d.ellipse([x + ox, y + oy, x + ox + w, y + oy + h], fill=(25, 22, 20, 255))
    return img.filter(ImageFilter.GaussianBlur(0.5))


def t_planks():
    s = SIZE
    boards = 8
    bh = s // boards
    out = np.zeros((s, s, 3))
    r = np.random.default_rng(61)
    grain = pfbm(s, 2, 48, seed=62)
    knots = pnoise(s, 10, 10, seed=63)
    for b in range(boards):
        tone = r.uniform(0.82, 1.12)
        base = np.array([0.50, 0.36, 0.22]) * tone
        rows = slice(b * bh, (b + 1) * bh)
        g = grain[rows]
        col = lerp3(base * 0.8, base * 1.15, g)
        out[rows] = col
        out[b * bh:b * bh + 2] *= 0.35
        # uç birleşimi
        cut = int(r.uniform(0, s))
        out[rows, cut:cut + 2] *= 0.4
    out *= (1 - ss(0.82, 0.95, knots) * 0.4)[..., None]
    return to_img(out)


def t_logwall():
    """Yatay kütüklerden duvar (her kütük silindir gibi gölgelenir)."""
    s = SIZE
    logs = 4
    lh = s / logs
    y = np.arange(s)
    t = (y % lh) / lh
    shade = np.sin(t * np.pi) ** 0.6
    bark = np.asarray(t_bark().resize((s, s)).transpose(Image.ROTATE_90))[..., :3] / 255.0
    col = bark * (0.35 + 0.75 * shade)[:, None, None]
    gap = (t < 0.04) | (t > 0.96)
    col[gap] = col[gap] * 0.3 + np.array([0.18, 0.15, 0.12]) * 0.7
    return to_img(col)


def t_stonewall():
    s = SIZE
    r = np.random.default_rng(71)
    pts = r.uniform(0, s, (70, 2))
    pts[:, 1] = (pts[:, 1] // (s / 7)) * (s / 7) + r.uniform(10, s / 7 - 10, len(pts))
    yy, xx = np.mgrid[0:s, 0:s]
    d1 = np.full((s, s), 1e9); d2 = np.full((s, s), 1e9); idx = np.zeros((s, s), int)
    for i, (px, py) in enumerate(pts):
        for ox in (-s, 0, s):
            for oy in (-s, 0, s):
                dx = (xx - px - ox) * 0.75
                dy = yy - py - oy
                d = np.sqrt(dx * dx + dy * dy * 1.6)
                closer = d < d1
                d2 = np.where(closer, d1, np.minimum(d2, d))
                idx = np.where(closer, i, idx)
                d1 = np.where(closer, d, d1)
    edge = d2 - d1
    tones = r.uniform(0.75, 1.2, len(pts))
    hue = r.uniform(-0.04, 0.04, len(pts))
    rock = np.asarray(t_rock())[..., :3] / 255.0
    col = rock * tones[idx][..., None]
    col[..., 0] += hue[idx]
    col *= (0.55 + 0.45 * ss(1, 6, edge))[..., None]
    mortar = edge < 3.0
    col[mortar] = np.array([0.55, 0.52, 0.47]) * (0.8 + 0.2 * pnoise(s, 64, seed=72)[mortar])[..., None]
    return to_img(col)


def t_shingles():
    s = SIZE
    rows = 10
    rh = s / rows
    out = np.zeros((s, s, 3))
    r = np.random.default_rng(81)
    grain = pfbm(s, 40, 3, seed=82)
    for i in range(rows):
        y0 = int(i * rh); y1 = int((i + 1) * rh)
        off = (i % 2) * 0.5
        n = 8
        for k in range(n + 1):
            x0 = int(((k + off) / n) * s) % s
            w = s // n
            tone = r.uniform(0.75, 1.15)
            cols = (np.arange(x0, x0 + w)) % s
            base = np.array([0.36, 0.27, 0.20]) * tone
            seg = lerp3(base * 0.8, base * 1.1, grain[y0:y1][:, cols])
            grad = np.linspace(1.0, 0.65, y1 - y0)[:, None, None]
            out[y0:y1][:, cols] = seg * grad
            out[y0:y1, cols[0] % s] *= 0.4
    return to_img(out)


def t_rooftiles():
    s = SIZE
    rows = 8
    rh = s / rows
    out = np.zeros((s, s, 3))
    y = np.arange(s)[:, None]
    x = np.arange(s)[None, :]
    n = 8
    for i in range(rows):
        off = (i % 2) * 0.5
        u = ((x / s * n) + off) % 1.0
        arch = np.sqrt(np.clip(1 - (u * 2 - 1) ** 2, 0, 1))
        t = (y - i * rh) / rh
        sel = ((y >= i * rh) & (y < (i + 1) * rh)).astype(float)
        shade = (0.55 + 0.45 * arch) * (1.0 - 0.35 * t)
        out += sel[..., None] * np.array([0.62, 0.27, 0.16]) * shade[..., None]
    out *= (0.88 + 0.24 * pfbm(s, 8, seed=91))[..., None]
    return to_img(out)


def t_plaster():
    s = SIZE
    base = lerp3((0.80, 0.76, 0.66), (0.92, 0.89, 0.80), pfbm(s, 6, seed=101))
    base *= (0.94 + 0.08 * pnoise(s, 128, seed=102))[..., None]
    stain = ss(0.7, 0.9, pfbm(s, 3, seed=103))
    base = lerp3(base, base * 0.82, stain)
    return to_img(base)


def t_soil():
    s = SIZE
    y = np.arange(s)[:, None]
    furrow = (np.sin(y / s * 2 * np.pi * 4) * 0.5 + 0.5)
    base = lerp3((0.20, 0.13, 0.08), (0.42, 0.30, 0.19), furrow * 0.7 + pfbm(s, 8, seed=111) * 0.3)
    base *= (0.85 + 0.3 * pnoise(s, 128, seed=112))[..., None]
    return to_img(base)


def t_canvas():
    s = SIZE
    x = np.arange(s)
    weave = (np.sin(x * 0.9)[None, :] * np.sin(x * 0.9)[:, None]) * 0.05
    base = lerp3((0.62, 0.56, 0.42), (0.74, 0.68, 0.52), pfbm(s, 4, seed=121)) * (1 + weave)[..., None]
    return to_img(base)


def t_metal():
    s = SIZE
    base = np.ones((s, s, 3)) * 0.62
    base *= (0.9 + 0.1 * pfbm(s, 2, 60, seed=131))[..., None]
    rust = ss(0.72, 0.88, pfbm(s, 6, seed=132))
    base = lerp3(base, (0.45, 0.25, 0.12), rust * 0.6)
    return to_img(base)


def t_ironore():
    s = SIZE
    rock = np.asarray(t_rock())[..., :3] / 255.0
    spots = ss(0.66, 0.74, pfbm(s, 14, seed=141))
    col = lerp3(rock * 0.85, (0.55, 0.28, 0.14), spots)
    glints = ss(0.93, 0.97, pnoise(s, 90, seed=142))
    col = lerp3(col, (0.75, 0.72, 0.68), glints * spots)
    return to_img(col)


def t_white():
    return Image.new('RGBA', (16, 16), (255, 255, 255, 255))


# --- alfa kesimli (yaprak / çim) dokular
def _leafshape(d, cx, cy, L, W, ang, col):
    pts = []
    for i in range(14):
        t = i / 13
        w = math.sin(t * math.pi) ** 0.8 * W
        pts.append((t * L, w))
    for i in range(13, -1, -1):
        t = i / 13
        w = math.sin(t * math.pi) ** 0.8 * W
        pts.append((t * L, -w))
    ca, sa = math.cos(ang), math.sin(ang)
    poly = [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in pts]
    d.polygon(poly, fill=col)
    d.line([(cx, cy), (cx + L * ca * 0.9, cy + L * sa * 0.9)], fill=tuple(int(c * 0.75) for c in col[:3]) + (255,), width=2)


def t_leaves(kind='oak'):
    S2 = 1024
    img = Image.new('RGBA', (S2, S2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(151 if kind == 'oak' else 152)
    # dallar
    for _ in range(7):
        x0, y0 = S2 * 0.5 + r.uniform(-80, 80), S2 * 0.98
        x1, y1 = r.uniform(0.1, 0.9) * S2, r.uniform(0.1, 0.5) * S2
        d.line([(x0, y0), (x1, y1)], fill=(70, 52, 35, 255), width=int(r.uniform(5, 10)))
    if kind == 'oak':
        base = np.array([62, 98, 38])
        n, L = 260, (60, 110)
    else:   # huş: daha açık, küçük
        base = np.array([88, 128, 46])
        n, L = 340, (45, 80)
    for _ in range(n):
        a = r.uniform(0, 2 * np.pi)
        rr = math.sqrt(r.uniform(0, 1)) * S2 * 0.46
        cx, cy = S2 * 0.5 + math.cos(a) * rr, S2 * 0.45 + math.sin(a) * rr * 0.9
        t = r.uniform(0.65, 1.3)
        col = tuple(int(min(255, c * t)) for c in base) + (255,)
        if r.random() < 0.08:
            col = (int(150 * t), int(140 * t), int(50 * t), 255)
        _leafshape(d, cx, cy, r.uniform(*L), r.uniform(L[0] * 0.25, L[0] * 0.4), r.uniform(0, 2 * np.pi), col)
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def t_pine():
    S2 = 1024
    img = Image.new('RGBA', (S2, S2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(161)
    # dal: alttan (gövde tarafı) yukarı doğru, iğneler yanlara
    d.line([(S2 / 2, S2), (S2 / 2, 10)], fill=(70, 50, 34, 255), width=10)
    for k in range(14):
        y = S2 - (k + 1) * S2 / 15
        side = 1 if k % 2 else -1
        L = (1 - k / 15) * S2 * 0.38 + 60
        d.line([(S2 / 2, y), (S2 / 2 + side * L, y - L * 0.35)], fill=(64, 48, 32, 255), width=5)
        for j in range(int(L / 6)):
            t = j / (L / 6)
            px, py = S2 / 2 + side * L * t, y - L * 0.35 * t
            for sgn in (-1, 1):
                ln = r.uniform(26, 40) * (1 - 0.4 * t)
                a = math.atan2(-0.35, side) + sgn * r.uniform(0.7, 1.1)
                g = r.uniform(0.7, 1.15)
                d.line([(px, py), (px + math.cos(a) * ln * 1.3, py + math.sin(a) * ln * 1.3)],
                       fill=(int(34 * g), int(62 * g), int(40 * g), 255), width=6)
    for j in range(80):
        y = r.uniform(0, S2)
        for sgn in (-1, 1):
            a = -math.pi / 2 + sgn * r.uniform(0.5, 1.2)
            d.line([(S2 / 2, y), (S2 / 2 + math.cos(a) * 40, y + math.sin(a) * 40)], fill=(38, 66, 43, 255), width=6)
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def t_grassblades():
    S2 = 1024
    img = Image.new('RGBA', (S2, S2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(171)
    for _ in range(150):
        x = float(np.clip(S2 / 2 + r.normal(0, S2 * 0.2), 40, S2 - 40))
        h = (0.22 + 0.76 * r.random() ** 1.6) * S2
        w = r.uniform(6, 13)
        bend = r.uniform(-120, 120)
        g = r.uniform(0.75, 1.2)
        dry = r.random() < 0.12
        top = (int(150 * g), int(150 * g), int(70 * g)) if dry else (int(105 * g), int(150 * g), int(55 * g))
        bot = (int(40 * g), int(70 * g), int(25 * g))
        steps = 12
        for i in range(steps):
            t0, t1 = i / steps, (i + 1) / steps
            x0 = x + bend * t0 ** 2; x1 = x + bend * t1 ** 2
            w0 = w * (1 - t0); w1 = w * (1 - t1)
            y0 = S2 - h * t0; y1 = S2 - h * t1
            c = tuple(int(bot[k] + (top[k] - bot[k]) * t0) for k in range(3)) + (255,)
            d.polygon([(x0 - w0, y0), (x0 + w0, y0), (x1 + w1, y1), (x1 - w1, y1)], fill=c)
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def t_flowers():
    S2 = 1024
    img = t_grassblades().resize((S2, S2))
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(181)
    cols = [(240, 230, 240), (250, 210, 60), (200, 80, 160), (120, 120, 230), (240, 90, 60)]
    for _ in range(26):
        x, y = r.uniform(60, S2 - 60), r.uniform(S2 * 0.15, S2 * 0.6)
        c = cols[r.integers(len(cols))]
        d.line([(x, y), (x + r.uniform(-20, 20), S2)], fill=(60, 100, 40, 255), width=4)
        for k in range(6):
            a = k / 6 * 2 * math.pi
            d.ellipse([x + math.cos(a) * 14 - 11, y + math.sin(a) * 14 - 11, x + math.cos(a) * 14 + 11,
                       y + math.sin(a) * 14 + 11], fill=c + (255,))
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(240, 200, 40, 255))
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def t_noise():
    """Arazide tekrarı kırmak için büyük ölçekli gürültü (gri)."""
    s = SIZE
    n = pfbm(s, 4, seed=191, octaves=6)
    return to_img(np.dstack([n, pfbm(s, 8, seed=192), pfbm(s, 2, seed=193)]))


TEXTURES = {
    'grass': (t_grass, True), 'dirt': (t_dirt, True), 'rock': (t_rock, True), 'sand': (t_sand, True),
    'bark': (t_bark, True), 'birch': (t_birch, True), 'planks': (t_planks, True), 'logwall': (t_logwall, True),
    'stonewall': (t_stonewall, True), 'shingles': (t_shingles, True), 'rooftiles': (t_rooftiles, True),
    'plaster': (t_plaster, True), 'soil': (t_soil, True), 'canvas': (t_canvas, True), 'metal': (t_metal, True),
    'ironore': (t_ironore, True), 'white': (t_white, True), 'noise': (t_noise, True),
    'leaves_oak': (lambda: t_leaves('oak'), False), 'leaves_birch': (lambda: t_leaves('birch'), False),
    'pine': (t_pine, False), 'grassblades': (t_grassblades, False), 'flowers': (t_flowers, False),
}

_tex = {}


def get_texture(name):
    if name in _tex:
        return _tex[name]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + '.png')
    if os.path.exists(path):
        img = Image.open(path).convert('RGBA')
    else:
        fn, _ = TEXTURES[name]
        img = fn()
        img.save(path)
    repeat = TEXTURES[name][1]
    arr = np.asarray(img.convert('RGBA'))[::-1]   # panda: satırlar alttan yukarı
    h, w = arr.shape[:2]
    tex = Texture(name)
    tex.setup_2d_texture(w, h, Texture.T_unsigned_byte, Texture.F_rgba8)
    # panda BGRA bekler
    bgra = arr[..., [2, 1, 0, 3]].copy()
    tex.set_ram_image(bgra.tobytes())
    if not repeat:
        _cutout_mips(tex, np.asarray(img.convert('RGBA')))
    tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_anisotropic_degree(8)
    mode = SamplerState.WM_repeat if repeat else SamplerState.WM_clamp
    tex.set_wrap_u(mode)
    tex.set_wrap_v(mode)
    _tex[name] = tex
    return tex


def _cutout_mips(tex, rgba):
    """Alfa kesimli dokularda uzaktaki yaprakların kaybolmaması için kapsama korumalı mipmap."""
    TH = 0.42
    a0 = rgba[..., 3] / 255.0
    cov0 = (a0 > TH).mean()
    img = Image.fromarray(rgba, 'RGBA')
    w, h = img.size
    level = 1
    while w > 1 or h > 1:
        w, h = max(1, w // 2), max(1, h // 2)
        im = img.resize((w, h), Image.BOX)
        arr = np.asarray(im).astype(np.float64)
        a = arr[..., 3] / 255.0
        lo, hi = 0.5, 6.0
        for _ in range(14):
            mid = (lo + hi) / 2
            if (np.clip(a * mid, 0, 1) > TH).mean() < cov0:
                lo = mid
            else:
                hi = mid
        arr[..., 3] = np.clip(a * hi, 0, 1) * 255
        # renkleri alfa ağırlıklı ortalamadan gelen koyulaşmaya karşı düzelt
        arr = arr.astype(np.uint8)[::-1]
        data = arr[..., [2, 1, 0, 3]].copy().tobytes()
        pta = PTA_uchar.empty_array(len(data))
        memoryview(pta)[:] = data
        tex.set_ram_mipmap_image(level, CPTA_uchar(pta))
        level += 1


def prewarm():
    for k in TEXTURES:
        get_texture(k)


# =========================================================================== malzemeler
# malzeme adı -> doku adı (aynı doku farklı malzemelerde kullanılabilir)
MATERIALS = {
    'bark': 'bark', 'birch': 'birch', 'leaves_oak': 'leaves_oak', 'leaves_birch': 'leaves_birch', 'pine': 'pine',
    'rock': 'rock', 'ironore': 'ironore', 'planks': 'planks', 'logwall': 'logwall', 'stonewall': 'stonewall',
    'shingles': 'shingles', 'rooftiles': 'rooftiles', 'plaster': 'plaster', 'soil': 'soil', 'canvas': 'canvas',
    'metal': 'metal', 'paint': 'white', 'plain': 'white', 'grassblades': 'grassblades', 'flowers': 'flowers',
    'dirt': 'dirt', 'grass': 'grass', 'glass': 'white',
}
TWO_SIDED = {'leaves_oak', 'leaves_birch', 'pine', 'grassblades', 'flowers', 'canvas'}

_states = {}


def state(mat):
    if mat not in _states:
        st = RenderState.make(TextureAttrib.make(get_texture(MATERIALS.get(mat, 'white'))))
        if mat in TWO_SIDED:
            st = st.add_attrib(CullFaceAttrib.make(CullFaceAttrib.M_cull_none))
        _states[mat] = st
    return _states[mat]

"""İnşaat sistemi (The Forest tarzı): plan (hayalet) yerleştir, malzeme ekledikçe parçalar tek tek belirsin.

Her yapı tipi bir 'spec' üretir: sıralı parça listesi (her parça bir kaynak türünden belirli miktar ister),
duvar çarpışma çizgileri, zemin (üzerinde yürünebilir) ve etkileşim noktaları.
"""
import math
import numpy as np
from ursina import Entity, Vec3, color as ucolor
from panda3d.core import GeomNode, TransparencyAttrib

from config import *
from geometry import (Model, tube_tris, box_tris, cylinder_tris, cone_tris, sphere_tris, prism_tris, quad_tris, trs,
                      look_rot, make_geom, rng)
from shaders import world_shader, ghost_shader, glow_shader
from environment import SHADOW_MASK
import materials

LOG_R = 0.15
COURSE = LOG_R * 2 * 0.9


class Part:
    __slots__ = ('res', 'units', 'model', 'baked', 'tag')

    def __init__(self, res, units, model, tag=None):
        self.res, self.units, self.model, self.tag = res, units, model, tag
        self.baked = None


class Spec:
    def __init__(self, kind, w, d):
        self.kind = kind
        self.w, self.d = w, d
        self.parts = []
        self.walls = []          # (ax, az, bx, bz)
        self.floor = None        # (yarı genişlik, yarı derinlik, y)
        self.points = []         # (tür, x, y, z)
        self.height = 3.0
        self.lights = []         # (x, y, z, güç)
        self.fires = []          # (x, y, z)
        self.solid_circles = []  # (x, z, r)

    def add(self, res, units, model, tag=None):
        self.parts.append(Part(res, units, model, tag))

    def cost(self):
        c = {}
        for p in self.parts:
            if p.res:
                c[p.res] = c.get(p.res, 0) + p.units
        return c


# =========================================================================== geometri yardımcıları
def _md():
    return Model()


def log_piece(ax, az, bx, bz, y, r=LOG_R, ext=0.0, mat='bark'):
    md = _md()
    dx, dz = bx - ax, bz - az
    L = math.hypot(dx, dz)
    ux, uz = dx / L, dz / L
    a = np.array([ax - ux * ext, y, az - uz * ext])
    b = np.array([bx + ux * ext, y, bz + uz * ext])
    rr = rng()
    mid = (a + b) / 2 + np.array([0, rr.uniform(-0.01, 0.01), 0])
    t, n, uv = tube_tris([a, mid, b], [r, r * rr.uniform(0.93, 1.0), r * rr.uniform(0.9, 0.97)], 9)
    uv = uv.copy()
    uv[:, :, 0] *= 2 * math.pi * r / 1.2
    uv[:, :, 1] /= 1.2
    pr = np.zeros(t.shape[:2] + (4,)); pr[:, :, 2] = 0.8
    md[mat].add(t, (1, 1, 1, 1), normals=n, uv=uv, vparams=pr)
    for end, sgn in ((a, -1), (b, 1)):
        m = trs(tuple(end)) @ look_rot(ux * sgn, 0, uz * sgn)
        md['planks'].add(cylinder_tris(r * 0.97, r * 0.97, 0.012, 9), (1.25, 1.05, 0.8, 1), m=m, uv_scale=0.35)
    return md


def box_piece(mat, sx, sy, sz, pos, rot=(0, 0, 0), color=(1, 1, 1, 1), uv_scale=1.0, spec=0.0, bump=0.4, bottom=False):
    md = _md()
    md[mat].add(box_tris(sx, sy, sz, bottom=bottom), color, m=trs(pos, rot), uv_scale=uv_scale, spec=spec, bump=bump)
    return md


def _intervals(L, cuts):
    segs = [(0.0, L)]
    for c0, c1 in cuts:
        out = []
        for a, b in segs:
            if c1 <= a or c0 >= b:
                out.append((a, b))
                continue
            if c0 > a:
                out.append((a, c0))
            if c1 < b:
                out.append((c1, b))
        segs = out
    return [s for s in segs if s[1] - s[0] > 0.25]


def log_wall(spec, ax, az, bx, bz, H, openings=(), ext=0.28, phase=0.0, res='kutuk', mat='bark'):
    """Üst üste kütüklerden duvar. openings: (u0, u1, y0, y1) duvar boyunca metre."""
    L = math.hypot(bx - ax, bz - az)
    ux, uz = (bx - ax) / L, (bz - az) / L
    y = LOG_R + phase * COURSE * 0.5
    while y < H - LOG_R * 0.3:
        cuts = [(u0, u1) for (u0, u1, y0, y1) in openings if y0 - LOG_R * 0.5 < y < y1 + LOG_R * 0.5]
        segs = _intervals(L, cuts)
        for s0, s1 in segs:
            e0 = ext if s0 < 0.01 else 0.0
            e1 = ext if s1 > L - 0.01 else 0.0
            p0x, p0z = ax + ux * s0, az + uz * s0
            p1x, p1z = ax + ux * s1, az + uz * s1
            md = log_piece(p0x - ux * e0, p0z - uz * e0, p1x + ux * e1, p1z + uz * e1, y, mat=mat)
            spec.add(res, 1, md, 'wall')
        y += COURSE
    _wall_segments(spec, ax, az, bx, bz, openings, ux, uz, L)


def _wall_segments(spec, ax, az, bx, bz, openings, ux, uz, L):
    cuts = [(u0, u1) for (u0, u1, y0, y1) in openings if y0 < 0.3]   # yalnız kapılar geçit verir
    for s0, s1 in _intervals(L, cuts) or []:
        spec.walls.append((ax + ux * s0, az + uz * s0, ax + ux * s1, az + uz * s1))


def stone_wall(spec, ax, az, bx, bz, H, thick=0.4, openings=(), res='tas', mat='stonewall', block=1.3, ch=0.55,
               color=(1, 1, 1, 1)):
    L = math.hypot(bx - ax, bz - az)
    ux, uz = (bx - ax) / L, (bz - az) / L
    ang = math.degrees(math.atan2(ux, uz)) - 90
    y = 0.0
    row = 0
    while y < H - 0.05:
        h = min(ch, H - y)
        cuts = [(u0, u1) for (u0, u1, y0, y1) in openings if y0 - 0.01 < y + h / 2 < y1 + 0.01]
        for s0, s1 in _intervals(L, cuts):
            # bloklara böl (sıra ofsetli)
            pos = s0
            first = block * (0.5 if row % 2 else 1.0)
            edges = [s0]
            nxt = s0 + first
            while nxt < s1 - 0.3:
                edges.append(nxt)
                nxt += block
            edges.append(s1)
            for e0, e1 in zip(edges[:-1], edges[1:]):
                cx, cz = ax + ux * (e0 + e1) / 2, az + uz * (e0 + e1) / 2
                md = box_piece(mat, e1 - e0, h, thick, (cx, y + h / 2, cz), (0, ang + 90, 0), color=color,
                               uv_scale=2.4, bump=1.0)
                spec.add(res, 1, md, 'wall')
        y += h
        row += 1
    _wall_segments(spec, ax, az, bx, bz, openings, ux, uz, L)


def gable_logs(spec, W, D, H, rise, z_face, res='kutuk'):
    """Ön/arka alınlık üçgenini kütüklerle doldur (x boyunca)."""
    y = H + LOG_R
    while y < H + rise - 0.2:
        k = 1 - (y - H) / rise
        half = W / 2 * k
        if half < 0.3:
            break
        spec.add(res, 1, log_piece(-half, z_face, half, z_face, y), 'gable')
        y += COURSE


def gable_roof(spec, W, D, H, rise, mat='shingles', res='kutuk', strip=1.1, over=0.45, color=(1, 1, 1, 1), thick=0.12):
    ang = math.degrees(math.atan2(rise, W / 2))
    slope_len = math.hypot(W / 2, rise) + over
    n = max(2, int(math.ceil((D + 2 * over) / strip)))
    sl = (D + 2 * over) / n
    for sgn in (-1, 1):
        for i in range(n):
            z = -D / 2 - over + (i + 0.5) * sl
            cx = sgn * (W / 4 + over * 0.5 * math.cos(math.radians(ang)) * 0.0)
            # panelin merkezini eğim boyunca yerleştir
            mid_x = sgn * (W / 2 + over * math.cos(math.radians(ang))) / 2
            mid_y = H + rise / 2 - over * math.sin(math.radians(ang)) / 2 + thick
            md = _md()
            m = trs((sgn * (slope_len / 2 - over) * math.cos(math.radians(ang)) * 1.0 if False else mid_x, mid_y, z),
                    (0, 0, -sgn * ang))
            md[mat].add(box_tris(slope_len, thick, sl * 1.02), color, m=m, uv_scale=1.6, bump=0.9)
            spec.add(res, 1, md, 'roof')
    md = _md()
    md['planks'].add(box_tris(0.25, 0.25, D + 2 * over + 0.05), (0.7, 0.6, 0.5, 1),
                     m=trs((0, H + rise + thick * 0.6, 0), (0, 0, 45)), uv_scale=0.8)
    spec.add(None, 0, md, 'roof')


def plank_floor(spec, W, D, res='kutuk', pieces=3, y=0.0):
    for i in range(pieces):
        z0 = -D / 2 + i * D / pieces
        md = box_piece('planks', W, 0.12, D / pieces, (0, y - 0.06, z0 + D / pieces / 2), uv_scale=2.0, bump=0.6)
        spec.add(res, 1, md, 'floor')


def foundation(spec, W, D, depth, res='tas', mat='stonewall'):
    if depth < 0.05:
        return
    for (cx, cz, sx, sz) in [(0, -D / 2, W + 0.1, 0.35), (0, D / 2, W + 0.1, 0.35),
                             (-W / 2, 0, 0.35, D - 0.3), (W / 2, 0, 0.35, D - 0.3)]:
        md = box_piece(mat, sx, depth + 0.12, sz, (cx, -depth / 2 - 0.06, cz), uv_scale=2.4, bump=1.0)
        spec.add(res, max(1, int(round((max(sx, sz) * depth) / 2.5))), md, 'found')


def door(spec, x, z, rot, w=1.0, h=2.05, open_angle=-75):
    md = _md()
    a = math.radians(rot)
    for sx in (-1, 1):
        md['planks'].add(box_tris(0.1, h, 0.12, bottom=True), (0.8, 0.7, 0.6, 1),
                         m=trs((x, 0, z), (0, rot, 0)) @ trs((sx * (w / 2 + 0.05), 0, 0)), uv_scale=1.0)
    md['planks'].add(box_tris(w + 0.3, 0.12, 0.14), (0.8, 0.7, 0.6, 1), m=trs((x, 0, z), (0, rot, 0)) @ trs((0, h + 0.06, 0)),
                     uv_scale=1.0)
    leaf = trs((x, 0, z), (0, rot, 0)) @ trs((-w / 2, 0, 0), (0, open_angle, 0)) @ trs((w / 2, 0, 0))
    md['planks'].add(box_tris(w - 0.05, h - 0.05, 0.06, bottom=True), (0.75, 0.6, 0.48, 1), m=leaf, uv_scale=1.2)
    md['metal'].add(sphere_tris(0.035, 1), (0.6, 0.5, 0.2, 1), m=leaf @ trs((w * 0.38, 1.0, 0.05)), spec=0.9)
    spec.add(None, 0, md, 'deco')


def window(spec, x, y, z, rot, w=1.0, h=0.9, frame_col=(0.85, 0.78, 0.7, 1)):
    md = _md()
    M = trs((x, y, z), (0, rot, 0))
    t = 0.08
    for (px, py, sx, sy) in [(0, h / 2 + t / 2, w + 2 * t, t), (0, -h / 2 - t / 2, w + 2 * t, t),
                             (-w / 2 - t / 2, 0, t, h), (w / 2 + t / 2, 0, t, h), (0, 0, 0.05, h), (0, 0, w, 0.05)]:
        md['planks'].add(box_tris(sx, sy, 0.16), frame_col, m=M @ trs((px, py, 0)), uv_scale=0.8)
    md['glass'].add(box_tris(w, h, 0.02), (0.30, 0.38, 0.42, 1), m=M, spec=1.0, uv_mode='none')
    md['planks'].add(box_tris(w + 0.3, 0.06, 0.28), frame_col, m=M @ trs((0, -h / 2 - 0.1, 0.06)), uv_scale=0.8)
    spec.add(None, 0, md, 'deco')


def bed_model(x, z, rot):
    md = _md()
    M = trs((x, 0, z), (0, rot, 0))
    for sx in (-0.45, 0.45):
        for sz in (-0.95, 0.95):
            md['planks'].add(box_tris(0.1, 0.45 if sz < 0 else 0.8, 0.1, bottom=True), (0.9, 0.75, 0.6, 1),
                             m=M @ trs((sx, 0, sz)), uv_scale=0.6)
    md['planks'].add(box_tris(1.0, 0.12, 2.0), (0.9, 0.75, 0.6, 1), m=M @ trs((0, 0.32, 0)), uv_scale=1.0)
    md['canvas'].add(box_tris(0.92, 0.16, 1.9), (1.0, 0.98, 0.94, 1), m=M @ trs((0, 0.46, 0)), uv_scale=0.5)
    md['canvas'].add(box_tris(0.95, 0.08, 1.25), (0.55, 0.12, 0.10, 1), m=M @ trs((0, 0.56, -0.3)), uv_scale=0.4)
    md['canvas'].add(box_tris(0.6, 0.12, 0.35), (1, 1, 1, 1), m=M @ trs((0, 0.6, 0.7)), uv_scale=0.4)
    return md


def table_model(x, z, rot):
    md = _md()
    M = trs((x, 0, z), (0, rot, 0))
    md['planks'].add(box_tris(1.2, 0.07, 0.8), (1, 0.9, 0.8, 1), m=M @ trs((0, 0.76, 0)), uv_scale=1.0)
    for sx in (-0.5, 0.5):
        for sz in (-0.32, 0.32):
            md['planks'].add(box_tris(0.07, 0.74, 0.07, bottom=True), (0.9, 0.8, 0.7, 1), m=M @ trs((sx, 0, sz)))
    for sz in (-0.6, 0.6):
        C = M @ trs((0, 0, sz), (0, 0 if sz > 0 else 180, 0))
        md['planks'].add(box_tris(0.42, 0.05, 0.42), (0.95, 0.85, 0.75, 1), m=C @ trs((0, 0.45, 0)))
        for sx in (-0.18, 0.18):
            for s2 in (-0.18, 0.18):
                md['planks'].add(box_tris(0.04, 0.45, 0.04, bottom=True), (0.9, 0.8, 0.7, 1), m=C @ trs((sx, 0, s2)))
        md['planks'].add(box_tris(0.42, 0.45, 0.04, bottom=True), (0.95, 0.85, 0.75, 1), m=C @ trs((0, 0.45, 0.2)))
    return md


def chimney(spec, x, z, top, base=0.0, res='tas'):
    y = base
    while y < top:
        h = min(0.6, top - y)
        spec.add(res, 1, box_piece('stonewall', 0.8, h, 0.8, (x, y + h / 2, z), uv_scale=1.6, bump=1.0), 'wall')
        y += h


# =========================================================================== yapı tipleri
def spec_cadir(depth):
    s = Spec('cadir', 2.6, 2.6)
    for x in (-1.1, 1.1):
        md = _md()
        t, n, uv = tube_tris([(x, 0, 0), (x * 0.0, 1.8, 0)], [0.05, 0.04], 6)
        s.add('kutuk', 1, log_piece(x, -1.2, x, 1.2, 0.08, r=0.06))
    s.add('kutuk', 1, log_piece(0, -1.4, 0, 1.4, 1.75, r=0.06))
    md = _md()
    for sgn in (-1, 1):
        ang = math.degrees(math.atan2(1.75, 1.2))
        md['canvas'].add(box_tris(math.hypot(1.75, 1.2) + 0.15, 0.03, 2.8), (0.62, 0.55, 0.38, 1),
                         m=trs((sgn * 0.62, 0.88, 0), (0, 0, -sgn * ang)), uv_scale=1.0)
    md['canvas'].add(prism_tris(2.5, 1.75, 0.02), (0.55, 0.48, 0.33, 1), m=trs((0, 0, -1.35)), uv_scale=1.0)
    s.add(None, 0, md, 'deco')
    md = _md()
    md['canvas'].add(box_tris(0.8, 0.1, 1.9), (0.5, 0.15, 0.12, 1), m=trs((0, 0.05, -0.1)), uv_scale=0.6)
    s.add(None, 0, md, 'deco')
    s.points.append(('sleep', 0, 0.3, 0.2))
    s.height = 2.0
    s.solid_circles.append((0, 0, 1.3))
    return s


def spec_kulube(depth, roof='shingles'):
    W, D, H, rise = 4.2, 4.6, 2.5, 1.5
    s = Spec('kulube', W, D)
    foundation(s, W, D, depth)
    plank_floor(s, W - 0.1, D - 0.1, pieces=2)
    door_o = [(W / 2 - 0.5 - 0.6, W / 2 - 0.5 + 0.5, 0, 2.05)]
    win_o = [(D / 2 - 0.55, D / 2 + 0.55, 0.95, 1.85)]
    log_wall(s, -W / 2, D / 2, W / 2, D / 2, H, door_o, phase=0)
    log_wall(s, W / 2, D / 2, W / 2, -D / 2, H, win_o, phase=1)
    log_wall(s, W / 2, -D / 2, -W / 2, -D / 2, H, (), phase=0)
    log_wall(s, -W / 2, -D / 2, -W / 2, D / 2, H, (), phase=1)
    gable_logs(s, W, D, H, rise, D / 2)
    gable_logs(s, W, D, H, rise, -D / 2)
    gable_roof(s, W + 0.3, D, H + 0.1, rise, mat=roof)
    door(s, -1.05, D / 2, 0)
    window(s, W / 2, 1.4, 0, 90, 1.0, 0.85)
    s.add(None, 0, bed_model(-1.2, -0.9, 0), 'deco')
    s.floor = (W / 2, D / 2, 0.0)
    s.points.append(('sleep', -1.2, 0.5, -0.9))
    s.height = H + rise + 0.4
    s.lights.append((0, 2.0, 0, 1.2))
    return s


def spec_kutuk_ev(depth):
    W, D, H, rise = 6.4, 8.0, 2.8, 2.0
    s = Spec('kutuk_ev', W, D + 2.0)
    foundation(s, W, D, depth)
    plank_floor(s, W - 0.1, D - 0.1, pieces=4)
    # sundurma (veranda)
    md = box_piece('planks', W, 0.12, 1.8, (0, -0.06, D / 2 + 0.95), uv_scale=2.0)
    s.add('kutuk', 2, md, 'floor')
    for x in (-W / 2 + 0.2, W / 2 - 0.2):
        s.add('kutuk', 1, _post(x, D / 2 + 1.7, H + 0.2))
    s.add('kutuk', 1, log_piece(-W / 2, D / 2 + 1.7, W / 2, D / 2 + 1.7, H + 0.1, r=0.12, ext=0.2))
    door_o = [(W / 2 - 0.55, W / 2 + 0.55, 0, 2.05)]
    fw = [(1.0, 2.1, 0.95, 1.9), (W - 2.1, W - 1.0, 0.95, 1.9)]
    log_wall(s, -W / 2, D / 2, W / 2, D / 2, H, door_o + [(0.9, 2.0, 0.95, 1.9), (W - 2.0, W - 0.9, 0.95, 1.9)], phase=0)
    log_wall(s, W / 2, D / 2, W / 2, -D / 2, H, [(1.5, 2.7, 0.95, 1.9), (D - 2.7, D - 1.5, 0.95, 1.9)], phase=1)
    log_wall(s, W / 2, -D / 2, -W / 2, -D / 2, H, [(W / 2 - 0.6, W / 2 + 0.6, 0.95, 1.9)], phase=0)
    log_wall(s, -W / 2, -D / 2, -W / 2, D / 2, H, [(1.5, 2.7, 0.95, 1.9)], phase=1)
    gable_logs(s, W, D, H, rise, D / 2)
    gable_logs(s, W, D, H, rise, -D / 2)
    chimney(s, -W / 2 - 0.3, -D / 2 + 1.5, H + rise + 0.9, -0.2)
    gable_roof(s, W + 0.4, D + 1.8, H + 0.1, rise, over=0.5)
    door(s, 0, D / 2, 0)
    for x in (-W / 2 + 1.45, W / 2 - 1.45):
        window(s, x, 1.42, D / 2, 0)
    for z in (D / 2 - 2.1, -D / 2 + 2.1):
        window(s, W / 2, 1.42, z, 90)
    window(s, 0, 1.42, -D / 2, 0)
    window(s, -W / 2, 1.42, D / 2 - 2.1, 90)
    s.add(None, 0, bed_model(W / 2 - 0.9, -D / 2 + 1.3, 0), 'deco')
    s.add(None, 0, table_model(-1.0, 0.8, 90), 'deco')
    s.floor = (W / 2, D / 2 + 1.8, 0.0)
    s.points.append(('sleep', W / 2 - 0.9, 0.5, -D / 2 + 1.3))
    s.height = H + rise + 1.0
    s.lights += [(0, 2.2, 0, 1.4), (0, 2.4, D / 2 + 1.0, 1.0)]
    return s


def _post(x, z, h, r=0.12, base=0.0):
    md = _md()
    t, n, uv = tube_tris([(x, base, z), (x, base + h, z)], [r, r * 0.92], 8)
    uv = uv.copy(); uv[:, :, 0] *= 2 * math.pi * r / 1.2; uv[:, :, 1] /= 1.2
    md['bark'].add(t, (1, 1, 1, 1), normals=n, uv=uv, bump=0.8)
    return md


def spec_tas_ev(depth):
    W, D, H, rise = 6.0, 6.6, 2.7, 1.8
    s = Spec('tas_ev', W, D)
    foundation(s, W, D, depth)
    plank_floor(s, W - 0.6, D - 0.6, pieces=3)
    T = 0.45
    stone_wall(s, -W / 2, D / 2, W / 2, D / 2, H, T, [(W / 2 - 0.55, W / 2 + 0.55, 0, 2.1), (0.8, 1.9, 0.9, 1.9)])
    stone_wall(s, W / 2, D / 2, W / 2, -D / 2, H, T, [(D / 2 - 0.55, D / 2 + 0.55, 0.9, 1.9)])
    stone_wall(s, W / 2, -D / 2, -W / 2, -D / 2, H, T, [(W / 2 - 0.55, W / 2 + 0.55, 0.9, 1.9)])
    stone_wall(s, -W / 2, -D / 2, -W / 2, D / 2, H, T, [])
    for zf in (D / 2, -D / 2):
        y = H
        while y < H + rise - 0.3:
            k = 1 - (y - H) / rise
            md = box_piece('stonewall', W * k, 0.45, T, (0, y + 0.225, zf), uv_scale=2.4, bump=1.0)
            s.add('tas', 1, md, 'gable')
            y += 0.45
    chimney(s, -W / 2 + 0.6, -D / 2 + 0.6, H + rise + 0.8, H)
    gable_roof(s, W + 0.4, D + 0.2, H, rise, mat='shingles', over=0.5)
    door(s, 0, D / 2, 0)
    window(s, -W / 2 + 1.35, 1.4, D / 2, 0)
    window(s, W / 2, 1.4, 0, 90)
    window(s, 0, 1.4, -D / 2, 0)
    s.add(None, 0, bed_model(-W / 2 + 1.1, -D / 2 + 1.4, 0), 'deco')
    s.add(None, 0, table_model(1.2, -0.8, 0), 'deco')
    s.floor = (W / 2, D / 2, 0.0)
    s.points.append(('sleep', -W / 2 + 1.1, 0.5, -D / 2 + 1.4))
    s.height = H + rise + 1.0
    s.lights.append((0, 2.2, 0, 1.4))
    return s


def spec_ciftlik_evi(depth):
    W, D, H, rise = 7.2, 9.0, 3.0, 2.3
    s = Spec('ciftlik_evi', W, D + 2.2)
    foundation(s, W, D, depth)
    plank_floor(s, W - 0.1, D - 0.1, pieces=4)
    md = box_piece('planks', W + 0.6, 0.12, 2.0, (0, -0.06, D / 2 + 1.05), uv_scale=2.0)
    s.add('kutuk', 2, md, 'floor')
    beam = (0.55, 0.38, 0.26, 1)
    walls = [(-W / 2, D / 2, W / 2, D / 2, [(W / 2 - 0.55, W / 2 + 0.55, 0, 2.1), (1.0, 2.2, 1.0, 2.0), (W - 2.2, W - 1.0, 1.0, 2.0)]),
             (W / 2, D / 2, W / 2, -D / 2, [(1.6, 2.8, 1.0, 2.0), (D - 2.8, D - 1.6, 1.0, 2.0)]),
             (W / 2, -D / 2, -W / 2, -D / 2, [(W / 2 - 0.6, W / 2 + 0.6, 1.0, 2.0)]),
             (-W / 2, -D / 2, -W / 2, D / 2, [(1.6, 2.8, 1.0, 2.0), (D - 2.8, D - 1.6, 1.0, 2.0)])]
    # alt taş sıra
    for ax, az, bx, bz, op in walls:
        stone_wall(s, ax, az, bx, bz, 0.7, 0.32, op, block=1.5, ch=0.7)
    s.walls = []
    for ax, az, bx, bz, op in walls:
        L = math.hypot(bx - ax, bz - az)
        ux, uz = (bx - ax) / L, (bz - az) / L
        ang = math.degrees(math.atan2(ux, uz)) - 90
        # ahşap direkler
        n = int(L / 1.5) + 1
        for i in range(n + 1):
            u = i * L / n
            px, pz = ax + ux * u, az + uz * u
            s.add('kutuk', 1, box_piece('planks', 0.2, H - 0.7, 0.24, (px, 0.7 + (H - 0.7) / 2, pz), (0, ang + 90, 0),
                                         color=beam, uv_scale=1.0), 'wall')
        # sıva paneller
        for i in range(n):
            u0, u1 = i * L / n, (i + 1) * L / n
            cuts = [(o0, o1) for (o0, o1, y0, y1) in op]
            for y0, y1 in ((0.7, 1.0), (1.0, 2.0), (2.0, H)):
                cut_here = [(o0, o1) for (o0, o1, oy0, oy1) in op if oy0 < (y0 + y1) / 2 < oy1]
                for s0, s1 in _intervals(u1 - u0, [(c0 - u0, c1 - u0) for c0, c1 in cut_here]):
                    cu = u0 + (s0 + s1) / 2
                    px, pz = ax + ux * cu, az + uz * cu
                    s.add('tas', 1, box_piece('plaster', s1 - s0, y1 - y0, 0.2, (px, (y0 + y1) / 2, pz), (0, ang + 90, 0),
                                               uv_scale=2.0, bump=0.5), 'wall')
        _wall_segments(s, ax, az, bx, bz, op, ux, uz, L)
        s.add(None, 0, box_piece('planks', L + 0.2, 0.2, 0.26, ((ax + bx) / 2, H - 0.1, (az + bz) / 2), (0, ang + 90, 0),
                                  color=beam), 'deco')
    for zf in (D / 2, -D / 2):
        y = H
        while y < H + rise - 0.25:
            k = 1 - (y - H) / rise
            s.add('tas', 1, box_piece('plaster', W * k, 0.5, 0.2, (0, y + 0.25, zf), uv_scale=2.0), 'gable')
            y += 0.5
    chimney(s, W / 2 - 1.0, -D / 2 + 1.5, H + rise + 1.0, H - 0.2)
    gable_roof(s, W + 0.4, D + 2.0, H, rise, mat='rooftiles', res='tas', over=0.55)
    for x in (-W / 2 + 0.2, W / 2 - 0.2, -W / 4, W / 4):
        s.add('kutuk', 1, _post(x, D / 2 + 1.9, H - 0.1, r=0.1))
    door(s, 0, D / 2, 0)
    for x in (-W / 2 + 1.6, W / 2 - 1.6):
        window(s, x, 1.5, D / 2, 0)
    for z in (D / 2 - 2.2, -D / 2 + 2.2):
        window(s, W / 2, 1.5, z, 90)
        window(s, -W / 2, 1.5, z, 90)
    window(s, 0, 1.5, -D / 2, 0)
    s.add(None, 0, bed_model(-W / 2 + 1.0, -D / 2 + 1.3, 0), 'deco')
    s.add(None, 0, bed_model(-W / 2 + 2.4, -D / 2 + 1.3, 0), 'deco')
    s.add(None, 0, table_model(1.4, 1.0, 90), 'deco')
    s.floor = (W / 2 + 0.3, D / 2 + 2.0, 0.0)
    s.points.append(('sleep', -W / 2 + 1.0, 0.5, -D / 2 + 1.3))
    s.height = H + rise + 1.2
    s.lights += [(0, 2.4, 0, 1.5), (0, 2.6, D / 2 + 1.1, 1.0)]
    return s


def spec_yatak(depth):
    s = Spec('yatak', 1.1, 2.1)
    s.add('kutuk', 4, bed_model(0, 0, 0), 'deco')
    s.points.append(('sleep', 0, 0.5, 0))
    s.height = 0.9
    s.solid_circles.append((0, 0, 0.6))
    return s


def spec_tarla(depth):
    W = 6.4
    s = Spec('tarla', W, W)
    for (ax, az, bx, bz) in [(-W / 2, -W / 2, W / 2, -W / 2), (W / 2, -W / 2, W / 2, W / 2), (W / 2, W / 2, -W / 2, W / 2),
                             (-W / 2, W / 2, -W / 2, -W / 2)]:
        s.add('kutuk', 1, log_piece(ax, az, bx, bz, 0.1, r=0.11, ext=0.12))
    md = _md()
    md['soil'].add(box_tris(W - 0.3, 0.06, W - 0.3, bottom=True), (0.75, 0.68, 0.62, 1), m=trs((0, -0.05, 0)), uv_scale=1.4,
                   bump=0.6)
    s.add(None, 0, md, 'deco')
    s.points.append(('farm', 0, 0, 0))
    s.height = 0.4
    return s


def spec_kumes(depth):
    s = Spec('kumes', 3.2, 3.0)
    for sx in (-1.1, 1.1):
        for sz in (-0.9, 0.9):
            s.add('kutuk', 1, _post(sx, sz, 0.75, r=0.07))
    s.add('kutuk', 2, box_piece('planks', 2.6, 0.1, 2.2, (0, 0.75, 0), uv_scale=1.4))
    for (x, z, sx, sz) in [(0, 1.05, 2.6, 0.1), (0, -1.05, 2.6, 0.1), (1.25, 0, 0.1, 2.0), (-1.25, 0, 0.1, 2.0)]:
        s.add('kutuk', 2, box_piece('planks', sx, 1.3, sz, (x, 1.45, z), color=(0.92, 0.78, 0.62, 1), uv_scale=1.4))
    md = _md()
    md['plain'].add(box_tris(0.55, 0.55, 0.12, bottom=True), (0.08, 0.06, 0.05, 1), m=trs((0.5, 0.82, 1.08)))
    md['planks'].add(box_tris(0.5, 0.05, 1.5), (1, 0.9, 0.8, 1), m=trs((0.5, 0.42, 1.65), (28, 0, 0)), uv_scale=1.0)
    s.add(None, 0, md, 'deco')
    rise = 0.8
    ang = math.degrees(math.atan2(rise, 1.4))
    for sgn in (-1, 1):
        s.add('kutuk', 2, box_piece('shingles', math.hypot(1.4, rise) + 0.35, 0.08, 2.9, (sgn * 0.72, 2.1 + rise / 2 + 0.03, 0),
                                     (0, 0, -sgn * ang), uv_scale=1.4))
    for zf in (1.05, -1.05):
        md = _md()
        md['planks'].add(prism_tris(2.6, rise, 0.1), (0.92, 0.78, 0.62, 1), m=trs((0, 2.1, zf)), uv_scale=1.4)
        s.add(None, 0, md, 'deco')
    s.height = 3.0
    s.solid_circles.append((0, 0, 1.45))
    s.points.append(('home', 0, 0, 2.5))
    return s


def spec_ahir(depth):
    W, D, H, rise = 8.0, 10.0, 3.6, 2.6
    s = Spec('ahir', W, D)
    foundation(s, W, D, depth)
    md = _md()
    md['dirt'].add(box_tris(W - 0.2, 0.05, D - 0.2), (0.8, 0.75, 0.65, 1), m=trs((0, -0.03, 0)), uv_scale=2.0)
    s.add(None, 0, md, 'floor')
    red = (0.70, 0.24, 0.18, 1)
    for ax, az, bx, bz, op in [(-W / 2, D / 2, W / 2, D / 2, [(W / 2 - 1.6, W / 2 + 1.6, 0, 3.0)]),
                               (W / 2, D / 2, W / 2, -D / 2, [(D / 2 - 0.7, D / 2 + 0.7, 1.4, 2.4)]),
                               (W / 2, -D / 2, -W / 2, -D / 2, [(W / 2 - 1.0, W / 2 + 1.0, 0, 2.4)]),
                               (-W / 2, -D / 2, -W / 2, D / 2, [(D / 2 - 0.7, D / 2 + 0.7, 1.4, 2.4)])]:
        stone_wall(s, ax, az, bx, bz, H, 0.14, op, res='kutuk', mat='planks', block=100.0, ch=0.6, color=red)
        L = math.hypot(bx - ax, bz - az)
        ux, uz = (bx - ax) / L, (bz - az) / L
        ang = math.degrees(math.atan2(ux, uz)) - 90
        for (o0, o1, y0, y1) in op:     # beyaz kapı/pencere çerçeveleri
            for u in (o0, o1):
                px, pz = ax + ux * u, az + uz * u
                s.add(None, 0, box_piece('planks', 0.14, y1 - y0, 0.2, (px, (y0 + y1) / 2, pz), (0, ang + 90, 0),
                                          color=(1.0, 0.97, 0.92, 1)), 'deco')
            cu = (o0 + o1) / 2
            s.add(None, 0, box_piece('planks', o1 - o0 + 0.14, 0.14, 0.2, (ax + ux * cu, y1, az + uz * cu), (0, ang + 90, 0),
                                      color=(1.0, 0.97, 0.92, 1)), 'deco')
    for sx in (-1, 1):
        for sz in (-1, 1):
            s.add('kutuk', 1, box_piece('planks', 0.3, H, 0.3, (sx * W / 2, H / 2, sz * D / 2), color=(0.95, 0.92, 0.88, 1)))
    for zf in (D / 2, -D / 2):
        y = H
        while y < H + rise - 0.3:
            k = 1 - (y - H) / rise
            s.add('kutuk', 1, box_piece('planks', W * k, 0.5, 0.14, (0, y + 0.25, zf), (0, 0, 0), color=red, uv_scale=2.4), 'gable')
            y += 0.5
    gable_roof(s, W + 0.3, D, H, rise, mat='shingles', over=0.6, color=(0.75, 0.75, 0.8, 1))
    md = _md()
    for x in (-2.2, 2.4):
        md['grass'].add(box_tris(1.2, 0.6, 0.8, bottom=True), (1.4, 1.2, 0.7, 1), m=trs((x, 0, -D / 2 + 1.0), (0, 10 * x, 0)),
                        uv_scale=0.6)
    md['planks'].add(box_tris(2.4, 0.5, 0.6, bottom=True), (0.9, 0.8, 0.7, 1), m=trs((-W / 2 + 0.6, 0, 0), (0, 90, 0)))
    s.add(None, 0, md, 'deco')
    s.floor = (W / 2, D / 2, 0.0)
    s.height = H + rise + 0.6
    s.points.append(('home', 0, 0, D / 2 + 4.0))
    return s


def spec_kuyu(depth):
    s = Spec('kuyu', 2.4, 2.4)
    for i in range(3):
        for k in range(6):
            a0 = k / 6 * 2 * math.pi + (i % 2) * 0.5
            md = _md()
            md['stonewall'].add(box_tris(0.62, 0.3, 0.3), (1, 1, 1, 1),
                                m=trs((math.cos(a0) * 0.85, 0.15 + i * 0.3, math.sin(a0) * 0.85), (0, -math.degrees(a0) + 90, 0)),
                                uv_scale=1.2, bump=1.0)
            s.add('tas', 1, md, 'wall')
    md = _md()
    md['glass'].add(cylinder_tris(0.7, 0.7, 0.02, 12), (0.04, 0.12, 0.14, 1), m=trs((0, 0.55, 0)), spec=1.0, uv_mode='none')
    s.add(None, 0, md, 'deco')
    for sx in (-1, 1):
        s.add('kutuk', 1, _post(sx * 0.95, 0, 2.3, r=0.07))
    md = _md()
    md['shingles'].add(prism_tris(2.6, 0.9, 1.5), (1, 1, 1, 1), m=trs((0, 2.25, 0)), uv_scale=1.0)
    md['planks'].add(cylinder_tris(0.06, 0.06, 1.9, 8), (1, 1, 1, 1), m=trs((-0.95, 1.8, 0), (0, 0, -90)), smooth='radial')
    md['metal'].add(cylinder_tris(0.15, 0.12, 0.25, 10), (0.5, 0.45, 0.4, 1), m=trs((0.2, 1.0, 0)), spec=0.5)
    s.add('kutuk', 2, md, 'roof')
    s.points.append(('water', 0, 0.8, 0))
    s.solid_circles.append((0, 0, 1.15))
    s.height = 3.2
    return s


def spec_ates(depth):
    s = Spec('ates', 1.6, 1.6)
    for k in range(8):
        a = k / 8 * 2 * math.pi
        md = _md()
        md['rock'].add(sphere_tris(0.18, 1, jitter=0.2, squash=0.7), (1, 1, 1, 1), m=trs((math.cos(a) * 0.6, 0.05, math.sin(a) * 0.6)),
                       uv_scale=0.5, bump=1.0)
        s.add('tas', 1, md)
    for k in range(3):
        a = k / 3 * math.pi
        s.add('kutuk', 1, log_piece(math.cos(a) * 0.45, math.sin(a) * 0.45, -math.cos(a) * 0.45, -math.sin(a) * 0.45,
                                    0.12 + k * 0.08, r=0.07))
    s.fires.append((0, 0.3, 0))
    s.lights.append((0, 0.8, 0, 3.0))
    s.points.append(('cook', 0, 0.3, 0))
    s.solid_circles.append((0, 0, 0.75))
    s.height = 0.6
    return s


def spec_cit(depth):
    s = Spec('cit', 4.0, 0.3)
    for x in (-2.0, 2.0):
        s.add('kutuk', 0 if x < 0 else 1, _post(x, 0, 1.2, r=0.08, base=-0.2))
    s.add('kutuk', 1, log_piece(-2.0, 0, 2.0, 0, 0.45, r=0.06, ext=0.1))
    s.add(None, 0, log_piece(-2.0, 0, 2.0, 0, 0.95, r=0.06, ext=0.1))
    s.walls.append((-2.0, 0, 2.0, 0))
    s.height = 1.2
    return s


def spec_pazar(depth):
    s = Spec('pazar', 4.4, 2.4)
    s.add('kutuk', 3, box_piece('planks', 4.0, 1.0, 1.0, (0, 0.5, 0.4), uv_scale=1.4))
    s.add('kutuk', 1, box_piece('planks', 4.2, 0.08, 1.2, (0, 1.04, 0.4), color=(1.1, 1, 0.9, 1), uv_scale=1.4))
    for sx in (-1.95, 1.95):
        for sz in (-0.6, 0.9):
            s.add('kutuk', 1, _post(sx, sz, 2.6, r=0.07))
    md = _md()
    for i in range(7):
        col = (0.75, 0.15, 0.12, 1) if i % 2 == 0 else (0.95, 0.92, 0.85, 1)
        md['canvas'].add(box_tris(0.6, 0.04, 2.2), col, m=trs((-1.8 + i * 0.6, 2.72, 0.2), (-12, 0, 0)), uv_scale=0.8)
    produce = [(0.98, 0.52, 0.12), (0.88, 0.14, 0.10), (0.95, 0.82, 0.36), (0.97, 0.53, 0.10), (0.5, 0.7, 0.25)]
    r = rng()
    for i, pc in enumerate(produce):
        cx = -1.6 + i * 0.8
        md['planks'].add(box_tris(0.7, 0.22, 0.6, bottom=True), (1, 1, 1, 1), m=trs((cx, 1.08, 0.4)), uv_scale=0.8)
        for k in range(6):
            md['plain'].add(sphere_tris(0.1, 1), pc + (1,), m=trs((cx + r.uniform(-0.2, 0.2), 1.36, 0.4 + r.uniform(-0.18, 0.18))),
                            smooth='sphere', spec=0.4)
    s.add('kutuk', 4, md)
    s.points.append(('trade', 0, 1.0, 1.4))
    s.walls.append((-2.0, 0.4, 2.0, 0.4))
    s.height = 2.9
    return s


def spec_fener(depth):
    s = Spec('fener', 0.6, 0.6)
    s.add('kutuk', 2, _post(0, 0, 3.0, r=0.08, base=-0.2))
    md = _md()
    md['planks'].add(box_tris(0.7, 0.08, 0.08), (1, 1, 1, 1), m=trs((0.3, 2.9, 0)))
    md['metal'].add(cone_tris(0.22, 0.18, 6), (0.15, 0.15, 0.15, 1), m=trs((0.6, 2.75, 0)), spec=0.6)
    md['metal'].add(cylinder_tris(0.13, 0.13, 0.03, 6), (0.15, 0.15, 0.15, 1), m=trs((0.6, 2.42, 0)), spec=0.6)
    s.add('demir', 1, md)
    s.lights.append((0.6, 2.55, 0, 4.0))
    s.fires.append((0.6, 2.58, 0))
    s.solid_circles.append((0, 0, 0.2))
    s.height = 3.1
    return s


def spec_yol(depth):
    s = Spec('yol', 3.0, 3.0)
    r = np.random.default_rng(5)
    for k in range(3):
        md = _md()
        for _ in range(9):
            x, z = r.uniform(-1.3, 1.3), -1.4 + k + r.uniform(0.0, 1.0)
            md['rock'].add(sphere_tris(r.uniform(0.3, 0.45), 1, jitter=0.15, squash=0.15), (0.95, 0.93, 0.9, 1),
                           m=trs((x, 0.0, min(1.3, z)), (0, r.uniform(0, 360), 0)), uv_scale=0.7, bump=1.0)
        s.add('tas', 1, md)
    s.height = 0.3
    return s


def spec_garaj(depth):
    W, D, H = 6.2, 8.0, 3.2
    s = Spec('garaj', W, D)
    foundation(s, W, D, depth)
    md = _md()
    md['rock'].add(box_tris(W - 0.2, 0.1, D - 0.2), (0.75, 0.75, 0.75, 1), m=trs((0, -0.05, 0)), uv_scale=4.0, bump=0.3)
    s.add('tas', 3, md, 'floor')
    stone_wall(s, -W / 2, D / 2, W / 2, D / 2, H, 0.4, [(0.6, W - 0.6, 0, 2.7)])
    stone_wall(s, W / 2, D / 2, W / 2, -D / 2, H, 0.4, [(D / 2 - 0.6, D / 2 + 0.6, 1.0, 2.0)])
    stone_wall(s, W / 2, -D / 2, -W / 2, -D / 2, H, 0.4, [])
    stone_wall(s, -W / 2, -D / 2, -W / 2, D / 2, H, 0.4, [(D / 2 - 0.6, D / 2 + 0.6, 1.0, 2.0)])
    s.add('kutuk', 1, box_piece('planks', W - 1.0, 0.3, 0.45, (0, 2.85, D / 2), uv_scale=1.0))
    n = 6
    for i in range(n):
        z = -D / 2 - 0.3 + (i + 0.5) * (D + 0.6) / n
        s.add('demir', 1, box_piece('metal', W + 0.6, 0.06, (D + 0.6) / n, (0, H + 0.25, z), (0, 0, 4),
                                     color=(0.7, 0.72, 0.75, 1), uv_scale=1.0, spec=0.6), 'roof')
    md = _md()
    md['planks'].add(box_tris(2.4, 0.08, 0.9), (1, 1, 1, 1), m=trs((-W / 2 + 0.75, 0.9, -1.0), (0, 90, 0)))
    for z in (-2.0, 0.0):
        md['planks'].add(box_tris(0.08, 0.9, 0.08, bottom=True), (1, 1, 1, 1), m=trs((-W / 2 + 0.75 - 0.35, 0, z)))
        md['planks'].add(box_tris(0.08, 0.9, 0.08, bottom=True), (1, 1, 1, 1), m=trs((-W / 2 + 0.75 + 0.35, 0, z)))
    md['metal'].add(box_tris(0.5, 0.3, 0.3, bottom=True), (0.7, 0.15, 0.1, 1), m=trs((-W / 2 + 0.75, 0.94, -1.5)), spec=0.7)
    md['metal'].add(cylinder_tris(0.06, 0.06, 0.3, 8), (0.3, 0.3, 0.32, 1), m=trs((-W / 2 + 0.7, 0.94, -0.4), (0, 0, 80)), spec=0.8)
    s.add(None, 0, md, 'deco')
    s.floor = (W / 2, D / 2, 0.0)
    s.points.append(('garage', -W / 2 + 1.2, 1.0, -1.0))
    s.height = H + 0.5
    s.lights.append((0, 2.8, 0, 1.6))
    return s


SPECS = {
    'cadir': ('Çadır', 'Barınma', spec_cadir, 'Basit bir sığınak. Uyuyabilirsin.', 0),
    'kulube': ('Kütük Kulübe', 'Barınma', spec_kulube, 'Küçük, sıcak bir kulübe: yatak ve pencere.', 0),
    'kutuk_ev': ('Büyük Kütük Ev', 'Barınma', spec_kutuk_ev, 'Verandalı, bacalı geniş kütük ev.', 0),
    'tas_ev': ('Taş Ev', 'Barınma', spec_tas_ev, 'Kalın taş duvarlı, sağlam bir ev.', 0),
    'ciftlik_evi': ('Çiftlik Evi', 'Barınma', spec_ciftlik_evi, 'Ahşap iskeletli, kiremit çatılı büyük ev.', 150),
    'yatak': ('Yatak', 'Barınma', spec_yatak, 'Evinin içine koyabileceğin bir yatak.', 0),
    'tarla': ('Tarla Parseli', 'Çiftlik', spec_tarla, '4x4 ekim alanı.', 0),
    'kumes': ('Kümes', 'Çiftlik', spec_kumes, 'Tavuk besleyebilmek için (6 tavuk).', 0),
    'ahir': ('Ahır', 'Çiftlik', spec_ahir, 'İnek ve koyunlar için (4 koyun / 3 inek).', 0),
    'kuyu': ('Kuyu', 'Çiftlik', spec_kuyu, 'Sulama kabını doldurmak için.', 0),
    'cit': ('Çit', 'Çiftlik', spec_cit, '4 metrelik ahşap çit.', 0),
    'pazar': ('Pazar Tezgâhı', 'Çiftlik', spec_pazar, 'Kendi tezgâhında alım-satım yap.', 0),
    'ates': ('Kamp Ateşi', 'Diğer', spec_ates, 'Işık ve yemek pişirme.', 0),
    'fener': ('Fener Direği', 'Diğer', spec_fener, 'Geceleri etrafı aydınlatır.', 0),
    'yol': ('Taş Yol', 'Diğer', spec_yol, '3x3 metre taş döşeme.', 0),
    'garaj': ('Garaj', 'Diğer', spec_garaj, 'Araba üretim tezgâhı içerir.', 0),
}
CATEGORIES = ['Barınma', 'Çiftlik', 'Diğer']
PREVIEW_COST = {}


def spec_for(kind, depth=0.3):
    return SPECS[kind][2](depth)


def cost_of(kind):
    if kind not in PREVIEW_COST:
        PREVIEW_COST[kind] = spec_for(kind, 0.3).cost()
    return PREVIEW_COST[kind]


# =========================================================================== yapı örneği
def _bake(model):
    out = {}
    for mat, mb in model.parts.items():
        a = mb.arrays()
        if a is not None:
            out[mat] = a
    return out


def _geomnode(baked_list):
    acc = {}
    for baked in baked_list:
        for mat, arrs in baked.items():
            acc.setdefault(mat, []).append(arrs)
    gn = GeomNode('b')
    for mat, lst in acc.items():
        arrs = [np.concatenate([a[i] for a in lst]) for i in range(5)]
        gn.add_geom(make_geom(*arrs), materials.state(mat))
    return gn


class Building:
    _next_id = 1

    def __init__(self, mgr, kind, x, z, rot, y=None, depth=None, built=0, uid=None):
        self.mgr = mgr
        self.kind = kind
        self.uid = uid or Building._next_id
        Building._next_id = max(Building._next_id, self.uid + 1)
        self.x, self.z, self.rot = x, z, rot
        if y is None:
            y, depth = mgr.base_height(kind, x, z, rot)
        self.y, self.depth = y, depth
        self.spec = spec_for(kind, depth)
        for p in self.spec.parts:
            p.baked = _bake(p.model)
            p.model = None
        self.built = built          # yapılmış parça sayısı (sırayla)
        self.root = Entity(position=(x, y, z), rotation_y=rot, shader=world_shader)
        self.ghost = Entity(parent=self.root, shader=ghost_shader)
        self.ghost.setTransparency(TransparencyAttrib.M_alpha)
        self.ghost.setDepthWrite(False)
        self.ghost.setBin('transparent', 5)
        self.ghost.hide(SHADOW_MASK)
        self.ghost.setTwoSided(True)
        self.built_np = None
        self.ghost_np = None
        self.data = {}
        self.fire_entities = []
        self._refresh()

    # ------------------------------------------------------------------
    @property
    def complete(self):
        return self.built >= len(self.spec.parts)

    @property
    def progress(self):
        tot = sum(max(p.units, 0) for p in self.spec.parts) or 1
        done = sum(max(p.units, 0) for p in self.spec.parts[:self.built])
        return done / tot

    def _skip_free(self):
        # ücretsiz parçalar (süsler) sıra kendilerine geldiğinde kendiliğinden belirir
        while self.built < len(self.spec.parts) and self.spec.parts[self.built].units == 0:
            self.built += 1

    def _next_paid(self):
        for i in range(self.built, len(self.spec.parts)):
            if self.spec.parts[i].units > 0:
                return i
        return None

    def remaining(self):
        c = {}
        for p in self.spec.parts[self.built:]:
            if p.res and p.units:
                c[p.res] = c.get(p.res, 0) + p.units
        return c

    def needed_next(self):
        i = self._next_paid()
        return None if i is None else self.spec.parts[i].res

    def add_material(self, inv):
        """Envanterden sıradaki parçanın malzemesini ekler. Döner: (eklenen kaynak, eksik kaynak)."""
        self._skip_free()
        if self.complete:
            return None, None
        p = self.spec.parts[self.built]
        if p.units == 0:
            self.built += 1
            self._skip_free()
            self._refresh()
            return 'free', None
        if inv.get(p.res, 0) < p.units:
            return None, p.res
        inv[p.res] -= p.units
        self.built += 1
        self._skip_free()
        self._refresh()
        return p.res, None

    def refund(self):
        c = {}
        for p in self.spec.parts[:self.built]:
            if p.res and p.units:
                c[p.res] = c.get(p.res, 0) + p.units
        return c

    def _refresh(self):
        self._skip_free()
        if self.built_np is not None:
            self.built_np.remove_node()
            self.built_np = None
        if self.ghost_np is not None:
            self.ghost_np.remove_node()
            self.ghost_np = None
        done = [p.baked for p in self.spec.parts[:self.built]]
        todo = [p.baked for p in self.spec.parts[self.built:]]
        if done:
            self.built_np = self.root.attachNewNode(_geomnode(done))
        if todo:
            self.ghost_np = self.ghost.attachNewNode(_geomnode(todo))
        self.ghost.enabled = bool(todo)
        if self.complete:
            self._on_complete()

    def _on_complete(self):
        if self.fire_entities or not self.spec.fires:
            return
        for (fx, fy, fz) in self.spec.fires:
            e = Entity(parent=self.root, position=(fx, fy, fz), shader=glow_shader)
            md = Model()
            if self.kind == 'ates':
                for k in range(4):
                    t = cone_tris(0.2 - k * 0.035, 0.75 - k * 0.12, 9, jitter=0.25)
                    ys = t[:, :, 1:2] / 0.75
                    vc = np.concatenate([np.clip(np.broadcast_to([1.0, 0.35, 0.05], t.shape) * (1 - ys) +
                                                 np.broadcast_to([1.0, 0.85, 0.35], t.shape) * ys, 0, 1),
                                         np.ones(t.shape[:2] + (1,))], 2)
                    md['plain'].add(t, vcolors=vc, m=trs((0.04 * (k % 2), -0.1, 0.03 * (k // 2)), (0, k * 40, 0)), uv_mode='none')
            else:
                md['plain'].add(sphere_tris(0.12, 1), (1.0, 0.85, 0.55, 1), uv_mode='none')
            e.attachNewNode(md.node('fire'))
            e.hide(SHADOW_MASK)
            self.fire_entities.append(e)
        self.mgr.on_complete(self)

    # ------------------------------------------------------------------ dünya dönüşümleri
    def to_local(self, x, z):
        dx, dz = x - self.x, z - self.z
        a = math.radians(self.rot)
        c, s = math.cos(a), math.sin(a)
        return dx * c - dz * s, dx * s + dz * c

    def to_world(self, lx, lz):
        a = math.radians(self.rot)
        c, s = math.cos(a), math.sin(a)
        return self.x + lx * c + lz * s, self.z - lx * s + lz * c

    def world_point(self, lx, ly, lz):
        x, z = self.to_world(lx, lz)
        return Vec3(x, self.y + ly, z)

    def ray_hit(self, o, d, max_d=5.0):
        """Yerel eksenlere hizalı kutu ile ışın kesişimi."""
        lox, loz = self.to_local(o[0], o[2])
        a = math.radians(self.rot)
        c, s = math.cos(a), math.sin(a)
        ldx, ldz = d[0] * c - d[2] * s, d[0] * s + d[2] * c
        lo = (lox, o[1] - self.y, loz)
        ld = (ldx, d[1], ldz)
        hw, hd = self.spec.w / 2 + 0.2, self.spec.d / 2 + 0.2
        lo_b = (-hw, -max(self.depth, 0.2) - 0.1, -hd)
        hi_b = (hw, self.spec.height, hd)
        t0, t1 = 0.0, max_d
        for i in range(3):
            if abs(ld[i]) < 1e-9:
                if lo[i] < lo_b[i] or lo[i] > hi_b[i]:
                    return None
                continue
            ta = (lo_b[i] - lo[i]) / ld[i]
            tb = (hi_b[i] - lo[i]) / ld[i]
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
            if t0 > t1:
                return None
        return t0

    def wall_segments_world(self):
        out = []
        if not self.complete:
            return out
        for ax, az, bx, bz in self.spec.walls:
            p = self.to_world(ax, az)
            q = self.to_world(bx, bz)
            out.append((p[0], p[1], q[0], q[1]))
        return out

    def circles_world(self):
        if not self.complete:
            return []
        out = []
        for cx, cz, r in self.spec.solid_circles:
            x, z = self.to_world(cx, cz)
            out.append((x, z, r))
        return out

    def floor_at(self, x, z):
        if self.spec.floor is None:
            return None
        built_floor = any(p.tag in ('floor', 'found') for p in self.spec.parts[:self.built])
        if not built_floor and not self.complete:
            return None
        lx, lz = self.to_local(x, z)
        hw, hd, fy = self.spec.floor
        if abs(lx) <= hw and abs(lz) <= hd:
            return self.y + fy
        return None

    def points_world(self):
        if not self.complete:
            return []
        return [(k, self.world_point(px, py, pz)) for (k, px, py, pz) in self.spec.points]

    def lights_world(self):
        if not self.complete:
            return []
        return [(self.world_point(lx, ly, lz), w) for (lx, ly, lz, w) in self.spec.lights]

    def destroy(self):
        self.root.removeNode()

    def to_dict(self):
        return {'uid': self.uid, 'kind': self.kind, 'x': round(self.x, 3), 'z': round(self.z, 3), 'rot': self.rot,
                'y': round(self.y, 3), 'depth': round(self.depth, 3), 'built': self.built, 'data': self.data}


# =========================================================================== yönetici
class BuildManager:
    def __init__(self, game):
        self.game = game
        self.buildings = []
        self.preview = None
        self.preview_kind = None
        self.preview_rot = 0.0
        self.preview_ok = False
        self.preview_pos = None

    def base_height(self, kind, x, z, rot):
        sp_w, sp_d = self._footprint(kind)
        t = self.game.terrain
        a = math.radians(rot)
        c, s = math.cos(a), math.sin(a)
        hs = []
        for fx in (-0.5, 0, 0.5):
            for fz in (-0.5, 0, 0.5):
                lx, lz = fx * sp_w, fz * sp_d
                wx, wz = x + lx * c + lz * s, z - lx * s + lz * c
                hs.append(t.ground_y(wx, wz))
        hmax, hmin = max(hs), min(hs)
        if kind in ('yol', 'tarla', 'ates', 'cit', 'fener', 'cadir', 'yatak', 'pazar', 'kumes', 'kuyu'):
            mean = sum(hs) / len(hs)
            y = (mean + hmax) / 2 if kind not in ('yol', 'tarla') else mean + 0.04
            if kind == 'yatak':
                fl = self.floor_height(x, z)
                if fl is not None:
                    y = fl
            return y, max(0.0, y - hmin)
        y = hmax + 0.25
        return y, y - hmin

    def _footprint(self, kind):
        if kind not in PREVIEW_COST:
            cost_of(kind)
        sp = _spec_cache(kind)
        return sp.w, sp.d

    # ------------------------------------------------------------------ önizleme
    def start_preview(self, kind):
        self.cancel_preview()
        self.preview_kind = kind
        sp = _spec_cache(kind)
        baked = [_bake(p.model) for p in sp.parts]
        self.preview = Entity(shader=ghost_shader)
        self.preview.attachNewNode(_geomnode(baked))
        self.preview.setTransparency(TransparencyAttrib.M_alpha)
        self.preview.setDepthWrite(False)
        self.preview.setBin('transparent', 6)
        self.preview.setTwoSided(True)
        self.preview.hide(SHADOW_MASK)

    def cancel_preview(self):
        if self.preview is not None:
            self.preview.removeNode()
        self.preview = None
        self.preview_kind = None

    def update_preview(self, eye, dirv):
        if self.preview is None:
            return
        g = self.game
        t = g.terrain.raycast((eye.x, eye.y, eye.z), (dirv.x, dirv.y, dirv.z), 14.0)
        if t is None:
            t = 6.0
        px, pz = eye.x + dirv.x * t, eye.z + dirv.z * t
        sp = _spec_cache(self.preview_kind)
        # kendinden en az biraz uzakta
        dx, dz = px - eye.x, pz - eye.z
        dist = math.hypot(dx, dz)
        need = max(sp.w, sp.d) / 2 + 1.2 if self.preview_kind not in ('yatak', 'yol', 'cit') else 1.5
        if dist < need and dist > 1e-3:
            px, pz = eye.x + dx / dist * need, eye.z + dz / dist * need
        px, pz = round(px * 4) / 4, round(pz * 4) / 4
        y, depth = self.base_height(self.preview_kind, px, pz, self.preview_rot)
        self.preview.position = (px, y, pz)
        self.preview.rotation_y = self.preview_rot
        ok, why = self.check_place(self.preview_kind, px, pz, self.preview_rot, y, depth)
        self.preview_ok, self.preview_why = ok, why
        self.preview_pos = (px, pz)
        self.preview.set_shader_input('ghost_col', (0.45, 1.0, 0.55, 0.5) if ok else (1.0, 0.35, 0.3, 0.5))

    def check_place(self, kind, x, z, rot, y, depth):
        g = self.game
        sp = _spec_cache(kind)
        r = max(sp.w, sp.d) / 2
        if depth > (2.2 if kind not in ('yol', 'tarla', 'yatak') else 0.9):
            return False, 'Zemin çok eğimli'
        for fx in (-0.5, 0.5):
            for fz in (-0.5, 0.5):
                if g.terrain.ground_y(x + fx * sp.w, z + fz * sp.d) < WATER_Y + 0.2:
                    return False, 'Suyun üzerine inşa edilemez'
        blockers = g.nature.blocked_area(x, z, r * 0.85)
        if blockers:
            kinds = {b.kind for b in blockers}
            if kinds & {'oak', 'pine', 'birch'}:
                return False, 'Önce buradaki ağaçları kes'
            return False, 'Önce buradaki kayaları kaz'
        for b in self.buildings:
            br = max(b.spec.w, b.spec.d) / 2
            dd = math.hypot(b.x - x, b.z - z)
            if kind == 'yatak' and b.spec.floor is not None and b.floor_at(x, z) is not None:
                continue
            small = kind in ('cit', 'fener', 'yol') or b.kind in ('cit', 'fener', 'yol')
            if dd < (r + br) * (0.55 if small else 0.82):
                return False, 'Başka bir yapıyla çakışıyor'
        if math.hypot(x - MERCHANT[0], z - MERCHANT[1]) < 6 + r:
            return False, 'Tüccarın alanına çok yakın'
        if kind == 'yatak':
            pass
        return True, ''

    def rotate_preview(self, step=15):
        self.preview_rot = (self.preview_rot + step) % 360

    def place(self):
        if self.preview is None or not self.preview_ok:
            return None
        kind = self.preview_kind
        px, pz = self.preview_pos
        b = Building(self, kind, px, pz, self.preview_rot)
        self.buildings.append(b)
        sp = b.spec
        self.game.grass.add_blocker(px, pz, max(sp.w, sp.d) / 2 + 0.3)
        self.game.nature.remove_for_building(px, pz, max(sp.w, sp.d) / 2)
        return b

    def remove(self, b):
        self.buildings.remove(b)
        b.destroy()

    def on_complete(self, b):
        self.game.on_building_complete(b)

    # ------------------------------------------------------------------ sorgular
    def raycast(self, o, d, max_d=5.0):
        best, bd = None, max_d
        for b in self.buildings:
            if abs(b.x - o[0]) > 25 or abs(b.z - o[2]) > 25:
                continue
            t = b.ray_hit(o, d, bd)
            if t is not None and t < bd:
                best, bd = b, t
        return best, bd

    def floor_height(self, x, z):
        best = None
        for b in self.buildings:
            if abs(b.x - x) > 14 or abs(b.z - z) > 14:
                continue
            f = b.floor_at(x, z)
            if f is not None and (best is None or f > best):
                best = f
        return best

    def segments_near(self, x, z, r=12):
        out = []
        for b in self.buildings:
            if abs(b.x - x) < r + 8 and abs(b.z - z) < r + 8:
                out += b.wall_segments_world()
        return out

    def circles_near(self, x, z, r=12):
        out = []
        for b in self.buildings:
            if abs(b.x - x) < r + 8 and abs(b.z - z) < r + 8:
                out += b.circles_world()
        return out

    def count(self, kind, complete=True):
        return sum(1 for b in self.buildings if b.kind == kind and (b.complete or not complete))

    def to_list(self):
        return [b.to_dict() for b in self.buildings]

    def from_list(self, lst):
        for d in lst:
            b = Building(self, d['kind'], d['x'], d['z'], d['rot'], d['y'], d['depth'], d['built'], d.get('uid'))
            b.data = d.get('data', {})
            self.buildings.append(b)
            sp = b.spec
            self.game.grass.add_blocker(b.x, b.z, max(sp.w, sp.d) / 2 + 0.3)


_SPEC_CACHE = {}


def _spec_cache(kind):
    if kind not in _SPEC_CACHE:
        _SPEC_CACHE[kind] = spec_for(kind, 0.4)
    return _SPEC_CACHE[kind]

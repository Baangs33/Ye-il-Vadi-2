"""Doğa: gerçekçi ağaç/kaya/çalı modelleri, kaynak yönetimi (kesme, kazma, toplama),
akışlı (oyuncu çevresinde üretilen) çim ve göl suyu."""
import math
import numpy as np
from ursina import Entity, Vec3, destroy
from panda3d.core import GeomNode, TransparencyAttrib, NodePath

from config import *
from geometry import (Model, MeshBuilder, tube_tris, quad_tris, quad_uv, sphere_tris, cylinder_tris, box_tris,
                      trs, mat_rot_y, fbm, smoothstep, make_geom, rng, value_noise)
from shaders import world_shader, water_shader
from terrain import dist_to_road, lake_mask
from environment import SHADOW_MASK
import materials

R = np.random.default_rng(2024)


# =========================================================================== yardımcılar
def basis(side, along, normal, pos):
    m = np.eye(4)
    m[:3, 0] = side
    m[:3, 1] = along
    m[:3, 2] = normal
    m[:3, 3] = pos
    return m


def _norm(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def add_tube(mb, pts, radii, seg, uv_size=1.2, color=(1, 1, 1, 1), bump=0.6, sway=0.0):
    t, n, uv = tube_tris(pts, radii, seg)
    r_avg = float(np.mean(radii))
    uv = uv.copy()
    uv[:, :, 0] *= 2 * math.pi * r_avg / uv_size
    uv[:, :, 1] /= uv_size
    pr = np.zeros(t.shape[:2] + (4,))
    if sway:
        ys = t[:, :, 1]
        pr[:, :, 0] = sway * np.clip((ys - ys.min()) / max(np.ptp(ys), 1e-6), 0, 1) ** 2
    pr[:, :, 2] = bump
    mb.add(t, color, normals=n, uv=uv, vparams=pr)


def add_card(mb, m, w, h, color, sway, transl, canopy_center=None, normal=None, up_bias=0.5):
    tris = quad_tris(w, h)
    flat = tris.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    tw = flat.reshape(-1, 3, 3)
    if canopy_center is not None:
        nn = tw - np.asarray(canopy_center)
        nn = nn / (np.linalg.norm(nn, axis=2, keepdims=True) + 1e-9)
        nn = nn * (1 - up_bias) + np.array([0, 1.0, 0]) * up_bias
    elif normal is not None:
        nn = np.broadcast_to(np.asarray(normal, dtype=float), (2, 3, 3)).copy()
    else:
        nn = np.broadcast_to(np.array([0, 1.0, 0]), (2, 3, 3)).copy()
    pr = np.zeros((2, 3, 4))
    pr[:, :, 0] = sway
    pr[:, :, 3] = transl
    col = np.asarray(color, dtype=float)
    mb.add(tw, col, normals=nn, uv=quad_uv(), vparams=pr)


# =========================================================================== ağaç modelleri
def make_oak(seed, birch=False):
    r = np.random.default_rng(seed)
    md = Model()
    h = r.uniform(4.2, 5.6) * (1.25 if birch else 1.0)
    r0 = r.uniform(0.30, 0.40) * (0.65 if birch else 1.0)
    lean = r.uniform(-0.4, 0.4, 2)
    pts, radii = [], []
    for i in range(7):
        t = i / 6
        pts.append((lean[0] * t * t + math.sin(t * 3 + seed) * 0.08, -0.4 + t * h, lean[1] * t * t))
        radii.append(r0 * (1 - 0.55 * t) * (1.25 if i == 0 else 1.0))
    bark = 'birch' if birch else 'bark'
    add_tube(md[bark], pts, radii, 9, uv_size=1.4 if not birch else 2.0, bump=0.9)
    top = np.array(pts[-1])
    center = top + np.array([0, 1.0 if not birch else 0.6, 0])
    rx, ry = (2.7, 2.1) if not birch else (1.8, 2.4)
    tips = []
    for k in range(r.integers(5, 8)):
        t0 = r.uniform(0.5, 0.92)
        i0 = int(t0 * 6)
        base = np.array(pts[i0])
        ang = r.uniform(0, 2 * math.pi)
        dirv = _norm([math.cos(ang), r.uniform(0.5, 1.1) * (1.4 if birch else 1.0), math.sin(ang)])
        L = r.uniform(1.6, 2.6) * (0.8 if birch else 1.0)
        mid = base + dirv * L * 0.5 + np.array([0, 0.2, 0])
        end = base + dirv * L
        add_tube(md[bark], [base, mid, end], [radii[i0] * 0.5, radii[i0] * 0.3, 0.04], 5, bump=0.6, sway=0.15)
        tips.append(end)
    leaf = 'leaves_birch' if birch else 'leaves_oak'
    tint = np.array([1.0, 1.0, 1.0]) * r.uniform(0.85, 1.1)
    n_cards = 74 if not birch else 60
    for k in range(n_cards):
        if k < len(tips) * 6:
            p = tips[k % len(tips)] + r.normal(0, 0.55, 3)
        else:
            d = _norm(r.normal(0, 1, 3))
            d[1] = abs(d[1]) * 0.8 + 0.1
            rr = r.uniform(0.55, 1.0)
            p = center + d * np.array([rx, ry, rx]) * rr
        size = r.uniform(1.5, 2.3) * (0.85 if birch else 1.0)
        yaw = r.uniform(0, 360); pitch = r.uniform(-70, 70); roll = r.uniform(-40, 40)
        m = trs(p - np.array([0, size * 0.5, 0]), (pitch, yaw, roll))
        rel = (p[1] - (center[1] - ry)) / (2 * ry)
        ao = 0.55 + 0.45 * np.clip(rel, 0, 1)
        col = tuple(np.clip(tint * r.uniform(0.85, 1.1), 0, 1.4)) + (ao,)
        add_card(md[leaf], m, size, size, col, sway=0.55 + 0.3 * rel, transl=0.75, canopy_center=center, up_bias=0.35)
    return md, h + ry + 1.0, r0


def make_pine(seed):
    r = np.random.default_rng(seed)
    md = Model()
    h = r.uniform(9.0, 13.0)
    r0 = r.uniform(0.32, 0.42)
    pts = [(0, -0.4 + i / 6 * (h + 0.4), 0) for i in range(7)]
    radii = [r0 * (1 - 0.88 * i / 6) + 0.02 for i in range(7)]
    radii[0] *= 1.25
    add_tube(md['bark'], pts, radii, 8, uv_size=1.3, bump=0.9)
    y = r.uniform(1.8, 2.6)
    tint = np.array([1.0, 1.0, 1.0]) * r.uniform(0.85, 1.1)
    while y < h + 0.3:
        t = y / h
        L = (1 - t) * 3.4 + 0.7
        n = r.integers(6, 9)
        off = r.uniform(0, 360)
        for k in range(n):
            a = math.radians(off + k * 360 / n + r.uniform(-12, 12))
            out = np.array([math.cos(a), 0, math.sin(a)])
            droop = r.uniform(-0.45, -0.15)
            along = _norm(out + np.array([0, droop, 0]))
            side = _norm(np.cross([0, 1, 0], along))
            nrm = _norm(np.cross(along, side))
            for tilt in (0.0, 1.0):
                if tilt:
                    nrm2 = _norm(nrm * 0.5 + side * 0.86)
                    side2 = _norm(np.cross(along, nrm2) * -1)
                    m = basis(side2, along, nrm2, np.array([0, y, 0]) + out * 0.1)
                else:
                    m = basis(side, along, nrm, np.array([0, y, 0]) + out * 0.1)
                ao = 0.5 + 0.5 * t
                col = tuple(np.clip(tint * r.uniform(0.85, 1.1), 0, 1.4)) + (ao,)
                add_card(md['pine'], m, L * 0.95, L, col, sway=0.25 + 0.5 * t, transl=0.45,
                         canopy_center=np.array([0, y + 1.5, 0]), up_bias=0.55)
        y += r.uniform(0.55, 0.8)
    # tepe
    for k in range(3):
        m = trs((0, h - 0.4, 0), (0, k * 60, 0))
        add_card(md['pine'], m, 0.9, 1.6, (1, 1, 1, 1), sway=0.8, transl=0.4, normal=(0, 1, 0))
    return md, h + 1.2, r0


def make_bush(seed, berries):
    r = np.random.default_rng(seed)
    md = Model()
    center = np.array([0, 0.6, 0])
    for k in range(22):
        d = _norm(r.normal(0, 1, 3)); d[1] = abs(d[1]) * 0.7
        p = center + d * np.array([0.9, 0.6, 0.9]) * r.uniform(0.3, 1.0)
        s = r.uniform(0.8, 1.2)
        m = trs(p - np.array([0, s * 0.5, 0]), (r.uniform(-60, 60), r.uniform(0, 360), r.uniform(-30, 30)))
        add_card(md['leaves_oak'], m, s, s, (0.85, 0.95, 0.8, 0.8), sway=0.4, transl=0.6, canopy_center=center)
    if berries:
        for k in range(26):
            d = _norm(r.normal(0, 1, 3)); d[1] = abs(d[1]) * 0.8
            p = center + d * np.array([0.95, 0.62, 0.95])
            md['plain'].add(sphere_tris(0.055, 1), (0.42, 0.04, 0.10, 1), m=trs(tuple(p)), smooth='sphere',
                            spec=0.6, uv_mode='none')
    return md


def _rock_noise(seed):
    def fn(v):
        n = (value_noise(v[:, 0] * 2.1 + seed, v[:, 2] * 2.1 + v[:, 1] * 1.3, seed) - 0.5) * 0.5 \
            + (value_noise(v[:, 0] * 5.0, v[:, 2] * 5.0 + v[:, 1] * 3.0, seed + 3) - 0.5) * 0.18
        return 1 + n
    return fn


def make_rock(seed, iron=False):
    r = np.random.default_rng(seed)
    md = Model()
    sq = r.uniform(0.55, 0.85)
    tris = sphere_tris(1.0, 2, noise_fn=_rock_noise(seed), squash=sq)
    mat = 'ironore' if iron else 'rock'
    md[mat].add(tris, (1, 1, 1, 1), m=trs((0, 0.1, 0), (r.uniform(-8, 8), r.uniform(0, 360), r.uniform(-8, 8))),
                uv_scale=1.4, bump=1.6, spec=0.08 if not iron else 0.25)
    return md, sq


def make_stump(r0):
    md = Model()
    md['bark'].add(cylinder_tris(r0 * 1.15, r0, 0.55, 9, caps=False), (1, 1, 1, 1), m=trs((0, -0.2, 0)), uv_mode='cyl',
                   uv_scale=1.2, smooth='radial', bump=0.9)
    md['planks'].add(cylinder_tris(r0, r0, 0.02, 9), (1.1, 0.95, 0.8, 1), m=trs((0, 0.35, 0)), uv_scale=0.6)
    return md


# =========================================================================== kaynaklar
class Res:
    __slots__ = ('id', 'kind', 'x', 'y', 'z', 's', 'rot', 'var', 'hp', 'alive', 'chunk', 'regrow', 'height', 'radius',
                 'berries')


KIND_INFO = {
    'oak':   dict(tool='balta', hp=6, yield_=('kutuk', 8)),
    'birch': dict(tool='balta', hp=4, yield_=('kutuk', 5)),
    'pine':  dict(tool='balta', hp=7, yield_=('kutuk', 10)),
    'rock':  dict(tool='kazma', hp=5, yield_=('tas', 7)),
    'iron':  dict(tool='kazma', hp=8, yield_=('demir', 4)),
    'bush':  dict(tool=None, hp=1, yield_=('bogurtlen', 4)),
}


class Nature:
    CHUNK = 32.0

    def __init__(self, terrain, quality=2):
        self.terrain = terrain
        self.quality = quality
        self.root = Entity(shader=world_shader)
        self.dyn = Entity(shader=world_shader)
        self.variants = {}
        self._make_variants()
        self.items = []
        self.chunks = {}
        self.chunk_np = {}
        self.dirty = set()
        self.falling = []
        self._place_all()
        for k in list(self.chunks):
            self._build_chunk(k)
        self.lake = self._build_lake()

    # ------------------------------------------------------------------ modeller
    def _bake(self, model):
        out = {}
        for mat, mb in model.parts.items():
            a = mb.arrays()
            if a is not None:
                out[mat] = a
        return out

    def _make_variants(self):
        v = self.variants
        for i in range(4):
            md, h, r0 = make_oak(100 + i)
            v[('oak', i)] = (self._bake(md), h, r0)
        for i in range(4):
            md, h, r0 = make_pine(200 + i)
            v[('pine', i)] = (self._bake(md), h, r0)
        for i in range(3):
            md, h, r0 = make_oak(300 + i, birch=True)
            v[('birch', i)] = (self._bake(md), h, r0)
        for i in range(5):
            md, sq = make_rock(400 + i)
            v[('rock', i)] = (self._bake(md), sq, 1.0)
        for i in range(3):
            md, sq = make_rock(500 + i, iron=True)
            v[('iron', i)] = (self._bake(md), sq, 1.0)
        for i in range(3):
            v[('bush', i)] = (self._bake(make_bush(600 + i, True)), 1.2, 0.8)
            v[('bush_empty', i)] = (self._bake(make_bush(600 + i, False)), 1.2, 0.8)
        for k in ('oak', 'pine', 'birch'):
            r0 = {'oak': 0.36, 'pine': 0.38, 'birch': 0.24}[k]
            v[('stump_' + k, 0)] = (self._bake(make_stump(r0)), 0.4, r0)

    # ------------------------------------------------------------------ yerleşim
    def _place_all(self):
        g = self.terrain.ground_y
        rr = np.random.default_rng(77)
        N = 9000
        xs = rr.uniform(-WORLD + 8, WORLD - 8, N)
        zs = rr.uniform(-WORLD + 8, WORLD - 8, N)
        d = np.hypot(xs, zs)
        forest = fbm(xs * 0.011 + 20, zs * 0.011 - 30, 4, 71)
        dens = smoothstep(0.42, 0.62, forest) * 0.95 + 0.04
        dens *= smoothstep(16, 55, d)
        dens *= 1 - smoothstep(175, 200, d) * 0.7
        rd = dist_to_road(xs, zs)
        ys = self.terrain.ground_y_vec(xs, zs)
        lm = lake_mask(xs, zs)
        ok = (rr.random(N) < dens) & (rd > 6) & (lm < 0.01) & (ys > WATER_Y + 0.6) & (ys < 70)
        ok &= np.hypot(xs - MERCHANT[0], zs - MERCHANT[1]) > 14
        sel = np.nonzero(ok)[0]
        placed = []
        for i in sel:
            x, z = xs[i], zs[i]
            if self.terrain.slope(x, z) > 0.55:
                continue
            if any(abs(x - px) < 2.6 and abs(z - pz) < 2.6 for px, pz in placed[-40:]):
                continue
            hgt = ys[i]
            pick = rr.random()
            pine_bias = float(smoothstep(10, 35, hgt) + smoothstep(0.55, 0.7, forest[i]) * 0.4)
            if pick < 0.35 + pine_bias * 0.5:
                kind = 'pine'
            elif pick < 0.82:
                kind = 'oak'
            else:
                kind = 'birch'
            placed.append((x, z))
            self._add(kind, x, z, rr)
            if len(placed) > 1900:
                break
        # merkeze yakın birkaç seyrek ağaç (manzara için)
        for _ in range(40):
            a = rr.uniform(0, 2 * math.pi); r = rr.uniform(22, 50)
            x, z = math.cos(a) * r, math.sin(a) * r
            if dist_to_road(x, z) > 7 and lake_mask(x, z) < 0.01:
                self._add('oak' if rr.random() < 0.7 else 'birch', x, z, rr)
        # kayalar
        for _ in range(420):
            x, z = rr.uniform(-WORLD + 6, WORLD - 6), rr.uniform(-WORLD + 6, WORLD - 6)
            if dist_to_road(x, z) < 5 or lake_mask(x, z) > 0.2 or math.hypot(x, z) < 12:
                continue
            y = g(x, z)
            iron = (y > 6 and rr.random() < 0.35) or rr.random() < 0.06
            self._add('iron' if iron else 'rock', x, z, rr)
        # böğürtlen çalıları
        for _ in range(320):
            x, z = rr.uniform(-170, 170), rr.uniform(-170, 170)
            if dist_to_road(x, z) < 5 or lake_mask(x, z) > 0.01 or g(x, z) < WATER_Y + 0.5:
                continue
            if fbm(x * 0.011 + 20, z * 0.011 - 30, 4, 71) < 0.38 and rr.random() < 0.6:
                continue
            self._add('bush', x, z, rr)
        # başlangıçta yakında birkaç çalı ve kaya
        for (x, z, k) in [(14, 6, 'bush'), (-12, 10, 'bush'), (-18, -14, 'rock'), (20, 18, 'rock'), (-25, 22, 'iron')]:
            self._add(k, x, z, rr)

    def _add(self, kind, x, z, rr):
        it = Res()
        it.id = len(self.items)
        it.kind = kind
        it.x, it.z = float(x), float(z)
        it.y = float(self.terrain.ground_y(x, z))
        it.rot = float(rr.uniform(0, 360))
        nvar = {'oak': 4, 'pine': 4, 'birch': 3, 'rock': 5, 'iron': 3, 'bush': 3}[kind]
        it.var = int(rr.integers(0, nvar))
        if kind in ('rock', 'iron'):
            it.s = float(rr.uniform(0.6, 1.6) if kind == 'rock' else rr.uniform(0.8, 1.3))
        elif kind == 'bush':
            it.s = float(rr.uniform(0.8, 1.2))
        else:
            it.s = float(rr.uniform(0.85, 1.25))
        it.hp = KIND_INFO[kind]['hp'] * (it.s if kind in ('rock', 'iron') else 1)
        it.alive = True
        it.regrow = 0.0
        it.berries = kind == 'bush'
        _, h, r0 = self.variants[(kind, it.var)]
        if kind in ('rock', 'iron'):
            it.height = h * it.s * 1.1 + 0.1
            it.radius = it.s * 1.0
        elif kind == 'bush':
            it.height = 1.3 * it.s
            it.radius = 0.9 * it.s
        else:
            it.height = h * it.s
            it.radius = r0 * it.s
        it.chunk = (int(math.floor(x / self.CHUNK)), int(math.floor(z / self.CHUNK)))
        self.chunks.setdefault(it.chunk, []).append(it)
        self.items.append(it)
        return it

    # ------------------------------------------------------------------ parça mesh'leri
    def _instance_arrays(self, it):
        if it.alive:
            key = ('bush_empty', it.var) if (it.kind == 'bush' and not it.berries) else (it.kind, it.var)
        else:
            if it.kind not in ('oak', 'pine', 'birch'):
                return None
            key = ('stump_' + it.kind, 0)
        baked = self.variants[key][0]
        s = it.s if it.alive or True else 1
        if it.kind in ('rock', 'iron'):
            m = trs((it.x, it.y - 0.25 * s, it.z), (0, it.rot, 0), (s, s, s))
        else:
            m = trs((it.x, it.y, it.z), (0, it.rot, 0), (s, s, s))
        return baked, m

    def _build_chunk(self, key):
        if key in self.chunk_np:
            self.chunk_np[key].removeNode()
        acc = {}
        for it in self.chunks.get(key, []):
            r = self._instance_arrays(it)
            if r is None:
                continue
            baked, m = r
            M3 = m[:3, :3]
            Ninv = np.linalg.inv(M3).T
            for mat, (P, N, Cc, U, Q) in baked.items():
                P2 = P @ M3.T + m[:3, 3]
                N2 = N @ Ninv.T
                N2 /= np.linalg.norm(N2, axis=1, keepdims=True) + 1e-9
                acc.setdefault(mat, []).append((P2, N2, Cc, U, Q))
        gn = GeomNode('chunk')
        for mat, lst in acc.items():
            arrs = [np.concatenate([a[i] for a in lst]) for i in range(5)]
            gn.add_geom(make_geom(*arrs), materials.state(mat))
        self.chunk_np[key] = self.root.attachNewNode(gn)

    def flush(self):
        for k in self.dirty:
            self._build_chunk(k)
        self.dirty.clear()

    # ------------------------------------------------------------------ sorgular
    def nearby(self, x, z, r):
        out = []
        cx0, cx1 = int(math.floor((x - r) / self.CHUNK)), int(math.floor((x + r) / self.CHUNK))
        cz0, cz1 = int(math.floor((z - r) / self.CHUNK)), int(math.floor((z + r) / self.CHUNK))
        for i in range(cx0, cx1 + 1):
            for j in range(cz0, cz1 + 1):
                for it in self.chunks.get((i, j), ()):
                    if abs(it.x - x) < r + it.radius and abs(it.z - z) < r + it.radius:
                        out.append(it)
        return out

    def colliders(self, x, z, r=3.0):
        out = []
        for it in self.nearby(x, z, r):
            if it.kind == 'bush':
                continue
            if not it.alive:
                if it.kind in ('oak', 'pine', 'birch'):
                    out.append((it.x, it.z, it.radius * 1.1, it.y + 0.35))
                continue
            if it.kind in ('rock', 'iron'):
                out.append((it.x, it.z, it.radius * 0.85, it.y + it.height * 0.8))
            else:
                out.append((it.x, it.z, it.radius + 0.12, it.y + 50))
        return out

    def raycast(self, o, d, max_d=3.5):
        best, bd = None, max_d
        for it in self.nearby(o[0], o[2], max_d + 2):
            if not it.alive:
                continue
            rad = it.radius + (0.25 if it.kind in ('oak', 'pine', 'birch') else 0.0)
            # 2B daire - ışın kesişimi
            fx, fz = o[0] - it.x, o[2] - it.z
            a = d[0] ** 2 + d[2] ** 2
            if a < 1e-8:
                continue
            b = 2 * (fx * d[0] + fz * d[2])
            c = fx * fx + fz * fz - rad * rad
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            t = (-b - math.sqrt(disc)) / (2 * a)
            if t < 0:
                t = 0 if c < 0 else None
            if t is None or t > bd:
                continue
            y = o[1] + d[1] * t
            top = it.y + (it.height if it.kind not in ('oak', 'pine', 'birch') else 3.0)
            if it.y - 0.5 <= y <= top:
                best, bd = it, t
        return best, bd

    def blocked_area(self, x, z, r):
        return [it for it in self.nearby(x, z, r) if it.alive and it.kind != 'bush'
                and math.hypot(it.x - x, it.z - z) < r + it.radius]

    # ------------------------------------------------------------------ etkileşim
    def hit(self, it, tool, power=1.0):
        info = KIND_INFO[it.kind]
        if it.kind == 'bush':
            if not it.berries:
                return 'empty', None
            it.berries = False
            it.regrow = DAY_LENGTH * 0.8
            self.dirty.add(it.chunk)
            return 'picked', ('bogurtlen', int(3 + R.integers(0, 3)))
        if info['tool'] != tool:
            return 'wrongtool', info['tool']
        it.hp -= power
        if it.hp > 0:
            return 'hit', None
        it.alive = False
        self.dirty.add(it.chunk)
        kind, n = info['yield_']
        n = int(round(n * (it.s if it.kind not in ('rock', 'iron') else it.s ** 0.7)))
        if it.kind in ('oak', 'pine', 'birch'):
            self._spawn_fall(it)
        return 'done', (kind, max(1, n))

    def _spawn_fall(self, it):
        baked = self.variants[(it.kind, it.var)][0]
        gn = GeomNode('fall')
        for mat, arrs in baked.items():
            gn.add_geom(make_geom(*arrs), materials.state(mat))
        e = Entity(shader=world_shader, position=(it.x, it.y + 0.3, it.z), rotation_y=it.rot, scale=it.s)
        e.attachNewNode(gn)
        self.falling.append([e, 0.0, R.uniform(0, 360)])

    def remove_for_building(self, x, z, r):
        """İnşaat alanındaki çalıları kaldırır (ağaç ve kayalar engel sayılır)."""
        for it in self.nearby(x, z, r):
            if it.kind == 'bush' and math.hypot(it.x - x, it.z - z) < r:
                it.alive = False
                self.dirty.add(it.chunk)

    def update(self, dt):
        for f in list(self.falling):
            e, t, dirn = f
            t += dt
            f[1] = t
            k = min(1.0, (t / 2.2) ** 2)
            e.rotation_x = 88 * k
            e.rotation_y = dirn
            if t > 2.2:
                e.y -= dt * 1.0
                e.alpha = max(0, 1 - (t - 2.2) / 1.6) if False else 1
            if t > 4.5:
                destroy(e)
                self.falling.remove(f)
        for it in self.items:
            pass
        if self.dirty:
            self.flush()

    def tick_regrow(self, dt):
        for it in self.items:
            if it.kind == 'bush' and it.alive and not it.berries:
                it.regrow -= dt
                if it.regrow <= 0:
                    it.berries = True
                    self.dirty.add(it.chunk)

    # ------------------------------------------------------------------ kayıt
    def to_dict(self):
        return {'dead': [it.id for it in self.items if not it.alive],
                'picked': [[it.id, round(it.regrow, 1)] for it in self.items if it.kind == 'bush' and not it.berries],
                'hp': [[it.id, round(it.hp, 2)] for it in self.items if it.alive and it.hp < KIND_INFO[it.kind]['hp'] * (it.s if it.kind in ('rock', 'iron') else 1)]}

    def from_dict(self, d):
        for i in d.get('dead', []):
            if 0 <= i < len(self.items):
                self.items[i].alive = False
                self.dirty.add(self.items[i].chunk)
        for i, rg in d.get('picked', []):
            if 0 <= i < len(self.items):
                self.items[i].berries = False
                self.items[i].regrow = rg
                self.dirty.add(self.items[i].chunk)
        for i, hp in d.get('hp', []):
            if 0 <= i < len(self.items):
                self.items[i].hp = hp
        self.flush()

    # ------------------------------------------------------------------ göl
    def _build_lake(self):
        px, pz, pr = LAKE
        R2 = pr * 1.5
        step = 1.5
        xs = np.arange(-R2, R2 + 0.01, step)
        X, Z = np.meshgrid(xs, xs)
        X = X + px; Z = Z + pz
        H = self.terrain.ground_y_vec(X, Z)
        tris, uvs = [], []
        n = len(xs)
        for i in range(n - 1):
            for j in range(n - 1):
                hs = [H[j, i], H[j, i + 1], H[j + 1, i], H[j + 1, i + 1]]
                if min(hs) > WATER_Y + 0.3:
                    continue
                p00 = (X[j, i] - px, 0, Z[j, i] - pz); p10 = (X[j, i + 1] - px, 0, Z[j, i + 1] - pz)
                p01 = (X[j + 1, i] - px, 0, Z[j + 1, i] - pz); p11 = (X[j + 1, i + 1] - px, 0, Z[j + 1, i + 1] - pz)
                d00, d10, d01, d11 = [WATER_Y - h for h in hs]
                tris += [(p00, p01, p11), (p00, p11, p10)]
                uvs += [((d00, 0), (d01, 0), (d11, 0)), ((d00, 0), (d11, 0), (d10, 0))]
        mb = MeshBuilder().add(np.array(tris), (1, 1, 1, 1), uv=np.array(uvs), normals=np.tile([0, 1.0, 0], (len(tris), 3, 1)))
        e = Entity(shader=water_shader, position=(px, WATER_Y, pz))
        gn = GeomNode('lake')
        gn.add_geom(mb.geom())
        e.attachNewNode(gn)
        e.set_shader_input('tex_noise', materials.get_texture('noise'))
        e.setTransparency(TransparencyAttrib.M_alpha)
        e.setBin('transparent', 10)
        e.setDepthWrite(False)
        e.hide(SHADOW_MASK)
        return e


# =========================================================================== akışlı çim
class GrassField:
    CH = 16.0

    def __init__(self, terrain, nature, density=1.0, radius=56.0):
        self.terrain = terrain
        self.nature = nature
        self.density = density
        self.radius = radius
        self.root = Entity(shader=world_shader)
        self.root.hide(SHADOW_MASK)
        self.chunks = {}
        self.blockers = []        # (x, z, yarıçap) - binalar/tarlalar
        self.queue = []

    def set_density(self, density, radius):
        self.density = density
        self.radius = radius
        for k in list(self.chunks):
            np_ = self.chunks.pop(k)
            if np_ is not None:
                np_.removeNode()

    def add_blocker(self, x, z, r):
        self.blockers.append((x, z, r))
        for k in list(self.chunks):
            cx, cz = (k[0] + 0.5) * self.CH, (k[1] + 0.5) * self.CH
            if abs(cx - x) < r + self.CH and abs(cz - z) < r + self.CH:
                np_ = self.chunks.pop(k)
                if np_ is not None:
                    np_.removeNode()

    def _gen(self, key):
        i, j = key
        seed = (i * 73856093) ^ (j * 19349663) & 0xFFFFFFF
        r = np.random.default_rng(abs(seed))
        n = int(700 * self.density)
        if n <= 0:
            return None
        xs = (i + r.random(n)) * self.CH
        zs = (j + r.random(n)) * self.CH
        ys = self.terrain.ground_y_vec(xs, zs)
        keep = ys > WATER_Y + 0.25
        keep &= dist_to_road(xs, zs) > 2.6
        keep &= lake_mask(xs, zs) < 0.05
        dens = fbm(xs * 0.05, zs * 0.05, 2, 90)
        keep &= r.random(n) < 0.35 + 0.9 * dens
        for bx, bz, br in self.blockers:
            keep &= np.hypot(xs - bx, zs - bz) > br
        e = 0.8
        sl = np.abs(self.terrain.ground_y_vec(xs + e, zs) - self.terrain.ground_y_vec(xs - e, zs)) + \
            np.abs(self.terrain.ground_y_vec(xs, zs + e) - self.terrain.ground_y_vec(xs, zs - e))
        keep &= sl < 1.3
        xs, zs, ys = xs[keep], zs[keep], ys[keep]
        m = len(xs)
        if m == 0:
            return None
        flower = r.random(m) < 0.07
        w = r.uniform(0.38, 0.7, m); h = r.uniform(0.28, 0.58, m)
        yaw = np.radians(r.uniform(0, 180, m))
        tint = r.uniform(0.85, 1.05, m)
        dry = r.random(m) < 0.15
        col = np.stack([tint * np.where(dry, 1.12, 1.0), tint, tint * np.where(dry, 0.8, 0.95)], -1)
        lean = r.uniform(-0.08, 0.08, (m, 2))
        quads_p, quads_c, quads_f = [], [], []
        for extra in (0.0, np.pi / 2):
            ca, sa = np.cos(yaw + extra), np.sin(yaw + extra)
            hw = w / 2
            bl = np.stack([xs - ca * hw, ys - 0.04, zs + sa * hw], -1)
            br = np.stack([xs + ca * hw, ys - 0.04, zs - sa * hw], -1)
            tl = np.stack([xs - ca * hw + lean[:, 0], ys - 0.04 + h, zs + sa * hw + lean[:, 1]], -1)
            tr = np.stack([xs + ca * hw + lean[:, 0], ys - 0.04 + h, zs - sa * hw + lean[:, 1]], -1)
            # quad_tris sırası: (a, c, b), (a, d, c) ; a=bl b=br c=tr d=tl
            t1 = np.stack([bl, tr, br], 1)
            t2 = np.stack([bl, tl, tr], 1)
            quads_p.append(np.stack([t1, t2], 1))
        P = np.concatenate(quads_p, 0).reshape(-1, 3, 3)
        F = np.concatenate([flower, flower])
        C2 = np.concatenate([col, col])
        VC = np.ones((len(C2), 2, 3, 4))
        VC[..., :3] = C2[:, None, None, :]
        bot = 0.78
        VC[:, 0, 0, :3] *= bot; VC[:, 0, 2, :3] *= bot; VC[:, 1, 0, :3] *= bot
        VC = VC.reshape(-1, 3, 4)
        UV = np.tile(quad_uv(), (len(C2), 1, 1))
        PR = np.zeros((len(C2), 2, 3, 4))
        PR[:, 0, :, 0] = (0, 0.9, 0); PR[:, 1, :, 0] = (0, 0.9, 0.9); PR[..., 3] = 0.22
        PR = PR.reshape(-1, 3, 4)
        NR = np.zeros_like(P); NR[..., 1] = 1.0
        F3 = np.repeat(F, 2)
        mbg = MeshBuilder(); mbf = MeshBuilder()
        if (~F3).any():
            mbg.add(P[~F3], vcolors=VC[~F3], uv=UV[~F3], normals=NR[~F3], vparams=PR[~F3])
        if F3.any():
            mbf.add(P[F3], vcolors=VC[F3], uv=UV[F3], normals=NR[F3], vparams=PR[F3])
        gn = GeomNode('grass')
        for mb, mat in ((mbg, 'grassblades'), (mbf, 'flowers')):
            g = mb.geom()
            if g is not None:
                gn.add_geom(g, materials.state(mat))
        return self.root.attachNewNode(gn)

    def update(self, px, pz, budget=2):
        R_ = self.radius
        ci0, ci1 = int(math.floor((px - R_) / self.CH)), int(math.floor((px + R_) / self.CH))
        cj0, cj1 = int(math.floor((pz - R_) / self.CH)), int(math.floor((pz + R_) / self.CH))
        want = set()
        for i in range(ci0, ci1 + 1):
            for j in range(cj0, cj1 + 1):
                cx, cz = (i + 0.5) * self.CH, (j + 0.5) * self.CH
                if math.hypot(cx - px, cz - pz) < R_ + self.CH * 0.7:
                    want.add((i, j))
        for k in list(self.chunks):
            if k not in want:
                np_ = self.chunks.pop(k)
                if np_ is not None:
                    np_.removeNode()
        missing = sorted([k for k in want if k not in self.chunks],
                         key=lambda k: math.hypot((k[0] + 0.5) * self.CH - px, (k[1] + 0.5) * self.CH - pz))
        for k in missing[:budget]:
            self.chunks[k] = self._gen(k)

    def fill_all(self, px, pz):
        self.update(px, pz, budget=10000)

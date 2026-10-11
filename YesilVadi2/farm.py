"""Tarla parselleri: çapalama, ekim, sulama, büyüme ve hasat. Ekin modelleri prosedüreldir."""
import math
import numpy as np
from ursina import Entity, Vec3
from panda3d.core import NodePath

from config import *
from geometry import Model, box_tris, cylinder_tris, cone_tris, sphere_tris, quad_tris, quad_uv, trs, rng
from shaders import world_shader, glow_shader
from environment import SHADOW_MASK

TILE = 1.4
N = 4
WATER_TIME = DAY_LENGTH * 0.55
LEAF = (0.30, 0.55, 0.18, 1)
LEAF_D = (0.20, 0.42, 0.13, 1)


def _blade(md, x, z, h, w, lean, yaw, col, sway=0.9):
    tris = np.array([[(-w / 2, 0, 0), (0, h, lean), (w / 2, 0, 0)]], dtype=float)
    md['plain'].add(tris, col, m=trs((x, 0.06, z), (0, yaw, 0)), uv_mode='none', sway=sway, transl=0.4,
                    normals=np.tile([0, 1.0, 0], (1, 3, 1)))


def _sprout(md, x, z, s=1.0):
    md['plain'].add(cylinder_tris(0.015, 0.015, 0.16 * s, 4), LEAF, m=trs((x, 0.06, z)), uv_mode='none', sway=0.4)
    for a in (0, 180):
        md['plain'].add(sphere_tris(0.07 * s, 1, squash=0.3), (0.42, 0.72, 0.28, 1),
                        m=trs((x, 0.06 + 0.16 * s, z), (0, a, 0)) @ trs((0.06 * s, 0, 0)), uv_mode='none', sway=0.5,
                        smooth='sphere', transl=0.4)


def _spots(n, r):
    w = TILE * 0.9
    if n == 1:
        return [(0.0, 0.0)]
    if n == 4:
        return [(x, z) for x in (-0.3, 0.3) for z in (-0.3, 0.3)]
    pts = []
    for i in range(3):
        for k in range(3):
            pts.append(((k - 1) * w / 3 + r.uniform(-0.05, 0.05), (i - 1) * w / 3 + r.uniform(-0.05, 0.05)))
    return pts


def crop_model(kind, stage):
    r = np.random.default_rng(hash((kind, stage)) & 0xFFFF)
    md = Model()
    if stage == 0:
        for x, z in _spots(4, r):
            _sprout(md, x, z, 1.0)
        return md
    if kind == 'bugday':
        h = [0, 0.45, 0.85, 1.1][stage]
        col = (0.85, 0.70, 0.32, 1) if stage == 3 else (0.36, 0.60, 0.22, 1)
        for x, z in _spots(9, r):
            for k in range(6):
                yaw = r.uniform(0, 360); lean = r.uniform(0.04, 0.15)
                _blade(md, x + r.uniform(-0.06, 0.06), z + r.uniform(-0.06, 0.06), h * r.uniform(0.85, 1.1), 0.05, lean, yaw, col)
                if stage >= 2:
                    ec = (0.95, 0.80, 0.40, 1) if stage == 3 else (0.55, 0.70, 0.30, 1)
                    md['plain'].add(sphere_tris(0.022, 1, squash=4.0), ec,
                                    m=trs((x, 0.06, z), (0, yaw, 0)) @ trs((0, h + 0.03, lean)), uv_mode='none', sway=1.0,
                                    smooth='sphere')
    elif kind == 'havuc':
        for x, z in _spots(4, r):
            n = [0, 6, 8, 10][stage]
            hh = [0, 0.28, 0.42, 0.5][stage]
            for k in range(n):
                _blade(md, x, z, hh * r.uniform(0.8, 1.1), 0.1, r.uniform(0.1, 0.25), k / n * 360, (0.28, 0.55, 0.18, 1), 0.7)
            if stage >= 2:
                rr = 0.07 if stage == 2 else 0.11
                md['plain'].add(cylinder_tris(rr, rr * 1.15, 0.12, 9), (0.95, 0.48, 0.12, 1), m=trs((x, 0.0, z)),
                                uv_mode='none', smooth='radial', spec=0.2)
    elif kind == 'domates':
        for x, z in _spots(4, r)[:2] + [(0.3, -0.3), (-0.3, 0.3)][:0]:
            hh = [0, 0.5, 0.9, 1.1][stage]
            md['planks'].add(cylinder_tris(0.02, 0.02, hh + 0.3, 5), (1, 1, 1, 1), m=trs((x + 0.08, 0, z)))
            for k in range([0, 3, 5, 7][stage]):
                y = 0.15 + (k + 1) / 8 * hh
                for c in range(5):
                    a = r.uniform(0, 360)
                    _blade(md, x, z, 0.0001, 0.0, 0, 0, LEAF) if False else None
                    md['plain'].add(sphere_tris(0.12, 1, squash=0.35), LEAF,
                                    m=trs((x, 0.06 + y, z), (r.uniform(-30, 30), a, 0)) @ trs((0.1, 0, 0)), uv_mode='none',
                                    sway=0.5, transl=0.4, jitter=0.1)
            if stage >= 2:
                col = (0.85, 0.12, 0.08, 1) if stage == 3 else (0.45, 0.65, 0.22, 1)
                for k in range(6 if stage == 3 else 4):
                    a = r.uniform(0, 2 * math.pi)
                    md['plain'].add(sphere_tris(0.075 if stage == 3 else 0.05, 2), col,
                                    m=trs((x + math.cos(a) * 0.17, r.uniform(0.3, hh), z + math.sin(a) * 0.17)),
                                    uv_mode='none', smooth='sphere', spec=0.8, sway=0.4)
    elif kind == 'misir':
        for x, z in _spots(4, r):
            hh = [0, 0.6, 1.3, 1.9][stage]
            md['plain'].add(cylinder_tris(0.03, 0.02, hh, 6), (0.40, 0.58, 0.22, 1), m=trs((x, 0.05, z)), uv_mode='none',
                            smooth='radial', sway=0.3)
            for k in range([0, 3, 5, 7][stage]):
                y = 0.2 + k * hh / 7.5
                yaw = k * 137
                _blade(md, x, z, 0.0, 0.0, 0, 0, LEAF) if False else None
                tris = np.array([[(-0.04, 0, 0), (0.0, 0.05, 0.55), (0.04, 0, 0)]], dtype=float)
                md['plain'].add(tris, (0.38, 0.6, 0.22, 1), m=trs((x, 0.05 + y, z), (-20, yaw, 0)), uv_mode='none',
                                sway=0.7, transl=0.4, normals=np.tile([0, 1.0, 0], (1, 3, 1)))
            if stage == 3:
                for side in (1, -1):
                    md['plain'].add(sphere_tris(0.05, 2, squash=3.0), (0.98, 0.85, 0.30, 1),
                                    m=trs((x + side * 0.07, 0.05 + hh * 0.55, z), (0, 0, side * 20)), uv_mode='none',
                                    smooth='sphere', sway=0.3)
                md['plain'].add(cone_tris(0.04, 0.25, 5), (0.85, 0.75, 0.45, 1), m=trs((x, 0.05 + hh, z)), uv_mode='none', sway=0.6)
    elif kind == 'balkabagi':
        n = [0, 5, 8, 10][stage]
        for k in range(n):
            a = k / max(n, 1) * 2 * math.pi + r.uniform(-0.2, 0.2)
            rr = r.uniform(0.2, 0.55)
            x, z = math.cos(a) * rr, math.sin(a) * rr
            md['plain'].add(cylinder_tris(0.015, 0.015, 0.22, 4), LEAF_D, m=trs((x, 0.05, z)), uv_mode='none', sway=0.4)
            md['plain'].add(sphere_tris(0.2, 1, squash=0.22), LEAF, m=trs((x, 0.27, z), (r.uniform(-15, 15), r.uniform(0, 360), 0)),
                            uv_mode='none', sway=0.5, transl=0.4, jitter=0.1, smooth='sphere')
        if stage >= 2:
            big = stage == 3
            col = (0.95, 0.50, 0.10, 1) if big else (0.42, 0.58, 0.20, 1)
            rr = 0.36 if big else 0.18
            for k in range(8):
                md['plain'].add(sphere_tris(rr * 0.55, 2, squash=1.4), col,
                                m=trs((0.1, rr * 0.8, 0.1), (0, k * 45, 0)) @ trs((rr * 0.42, 0, 0)), uv_mode='none',
                                smooth='sphere', spec=0.4)
            md['plain'].add(cylinder_tris(0.04, 0.03, 0.18, 5), (0.4, 0.45, 0.2, 1), m=trs((0.1, rr * 1.55, 0.1), (0, 0, 12)),
                            uv_mode='none')
    return md


def soil_model(wet):
    md = Model()
    c = (0.62, 0.55, 0.5, 1) if wet else (1.0, 0.92, 0.82, 1)
    md['soil'].add(box_tris(TILE * 0.92, 0.12, TILE * 0.92, bottom=True), c, m=trs((0, -0.04, 0)), uv_scale=1.4, bump=0.8,
                   spec=0.25 if wet else 0.0)
    return md


class FarmCache:
    def __init__(self):
        self.crops = {}
        self.soil = {w: soil_model(w).nodepath('soil') for w in (False, True)}

    def crop(self, kind, stage):
        k = (kind, stage)
        if k not in self.crops:
            self.crops[k] = crop_model(kind, stage).nodepath('crop')
        return self.crops[k]


class Plot:
    """Bir 'tarla' yapısına bağlı 4x4 karo."""

    def __init__(self, game, building, cache):
        self.game = game
        self.b = building
        self.cache = cache
        self.holder = building.root.attachNewNode('farm')
        if 'tiles' not in building.data:
            building.data['tiles'] = {}
        self.nodes = {}
        for key in list(building.data['tiles']):
            self.refresh(key)

    @staticmethod
    def key(i, j):
        return '%d,%d' % (i, j)

    def center_local(self, i, j):
        return (i - (N - 1) / 2) * TILE, (j - (N - 1) / 2) * TILE

    def tile_at_world(self, x, z):
        lx, lz = self.b.to_local(x, z)
        i = int(math.floor(lx / TILE + N / 2))
        j = int(math.floor(lz / TILE + N / 2))
        if 0 <= i < N and 0 <= j < N:
            return i, j
        return None

    def get(self, i, j):
        return self.b.data['tiles'].get(self.key(i, j))

    def refresh(self, key):
        t = self.b.data['tiles'].get(key)
        if key in self.nodes:
            self.nodes.pop(key).remove_node()
        if not t:
            return
        tilled, wet, crop, growth = t
        i, j = map(int, key.split(','))
        lx, lz = self.center_local(i, j)
        h = self.holder.attach_new_node('tile')
        h.set_pos(lx, 0.0, lz)
        self.cache.soil[wet > 0].instance_to(h)
        if crop:
            st = self.stage(t)
            c = h.attach_new_node('c')
            c.set_h((i * 7 + j * 3) % 4 * 90)
            self.cache.crop(crop, st).instance_to(c)
        self.nodes[key] = h

    @staticmethod
    def stage(t):
        tilled, wet, crop, growth = t
        if not crop:
            return -1
        return min(3, int(growth / (CROPS[crop]['time'] / 3)))

    def act(self, ij, tool, seed_kind=None):
        k = self.key(*ij)
        tiles = self.b.data['tiles']
        t = tiles.get(k)
        if t and t[2] and self.stage(t) >= 3:
            kind = t[2]
            tiles[k] = [True, 0, None, 0]
            self.refresh(k)
            return 'harvest', kind
        if tool == 'capa':
            if t and t[2]:
                return 'busy', None
            if t and t[0]:
                return 'already', None
            tiles[k] = [True, 0, None, 0]
            self.refresh(k)
            return 'tilled', None
        if tool == 'kova':
            if not t:
                return 'untilled', None
            t[1] = WATER_TIME
            self.refresh(k)
            return 'watered', None
        if tool == 'tohum':
            if not t:
                return 'untilled', None
            if t[2]:
                return 'busy', None
            t[2] = seed_kind
            t[3] = 0.0
            self.refresh(k)
            return 'planted', None
        return None, None

    def update(self, dt, raining):
        for k, t in self.b.data['tiles'].items():
            changed = False
            if raining and t[1] < WATER_TIME * 0.5:
                if t[1] <= 0:
                    changed = True
                t[1] = WATER_TIME
            if t[1] > 0:
                t[1] -= dt
                if t[2] and self.stage(t) < 3:
                    before = self.stage(t)
                    t[3] += dt
                    if self.stage(t) != before:
                        changed = True
                if t[1] <= 0:
                    t[1] = 0
                    changed = True
            if changed:
                self.refresh(k)

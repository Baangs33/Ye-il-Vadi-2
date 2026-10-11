"""Büyük, dokulu arazi: yükseklik haritası, doku karışım ağırlıkları, parça (chunk) mesh'leri."""
import math
import numpy as np
from ursina import Entity
from panda3d.core import GeomNode, NodePath

from config import *
from geometry import fbm, ridged, smoothstep, make_geom, rng
from shaders import terrain_shader
import materials


def dist_to_road(x, z):
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    best = np.full(np.broadcast(x, z).shape, 1e9)
    for (ax, az), (bx, bz) in zip(ROAD[:-1], ROAD[1:]):
        dx, dz = bx - ax, bz - az
        L2 = dx * dx + dz * dz
        t = np.clip(((x - ax) * dx + (z - az) * dz) / L2, 0, 1)
        best = np.minimum(best, np.hypot(x - (ax + t * dx), z - (az + t * dz)))
    return best


def lake_mask(x, z):
    px, pz, pr = LAKE
    dp = np.hypot(x - px, z - pz)
    pr_eff = pr * (0.75 + 0.5 * fbm(x * 0.03 + 3, z * 0.03 + 7, 3, 4))
    return 1 - smoothstep(pr_eff * 0.3, pr_eff * 1.25, dp)


def height_fn(x, z):
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    d = np.hypot(x * 1.0, z * 0.95)
    rolling = (fbm(x * 0.012 + 40, z * 0.012 + 10, 4, 1) - 0.5) * 9.0
    detail = (fbm(x * 0.05, z * 0.05, 3, 2) - 0.5) * 1.6
    hills = ridged(x * 0.008 + 5, z * 0.008 + 9, 4, 3) ** 1.5 * 26.0 * smoothstep(45, 120, d)
    h = rolling * (0.35 + 0.65 * smoothstep(20, 80, d)) + detail + hills
    mount = smoothstep(125, 215, d) ** 1.6 * (30 + 34 * ridged(x * 0.017, z * 0.017, 5, 6))
    h = h + mount
    # yol hafifçe düzleşir
    rd = dist_to_road(x, z)
    road_k = 1 - smoothstep(3.0, 9.0, rd)
    h = h * (1 - 0.6 * road_k) + h * 0.6 * road_k * 0.65 - road_k * 0.08
    k = lake_mask(x, z)
    h = h * (1 - k) + (WATER_Y - 3.0) * k
    return h


class Terrain:
    CHUNK = 40.0

    def __init__(self):
        S = WORLD + 16
        st = TERRAIN_STEP
        self.xs = np.arange(-S, S + 0.001, st)
        self.n = len(self.xs)
        self.x0 = self.xs[0]
        X, Z = np.meshgrid(self.xs, self.xs, indexing='xy')
        self.H = height_fn(X, Z)
        gz, gx = np.gradient(self.H, st)
        nrm = np.stack([-gx, np.ones_like(gx), -gz], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        self.N = nrm
        W = self._weights(X, Z, self.H, 1 - nrm[..., 1])
        self.entity = Entity(shader=terrain_shader)
        for name in ('grass', 'dirt', 'rock', 'sand', 'noise'):
            self.entity.set_shader_input('tex_' + name, materials.get_texture(name))
        self._build(X, Z, W)

    def _weights(self, X, Z, H, slope):
        grass = np.ones_like(X)
        n1 = fbm(X * 0.04, Z * 0.04, 3, 50)
        dirt = smoothstep(0.62, 0.78, n1) * 0.8
        rd = dist_to_road(X, Z) + (fbm(X * 0.3, Z * 0.3, 2, 51) - 0.5) * 1.2
        dirt = np.maximum(dirt, 1 - smoothstep(1.6, 2.8, rd))
        rock = np.clip(smoothstep(0.22, 0.42, slope) + smoothstep(32, 48, H), 0, 1)
        sand = smoothstep(WATER_Y + 1.2, WATER_Y + 0.3, H) * (lake_mask(X, Z) > 0.005)
        grass = np.clip(1 - dirt - rock - sand, 0.0, 1)
        w = np.stack([grass, dirt, rock, sand], -1) + 1e-3
        return w / w.sum(-1, keepdims=True)

    def _build(self, X, Z, W):
        P = np.stack([X, self.H, Z], -1)
        N = self.N
        C = W
        n = self.n
        cells = int(round(self.CHUNK / TERRAIN_STEP))
        for ci in range(0, n - 1, cells):
            for cj in range(0, n - 1, cells):
                i1 = min(ci + cells, n - 1)
                j1 = min(cj + cells, n - 1)
                p = P[cj:j1 + 1, ci:i1 + 1]; nn = N[cj:j1 + 1, ci:i1 + 1]; cc = C[cj:j1 + 1, ci:i1 + 1]

                def tri(a):
                    a00, a10, a01, a11 = a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]
                    t1 = np.stack([a00, a01, a11], -2).reshape(-1, 3, a.shape[-1])
                    t2 = np.stack([a00, a11, a10], -2).reshape(-1, 3, a.shape[-1])
                    t = np.concatenate([t1, t2])
                    return t[:, ::-1]           # y-up-left sarım düzeltmesi
                tp, tn, tc = tri(p), tri(nn), tri(cc)
                m = len(tp) * 3
                geom = make_geom(tp.reshape(-1, 3), tn.reshape(-1, 3), tc.reshape(-1, 4),
                                 np.zeros((m, 2)), np.zeros((m, 4)))
                gn = GeomNode('terrain')
                gn.add_geom(geom)
                self.entity.attachNewNode(gn)

    # ------------------------------------------------------------------
    def ground_y(self, x, z):
        fx = (x - self.x0) / TERRAIN_STEP
        fz = (z - self.x0) / TERRAIN_STEP
        i = int(min(max(math.floor(fx), 0), self.n - 2))
        j = int(min(max(math.floor(fz), 0), self.n - 2))
        u, v = fx - i, fz - j
        H = self.H
        h00, h10, h01, h11 = H[j, i], H[j, i + 1], H[j + 1, i], H[j + 1, i + 1]
        if v >= u:
            return h00 + (h11 - h01) * u + (h01 - h00) * v
        return h00 + (h10 - h00) * u + (h11 - h10) * v

    def ground_y_vec(self, x, z):
        x = np.asarray(x, dtype=np.float64); z = np.asarray(z, dtype=np.float64)
        fx = (x - self.x0) / TERRAIN_STEP
        fz = (z - self.x0) / TERRAIN_STEP
        i = np.clip(np.floor(fx).astype(int), 0, self.n - 2)
        j = np.clip(np.floor(fz).astype(int), 0, self.n - 2)
        u, v = fx - i, fz - j
        H = self.H
        h00, h10, h01, h11 = H[j, i], H[j, i + 1], H[j + 1, i], H[j + 1, i + 1]
        a = h00 + (h11 - h01) * u + (h01 - h00) * v
        b = h00 + (h10 - h00) * u + (h11 - h10) * v
        return np.where(v >= u, a, b)

    def normal(self, x, z):
        e = 0.6
        hx = self.ground_y(x + e, z) - self.ground_y(x - e, z)
        hz = self.ground_y(x, z + e) - self.ground_y(x, z - e)
        n = np.array([-hx, 2 * e, -hz])
        return n / np.linalg.norm(n)

    def slope(self, x, z):
        return 1 - self.normal(x, z)[1]

    def raycast(self, origin, direction, max_d=8.0, step=0.15):
        ox, oy, oz = origin
        dx, dy, dz = direction
        t = 0.0
        prev = oy - self.ground_y(ox, oz)
        while t < max_d:
            t += step
            x, y, z = ox + dx * t, oy + dy * t, oz + dz * t
            g = self.ground_y(x, z)
            if y <= g:
                # ikiye bölme ile hassaslaştır
                lo, hi = t - step, t
                for _ in range(6):
                    mid = (lo + hi) / 2
                    if oy + dy * mid <= self.ground_y(ox + dx * mid, oz + dz * mid):
                        hi = mid
                    else:
                        lo = mid
                return hi
        return None

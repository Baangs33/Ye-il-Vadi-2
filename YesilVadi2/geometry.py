"""Prosedürel geometri çekirdeği.

Koordinatlar Ursina ile aynıdır: x sağ, y yukarı, z ileri (metre).
Her köşe şu verileri taşır: konum, normal, renk (RGBA, doku ile çarpılır), uv ve
'params' = (rüzgâr salınımı, parlaklık/specular, kabartma gücü, yarı saydamlık).
Bir model birden fazla malzeme (doku) içerebilir: Model -> {malzeme adı: MeshBuilder}.
"""
import math
import numpy as np
from panda3d.core import (GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData, Geom, GeomTriangles,
                          GeomNode, InternalName, RenderState, TextureAttrib, NodePath, TransparencyAttrib)

_rng = np.random.default_rng(11)


def rng():
    return _rng


# =========================================================================== matrisler
def mat_translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def mat_scale(sx, sy=None, sz=None):
    if sy is None:
        sy = sz = sx
    return np.diag([sx, sy, sz, 1.0])


def mat_rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def mat_rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def mat_rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def trs(pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    if np.isscalar(scale):
        scale = (scale, scale, scale)
    return mat_translate(*pos) @ mat_rot_y(rot[1]) @ mat_rot_x(rot[0]) @ mat_rot_z(rot[2]) @ mat_scale(*scale)


def look_rot(dx, dy, dz):
    """+y eksenini (dx,dy,dz) yönüne çeviren döndürme matrisi."""
    v = np.array([dx, dy, dz], dtype=float)
    v /= np.linalg.norm(v) + 1e-12
    up = np.array([0, 1.0, 0])
    axis = np.cross(up, v)
    s = np.linalg.norm(axis)
    c = float(np.dot(up, v))
    m = np.eye(4)
    if s < 1e-8:
        if c < 0:
            m[1, 1] = -1; m[0, 0] = -1
        return m
    axis /= s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + K * s + K @ K * (1 - c)
    m[:3, :3] = R
    return m


# =========================================================================== ilkel şekiller (n,3,3)
def _quad(a, b, c, d):
    return [(a, b, c), (a, c, d)]


def box_tris(sx=1.0, sy=1.0, sz=1.0, bottom=False):
    x, y, z = sx / 2, sy / 2, sz / 2
    oy = y if bottom else 0.0
    p = [(-x, -y + oy, -z), (x, -y + oy, -z), (x, y + oy, -z), (-x, y + oy, -z),
         (-x, -y + oy, z), (x, -y + oy, z), (x, y + oy, z), (-x, y + oy, z)]
    f = []
    f += _quad(p[0], p[3], p[2], p[1])
    f += _quad(p[5], p[6], p[7], p[4])
    f += _quad(p[4], p[7], p[3], p[0])
    f += _quad(p[1], p[2], p[6], p[5])
    f += _quad(p[3], p[7], p[6], p[2])
    f += _quad(p[4], p[0], p[1], p[5])
    return np.array(f, dtype=np.float64)


def cylinder_tris(r0=0.5, r1=0.5, h=1.0, seg=8, caps=True, jitter=0.0):
    tris = []
    ang = [2 * math.pi * i / seg for i in range(seg)]
    jit = [1 + (_rng.uniform(-jitter, jitter) if jitter else 0) for _ in range(seg)]
    bot = [(math.cos(a) * r0 * j, 0.0, math.sin(a) * r0 * j) for a, j in zip(ang, jit)]
    top = [(math.cos(a) * r1 * j, h, math.sin(a) * r1 * j) for a, j in zip(ang, jit)]
    for i in range(seg):
        k = (i + 1) % seg
        if r1 > 1e-6:
            tris += _quad(bot[i], top[i], top[k], bot[k])
        else:
            tris.append((bot[i], (0.0, h, 0.0), bot[k]))
        if caps:
            tris.append(((0.0, 0.0, 0.0), bot[i], bot[k]))
            if r1 > 1e-6:
                tris.append(((0.0, h, 0.0), top[k], top[i]))
    return np.array(tris, dtype=np.float64)


def cone_tris(r=0.5, h=1.0, seg=8, jitter=0.0):
    return cylinder_tris(r, 0.0, h, seg, True, jitter)


_ICO = {}


def _ico(sub):
    if sub in _ICO:
        return _ICO[sub]
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    v = [np.array(p, dtype=float) / np.linalg.norm(p) for p in v]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
         (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(sub):
        cache = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]
        nf = []
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    _ICO[sub] = (np.array(v), np.array(f))
    return _ICO[sub]


def sphere_tris(r=0.5, sub=1, jitter=0.0, squash=1.0, noise_fn=None):
    v, f = _ico(sub)
    v = v.copy()
    if noise_fn is not None:
        v *= noise_fn(v)[:, None]
    elif jitter:
        v *= (1 + _rng.uniform(-jitter, jitter, (len(v), 1)))
    v = v * r
    v[:, 1] *= squash
    tris = v[f]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    c = tris.mean(axis=1)
    flip = (n * c).sum(axis=1) < 0
    tris[flip] = tris[flip][:, ::-1]
    return tris


def prism_tris(w=1.0, h=1.0, d=1.0):
    x, z = w / 2, d / 2
    a0, b0, c0 = (-x, 0, -z), (x, 0, -z), (0, h, -z)
    a1, b1, c1 = (-x, 0, z), (x, 0, z), (0, h, z)
    tris = [(a0, c0, b0), (a1, b1, c1)]
    tris += _quad(a0, a1, c1, c0)
    tris += _quad(b0, c0, c1, b1)
    tris += _quad(a0, b0, b1, a1)
    return np.array(tris, dtype=np.float64)


def quad_tris(w=1.0, h=1.0):
    """XY düzleminde, alt kenarı y=0 olan dikey dörtgen (+z'ye bakar)."""
    a, b, c, d = (-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, h, 0), (-w / 2, h, 0)
    return np.array([(a, c, b), (a, d, c)], dtype=np.float64)


def quad_uv():
    return np.array([[(0, 0), (1, 1), (1, 0)], [(0, 0), (0, 1), (1, 1)]], dtype=np.float64)


def tube_tris(points, radii, seg=8):
    """Bir eğri boyunca konik boru. points: (k,3), radii: (k,). Silindirik uv ve yumuşak normaller döner."""
    pts = np.asarray(points, dtype=float)
    k = len(pts)
    rings = []
    for i in range(k):
        if i == 0:
            d = pts[1] - pts[0]
        elif i == k - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i + 1] - pts[i - 1]
        d /= np.linalg.norm(d) + 1e-12
        a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 0, 1.0])
        u = np.cross(d, a); u /= np.linalg.norm(u)
        v = np.cross(d, u)
        ring = []
        for s in range(seg + 1):
            ang = 2 * math.pi * s / seg
            n = math.cos(ang) * u + math.sin(ang) * v
            ring.append((pts[i] + n * radii[i], n))
        rings.append(ring)
    tris, nrm, uvs = [], [], []
    length = 0.0
    vcoord = [0.0]
    for i in range(1, k):
        length += np.linalg.norm(pts[i] - pts[i - 1])
        vcoord.append(length)
    for i in range(k - 1):
        for s in range(seg):
            p00, n00 = rings[i][s]; p01, n01 = rings[i][s + 1]
            p10, n10 = rings[i + 1][s]; p11, n11 = rings[i + 1][s + 1]
            u0, u1 = s / seg, (s + 1) / seg
            v0, v1 = vcoord[i], vcoord[i + 1]
            tris += [(p00, p10, p11), (p00, p11, p01)]
            nrm += [(n00, n10, n11), (n00, n11, n01)]
            uvs += [((u0, v0), (u0, v1), (u1, v1)), ((u0, v0), (u1, v1), (u1, v0))]
    return np.array(tris), np.array(nrm), np.array(uvs)


# =========================================================================== uv üretimi
def box_uv(tris, scale=1.0):
    """Yüz normalinin baskın eksenine göre düzlemsel izdüşüm (metre / scale)."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ax = np.argmax(np.abs(n), axis=1)
    uv = np.zeros(tris.shape[:2] + (2,))
    for a, (i, j) in enumerate(((2, 1), (0, 2), (0, 1))):
        sel = ax == a
        uv[sel, :, 0] = tris[sel][:, :, i]
        uv[sel, :, 1] = tris[sel][:, :, j]
    return uv / scale


def cyl_uv(tris, scale=1.0, radius=0.5):
    ang = np.arctan2(tris[:, :, 2], tris[:, :, 0])
    u = ang / (2 * np.pi)
    # dikiş: aynı üçgende 1'e sarılan köşeleri düzelt
    span = u.max(axis=1, keepdims=True) - u.min(axis=1, keepdims=True)
    u = np.where((span > 0.5) & (u < 0), u + 1, u)
    circumference = 2 * np.pi * radius
    return np.stack([u * circumference / scale, tris[:, :, 1] / scale], -1)


# =========================================================================== inşa edici
class MeshBuilder:
    def __init__(self):
        self.p, self.n, self.c, self.uv, self.pr = [], [], [], [], []

    def __len__(self):
        return sum(len(a) for a in self.p)

    def add(self, tris, color=(1, 1, 1, 1), m=None, jitter=0.0, vcolors=None, normals=None, smooth=None,
            uv=None, uv_mode='box', uv_scale=1.0, sway=0.0, spec=0.0, bump=0.0, transl=0.0, vparams=None):
        tris = np.asarray(tris, dtype=np.float64)
        if len(tris) == 0:
            return self
        n = len(tris)
        # uv (yerel uzayda, dönüşümden önce)
        if uv is None:
            if uv_mode == 'box':
                uv = box_uv(tris, uv_scale)
            elif uv_mode == 'cyl':
                rr = np.abs(tris[:, :, [0, 2]]).max() or 0.5
                uv = cyl_uv(tris, uv_scale, rr)
            else:
                uv = np.zeros((n, 3, 2))
        uv = np.asarray(uv, dtype=np.float64)
        if uv.shape[0] != n:
            uv = np.broadcast_to(uv, (n, 3, 2)).copy()
        # normaller
        if normals is not None:
            nn = np.asarray(normals, dtype=np.float64).copy()
        elif smooth == 'sphere':
            c = tris.reshape(-1, 3).mean(axis=0)
            nn = tris - c
        elif smooth == 'radial':
            nn = tris.copy(); nn[:, :, 1] = 0
        else:
            fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            nn = np.repeat(fn[:, None, :], 3, axis=1)
        if smooth == 'radial':
            fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
            cap = np.abs(fn[:, 1]) > 0.9
            nn[cap] = fn[cap][:, None, :]
        if m is not None:
            flat = tris.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
            tris = flat.reshape(-1, 3, 3)
            nn = (nn.reshape(-1, 3) @ np.linalg.inv(m[:3, :3])).reshape(-1, 3, 3)
            if np.linalg.det(m[:3, :3]) < 0:
                tris = tris[:, ::-1]; nn = nn[:, ::-1]; uv = uv[:, ::-1]
        nn = nn / (np.linalg.norm(nn, axis=2, keepdims=True) + 1e-12)
        if vcolors is not None:
            cc = np.asarray(vcolors, dtype=np.float64)
            if cc.ndim == 2:
                cc = np.repeat(cc[:, None, :], 3, axis=1)
            cc = cc.copy()
            if cc.shape[2] == 3:
                cc = np.concatenate([cc, np.ones(cc.shape[:2] + (1,))], 2)
        else:
            col = np.array(color, dtype=np.float64)
            if len(col) == 3:
                col = np.append(col, 1.0)
            cc = np.tile(col, (n, 3, 1))
            if jitter:
                cc[:, :, :3] *= 1 + _rng.uniform(-jitter, jitter, (n, 1, 1))
        if vparams is not None:
            pp = np.asarray(vparams, dtype=np.float64)
            if pp.ndim == 1:
                pp = np.tile(pp, (n, 3, 1))
        else:
            pp = np.zeros((n, 3, 4))
            if sway:
                ys = tris[:, :, 1]
                y0, y1 = ys.min(), ys.max()
                pp[:, :, 0] = sway * ((ys - y0) / max(y1 - y0, 1e-6)) ** 1.5
            pp[:, :, 1] = spec
            pp[:, :, 2] = bump
            pp[:, :, 3] = transl
        self.p.append(tris); self.n.append(nn); self.c.append(cc); self.uv.append(uv); self.pr.append(pp)
        return self

    def extend(self, other, m=None):
        for p, n, c, uv, pr in zip(other.p, other.n, other.c, other.uv, other.pr):
            if m is not None:
                p = (p.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]).reshape(-1, 3, 3)
                n = (n.reshape(-1, 3) @ np.linalg.inv(m[:3, :3])).reshape(-1, 3, 3)
                n = n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-12)
            self.p.append(p); self.n.append(n); self.c.append(c); self.uv.append(uv); self.pr.append(pr)
        return self

    def arrays(self):
        if not self.p:
            return None
        # Ursina 'y-up-left' koordinat sistemi: sarım ters çevrilir
        P = np.concatenate(self.p)[:, ::-1].reshape(-1, 3).astype(np.float32)
        N = np.concatenate(self.n)[:, ::-1].reshape(-1, 3).astype(np.float32)
        C = np.concatenate(self.c)[:, ::-1].reshape(-1, 4).astype(np.float32)
        U = np.concatenate(self.uv)[:, ::-1].reshape(-1, 2).astype(np.float32)
        Q = np.concatenate(self.pr)[:, ::-1].reshape(-1, 4).astype(np.float32)
        return P, N, C, U, Q

    def geom(self):
        a = self.arrays()
        return None if a is None else make_geom(*a)


class Model:
    """Malzeme adına göre gruplanmış MeshBuilder'lar."""

    def __init__(self):
        self.parts = {}

    def __getitem__(self, mat):
        if mat not in self.parts:
            self.parts[mat] = MeshBuilder()
        return self.parts[mat]

    def extend(self, other, m=None):
        for k, mb in other.parts.items():
            self[k].extend(mb, m)
        return self

    def node(self, name='model'):
        import materials
        gn = GeomNode(name)
        for mat, mb in self.parts.items():
            g = mb.geom()
            if g is not None:
                gn.add_geom(g, materials.state(mat))
        return gn

    def nodepath(self, name='model'):
        return NodePath(self.node(name))

    def tri_count(self):
        return sum(len(mb) for mb in self.parts.values())


_FORMAT = None


def vformat():
    global _FORMAT
    if _FORMAT is None:
        fmt = GeomVertexFormat()
        for name, ncomp, contents in ((InternalName.get_vertex(), 3, Geom.C_point),
                                      (InternalName.get_normal(), 3, Geom.C_normal),
                                      (InternalName.get_color(), 4, Geom.C_color),
                                      (InternalName.get_texcoord(), 2, Geom.C_texcoord),
                                      (InternalName.make('params'), 4, Geom.C_other)):
            a = GeomVertexArrayFormat()
            a.add_column(name, ncomp, Geom.NT_float32, contents)
            fmt.add_array(a)
        _FORMAT = GeomVertexFormat.register_format(fmt)
    return _FORMAT


def make_geom(P, N, C, U, Q, dynamic=False):
    n = len(P)
    vdata = GeomVertexData('v', vformat(), Geom.UH_dynamic if dynamic else Geom.UH_static)
    vdata.unclean_set_num_rows(n)
    for i, arr in enumerate((P, N, C, U, Q)):
        mv = memoryview(vdata.modify_array(i)).cast('B').cast('f')
        mv[:] = np.ascontiguousarray(arr, dtype=np.float32).ravel()
    prim = GeomTriangles(Geom.UH_static)
    prim.set_nonindexed_vertices(0, n)
    geom = Geom(vdata)
    geom.add_primitive(prim)
    return geom


# =========================================================================== gürültü
def _hash2(ix, iz, seed):
    ix = np.asarray(ix, dtype=np.int64)
    iz = np.asarray(iz, dtype=np.int64)
    h = (ix * 374761393 + iz * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF) / 65535.0


def value_noise(x, z, seed=0):
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    ix, iz = np.floor(x), np.floor(z)
    fx, fz = x - ix, z - iz
    ux, uz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
    ix = ix.astype(np.int64); iz = iz.astype(np.int64)
    a = _hash2(ix, iz, seed); b = _hash2(ix + 1, iz, seed)
    c = _hash2(ix, iz + 1, seed); d = _hash2(ix + 1, iz + 1, seed)
    return (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uz


def fbm(x, z, octaves=4, seed=0, lac=2.03, gain=0.5):
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        total = total + value_noise(np.asarray(x) * freq, np.asarray(z) * freq, seed + o * 17) * amp
        norm += amp
        amp *= gain
        freq *= lac
    return total / norm


def ridged(x, z, octaves=4, seed=0):
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        n = 1 - np.abs(value_noise(np.asarray(x) * freq, np.asarray(z) * freq, seed + o * 31) * 2 - 1)
        total = total + n * n * amp
        norm += amp
        amp *= 0.5
        freq *= 2.1
    return total / norm


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

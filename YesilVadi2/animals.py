"""Hayvanlar: tavuk, koyun, inek. Barınaklarının çevresinde dolaşır, ürün verir."""
import math
import numpy as np
from ursina import Entity, Vec3

from config import *
from geometry import Model, tube_tris, box_tris, cylinder_tris, cone_tris, sphere_tris, trs, rng
from shaders import world_shader

R = np.random.default_rng(9)


def _ent(parent, md, pos=(0, 0, 0)):
    e = Entity(parent=parent, position=pos, shader=world_shader)
    e.attachNewNode(md.node('p'))
    return e


def _sph(md, mat, r, pos, scale=(1, 1, 1), col=(1, 1, 1, 1), rot=(0, 0, 0), spec=0.1):
    md[mat].add(sphere_tris(r, 2), col, m=trs(pos, rot, scale), smooth='sphere', uv_mode='box', uv_scale=0.5, spec=spec)


def _leg(md, mat, top, bot, r0, r1, col, hoof=None):
    t, n, uv = tube_tris([top, ((top[0] + bot[0]) / 2, (top[1] + bot[1]) / 2 + 0.01, (top[2] + bot[2]) / 2), bot], [r0, (r0 + r1) / 2, r1], 7)
    md[mat].add(t, col, normals=n, uv=uv)
    if hoof:
        md['plain'].add(cylinder_tris(r1 * 1.1, r1 * 1.15, 0.08, 7), hoof, m=trs((bot[0], bot[1] - 0.04, bot[2])), smooth='radial')


def build_chicken(root):
    white = (0.97, 0.95, 0.90, 1)
    b = Model()
    _sph(b, 'canvas', 0.22, (0, 0.38, 0), (0.9, 0.95, 1.25), white)
    _sph(b, 'canvas', 0.13, (0, 0.5, -0.24), (0.8, 1.2, 0.8), white, rot=(-30, 0, 0))
    for sx in (-1, 1):
        _sph(b, 'canvas', 0.13, (sx * 0.18, 0.4, -0.02), (0.35, 0.8, 1.2), (0.93, 0.9, 0.84, 1))
    body = _ent(root, b)
    h = Model()
    _sph(h, 'canvas', 0.1, (0, 0, 0), (1, 1.1, 1), white)
    h['plain'].add(cone_tris(0.035, 0.09, 6), (0.95, 0.68, 0.15, 1), m=trs((0, -0.01, 0.09), (90, 0, 0)), smooth='radial')
    for k in range(3):
        _sph(h, 'plain', 0.035, (0, 0.11 + 0.01 * (k % 2), -0.03 + k * 0.035), (0.6, 1, 1), (0.85, 0.12, 0.1, 1))
    _sph(h, 'plain', 0.03, (0, -0.07, 0.07), (0.6, 1.3, 1), (0.85, 0.12, 0.1, 1))
    for sx in (-1, 1):
        _sph(h, 'plain', 0.018, (sx * 0.065, 0.025, 0.06), col=(0.05, 0.05, 0.05, 1), spec=1.0)
    head = _ent(root, h, (0, 0.6, 0.2))
    legs = []
    for sx in (-0.08, 0.08):
        l = Model()
        _leg(l, 'plain', (0, 0, 0), (0, -0.2, 0.02), 0.022, 0.016, (0.95, 0.65, 0.15, 1))
        for a in (-30, 0, 30):
            l['plain'].add(cylinder_tris(0.01, 0.008, 0.08, 4), (0.95, 0.65, 0.15, 1), m=trs((0, -0.2, 0.02), (90, a, 0)))
        legs.append(_ent(root, l, (sx, 0.22, 0)))
    return body, head, legs, 0.75


def build_cow(root):
    white = (0.93, 0.91, 0.88, 1)
    black = (0.10, 0.09, 0.09, 1)
    pink = (0.92, 0.66, 0.64, 1)
    b = Model()
    _sph(b, 'canvas', 0.6, (0, 1.12, 0), (0.85, 0.85, 1.65), white)
    for (x, y, z, s) in [(0.42, 1.25, 0.3, 0.32), (-0.45, 1.1, -0.5, 0.36), (0.1, 1.55, -0.2, 0.33), (-0.3, 1.35, 0.55, 0.25)]:
        _sph(b, 'canvas', s, (x * 1.02, y, z), (0.55, 1, 1.2), black)
    _sph(b, 'plain', 0.2, (0, 0.62, -0.42), (1, 0.6, 1), pink)
    t, n, uv = tube_tris([(0, 1.35, -0.95), (0, 1.0, -1.08), (0, 0.65, -1.05)], [0.04, 0.03, 0.025], 5)
    b['plain'].add(t, white, normals=n, uv=uv)
    _sph(b, 'plain', 0.08, (0, 0.6, -1.05), (0.8, 1.4, 0.8), black)
    body = _ent(root, b)
    h = Model()
    _sph(h, 'canvas', 0.3, (0, 0, 0.12), (0.8, 0.85, 1.2), white)
    _sph(h, 'plain', 0.2, (0, -0.12, 0.42), (1.1, 0.8, 0.8), pink)
    for sx in (-1, 1):
        _sph(h, 'plain', 0.04, (sx * 0.07, -0.12, 0.57), (1, 0.8, 0.6), (0.3, 0.15, 0.15, 1))
        _sph(h, 'plain', 0.035, (sx * 0.2, 0.1, 0.3), col=(0.04, 0.04, 0.04, 1), spec=1.0)
        h['plain'].add(cone_tris(0.045, 0.22, 6), (0.9, 0.86, 0.7, 1), m=trs((sx * 0.2, 0.2, 0.0), (0, 0, -sx * 50)), smooth='radial')
        _sph(h, 'canvas', 0.1, (sx * 0.3, 0.08, -0.02), (1.5, 0.5, 0.8), white)
    _sph(h, 'canvas', 0.16, (0.05, 0.18, 0.15), (1, 0.6, 1), black)
    head = _ent(root, h, (0, 1.35, 1.0))
    legs = []
    for sx in (-0.3, 0.3):
        for sz in (-0.65, 0.6):
            l = Model()
            _leg(l, 'canvas', (0, 0, 0), (0, -0.82, 0), 0.11, 0.075, white, hoof=(0.18, 0.14, 0.12, 1))
            legs.append(_ent(root, l, (sx, 0.9, sz)))
    return body, head, legs, 1.75


def build_sheep(root):
    wool = (0.95, 0.93, 0.88, 1)
    face = (0.16, 0.14, 0.13, 1)
    b = Model()
    r = np.random.default_rng(3)
    for _ in range(16):
        x, y, z = r.uniform(-0.25, 0.25), r.uniform(0.78, 1.0), r.uniform(-0.48, 0.48)
        b['canvas'].add(sphere_tris(r.uniform(0.25, 0.33), 2, jitter=0.06), wool, m=trs((x, y, z)), smooth='sphere', uv_scale=0.3)
    body = _ent(root, b)
    h = Model()
    _sph(h, 'plain', 0.15, (0, 0, 0.1), (0.8, 0.95, 1.4), face)
    _sph(h, 'canvas', 0.16, (0, 0.13, 0.0), (1, 0.8, 1), wool)
    for sx in (-1, 1):
        _sph(h, 'plain', 0.07, (sx * 0.15, 0.04, -0.02), (1.6, 0.5, 0.8), face)
        _sph(h, 'plain', 0.022, (sx * 0.08, 0.04, 0.22), col=(0.9, 0.85, 0.6, 1), spec=1.0)
    head = _ent(root, h, (0, 1.05, 0.62))
    legs = []
    for sx in (-0.18, 0.18):
        for sz in (-0.36, 0.36):
            l = Model()
            _leg(l, 'plain', (0, 0, 0), (0, -0.62, 0), 0.05, 0.04, face)
            legs.append(_ent(root, l, (sx, 0.64, sz)))
    return body, head, legs, 1.45


BUILD = {'tavuk': build_chicken, 'inek': build_cow, 'koyun': build_sheep}
SPEED = {'tavuk': 1.4, 'inek': 0.85, 'koyun': 1.0}
RADIUS = {'tavuk': 0.3, 'inek': 0.9, 'koyun': 0.6}
SOUND = {'tavuk': 'cluck', 'inek': 'moo', 'koyun': 'baa'}


class Animal:
    def __init__(self, game, kind, home_uid, x=None, z=None, timer=None):
        self.game = game
        self.kind = kind
        self.home_uid = home_uid
        hx, hz = self.home_center()
        if x is None:
            x, z = hx + R.uniform(-2, 2), hz + R.uniform(-2, 2)
        self.root = Entity(shader=world_shader, position=(x, game.terrain.ground_y(x, z), z), rotation_y=R.uniform(0, 360))
        self.body, self.head, self.legs, self.top = BUILD[kind](self.root)
        self.heading = self.root.rotation_y
        self.target = None
        self.wait = R.uniform(0, 3)
        self.phase = R.uniform(0, 10)
        self.peck = 0.0
        self.timer = ANIMALS[kind]['time'] * R.uniform(0.3, 1.0) if timer is None else timer
        self.ready = self.timer <= 0
        self.sound_cd = R.uniform(10, 40)

    def home(self):
        for b in self.game.builds.buildings:
            if b.uid == self.home_uid:
                return b
        return None

    def home_center(self):
        b = self.home()
        if b is None:
            return self.game.player.pos.x, self.game.player.pos.z
        for k, p in b.points_world():
            if k == 'home':
                return p.x, p.z
        p = b.world_point(0, 0, b.spec.d / 2 + 3)
        return p.x, p.z

    @property
    def pos(self):
        return self.root.x, self.root.z

    def update(self, dt, t, others, sounds, player_pos):
        x, z = self.pos
        hx, hz = self.home_center()
        moving = 0.0
        if self.target is None:
            self.wait -= dt
            if self.wait <= 0:
                for _ in range(8):
                    tx, tz = hx + R.uniform(-6, 6), hz + R.uniform(-6, 6)
                    if self.game.terrain.ground_y(tx, tz) > WATER_Y + 0.3:
                        self.target = (tx, tz)
                        break
            elif self.peck <= 0 and R.random() < dt * 0.4:
                self.peck = 1.4
        else:
            tx, tz = self.target
            dx, dz = tx - x, tz - z
            d = math.hypot(dx, dz)
            if d < 0.3:
                self.target = None
                self.wait = R.uniform(2, 7)
            else:
                want = math.degrees(math.atan2(dx, dz))
                diff = (want - self.heading + 180) % 360 - 180
                self.heading += diff * min(1, dt * 3)
                if abs(diff) < 50:
                    a = math.radians(self.heading)
                    x += math.sin(a) * SPEED[self.kind] * dt
                    z += math.cos(a) * SPEED[self.kind] * dt
                    moving = 1.0
        r = RADIUS[self.kind]
        for o in others:
            if o is self:
                continue
            ox, oz = o.pos
            dx, dz = x - ox, z - oz
            d = math.hypot(dx, dz)
            rr = r + RADIUS[o.kind]
            if 1e-4 < d < rr:
                x += dx / d * (rr - d) * 0.5
                z += dz / d * (rr - d) * 0.5
        px, pz = player_pos
        dx, dz = x - px, z - pz
        d = math.hypot(dx, dz)
        if 1e-4 < d < r + 0.45:
            x += dx / d * (r + 0.45 - d)
            z += dz / d * (r + 0.45 - d)
        x, z = self.game.resolve_collision(x, z, self.root.y, r, animals=False)
        self.root.x, self.root.z = x, z
        gy = self.game.ground_height(x, z, self.root.y + 0.5)
        self.root.y += (gy - self.root.y) * min(1, dt * 10)
        self.root.rotation_y = self.heading
        self.phase += dt * (9 if self.kind == 'tavuk' else 5) * moving
        sw = math.sin(self.phase) * (40 if self.kind == 'tavuk' else 25) * moving
        pattern = (1, -1) if len(self.legs) == 2 else (1, -1, -1, 1)
        for l, sgn in zip(self.legs, pattern):
            l.rotation_x = sw * sgn
        self.body.y = abs(math.sin(self.phase)) * 0.03 * moving
        if self.peck > 0:
            self.peck -= dt
            k = math.sin(min(1, (1.4 - self.peck) / 1.4) * math.pi)
            self.head.rotation_x = (55 if self.kind != 'tavuk' else 70 * abs(math.sin(self.peck * 12))) * k
        else:
            self.head.rotation_x = math.sin(t * 0.8 + self.phase) * 5
            self.head.rotation_y = math.sin(t * 0.5 + self.phase) * 12
        if not self.ready:
            self.timer -= dt
            if self.timer <= 0:
                self.ready = True
        self.sound_cd -= dt
        if self.sound_cd <= 0:
            self.sound_cd = R.uniform(15, 45)
            dist = math.hypot(x - px, z - pz)
            if dist < 35:
                sounds.play(SOUND[self.kind], volume=max(0.05, 0.5 * (1 - dist / 35)))

    def collect(self):
        if not self.ready:
            return None
        self.ready = False
        self.timer = ANIMALS[self.kind]['time']
        return ANIMALS[self.kind]['product']

    def destroy(self):
        self.root.removeNode()

    def to_list(self):
        return [self.kind, self.home_uid, round(self.root.x, 2), round(self.root.z, 2), round(self.timer, 1)]

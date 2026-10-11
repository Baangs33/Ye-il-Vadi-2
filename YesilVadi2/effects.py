"""Parçacıklar (talaş, taş kırıntısı, toprak, su), kamp ateşi kıvılcımları ve ateş böcekleri."""
import math
import numpy as np
from ursina import Entity, Vec3, color, destroy

from environment import SHADOW_MASK
from geometry import Model, sphere_tris
from shaders import glow_shader

R = np.random.default_rng(31)


class Particles:
    def __init__(self):
        self.items = []

    def emit(self, pos, col, count=10, speed=2.0, up=2.5, life=0.8, size=0.06, grav=9.0, spread=0.15, glow=False):
        for _ in range(count):
            off = Vec3(R.uniform(-spread, spread), R.uniform(0, spread), R.uniform(-spread, spread))
            c = col() if callable(col) else col
            e = Entity(model='cube', color=c, position=Vec3(*pos) + off, scale=size * R.uniform(0.6, 1.4),
                       rotation=(R.uniform(0, 360), R.uniform(0, 360), 0))
            e.hide(SHADOW_MASK)
            a = R.uniform(0, 2 * math.pi)
            sp = R.uniform(0.3, 1.0) * speed
            vel = Vec3(math.cos(a) * sp, R.uniform(0.5, 1.0) * up, math.sin(a) * sp)
            lf = life * R.uniform(0.7, 1.2)
            self.items.append([e, vel, lf, lf, grav, R.uniform(-400, 400)])

    def update(self, dt):
        keep = []
        for p in self.items:
            e, vel, life, mx, grav, spin = p
            life -= dt
            p[2] = life
            if life <= 0:
                destroy(e)
                continue
            vel.y -= grav * dt
            e.position += vel * dt
            e.rotation_x += spin * dt
            k = life / mx
            if k < 0.3:
                e.scale *= 0.92
            keep.append(p)
        self.items = keep


class Fireflies:
    def __init__(self, terrain, center=(0, 0), n=60):
        self.terrain = terrain
        mb = Model()
        mb['plain'].add(sphere_tris(0.05, 1), (1, 1, 1, 1), uv_mode='none')
        self.items = []
        for _ in range(n):
            e = Entity(shader=glow_shader)
            e.attachNewNode(mb.node('fly'))
            e.setColorScale(0.75, 1.0, 0.35, 1)
            e.hide(SHADOW_MASK)
            a = R.uniform(0, 2 * math.pi)
            r = R.uniform(3, 45)
            self.items.append([e, Vec3(center[0] + math.cos(a) * r, 0, center[1] + math.sin(a) * r), R.uniform(0, 100),
                               R.uniform(0.5, 1.5)])

    def recenter(self, x, z):
        for f in self.items:
            if math.hypot(f[1].x - x, f[1].z - z) > 50:
                a = R.uniform(0, 2 * math.pi)
                r = R.uniform(5, 40)
                f[1] = Vec3(x + math.cos(a) * r, 0, z + math.sin(a) * r)

    def update(self, dt, t, night, rain):
        vis = night > 0.3 and rain < 0.3
        for f in self.items:
            e, home, ph, sp = f
            e.enabled = vis
            if not vis:
                continue
            x = home.x + math.sin(t * 0.3 * sp + ph) * 2.5
            z = home.z + math.cos(t * 0.23 * sp + ph * 1.3) * 2.5
            y = self.terrain.ground_y(x, z) + 0.8 + math.sin(t * 0.7 + ph) * 0.5
            e.position = (x, y, z)
            e.set_shader_input('glow', max(0.0, math.sin(t * 2.0 * sp + ph)) * 3.0 * night)

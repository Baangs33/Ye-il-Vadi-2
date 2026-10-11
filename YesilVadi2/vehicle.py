"""Kamyonet: prosedürel model, basit araç fiziği, arazi takibi ve çarpışma."""
import math
import numpy as np
from ursina import Entity, Vec3, held_keys, camera, lerp

from config import *
from geometry import Model, box_tris, cylinder_tris, sphere_tris, tube_tris, prism_tris, trs
from shaders import world_shader, glow_shader
from environment import SHADOW_MASK

WHEELS = [(-0.86, 1.42), (0.86, 1.42), (-0.86, -1.38), (0.86, -1.38)]   # (x, z)
WHEEL_R = 0.42


def car_body(paint=(0.62, 0.08, 0.06)):
    md = Model()
    P = tuple(paint) + (1,)
    pk = dict(spec=0.9, uv_scale=2.0)

    def bx(sx, sy, sz, pos, col=P, mat='paint', rot=(0, 0, 0), **kw):
        a = dict(pk); a.update(kw)
        md[mat].add(box_tris(sx, sy, sz), col, m=trs(pos, rot), **a)
    bx(1.86, 0.5, 4.5, (0, 0.82, 0))                         # alt gövde
    bx(1.8, 0.22, 1.3, (0, 1.16, 1.55), rot=(-4, 0, 0))     # kaput
    bx(1.82, 0.12, 0.5, (0, 1.12, 2.2), rot=(-14, 0, 0))     # ön eğim
    bx(1.76, 0.18, 1.7, (0, 1.16, 0.05))                     # kabin tabanı
    for sx in (-1, 1):
        bx(0.06, 0.62, 1.55, (sx * 0.88, 1.56, 0.0))         # kabin yan
        md['glass'].add(box_tris(0.07, 0.42, 1.2), (0.12, 0.15, 0.18, 1), m=trs((sx * 0.885, 1.6, 0.05)), spec=1.0, uv_mode='none')
    bx(1.76, 0.08, 1.45, (0, 1.9, -0.05))                    # tavan
    md['glass'].add(box_tris(1.66, 0.62, 0.06), (0.12, 0.15, 0.18, 1), m=trs((0, 1.58, 0.82), (-28, 0, 0)), spec=1.0, uv_mode='none')
    md['glass'].add(box_tris(1.66, 0.45, 0.06), (0.12, 0.15, 0.18, 1), m=trs((0, 1.62, -0.78)), spec=1.0, uv_mode='none')
    bx(1.76, 0.62, 0.06, (0, 1.55, -0.8), col=P)
    for sx in (-1, 1):                                       # kasa yanları
        bx(0.08, 0.5, 2.0, (sx * 0.89, 1.32, -1.28))
    bx(1.86, 0.5, 0.08, (0, 1.32, -2.25))
    md['planks'].add(box_tris(1.7, 0.05, 2.0), (0.8, 0.7, 0.6, 1), m=trs((0, 1.1, -1.28)), uv_scale=1.0)
    md['metal'].add(box_tris(1.95, 0.18, 0.22), (0.75, 0.76, 0.78, 1), m=trs((0, 0.68, 2.3)), spec=1.0, uv_scale=1.0)
    md['metal'].add(box_tris(1.95, 0.18, 0.2), (0.75, 0.76, 0.78, 1), m=trs((0, 0.68, -2.3)), spec=1.0, uv_scale=1.0)
    md['plain'].add(box_tris(1.2, 0.3, 0.05), (0.06, 0.06, 0.06, 1), m=trs((0, 0.98, 2.26)), spec=0.5)
    for k in range(5):
        md['metal'].add(box_tris(1.15, 0.025, 0.06), (0.6, 0.6, 0.62, 1), m=trs((0, 0.88 + k * 0.05, 2.28)), spec=1.0)
    for sx in (-1, 1):                                       # çamurluklar
        for z in (1.42, -1.38):
            md['paint'].add(box_tris(0.3, 0.12, 1.1), P, m=trs((sx * 0.86, 1.08, z)), **pk)
        md['metal'].add(box_tris(0.05, 0.12, 0.2), (0.15, 0.15, 0.15, 1), m=trs((sx * 0.97, 1.7, 0.75)), spec=0.6)
    # iç: koltuk ve direksiyon
    md['canvas'].add(box_tris(1.5, 0.15, 0.55), (0.25, 0.2, 0.18, 1), m=trs((0, 1.32, -0.3)), uv_scale=0.4)
    md['canvas'].add(box_tris(1.5, 0.6, 0.15), (0.25, 0.2, 0.18, 1), m=trs((0, 1.6, -0.6), (-10, 0, 0)), uv_scale=0.4)
    t, n, uv = tube_tris([(-0.42 + 0.18 * math.cos(a), 1.62 + 0.18 * math.sin(a) * 0.8, 0.5 + 0.18 * math.sin(a) * 0.5)
                          for a in np.linspace(0, 2 * math.pi, 13)], [0.02] * 13, 5)
    md['plain'].add(t, (0.08, 0.08, 0.08, 1), normals=n, uv=uv, spec=0.4)
    md['plain'].add(box_tris(1.7, 0.3, 0.3), (0.12, 0.11, 0.1, 1), m=trs((0, 1.4, 0.62)))
    return md


def lights_model():
    md = Model()
    for sx in (-0.7, 0.7):
        md['plain'].add(cylinder_tris(0.11, 0.11, 0.05, 10), (1.0, 0.95, 0.8, 1), m=trs((sx, 0.98, 2.27), (90, 0, 0)), uv_mode='none')
    return md


def tail_model():
    md = Model()
    for sx in (-0.75, 0.75):
        md['plain'].add(box_tris(0.22, 0.14, 0.04), (0.9, 0.08, 0.05, 1), m=trs((sx, 1.3, -2.3)), uv_mode='none')
    return md


def wheel_model():
    md = Model()
    tire = cylinder_tris(WHEEL_R, WHEEL_R, 0.3, 18)
    md['plain'].add(tire, (0.07, 0.07, 0.07, 1), m=trs((0.15, 0, 0), (0, 0, 90)), smooth='radial', spec=0.15)
    md['metal'].add(cylinder_tris(0.24, 0.24, 0.31, 12), (0.7, 0.7, 0.72, 1), m=trs((0.155, 0, 0), (0, 0, 90)), spec=1.0,
                    uv_scale=0.5)
    for k in range(5):
        md['metal'].add(box_tris(0.02, 0.06, 0.4), (0.5, 0.5, 0.52, 1), m=trs((0.17, 0, 0), (k * 36, 0, 0)), spec=1.0)
    return md


class Car:
    def __init__(self, game, x, z, heading=0.0, paint=(0.62, 0.08, 0.06)):
        self.game = game
        self.root = Entity(shader=world_shader, position=(x, game.terrain.ground_y(x, z), z), rotation_y=heading)
        self.body = Entity(parent=self.root, shader=world_shader)
        self.body.attachNewNode(car_body(paint).node('car'))
        self.head_l = Entity(parent=self.body, shader=glow_shader)
        self.head_l.attachNewNode(lights_model().node('hl'))
        self.head_l.hide(SHADOW_MASK)
        self.tail_l = Entity(parent=self.body, shader=glow_shader)
        self.tail_l.attachNewNode(tail_model().node('tl'))
        self.wheels = []
        wm = wheel_model()
        for (wx, wz) in WHEELS:
            pivot = Entity(parent=self.root, position=(wx, WHEEL_R, wz), shader=world_shader)
            spin = Entity(parent=pivot, shader=world_shader)
            np_ = spin.attachNewNode(wm.node('w'))
            if wx < 0:
                np_.set_sx(-1)
            self.wheels.append((pivot, spin))
        self.heading = heading
        self.speed = 0.0
        self.steer = 0.0
        self.spin = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.y_v = 0.0
        self.engine_on = False
        self.cam_mode = 0
        self.cam_yaw = heading
        self._settle()

    @property
    def pos(self):
        return self.root.x, self.root.z

    def _wheel_world(self, wx, wz, x=None, z=None, h=None):
        x = self.root.x if x is None else x
        z = self.root.z if z is None else z
        a = math.radians(self.heading if h is None else h)
        return x + wx * math.cos(a) + wz * math.sin(a), z - wx * math.sin(a) + wz * math.cos(a)

    def _settle(self):
        hs = []
        for wx, wz in WHEELS:
            px, pz = self._wheel_world(wx, wz)
            hs.append(self.game.ground_height(px, pz, 999))
        fl, fr, rl, rr = hs
        front, rear = (fl + fr) / 2, (rl + rr) / 2
        left, right = (fl + rl) / 2, (fr + rr) / 2
        self.root.y = sum(hs) / 4
        self.pitch = -math.degrees(math.atan2(front - rear, 2.8))
        self.roll = math.degrees(math.atan2(left - right, 1.72))

    def update(self, dt, driving):
        g = self.game
        throttle = steer_in = brake = 0
        if driving:
            throttle = held_keys['w'] - held_keys['s']
            steer_in = held_keys['d'] - held_keys['a']
            brake = held_keys['space']
        on_road = g.on_road(self.root.x, self.root.z)
        top = 24.0 if on_road else 15.0
        if throttle > 0:
            self.speed += (9.0 if self.speed >= 0 else 16.0) * dt
        elif throttle < 0:
            self.speed -= (14.0 if self.speed > 0 else 6.0) * dt
        else:
            self.speed -= math.copysign(min(abs(self.speed), (2.5 if on_road else 4.0) * dt), self.speed)
        if brake:
            self.speed -= math.copysign(min(abs(self.speed), 20 * dt), self.speed)
        # yokuş etkisi
        self.speed -= math.sin(math.radians(-self.pitch)) * 7.0 * dt
        self.speed = max(-6.0, min(top, self.speed))
        self.steer += (steer_in * 32 - self.steer) * min(1, dt * 5)
        turn = self.steer * self.speed / 2.8 * (0.55 if brake else 1.0) * 0.9
        nh = self.heading + turn * dt * (1 if self.speed >= 0 else 1)
        a = math.radians(nh)
        nx = self.root.x + math.sin(a) * self.speed * dt
        nz = self.root.z + math.cos(a) * self.speed * dt
        # çarpışma: gövdeyi üç dairelik zincirle temsil et
        hit = False
        for off in (1.5, 0.0, -1.5):
            cx, cz = nx + math.sin(a) * off, nz + math.cos(a) * off
            rx, rz = g.resolve_collision(cx, cz, self.root.y + 0.5, 1.0, animals=True, car=False)
            if abs(rx - cx) > 1e-4 or abs(rz - cz) > 1e-4:
                nx += rx - cx
                nz += rz - cz
                hit = True
        lim = WORLD - 6
        if abs(nx) > lim or abs(nz) > lim:
            hit = True
            nx, nz = max(-lim, min(lim, nx)), max(-lim, min(lim, nz))
        # çok dik yamaç veya derin su
        fh = g.terrain.ground_y(nx + math.sin(a) * 2.2, nz + math.cos(a) * 2.2)
        if fh - self.root.y > 1.4 or fh < WATER_Y - 0.6:
            hit = True
            nx, nz = self.root.x, self.root.z
        if hit:
            if abs(self.speed) > 6:
                g.sounds.play('crash', min(0.7, abs(self.speed) / 20))
            self.speed = -math.copysign(min(abs(self.speed) * 0.25, 1.5), self.speed)
            nh = self.heading
        self.heading = nh
        self.root.x, self.root.z = nx, nz
        self.root.rotation_y = self.heading
        hs = []
        for wx, wz in WHEELS:
            px, pz = self._wheel_world(wx, wz)
            hs.append(g.ground_height(px, pz, self.root.y + 1.2))
        fl, fr, rl, rr = hs
        target_y = sum(hs) / 4
        self.root.y += (target_y - self.root.y) * min(1, dt * 12)
        tp = -math.degrees(math.atan2((fl + fr) / 2 - (rl + rr) / 2, 2.8))
        tr = math.degrees(math.atan2((fl + rl) / 2 - (fr + rr) / 2, 1.72))
        self.pitch += (tp - self.pitch) * min(1, dt * 8)
        self.roll += (tr - self.roll) * min(1, dt * 8)
        self.body.rotation_x = self.pitch
        self.body.rotation_z = self.roll + self.steer * self.speed * 0.004
        self.spin += self.speed / WHEEL_R * dt * 57.3
        for i, (pivot, spin) in enumerate(self.wheels):
            wy = hs[i] - self.root.y + WHEEL_R
            pivot.y = wy
            pivot.rotation_y = self.steer if i < 2 else 0
            spin.rotation_x = self.spin
        night = g.env.night
        self.head_l.set_shader_input('glow', 1.0 + 3.0 * night if self.engine_on else 0.3)
        self.tail_l.set_shader_input('glow', (1.6 if brake or throttle < 0 else 0.7) if self.engine_on else 0.25)

    def camera_update(self, dt, mouse_dx, mouse_dy):
        self.cam_yaw += mouse_dx
        rel = (self.cam_yaw - self.heading + 180) % 360 - 180
        if abs(mouse_dx) < 1e-4:
            self.cam_yaw -= rel * min(1, dt * 1.5)
        if self.cam_mode == 0:
            yaw = math.radians(self.cam_yaw)
            dist = 7.5
            tx, ty, tz = self.root.x, self.root.y + 1.6, self.root.z
            cx, cz = tx - math.sin(yaw) * dist, tz - math.cos(yaw) * dist
            cy = max(ty + 1.6, self.game.terrain.ground_y(cx, cz) + 0.8)
            camera.position = lerp(camera.position, Vec3(cx, cy, cz), min(1, dt * 10))
            d = Vec3(tx, ty + 0.4, tz) - camera.position
            camera.rotation = (math.degrees(math.atan2(-d.y, math.hypot(d.x, d.z))), math.degrees(math.atan2(d.x, d.z)), 0)
        else:
            a = math.radians(self.heading)
            ox, oz = -0.42, 0.0
            px = self.root.x + ox * math.cos(a) + oz * math.sin(a)
            pz = self.root.z - ox * math.sin(a) + oz * math.cos(a)
            camera.position = Vec3(px, self.root.y + 1.95, pz)
            camera.rotation = (8 - self.pitch * 0.5, self.cam_yaw, -self.roll * 0.5)

    def exit_point(self):
        a = math.radians(self.heading)
        return self.root.x - 1.8 * math.cos(a), self.root.z + 1.8 * math.sin(a)

    def to_dict(self):
        return {'x': round(self.root.x, 2), 'z': round(self.root.z, 2), 'h': round(self.heading, 1)}

"""Birinci şahıs karakter: fare ile bakış, yürüme/koşma/zıplama, çarpışma, açlık/enerji
ve ayrı bir kamera katmanında çizilen eller + aletler (viewmodel)."""
import math
import numpy as np
from ursina import Vec3, camera, held_keys, mouse, window
from panda3d.core import (NodePath, Camera, PerspectiveLens, GeomNode, TransparencyAttrib)

from config import *
from geometry import Model, tube_tris, box_tris, cylinder_tris, cone_tris, sphere_tris, prism_tris, trs
from shaders import viewmodel_shader, glow_shader
import materials

EYE = 1.68
RADIUS = 0.38
GRAVITY = 22.0
JUMP_V = 7.2
STEP = 0.5


# =========================================================================== alet modelleri
def _handle(md, length, r=0.028, y0=0.0):
    md['planks'].add(cylinder_tris(r, r * 0.92, length, 8), (1.05, 0.9, 0.75, 1), m=trs((0, y0, 0)), uv_mode='cyl',
                     uv_scale=0.6, smooth='radial', spec=0.2)


def taper_extrude(profile, th_fn):
    """(y, z) profilini x yönünde kalınlığı z'ye göre değişen bir katıya dönüştürür."""
    pts = [np.array(p, dtype=float) for p in profile]
    n = len(pts)
    c = np.mean(pts, axis=0)
    tris = []
    L = lambda p: (th_fn(p[1]) / 2, p[0], p[1])
    Rr = lambda p: (-th_fn(p[1]) / 2, p[0], p[1])
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        tris.append((L(c), L(a), L(b)))
        tris.append((Rr(c), Rr(b), Rr(a)))
        tris.append((L(a), Rr(a), Rr(b)))
        tris.append((L(a), Rr(b), L(b)))
    t = np.array(tris)
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    cen = t.mean(axis=1) - np.array([0, c[0], c[1]])
    flip = (nrm * cen).sum(axis=1) < 0
    t[flip] = t[flip][:, ::-1]
    return t


def tool_model(kind, seed_color=(0.8, 0.6, 0.3)):
    md = Model()
    metal = (0.78, 0.80, 0.84, 1)
    if kind == 'balta':
        _handle(md, 0.78, 0.026, -0.1)
        prof = [(0.04, -0.05), (0.045, 0.06), (0.11, 0.17), (0.10, 0.2), (-0.07, 0.2), (-0.09, 0.17), (-0.035, 0.06), (-0.035, -0.05)]
        head = taper_extrude(prof, lambda z: 0.055 - 0.05 * max(0.0, min(1.0, (z - 0.02) / 0.18)))
        md['plain'].add(head, (0.36, 0.37, 0.40, 1), m=trs((0, 0.6, 0)), spec=0.8, uv_mode='none')
        edge = [(0.10, 0.185), (0.112, 0.2), (-0.072, 0.2), (-0.088, 0.185)]
        md['plain'].add(taper_extrude(edge, lambda z: 0.012), (0.8, 0.82, 0.86, 1), m=trs((0, 0.6, 0.002)), spec=1.0, uv_mode='none')
    elif kind == 'kazma':
        _handle(md, 0.8, 0.026, -0.1)
        pts = [(0, 0.66, -0.32), (0, 0.72, -0.1), (0, 0.74, 0.0), (0, 0.72, 0.1), (0, 0.66, 0.32)]
        t, n, uv = tube_tris(pts, [0.006, 0.03, 0.035, 0.03, 0.006], 6)
        md['metal'].add(t, (0.55, 0.56, 0.58, 1), normals=n, uv=uv, spec=0.7)
    elif kind == 'capa':
        _handle(md, 1.15, 0.024, -0.4)
        md['metal'].add(box_tris(0.05, 0.05, 0.12), metal, m=trs((0, 0.73, 0.05)), spec=0.8, uv_scale=0.3)
        md['metal'].add(box_tris(0.22, 0.16, 0.012), (0.6, 0.6, 0.62, 1), m=trs((0, 0.66, 0.12), (-20, 0, 0)), spec=0.8,
                        uv_scale=0.3)
    elif kind == 'kova':
        md['metal'].add(cylinder_tris(0.13, 0.11, 0.24, 14), (0.55, 0.68, 0.62, 1), m=trs((0, 0.0, 0)), uv_mode='cyl',
                        smooth='radial', spec=0.8, uv_scale=0.4)
        t, n, uv = tube_tris([(0, 0.06, 0.08), (0, 0.16, 0.28), (0, 0.2, 0.36)], [0.025, 0.018, 0.016], 6)
        md['metal'].add(t, (0.5, 0.62, 0.56, 1), normals=n, uv=uv, spec=0.8)
        md['metal'].add(cylinder_tris(0.04, 0.05, 0.03, 8), (0.5, 0.62, 0.56, 1), m=trs((0, 0.2, 0.36), (55, 0, 0)), spec=0.8)
        t, n, uv = tube_tris([(0, 0.22, -0.1), (0, 0.32, -0.02), (0, 0.24, 0.08)], [0.012, 0.012, 0.012], 5)
        md['metal'].add(t, (0.45, 0.55, 0.5, 1), normals=n, uv=uv, spec=0.8)
    elif kind == 'tohum':
        md['canvas'].add(sphere_tris(0.12, 2, squash=1.15), (0.9, 0.82, 0.62, 1), m=trs((0, 0.05, 0)), smooth='sphere',
                         uv_scale=0.25)
        md['canvas'].add(cylinder_tris(0.05, 0.08, 0.08, 10), (0.85, 0.75, 0.55, 1), m=trs((0, 0.16, 0)), uv_scale=0.25)
        md['plain'].add(cylinder_tris(0.055, 0.055, 0.02, 10), (0.4, 0.22, 0.12, 1), m=trs((0, 0.19, 0)))
        md['plain'].add(sphere_tris(0.04, 1), tuple(seed_color) + (1,), m=trs((0, 0.05, 0.11)), smooth='sphere')
    elif kind == 'cekic':
        _handle(md, 0.42, 0.02, -0.1)
        md['metal'].add(box_tris(0.05, 0.06, 0.16), metal, m=trs((0, 0.34, 0.0)), spec=0.9, uv_scale=0.3)
        md['metal'].add(cylinder_tris(0.03, 0.03, 0.05, 8), metal, m=trs((0, 0.34, 0.08), (90, 0, 0)), spec=0.9)
    elif kind == 'yemek':
        md['plain'].add(sphere_tris(0.06, 1), (0.45, 0.05, 0.12, 1), m=trs((0, 0.02, 0)), smooth='sphere', spec=0.6)
        md['plain'].add(sphere_tris(0.05, 1), (0.4, 0.04, 0.1, 1), m=trs((0.06, 0.04, 0.03)), smooth='sphere', spec=0.6)
    return md


def arm_model():
    md = Model()
    sleeve = (0.42, 0.18, 0.14, 1)
    skin = (0.85, 0.62, 0.48, 1)
    t, n, uv = tube_tris([(0, 0, -0.45), (0, 0, -0.05)], [0.075, 0.062], 10)
    md['canvas'].add(t, sleeve, normals=n, uv=uv * 2)
    md['canvas'].add(cylinder_tris(0.066, 0.066, 0.05, 10), (0.35, 0.15, 0.12, 1), m=trs((0, 0, -0.06), (90, 0, 0)))
    md['plain'].add(sphere_tris(0.06, 2, squash=0.7), skin, m=trs((0, 0.0, 0.03), (0, 0, 0), (0.85, 1, 1.3)),
                    smooth='sphere', spec=0.15)
    for k in range(4):
        md['plain'].add(cylinder_tris(0.014, 0.012, 0.07, 6), skin, m=trs((-0.03 + k * 0.02, 0.02, 0.08), (75, 0, 0)),
                        smooth='radial')
    md['plain'].add(cylinder_tris(0.017, 0.014, 0.06, 6), skin, m=trs((0.045, -0.01, 0.05), (60, 0, -40)), smooth='radial')
    return md


def _np(md, shader=None):
    np_ = md.nodepath('vm')
    return np_


# =========================================================================== oyuncu
class Player:
    def __init__(self, game):
        self.game = game
        self.pos = Vec3(SPAWN[0], 0, SPAWN[1])
        self.pos.y = game.ground_height(self.pos.x, self.pos.z, 100)
        self.vel = Vec3(0, 0, 0)
        self.yaw = 20.0
        self.pitch = 0.0
        self.grounded = True
        self.hunger = 100.0
        self.energy = 100.0
        self.sens = 1.0
        self.fov = 75.0
        self.bob = 0.0
        self.bob_amt = 0.0
        self.land_dip = 0.0
        self.swing_t = 0.0
        self.swing_cb = None
        self.swing_kind = 'chop'
        self.equip_t = 0.0
        self.tool = None
        self.input_enabled = True
        self.in_vehicle = None
        self.last_fall_v = 0.0
        self.sprinting = False
        self.footstep_t = 0.0
        self._build_viewmodel()

    # ------------------------------------------------------------------ viewmodel
    def _build_viewmodel(self):
        self.vm_root = NodePath('viewmodel_scene')
        self.vm_root.set_shader(self._compile(viewmodel_shader))
        self.vm_root.set_shader_input('emissive', 0.0)
        self.vm_root.set_shader_input('wetness', 0.0)
        self.vm_holder = self.vm_root.attach_new_node('holder')
        lens = PerspectiveLens()
        lens.set_fov(72)
        lens.set_near_far(0.01, 20)
        self.vm_lens = lens
        cam = Camera('vm_cam', lens)
        self.vm_cam = self.vm_holder.attach_new_node(cam)
        dr = base.win.make_display_region()
        dr.set_sort(10)
        dr.set_clear_depth_active(True)
        dr.set_camera(self.vm_cam)
        self.vm_dr = dr
        self.vm_offset = self.vm_holder.attach_new_node('offset')
        self.arm_r = self.vm_offset.attach_new_node('arm_r')
        self.arm_r.set_pos(0.21, -0.19, 0.46)
        self.arm_r_model = arm_model().nodepath('armr')
        self.arm_r_model.reparent_to(self.arm_r)
        self.hand_r = self.arm_r.attach_new_node('hand_r')
        self.hand_r.set_pos(0, 0.0, 0.05)
        # gece feneri (sol elde tutulur)
        lm = Model()
        dark = (0.12, 0.11, 0.1, 1)
        lm['metal'].add(cylinder_tris(0.075, 0.08, 0.03, 10), dark, m=trs((0, 0.0, 0)), spec=0.6, smooth='radial')
        lm['metal'].add(cylinder_tris(0.08, 0.03, 0.07, 10), dark, m=trs((0, 0.2, 0)), spec=0.6, smooth='radial')
        for k in range(4):
            a = k * math.pi / 2 + 0.5
            lm['metal'].add(box_tris(0.01, 0.2, 0.01), dark, m=trs((math.cos(a) * 0.068, 0.1, math.sin(a) * 0.068)), spec=0.6)
        t, n, uv = tube_tris([(-0.05, 0.27, 0), (0, 0.33, 0), (0.05, 0.27, 0)], [0.008, 0.008, 0.008], 5)
        lm['metal'].add(t, dark, normals=n, uv=uv)
        hand = Model()
        skin = (0.85, 0.62, 0.48, 1)
        hand['plain'].add(sphere_tris(0.05, 2, squash=0.8), skin, m=trs((0, 0.37, 0), (0, 0, 0), (1.1, 1, 1.2)), smooth='sphere')
        tt, nn, uu = tube_tris([(0.0, 0.38, -0.02), (-0.12, 0.12, -0.22)], [0.048, 0.06], 8)
        hand['canvas'].add(tt, (0.42, 0.18, 0.14, 1), normals=nn, uv=uu)
        self.arm_l = self.vm_offset.attach_new_node('arm_l')
        self.arm_l.set_pos(-0.34, -0.17, 0.7)
        self.arm_l.set_scale(0.5)
        self.lantern = self.arm_l.attach_new_node('lantern')
        lm.nodepath('lantern').reparent_to(self.lantern)
        hand.nodepath('lhand').reparent_to(self.arm_l)
        gl = Model()
        gl['plain'].add(cylinder_tris(0.06, 0.06, 0.17, 10), (1, 1, 1, 1), m=trs((0, 0.03, 0)), uv_mode='none', smooth='radial')
        gl['plain'].add(sphere_tris(0.025, 1), (1, 1, 1, 1), m=trs((0, 0.1, 0)), uv_mode='none')
        self.lantern_glass = gl.nodepath('lglass')
        self.lantern_glass.reparent_to(self.lantern)
        self.lantern_glass.set_shader_input('emissive', 1.6)
        self.lantern_glass.set_color_scale(1.0, 0.72, 0.38, 1)
        self.arm_l.hide()
        self.tool_np = None
        self.seed_color = (0.8, 0.6, 0.3)

    def _compile(self, sh):
        sh.compile()
        return sh._shader

    def set_tool(self, kind, seed_color=None):
        if kind == self.tool and seed_color == self.seed_color:
            return
        self.tool = kind
        if seed_color is not None:
            self.seed_color = seed_color
        if self.tool_np is not None:
            self.tool_np.remove_node()
        md = tool_model(kind, self.seed_color)
        # aletin sapı (+y) ileri-yukarı bakacak şekilde modeli döndür
        if kind in ('balta', 'kazma', 'cekic'):
            m = trs((0.0, -0.02, 0.04), (58, -8, -10), 0.85)
        elif kind == 'capa':
            m = trs((0.0, -0.05, 0.0), (62, -8, -10), 0.75)
        elif kind == 'kova':
            m = trs((-0.02, -0.16, 0.08), (0, -70, 0), 0.9)
        else:
            m = trs((0.0, -0.06, 0.08), (0, 0, 0), 0.9)
        md2 = Model()
        md2.extend(md, m)
        self.tool_np = md2.nodepath('tool')
        self.tool_np.reparent_to(self.hand_r)
        self.equip_t = 0.35

    # ------------------------------------------------------------------ yardımcılar
    @property
    def eye(self):
        return Vec3(self.pos.x, self.pos.y + EYE - self.land_dip, self.pos.z)

    def look_dir(self):
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        return Vec3(sy * cp, -sp, cy * cp)

    def start_swing(self, cb, kind='chop', cost=SWING_COST):
        if self.swing_t > 0 or self.equip_t > 0.1:
            return False
        if self.energy < cost:
            self.game.hud.message('Çok yorgunsun! Biraz dinlen ya da bir şeyler ye.', 'red')
            return False
        self.energy -= cost
        self.swing_t = 0.55 if kind in ('chop', 'mine', 'hoe') else 0.45
        self.swing_dur = self.swing_t
        self.swing_kind = kind
        self.swing_cb = cb
        return True

    # ------------------------------------------------------------------ güncelleme
    def update(self, dt, t):
        g = self.game
        if self.in_vehicle is not None:
            self.vm_root.hide()
            return
        self.vm_root.show()
        # bakış
        if self.input_enabled and mouse.locked:
            self.yaw += mouse.velocity[0] * 120 * self.sens
            self.pitch -= mouse.velocity[1] * 120 * self.sens
            self.pitch = max(-88, min(88, self.pitch))
        # hareket
        wish = Vec3(0, 0, 0)
        if self.input_enabled:
            fwd = held_keys['w'] + held_keys['up arrow'] - held_keys['s'] - held_keys['down arrow']
            side = held_keys['d'] + held_keys['right arrow'] - held_keys['a'] - held_keys['left arrow']
            yr = math.radians(self.yaw)
            f = Vec3(math.sin(yr), 0, math.cos(yr))
            r = Vec3(math.cos(yr), 0, -math.sin(yr))
            wish = f * fwd + r * side
            if wish.length() > 0:
                wish = wish.normalized()
        hungry = self.hunger <= 0
        want_sprint = self.input_enabled and held_keys['shift'] and wish.length() > 0 and self.energy > 2 and not hungry
        self.sprinting = want_sprint and held_keys['w']
        speed = 4.4 * (1.75 if self.sprinting else 1.0) * (0.65 if hungry else 1.0)
        if self.swing_t > 0:
            speed *= 0.6
        accel = 14.0 if self.grounded else 3.0
        target = wish * speed
        self.vel.x += (target.x - self.vel.x) * min(1, accel * dt)
        self.vel.z += (target.z - self.vel.z) * min(1, accel * dt)
        if self.sprinting:
            self.energy = max(0, self.energy - SPRINT_COST * dt)
        # yerçekimi ve zıplama
        self.vel.y -= GRAVITY * dt
        if self.input_enabled and held_keys['space'] and self.grounded and self.energy >= JUMP_COST * 0.5:
            self.vel.y = JUMP_V
            self.grounded = False
            self.energy = max(0, self.energy - JUMP_COST)
            g.sounds.play('jump', 0.25)
        nx = self.pos.x + self.vel.x * dt
        nz = self.pos.z + self.vel.z * dt
        nx, nz = g.resolve_collision(nx, nz, self.pos.y, RADIUS)
        # su ve harita sınırı
        lim = WORLD - 6
        nx, nz = max(-lim, min(lim, nx)), max(-lim, min(lim, nz))
        gh = g.ground_height(nx, nz, self.pos.y + STEP)
        if gh > self.pos.y + STEP and self.grounded:
            # çok dik: hareketi engelle
            nx, nz = self.pos.x, self.pos.z
            gh = g.ground_height(nx, nz, self.pos.y + STEP)
        self.pos.x, self.pos.z = nx, nz
        self.pos.y += self.vel.y * dt
        if self.pos.y <= gh:
            if not self.grounded and self.vel.y < -6:
                self.land_dip = min(0.25, -self.vel.y * 0.02)
                g.sounds.play('land', min(0.5, -self.vel.y * 0.03))
            self.pos.y = gh
            self.vel.y = 0
            self.grounded = True
        elif self.pos.y > gh + 0.08:
            self.grounded = False
        elif self.grounded:
            self.pos.y = gh
        # suya girince yavaşla ve batma
        if gh < WATER_Y - 0.9:
            self.vel.x *= 0.5; self.vel.z *= 0.5
        self.land_dip = max(0, self.land_dip - dt * 1.2)

        # enerji / açlık
        moving = math.hypot(self.vel.x, self.vel.z)
        self.hunger = max(0, self.hunger - dt * 100 / HUNGER_TIME * (1.6 if self.sprinting else 1.0))
        if not self.sprinting and self.swing_t <= 0:
            regen = ENERGY_REGEN * (0.35 if self.hunger < 15 else 1.0) * (0.0 if hungry else 1.0)
            if moving < 0.5:
                regen *= 1.6
            self.energy = min(100, self.energy + regen * dt)

        # kafa sallanması ve ayak sesi
        target_bob = (min(1, moving / 4.4) if self.grounded else 0) * getattr(self, 'bob_scale', 1.0)
        self.bob_amt += (target_bob - self.bob_amt) * min(1, dt * 8)
        self.bob += dt * (moving * 1.9 + 0.001)
        if self.grounded and moving > 1.0:
            self.footstep_t -= dt * moving / 4.4 * (1.25 if self.sprinting else 1)
            if self.footstep_t <= 0:
                self.footstep_t = 0.48
                g.sounds.play('step', 0.12 + (0.06 if self.sprinting else 0), None)

        # kamera
        bob_y = math.sin(self.bob * 2) * 0.045 * self.bob_amt
        bob_x = math.cos(self.bob) * 0.03 * self.bob_amt
        yr = math.radians(self.yaw)
        camera.position = Vec3(self.pos.x + math.cos(yr) * bob_x, self.pos.y + EYE - self.land_dip + bob_y,
                               self.pos.z - math.sin(yr) * bob_x)
        camera.rotation = (self.pitch, self.yaw, math.sin(self.bob) * 0.4 * self.bob_amt)
        fov_target = self.fov + (6 if self.sprinting else 0)
        camera.fov += (fov_target - camera.fov) * min(1, dt * 6)

        # viewmodel
        self._update_viewmodel(dt, t)

    def _update_viewmodel(self, dt, t):
        self.vm_holder.set_mat(camera.get_mat(render))
        self.vm_lens.set_aspect_ratio(base.camLens.get_aspect_ratio())
        sway_x = math.cos(self.bob) * 0.012 * self.bob_amt
        sway_y = abs(math.sin(self.bob)) * 0.016 * self.bob_amt + math.sin(t * 1.6) * 0.003
        lag = 0.0
        if mouse.locked:
            lag = mouse.velocity[0] * 0.6
        ex = self.equip_t / 0.35 if self.equip_t > 0 else 0
        self.equip_t = max(0, self.equip_t - dt)
        self.vm_offset.set_pos(sway_x - lag * 0.05, -sway_y - ex * 0.35 - (0.03 if self.sprinting else 0), 0)
        self.vm_offset.set_hpr(lag * 6, 0, 0)
        # salınım animasyonu
        h, p, r = 0, 0, 0
        if self.swing_t > 0:
            prev = self.swing_t
            self.swing_t = max(0, self.swing_t - dt)
            k = 1 - self.swing_t / self.swing_dur
            if self.swing_kind in ('chop', 'mine', 'hoe', 'build'):
                if k < 0.35:
                    a = k / 0.35
                    p = 55 * math.sin(a * math.pi / 2)
                    r = -10 * a
                elif k < 0.55:
                    a = (k - 0.35) / 0.2
                    p = 55 - 125 * a
                    r = -10 + 20 * a
                else:
                    a = (k - 0.55) / 0.45
                    p = -70 * (1 - a)
                    r = 10 * (1 - a)
                hit_at = 0.5
            elif self.swing_kind == 'water':
                p = -35 * math.sin(k * math.pi)
                r = -50 * math.sin(k * math.pi)
                hit_at = 0.4
            else:   # ekme / toplama / yeme
                p = -30 * math.sin(k * math.pi)
                hit_at = 0.45
            if prev / self.swing_dur > 1 - hit_at >= self.swing_t / self.swing_dur and self.swing_cb:
                cb, self.swing_cb = self.swing_cb, None
                cb()
        self.arm_r.set_hpr(h + 4, p - 6, r)
        night = self.game.env.night
        if night > 0.25 and self.tool not in ('kova',):
            self.arm_l.show()
        else:
            self.arm_l.hide()
        self.lantern.set_r(math.sin(t * 1.9) * 4 + math.cos(self.bob) * 6 * self.bob_amt)
        self.lantern.set_p(math.sin(t * 1.3) * 3)
        self.lantern_glass.set_shader_input('emissive', 1.4 + 0.25 * math.sin(t * 11) + 0.1 * math.sin(t * 27))
        self.vm_root.set_shader_input('emissive', 0.0)

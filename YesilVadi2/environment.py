"""Gökyüzü, gün döngüsü, güneş/ay gölgeleri ve hava durumu."""
import math
import numpy as np
from ursina import Entity, Vec3, camera
from ursina.lights import DirectionalLight
from panda3d.core import BitMask32, TransparencyAttrib, OmniBoundingVolume, LVecBase4f

from config import DAY_LENGTH, START_TIME
from geometry import MeshBuilder, sphere_tris, rng
from shaders import sky_shader, rain_shader
import materials

SHADOW_MASK = BitMask32.bit(0)
MAIN_MASK = BitMask32.bit(1)


def mix(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def sstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


# (güneş yüksekliği, gök üst, ufuk, güneş ışığı (doğrusal), ortam gök, ortam zemin, güneş parıltısı)
KEYS = [
    (-0.30, (0.010, 0.016, 0.040), (0.040, 0.055, 0.100), (0.10, 0.14, 0.26), (0.030, 0.045, 0.090), (0.010, 0.012, 0.020), (0.0, 0.0, 0.0)),
    (-0.06, (0.050, 0.070, 0.170), (0.320, 0.240, 0.300), (0.08, 0.07, 0.10), (0.100, 0.100, 0.170), (0.030, 0.025, 0.030), (0.6, 0.25, 0.12)),
    (0.04,  (0.170, 0.280, 0.520), (1.000, 0.620, 0.380), (2.40, 1.20, 0.50), (0.300, 0.280, 0.360), (0.120, 0.090, 0.060), (1.0, 0.55, 0.25)),
    (0.22,  (0.190, 0.400, 0.760), (0.740, 0.800, 0.860), (2.60, 2.30, 1.85), (0.380, 0.460, 0.640), (0.150, 0.130, 0.090), (0.9, 0.70, 0.45)),
    (0.60,  (0.160, 0.400, 0.820), (0.680, 0.800, 0.920), (2.75, 2.60, 2.35), (0.400, 0.500, 0.700), (0.170, 0.150, 0.110), (0.8, 0.75, 0.60)),
]


def sample_keys(e):
    if e <= KEYS[0][0]:
        return KEYS[0][1:]
    if e >= KEYS[-1][0]:
        return KEYS[-1][1:]
    for a, b in zip(KEYS[:-1], KEYS[1:]):
        if a[0] <= e <= b[0]:
            t = (e - a[0]) / (b[0] - a[0])
            t = t * t * (3 - 2 * t)
            return tuple(mix(x, y, t) for x, y in zip(a[1:], b[1:]))


class Environment:
    def __init__(self, render, shadow_res=2048, shadow_size=90.0):
        self.render = render
        self.time = START_TIME
        self.day = 1
        self.rain = 0.0
        self.raining = False
        self.rain_timer = 420.0
        self.wind = 1.0
        self.night = 0.0
        self.wet = 0.0
        self.focus = Vec3(0, 0, 0)
        self.extra_roots = []
        self.view_dist = 1.0
        self.exposure = 0.8
        base.camNode.setCameraMask(MAIN_MASK)

        mb = MeshBuilder().add(sphere_tris(1.0, 3), (1, 1, 1, 1), uv_mode='none')
        self.sky = Entity(shader=sky_shader, scale=250)
        self.sky.attachNewNode(_node(mb))
        self.sky.setTwoSided(True)
        self.sky.setBin('background', 0)
        self.sky.setDepthWrite(False)
        self.sky.setDepthTest(False)
        self.sky.hide(SHADOW_MASK)

        self.shadow_res = shadow_res
        self.shadow_size = shadow_size
        self.sun = DirectionalLight(shadow_map_resolution=(shadow_res, shadow_res))
        self._setup_lens()
        render.set_shader_input('exposure', 0.8)
        from panda3d.core import PTA_LVecBase4f
        self.lamps = PTA_LVecBase4f.empty_array(8)
        render.set_shader_input('lamps', self.lamps)
        render.set_shader_input('lamp_col', Vec3(1.0, 0.62, 0.30))
        self._build_rain()
        self.apply()

    def _setup_lens(self):
        lens = self.sun._light.get_lens()
        lens.set_film_size(self.shadow_size, self.shadow_size)
        lens.set_near_far(1, 400)
        lens.set_film_offset(0, 0)
        self.render.set_shader_input('shadow_texel', 1.0 / self.shadow_res * 0.85)

    def set_shadow_quality(self, res, size):
        if res <= 0:
            self.sun._light.set_shadow_caster(False)
            self.render.set_shader_input('shadow_texel', 0.0)
            self.shadow_res = 0
            return
        self.shadow_res = res
        self.shadow_size = size
        self.sun._light.set_shadow_caster(True, res, res)
        self._setup_lens()

    def _build_rain(self):
        R = rng()
        n = 3500
        x = R.uniform(-24, 24, n); z = R.uniform(-24, 24, n); y = R.uniform(-8, 16, n)
        L = R.uniform(0.35, 0.65, n)
        w = 0.012
        a = np.stack([x - w, y, z], -1); b = np.stack([x + w, y, z], -1)
        c = np.stack([x + w, y + L, z], -1); d = np.stack([x - w, y + L, z], -1)
        tris = np.concatenate([np.stack([a, d, c], 1), np.stack([a, c, b], 1)])
        vc = np.zeros((2 * n, 3, 4))
        vc[:, :, 0] = np.tile(R.random(n), 2)[:, None]
        vc[:, :, 1] = np.tile(R.uniform(0.3, 1.0, n), 2)[:, None]
        vc[:, :, 3] = 1
        mb = MeshBuilder().add(tris, vcolors=vc, uv_mode='none')
        self.rain_e = Entity(shader=rain_shader)
        self.rain_e.attachNewNode(_node(mb))
        self.rain_e.setTwoSided(True)
        self.rain_e.setTransparency(TransparencyAttrib.M_alpha)
        self.rain_e.setBin('transparent', 20)
        self.rain_e.setDepthWrite(False)
        self.rain_e.hide(SHADOW_MASK)
        for np_ in self.rain_e.findAllMatches('**/+GeomNode'):
            np_.node().set_bounds(OmniBoundingVolume())
            np_.node().set_final(True)

    def _set_common(self, r, light_dir, sun_c, amb_s, amb_g, sky_hor, glow, rk, sky_top):
        r.set_shader_input('sun_dir', light_dir)
        r.set_shader_input('sun_col', Vec3(*sun_c))
        r.set_shader_input('amb_sky', Vec3(*amb_s))
        r.set_shader_input('amb_ground', Vec3(*amb_g))
        r.set_shader_input('fog_col', Vec3(*sky_hor))
        r.set_shader_input('sun_glow', Vec3(*glow))
        r.set_shader_input('fog_density', 1 / 340 / self.view_dist * (1 + rk * 1.6))
        r.set_shader_input('cam_pos', camera.world_position)
        r.set_shader_input('wind_strength', self.wind)
        r.set_shader_input('sky_top', Vec3(*sky_top))
        r.set_shader_input('exposure', self.exposure)
        r.set_shader_input('shadow_texel', 1.0 / max(self.shadow_res, 1) * 0.85)
        r.set_shader_input('lamps', self.lamps)
        r.set_shader_input('lamp_col', Vec3(1.0, 0.62, 0.30))

    @property
    def clock(self):
        mins = int(self.time * 24 * 60)
        return '%02d:%02d' % (mins // 60, mins % 60)

    def is_night(self):
        return self.night > 0.5

    def update(self, dt, focus):
        prev = self.time
        self.time += dt / DAY_LENGTH
        new_day = False
        if self.time >= 1.0:
            self.time -= 1.0
        if prev < 0.25 <= self.time:
            self.day += 1
            new_day = True
        self.rain_timer -= dt
        if self.rain_timer <= 0:
            self.raining = not self.raining
            r = rng()
            self.rain_timer = r.uniform(120, 260) if self.raining else r.uniform(500, 1100)
        target = 1.0 if self.raining else 0.0
        self.rain += (target - self.rain) * min(1.0, dt * 0.2)
        self.wet += ((1.0 if self.rain > 0.4 else 0.0) - self.wet) * min(1.0, dt * (0.08 if self.rain > 0.4 else 0.02))
        self.wind = 1.0 + self.rain * 1.4
        self.focus = focus
        self.apply()
        return new_day

    def skip_to_morning(self):
        if self.time > 0.25:
            self.day += 1
        self.time = 0.26

    def apply(self):
        a = (self.time - 0.25) * 2 * math.pi
        sd = Vec3(math.cos(a), math.sin(a) * 0.92, -0.40).normalized()
        e = sd.y
        sky_top, sky_hor, sun_c, amb_s, amb_g, glow = sample_keys(e)
        self.night = sstep(0.0, -0.2, e)
        rk = self.rain
        grey = (sum(sky_hor) / 3,) * 3
        overcast = mix(grey, (0.50, 0.53, 0.57), 0.5 * (1 - self.night))
        sky_hor = mix(sky_hor, overcast, rk * 0.8)
        sky_top = mix(sky_top, mix(grey, (0.38, 0.41, 0.47), 0.5 * (1 - self.night)), rk * 0.8)
        sun_c = tuple(c * (1 - 0.8 * rk) for c in sun_c)
        amb_s = tuple(c * (1 - 0.1 * rk) for c in amb_s)
        glow = tuple(c * (1 - 0.8 * rk) for c in glow)
        light_dir = sd if e > 0 else -sd
        strength = sstep(0.0, 0.08, abs(e))
        sun_c = tuple(c * strength for c in sun_c)

        for r in [self.render] + self.extra_roots:
            self._set_common(r, light_dir, sun_c, amb_s, amb_g, sky_hor, glow, rk, sky_top)
        r = self.render
        r.set_shader_input('sun_dir', light_dir)
        r.set_shader_input('sun_col', Vec3(*sun_c))
        r.set_shader_input('amb_sky', Vec3(*amb_s))
        r.set_shader_input('amb_ground', Vec3(*amb_g))
        r.set_shader_input('fog_col', Vec3(*sky_hor))
        r.set_shader_input('sun_glow', Vec3(*glow))
        r.set_shader_input('fog_density', 1 / 340 / self.view_dist * (1 + rk * 1.6))
        r.set_shader_input('cam_pos', camera.world_position)
        r.set_shader_input('wind_strength', self.wind)
        r.set_shader_input('sky_top', Vec3(*sky_top))
        r.set_shader_input('wetness', self.wet)
        self.sky.set_shader_input('sun_dir', sd)
        self.sky.set_shader_input('sky_hor', Vec3(*sky_hor))
        self.sky.set_shader_input('sky_top', Vec3(*sky_top))
        self.sky.set_shader_input('sun_glow', Vec3(*glow))
        self.sky.set_shader_input('night', self.night * (1 - rk))
        self.sky.set_shader_input('cloud_cover', 0.3 + 0.7 * rk)
        lit = mix((1.0, 0.99, 0.97), (1.0, 0.78, 0.58), sstep(0.3, 0.05, e))
        lit = mix(lit, (0.13, 0.15, 0.22), self.night)
        lit = mix(lit, (0.62, 0.64, 0.68), rk * (1 - self.night))
        dark = mix((0.66, 0.70, 0.78), (0.58, 0.44, 0.46), sstep(0.3, 0.05, e))
        dark = mix(dark, (0.04, 0.05, 0.09), self.night)
        dark = mix(dark, (0.36, 0.38, 0.42), rk * (1 - self.night))
        self.sky.set_shader_input('cloud_lit', Vec3(*lit))
        self.sky.set_shader_input('cloud_dark', Vec3(*dark))
        self.sky.position = camera.world_position
        self.sky_hor = sky_hor

        f = self.focus
        snap = self.shadow_size / max(self.shadow_res, 1) * 2
        fx, fz = round(f.x / snap) * snap, round(f.z / snap) * snap
        target = Vec3(fx, f.y, fz)
        self.sun.position = target + light_dir * 200
        self.sun.lookAt(target)

        self.rain_e.position = camera.world_position
        self.rain_e.enabled = rk > 0.01
        self.rain_e.set_shader_input('rain_alpha', 0.4 * rk)
        self.rain_e.set_shader_input('rain_col', Vec3(*mix((0.72, 0.77, 0.85), (0.22, 0.25, 0.34), self.night)))


def _noop():
    pass


def _node(mb):
    from panda3d.core import GeomNode
    gn = GeomNode('m')
    gn.add_geom(mb.geom())
    return gn

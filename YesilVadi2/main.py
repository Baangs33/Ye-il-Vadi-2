"""YEŞİL VADİ — birinci şahıs hayatta kalma, inşaat ve çiftçilik oyunu (Python + Ursina/Panda3D).

Çalıştırmak için:
    pip install -r requirements.txt
    python main.py
"""
import os
import sys
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import settings as S
SETTINGS = S.load()
AUTOTEST = os.environ.get('VADI_AUTOTEST')

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'sync-video %d\n' % (1 if SETTINGS.get('vsync', 1) and not AUTOTEST else 0))
loadPrcFileData('', 'framebuffer-multisample 0\n')

from ursina import (Ursina, Entity, Vec2, Vec3, camera, window, color, held_keys, mouse, application, time as utime,
                    destroy, scene, Text)

app = Ursina(title='Yeşil Vadi', borderless=False, development_mode=False,
             fullscreen=bool(SETTINGS.get('fullscreen')) and not AUTOTEST, size=(1280, 720) if AUTOTEST else None)
window.color = color.black
for attr in ('exit_button', 'fps_counter', 'entity_counter', 'collider_counter', 'cog_button', 'cog_menu'):
    try:
        getattr(window, attr).enabled = False
    except Exception:
        pass
camera.clip_plane_near = 0.08
camera.clip_plane_far = 700

import numpy as np
from config import *
import materials
materials.prewarm()
from geometry import Model, box_tris, cylinder_tris, cone_tris, sphere_tris, tube_tris, trs, rng
from terrain import Terrain, dist_to_road, lake_mask
from environment import Environment, SHADOW_MASK
from nature import Nature, GrassField, KIND_INFO
from buildings import BuildManager, SPECS, cost_of
from farm import FarmCache, Plot, TILE as FARM_TILE
from player import Player
from animals import Animal
from vehicle import Car
from effects import Particles, Fireflies
from audio import SoundBank
from shaders import post_shader, world_shader, glow_shader
import ui as UIM
from ui import UI, item_name, item_price, FOOD

R = np.random.default_rng(5)


def tr_lower(s):
    return s.replace('I', 'ı').replace('İ', 'i').lower()
SLOTS = ['balta', 'kazma', 'capa', 'kova', 'tohum', 'cekic']
KIND_NAMES = {'oak': 'Meşe Ağacı', 'pine': 'Çam Ağacı', 'birch': 'Huş Ağacı', 'rock': 'Kaya', 'iron': 'Demir Damarı',
              'bush': 'Böğürtlen Çalısı'}
WATER_MAX = 12


def merchant_model():
    md = Model()
    wood = (1.0, 0.9, 0.8, 1)
    md['planks'].add(box_tris(2.0, 0.12, 3.6), wood, m=trs((0, 0.95, 0)), uv_scale=1.2)
    for sx in (-1, 1):
        md['planks'].add(box_tris(0.08, 0.5, 3.6), wood, m=trs((sx * 0.98, 1.25, 0)), uv_scale=1.2)
    for i in range(7):
        a = math.radians(-90 + i * 30)
        a2 = math.radians(-90 + (i + 1) * 30)
        x0, y0 = math.sin(a) * 1.05, math.cos(a) * 1.05
        x1, y1 = math.sin(a2) * 1.05, math.cos(a2) * 1.05
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
        if i < 6:
            md['canvas'].add(box_tris(math.hypot(x1 - x0, y1 - y0) + 0.02, 0.03, 3.4), (0.95, 0.9, 0.78, 1),
                             m=trs((cx, 1.5 + cy, 0), (0, 0, ang)), uv_scale=1.0)
    for sx in (-1.05, 1.05):
        for sz in (-1.2, 1.2):
            t = cylinder_tris(0.5, 0.5, 0.1, 14)
            md['planks'].add(t, (0.7, 0.55, 0.4, 1), m=trs((sx, 0.5, sz), (0, 0, 90)) @ trs((0, -0.05, 0)), uv_scale=0.6)
            md['metal'].add(cylinder_tris(0.52, 0.52, 0.05, 14, caps=False), (0.3, 0.3, 0.3, 1),
                            m=trs((sx, 0.5, sz), (0, 0, 90)) @ trs((0, -0.025, 0)), spec=0.5)
    md['planks'].add(box_tris(0.1, 0.1, 2.0), wood, m=trs((0, 0.85, 2.7)))
    for k in range(3):
        md['planks'].add(box_tris(0.6, 0.5, 0.6, bottom=True), (1, 0.9, 0.75, 1), m=trs((2.0 + (k % 2) * 0.1, (k // 2) * 0.5, -0.8 + k * 0.7), (0, k * 20, 0)), uv_scale=0.8)
    # tüccar
    coat = (0.25, 0.32, 0.45, 1)
    skin = (0.85, 0.65, 0.52, 1)
    P = (1.5, 0, 1.4)
    for sx in (-0.11, 0.11):
        t, n, uv = tube_tris([(P[0] + sx, 0.0, P[2]), (P[0] + sx, 0.85, P[2])], [0.075, 0.08], 8)
        md['canvas'].add(t, (0.25, 0.2, 0.17, 1), normals=n, uv=uv)
    t, n, uv = tube_tris([(P[0], 0.8, P[2]), (P[0], 1.2, P[2]), (P[0], 1.5, P[2])], [0.24, 0.22, 0.18], 10)
    md['canvas'].add(t, coat, normals=n, uv=uv)
    for sx in (-1, 1):
        t, n, uv = tube_tris([(P[0] + sx * 0.24, 1.45, P[2]), (P[0] + sx * 0.3, 1.15, P[2] + 0.1), (P[0] + sx * 0.25, 0.95, P[2] + 0.2)],
                             [0.07, 0.065, 0.06], 7)
        md['canvas'].add(t, coat, normals=n, uv=uv)
        md['plain'].add(sphere_tris(0.055, 1), skin, m=trs((P[0] + sx * 0.25, 0.92, P[2] + 0.22)), smooth='sphere')
    md['plain'].add(sphere_tris(0.15, 2), skin, m=trs((P[0], 1.68, P[2]), (0, 0, 0), (1, 1.12, 1)), smooth='sphere')
    md['plain'].add(sphere_tris(0.13, 2, squash=0.8), (0.55, 0.52, 0.48, 1), m=trs((P[0], 1.58, P[2] + 0.06)), smooth='sphere')
    md['canvas'].add(cylinder_tris(0.3, 0.3, 0.03, 14), (0.3, 0.22, 0.15, 1), m=trs((P[0], 1.8, P[2])))
    md['canvas'].add(cylinder_tris(0.16, 0.14, 0.18, 14), (0.3, 0.22, 0.15, 1), m=trs((P[0], 1.8, P[2])))
    for sx in (-0.05, 0.05):
        md['plain'].add(sphere_tris(0.018, 1), (0.05, 0.05, 0.05, 1), m=trs((P[0] + sx, 1.71, P[2] + 0.135)), spec=1)
    return md


class Game:
    def __init__(self):
        self.settings = SETTINGS
        self.t = 0.0
        self.state = 'menu'
        self.terrain = Terrain()
        self.nature = Nature(self.terrain)
        self.env = Environment(render)
        self.grass = GrassField(self.terrain, self.nature)
        self.builds = BuildManager(self)
        self.farm_cache = FarmCache()
        self.plots = {}
        self.sounds = SoundBank()
        self.particles = Particles()
        self.player = Player(self)
        self.env.extra_roots.append(self.player.vm_root)
        self.fireflies = Fireflies(self.terrain)
        self.merchant = Entity(shader=world_shader, position=(MERCHANT[0], self.terrain.ground_y(*MERCHANT), MERCHANT[1]),
                               rotation_y=-70)
        self.merchant.attachNewNode(merchant_model().node('merchant'))
        self.merchant_lamp = Entity(parent=self.merchant, shader=glow_shader, position=(1.0, 2.0, 1.85))
        lm = Model(); lm['plain'].add(sphere_tris(0.1, 1), (1, 0.85, 0.5, 1), uv_mode='none')
        self.merchant_lamp.attachNewNode(lm.node('l'))
        self.animals = []
        self.car = None
        self.car_parts = set()
        self.ui = UI(self)
        self.hud = self.ui.hud
        self.reset_state()
        self.target = None
        self.target_info = None
        self.hold_e = 0.0
        self.trade_is_stall = False
        self.autosave_t = 120
        self.bird_cd = 3
        self.fade = Entity(parent=camera.ui, model='quad', color=color.rgba(0, 0, 0, 0), scale=(4, 2), z=-0.9)
        self.fade_t = 0.0
        self.menu_angle = 0.0
        camera.shader = post_shader
        self.apply_settings()
        self.grass.fill_all(SPAWN[0], SPAWN[1])
        self.enter_menu()

    # ------------------------------------------------------------------ durum
    def reset_state(self):
        self.money = START_MONEY
        self.inventory = {}
        self.seeds = {'bugday': 8, 'havuc': 4}
        self.seed_kind = 'bugday'
        self.water = WATER_MAX
        self.slot = 0
        p = self.player
        p.pos = Vec3(SPAWN[0], self.terrain.ground_y(*SPAWN), SPAWN[1])
        p.vel = Vec3(0, 0, 0)
        p.yaw, p.pitch = 30.0, 4.0
        p.hunger, p.energy = 100.0, 100.0
        p.in_vehicle = None

    def clear_world(self):
        for a in self.animals:
            a.destroy()
        self.animals = []
        for b in list(self.builds.buildings):
            self.builds.remove(b)
        self.plots = {}
        if self.car is not None:
            self.car.root.removeNode()
            self.car = None
        self.car_parts = set()
        self.grass.blockers = []
        self.grass.set_density(*S.GRASS[int(self.settings['grass'])])
        for it in self.nature.items:
            changed = (not it.alive) or (it.kind == 'bush' and not it.berries) or it.hp != KIND_INFO[it.kind]['hp']
            it.alive = True
            it.berries = it.kind == 'bush'
            it.hp = KIND_INFO[it.kind]['hp'] * (it.s if it.kind in ('rock', 'iron') else 1)
            if changed:
                self.nature.dirty.add(it.chunk)
        self.nature.flush()
        self.env.time = START_TIME
        self.env.day = 1
        self.env.raining = False
        self.env.rain = 0
        self.env.rain_timer = 420

    def has_save(self):
        return os.path.exists(SAVE_FILE)

    def enter_menu(self):
        self.state = 'menu'
        mouse.locked = False
        mouse.visible = True
        self.ui.menu.show(True)
        self.hud.root.enabled = False
        self.player.vm_root.hide()
        self.ui.pause.show(False)
        self.ui.close_windows()

    def begin_play(self):
        self.state = 'play'
        self.ui.menu.show(False)
        self.ui.settings.root.enabled = False
        self.hud.root.enabled = True
        self.player.vm_root.show()
        self.lock_mouse(True)
        self.select_slot(self.slot, quiet=True)
        self.grass.fill_all(self.player.pos.x, self.player.pos.z)

    def new_game(self):
        self.sounds.play('click', 0.5)
        self.clear_world()
        self.reset_state()
        self.begin_play()
        self.hud.message('Vadiye hoş geldin! Önce ağaç kesip (Balta) kütük topla.', 'gold', 6)
        self.hud.message('B tuşu ile İnşaat Defteri\'ni açıp ilk barınağını kur.', 'info', 7)

    def continue_game(self):
        self.sounds.play('click', 0.5)
        self.clear_world()
        self.reset_state()
        try:
            self.load_game()
        except Exception as ex:
            print('Kayıt okunamadı:', ex)
        self.begin_play()
        self.hud.message('Tekrar hoş geldin!  Gün %d' % self.env.day, 'gold', 4)

    def to_main_menu(self):
        self.save_game()
        self.ui.pause.show(False)
        self.builds.cancel_preview()
        self.enter_menu()

    def quit_game(self):
        application.quit()

    def save_and_quit(self):
        self.save_game()
        application.quit()

    def set_paused(self, v):
        if v:
            self.state = 'paused'
            self.lock_mouse(False)
            self.ui.pause.show(True)
            self.hud.root.enabled = False
        else:
            self.state = 'play'
            self.ui.pause.show(False)
            self.ui.settings.root.enabled = False
            self.hud.root.enabled = True
            self.lock_mouse(True)

    def lock_mouse(self, v):
        mouse.locked = v
        mouse.visible = not v
        self.player.input_enabled = v

    # ------------------------------------------------------------------ ayarlar
    def apply_settings(self):
        st = self.settings
        res = S.SHADOW_RES[int(st['shadows'])]
        if res != self.env.shadow_res:
            self.env.set_shadow_quality(res, 90.0 if res < 4096 else 110.0)
        gd = S.GRASS[int(st['grass'])]
        if (gd[0], gd[1]) != (self.grass.density, self.grass.radius):
            self.grass.set_density(*gd)
        fogk, far = S.VIEW[int(st['view'])]
        self.env.view_dist = fogk
        camera.clip_plane_far = far
        self.player.fov = float(st['fov'])
        self.player.sens = float(st['sens']) * (-1 if st.get('invert') else 1)
        self.player.bob_scale = 1.0 if st.get('head_bob', 1) else 0.0
        self.sounds.master = float(st['volume']) / 100.0
        camera.set_shader_input('ao_amt', 0.55 if st['ao'] else 0.0)
        camera.set_shader_input('bloom_amt', 0.5 if st['bloom'] else 0.0)
        camera.set_shader_input('fxaa_on', 1.0 if st['fxaa'] else 0.0)
        camera.set_shader_input('near', camera.clip_plane_near)
        camera.set_shader_input('far', camera.clip_plane_far)
        if not AUTOTEST:
            try:
                if bool(st['fullscreen']) != bool(window.fullscreen):
                    window.fullscreen = bool(st['fullscreen'])
            except Exception:
                pass

    # ------------------------------------------------------------------ kayıt
    def save_game(self):
        if self.state == 'menu':
            return
        p = self.player
        data = {
            'money': self.money, 'inventory': self.inventory, 'seeds': self.seeds, 'seed_kind': self.seed_kind,
            'water': self.water, 'slot': self.slot,
            'player': [round(p.pos.x, 2), round(p.pos.y, 2), round(p.pos.z, 2), round(p.yaw, 1), round(p.pitch, 1),
                       round(p.hunger, 1), round(p.energy, 1)],
            'env': [self.env.day, self.env.time, self.env.raining, self.env.rain_timer],
            'nature': self.nature.to_dict(), 'buildings': self.builds.to_list(),
            'animals': [a.to_list() for a in self.animals],
            'car': self.car.to_dict() if self.car else None, 'car_parts': sorted(self.car_parts),
            'next_uid': __import__('buildings').Building._next_id,
        }
        with open(SAVE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        self.hud.message('Oyun kaydedildi.', 'good', 1.8)

    def load_game(self):
        with open(SAVE_FILE, encoding='utf-8') as f:
            d = json.load(f)
        self.money = d['money']
        self.inventory = {k: v for k, v in d['inventory'].items()}
        self.seeds = d['seeds']
        self.seed_kind = d.get('seed_kind', 'bugday')
        self.water = d.get('water', WATER_MAX)
        self.slot = d.get('slot', 0)
        p = self.player
        px, py, pz, yaw, pitch, hu, en = d['player']
        p.pos = Vec3(px, py, pz)
        p.yaw, p.pitch, p.hunger, p.energy = yaw, pitch, hu, en
        day, tm, raining, rt = d['env']
        self.env.day, self.env.time, self.env.raining, self.env.rain_timer = day, tm, raining, rt
        self.env.rain = 1.0 if raining else 0.0
        self.nature.from_dict(d['nature'])
        self.builds.from_list(d['buildings'])
        __import__('buildings').Building._next_id = d.get('next_uid', 1)
        for b in self.builds.buildings:
            if b.kind == 'tarla' and b.complete:
                self.plots[b.uid] = Plot(self, b, self.farm_cache)
        for kind, home, x, z, timer in d.get('animals', []):
            self.animals.append(Animal(self, kind, home, x, z, timer))
        if d.get('car'):
            c = d['car']
            self.car = Car(self, c['x'], c['z'], c['h'])
        self.car_parts = set(d.get('car_parts', []))

    # ------------------------------------------------------------------ fizik yardımcıları
    def ground_height(self, x, z, y_ref):
        g = self.terrain.ground_y(x, z)
        f = self.builds.floor_height(x, z)
        if f is not None and f <= y_ref + 0.55 and f > g:
            return f
        return g

    def resolve_collision(self, x, z, y, r, animals=False, car=True):
        circles = [(cx, cz, cr) for (cx, cz, cr, top) in self.nature.colliders(x, z, 4.0) if y < top]
        circles += self.builds.circles_near(x, z, 6)
        mx, mz = MERCHANT
        circles.append((mx + 0.0, mz, 2.0))
        if car and self.car is not None and self.player.in_vehicle is None:
            a = math.radians(self.car.heading)
            for off in (1.4, 0.0, -1.4):
                circles.append((self.car.root.x + math.sin(a) * off, self.car.root.z + math.cos(a) * off, 1.0))
        for cx, cz, cr in circles:
            dx, dz = x - cx, z - cz
            if abs(dx) > cr + r or abs(dz) > cr + r:
                continue
            d = math.hypot(dx, dz)
            if 1e-6 < d < cr + r:
                push = cr + r - d
                x += dx / d * push
                z += dz / d * push
        for (ax, az, bx, bz) in self.builds.segments_near(x, z, 4):
            vx, vz = bx - ax, bz - az
            L2 = vx * vx + vz * vz
            if L2 < 1e-6:
                continue
            t = max(0, min(1, ((x - ax) * vx + (z - az) * vz) / L2))
            px, pz = ax + vx * t, az + vz * t
            dx, dz = x - px, z - pz
            d = math.hypot(dx, dz)
            lim = r + 0.22
            if 1e-6 < d < lim:
                x += dx / d * (lim - d)
                z += dz / d * (lim - d)
        return x, z

    def on_road(self, x, z):
        if dist_to_road(x, z) < 3.5:
            return True
        for b in self.builds.buildings:
            if b.kind == 'yol' and b.complete and math.hypot(b.x - x, b.z - z) < 2.0:
                return True
        return False

    # ------------------------------------------------------------------ envanter / ekonomi
    def add_item(self, k, n=1):
        self.inventory[k] = self.inventory.get(k, 0) + n

    def eat(self, k=None):
        p = self.player
        if k is None:
            options = [(FOOD[i], i) for i, v in self.inventory.items() if v > 0 and FOOD.get(i)]
            if not options:
                self.hud.message('Yiyeceğin yok! Böğürtlen topla ya da hasat yap.', 'red')
                return
            options.sort(key=lambda fv: abs(fv[0] - (100 - p.hunger)))
            k = options[0][1]
        if self.inventory.get(k, 0) <= 0 or not FOOD.get(k):
            return
        if p.hunger > 97:
            self.hud.message('Tokum, şimdi yiyemem.')
            return
        self.inventory[k] -= 1
        p.hunger = min(100, p.hunger + FOOD[k])
        p.energy = min(100, p.energy + FOOD[k] * 0.4)
        self.sounds.play('eat', 0.5)
        self.hud.message('%s yedin (+%d)' % (item_name(k), FOOD[k]), 'good', 1.8)

    def buy_seed(self, k, n):
        cost = CROPS[k]['seed'] * n
        if self.money < cost:
            return
        self.money -= cost
        self.seeds[k] = self.seeds.get(k, 0) + n
        self.sounds.play('buy', 0.5)

    def sell(self, k, n=None):
        have = self.inventory.get(k, 0)
        n = have if n is None else min(n, have)
        if n <= 0:
            return
        self.inventory[k] -= n
        self.money += n * item_price(k)
        self.sounds.play('coin', 0.6)

    def sell_all(self):
        total = 0
        for k, v in list(self.inventory.items()):
            if v > 0 and item_price(k):
                total += v * item_price(k)
                self.inventory[k] = 0
        self.money += total
        self.sounds.play('coin', 0.7)
        self.hud.message('+%d altın' % total, 'gold')

    def animal_capacity(self, kind):
        a = ANIMALS[kind]
        homes = [b for b in self.builds.buildings if b.kind == a['home'] and b.complete]
        cap = len(homes) * a['per_home']
        have = sum(1 for an in self.animals if an.kind == kind)
        return cap, have

    def buy_animal(self, kind):
        a = ANIMALS[kind]
        cap, have = self.animal_capacity(kind)
        if have >= cap:
            self.hud.message('Önce bir %s inşa etmelisin.' % ('kümes' if a['home'] == 'kumes' else 'ahır'), 'red')
            return
        if self.money < a['price']:
            return
        homes = [b for b in self.builds.buildings if b.kind == a['home'] and b.complete]
        home = min(homes, key=lambda b: sum(1 for an in self.animals if an.home_uid == b.uid and an.kind == kind))
        self.money -= a['price']
        self.animals.append(Animal(self, kind, home.uid))
        self.sounds.play({'tavuk': 'cluck', 'koyun': 'baa', 'inek': 'moo'}[kind], 0.6)
        self.hud.message('Yeni %s barınağına gönderildi!' % tr_lower(a['name']), 'good')

    def buy_car(self):
        if self.car is not None or self.money < CAR_PRICE:
            return
        self.money -= CAR_PRICE
        self.car = Car(self, 4.6, -22.0, 200)
        self.sounds.play('coin', 0.7)
        self.hud.message('Kamyonet tüccarın yanında seni bekliyor! (E ile bin)', 'gold', 5)

    def craft_part(self, k):
        p = CAR_PARTS[k]
        if k in self.car_parts:
            return
        if any(self.inventory.get(rk, 0) < n for rk, n in p['cost'].items()) or self.money < p['money']:
            return
        for rk, n in p['cost'].items():
            self.inventory[rk] -= n
        self.money -= p['money']
        self.car_parts.add(k)
        self.sounds.play('build', 0.6)
        self.hud.message('%s üretildi!' % p['name'], 'good')

    def assemble_car(self):
        if self.car is not None or len(self.car_parts) < len(CAR_PARTS):
            return
        gar = self.current_garage or next((b for b in self.builds.buildings if b.kind == 'garaj' and b.complete), None)
        if gar is None:
            return
        p = gar.world_point(0, 0, gar.spec.d / 2 + 3.5)
        self.car = Car(self, p.x, p.z, gar.rot)
        self.car_parts = set()
        self.sounds.play('coin', 0.7)
        self.hud.message('Kamyonetin hazır! Yanına gidip E ile bin.', 'gold', 5)

    # ------------------------------------------------------------------ alet seçimi
    def select_slot(self, i, quiet=False):
        self.slot = i
        k = SLOTS[i]
        sc = CROPS[self.seed_kind]['color'] if k == 'tohum' else None
        self.player.set_tool(k, sc)
        name = UIM.TOOL_NAME[k]
        if k == 'tohum':
            name = '%s Tohumu  (R: değiştir)' % CROPS[self.seed_kind]['name']
        self.hud.set_tool(i, name)
        if not quiet:
            self.sounds.play('click', 0.25)

    def cycle_seed(self):
        i = CROP_ORDER.index(self.seed_kind)
        self.seed_kind = CROP_ORDER[(i + 1) % len(CROP_ORDER)]
        self.player.tool = None
        self.select_slot(self.slot)

    # ------------------------------------------------------------------ inşaat
    def choose_blueprint(self, kind):
        money = SPECS[kind][4]
        if money and self.money < money:
            self.hud.message('Bu yapı için %d altın gerekiyor.' % money, 'red')
            return
        self.ui.close_windows()
        self.builds.start_preview(kind)
        self.lock_mouse(True)
        self.slot = SLOTS.index('cekic')
        self.select_slot(self.slot, quiet=True)

    def place_blueprint(self):
        kind = self.builds.preview_kind
        if not self.builds.preview_ok:
            self.hud.message(self.builds.preview_why or 'Buraya yerleştirilemez.', 'red', 1.8)
            self.sounds.play('error', 0.3)
            return
        money = SPECS[kind][4]
        if money:
            if self.money < money:
                self.hud.message('Yeterli altının yok.', 'red')
                return
            self.money -= money
        b = self.builds.place()
        if b is not None:
            self.sounds.play('build', 0.5)
            self.hud.message('%s planı yerleştirildi. Malzeme eklemek için plana bakıp E\'ye bas (basılı tut).' % SPECS[kind][0], 'blue', 5)
            if not held_keys['shift']:
                self.builds.cancel_preview()

    def add_to_building(self, b):
        res, missing = b.add_material(self.inventory)
        if res:
            self.sounds.play('build', 0.35, None)
            pt = b.world_point(0, min(2.0, b.spec.height * 0.5), 0)
            if res != 'free':
                self.particles.emit(pt, color.rgb(0.6, 0.45, 0.3) if res == 'kutuk' else color.rgb(0.6, 0.6, 0.6), 4, 1.2, 1.5, 0.5, 0.05)
            if b.complete:
                self.hud.message('%s tamamlandı!' % SPECS[b.kind][0], 'gold', 4)
                self.sounds.play('coin', 0.5, 0.8)
            return True
        if missing:
            self.hud.message('%s gerekiyor.' % item_name(missing), 'red', 1.4)
            self.sounds.play('error', 0.25)
        return False

    def on_building_complete(self, b):
        if b.kind == 'tarla' and b.uid not in self.plots:
            self.plots[b.uid] = Plot(self, b, self.farm_cache)

    # ------------------------------------------------------------------ hedefleme
    def compute_target(self):
        p = self.player
        eye = p.eye
        d = p.look_dir()
        o = (eye.x, eye.y, eye.z)
        dv = (d.x, d.y, d.z)
        best = None
        bd = 4.0
        # bina / plan
        b, t = self.builds.raycast(o, dv, 6.0)
        if b is not None and t < bd + 1.5:
            info = None
            if b.kind == 'tarla' and b.complete and b.uid in self.plots:
                plot = self.plots[b.uid]
                # zemin düzlemi kesişimi
                if d.y < -0.05:
                    tt = (b.y - eye.y) / d.y
                    hx, hz = eye.x + d.x * tt, eye.z + d.z * tt
                    ij = plot.tile_at_world(hx, hz)
                    if ij is not None and tt < 4.5:
                        best = ('tile', (plot, ij), tt)
                        bd = tt
            if best is None:
                best = ('building', b, t)
                bd = t
        r, rt = self.nature.raycast(o, dv, 3.6)
        if r is not None and rt < bd:
            best = ('res', r, rt)
            bd = rt
        # hayvanlar
        for a in self.animals:
            c = a.root.world_position + Vec3(0, a.top * 0.6, 0)
            rel = c - eye
            proj = rel.x * d.x + rel.y * d.y + rel.z * d.z
            if 0 < proj < min(bd, 3.6):
                perp = (rel - d * proj).length()
                if perp < (0.5 if a.kind == 'tavuk' else 0.9):
                    best = ('animal', a, proj)
                    bd = proj
        # tüccar
        mpos = self.merchant.world_position + Vec3(0, 1.2, 0)
        rel = mpos - eye
        proj = rel.x * d.x + rel.y * d.y + rel.z * d.z
        if 0 < proj < min(bd, 5.0) and (rel - d * proj).length() < 2.2:
            best = ('merchant', None, proj)
            bd = proj
        # araba
        if self.car is not None:
            cpos = self.car.root.world_position + Vec3(0, 1.0, 0)
            rel = cpos - eye
            proj = rel.x * d.x + rel.y * d.y + rel.z * d.z
            if 0 < proj < min(bd + 0.5, 5.0) and (rel - d * proj).length() < 2.4:
                best = ('car', self.car, proj)
                bd = proj
        # su (göl)
        if best is None and d.y < -0.1:
            tw = self.terrain.raycast(o, dv, 5.0)
            if tw is not None:
                hy = eye.y + d.y * tw
                hx, hz = eye.x + d.x * tw, eye.z + d.z * tw
                if lake_mask(hx, hz) > 0.3 or self.terrain.ground_y(hx, hz) < WATER_Y:
                    best = ('water', None, tw)
        self.target = best
        self.target_info = self._describe(best)

    def _describe(self, tg):
        if tg is None:
            return None
        kind, obj, dist = tg
        tool = SLOTS[self.slot]
        if kind == 'res':
            it = obj
            info = {'name': KIND_NAMES[it.kind]}
            need = KIND_INFO[it.kind]['tool']
            if it.kind == 'bush':
                info['prompt'] = '[E] Böğürtlen topla' if it.berries else 'Meyveler henüz olgunlaşmadı'
            else:
                mx = KIND_INFO[it.kind]['hp'] * (it.s if it.kind in ('rock', 'iron') else 1)
                info['hp'] = max(0, it.hp / mx)
                info['prompt'] = '[Sol Tık] %s' % ('Kes' if need == 'balta' else 'Kaz') if tool == need else \
                    '%s gerekli' % UIM.TOOL_NAME[need]
            return info
        if kind == 'building':
            b = obj
            name = SPECS[b.kind][0]
            if not b.complete:
                rem = b.remaining()
                cost = b.spec.cost()
                parts = ['%s %d/%d' % (item_name(k), cost[k] - rem.get(k, 0), cost[k]) for k in ('kutuk', 'tas', 'demir') if k in cost]
                nxt = b.needed_next()
                have = self.inventory.get(nxt, 0) if nxt else 0
                pr = '[E] %s ekle (basılı tut)' % item_name(nxt) if nxt and have else ('%s gerekiyor' % item_name(nxt) if nxt else '')
                return {'name': '%s (plan)  •  %s' % (name, '  '.join(parts)), 'prompt': pr + '    [X] Planı kaldır',
                        'hp': b.progress, 'bar_col': UIM.BLUE}
            pr = ''
            pt = self._nearest_point(b)
            if pt == 'sleep':
                pr = '[E] Uyu'
            elif pt == 'water':
                pr = '[E] Sulama kabını doldur'
            elif pt == 'cook':
                pr = '[E] Çorba pişir (2 sebze)'
            elif pt == 'trade':
                pr = '[E] Tezgâhı aç'
            elif pt == 'garage':
                pr = '[E] Garaj tezgâhı'
            return {'name': name, 'prompt': pr}
        if kind == 'tile':
            plot, ij = obj
            t = plot.get(*ij)
            if t and t[2]:
                st = plot.stage(t)
                c = CROPS[t[2]]['name']
                if st >= 3:
                    return {'name': c + ' (olgun)', 'prompt': '[Sol Tık] Hasat et', 'hp': 1.0, 'bar_col': UIM.GOOD}
                pct = min(0.99, t[3] / CROPS[t[2]]['time'])
                return {'name': c, 'prompt': ('Islak' if t[1] > 0 else 'Susuz — sulamalısın!'), 'hp': pct,
                        'bar_col': UIM.GOOD if t[1] > 0 else UIM.BAD}
            if t:
                return {'name': 'Sürülmüş toprak' + ('  (ıslak)' if t[1] > 0 else ''),
                        'prompt': '[Sol Tık] Ek' if tool == 'tohum' else 'Tohum seç (5) ve ek'}
            return {'name': 'Tarla', 'prompt': '[Sol Tık] Çapala' if tool == 'capa' else 'Çapa ile sür (3)'}
        if kind == 'animal':
            a = obj
            prod = PRODUCTS[ANIMALS[a.kind]['product']]['name']
            return {'name': ANIMALS[a.kind]['name'], 'prompt': '[E] %s topla' % prod if a.ready else '%s henüz hazır değil' % prod}
        if kind == 'merchant':
            return {'name': 'Gezgin Tüccar', 'prompt': '[E] Alışveriş yap'}
        if kind == 'car':
            return {'name': 'Kamyonet', 'prompt': '[E] Bin'}
        if kind == 'water':
            return {'name': 'Göl', 'prompt': '[E] Sulama kabını doldur' if self.water < WATER_MAX else ''}
        return None

    def _nearest_point(self, b):
        pts = b.points_world()
        p = self.player.pos
        best, bd = None, 3.2
        for k, wp in pts:
            d = math.hypot(wp.x - p.x, wp.z - p.z)
            if k in ('home', 'farm'):
                continue
            if d < bd:
                best, bd = k, d
        return best

    # ------------------------------------------------------------------ eylemler
    def use_tool(self):
        tool = SLOTS[self.slot]
        tg = self.target
        p = self.player
        if tg is None:
            p.start_swing(lambda: None, 'chop' if tool in ('balta', 'kazma', 'cekic') else 'small', 1.0)
            return
        kind, obj, dist = tg
        if kind == 'res':
            it = obj
            need = KIND_INFO[it.kind]['tool']
            if it.kind == 'bush':
                self.interact()
                return
            if tool != need:
                self.hud.message('Bunun için %s kullanmalısın.' % UIM.TOOL_NAME[need], 'red', 1.5)
                return

            def hit():
                eye = p.eye
                hp = eye + p.look_dir() * max(0.3, dist - 0.1)
                res, data = self.nature.hit(it, tool, 1.0)
                if tool == 'balta':
                    self.sounds.play('chop', 0.6)
                    self.particles.emit(hp, color.rgb(0.75, 0.6, 0.42), 8, 2.0, 2.5, 0.7, 0.04)
                else:
                    self.sounds.play('mine', 0.55)
                    self.particles.emit(hp, color.rgb(0.55, 0.55, 0.55) if it.kind == 'rock' else color.rgb(0.6, 0.4, 0.3), 8, 2.0, 2.5, 0.7, 0.05)
                if res == 'done':
                    k, n = data
                    self.add_item(k, n)
                    if it.kind in ('oak', 'pine', 'birch'):
                        self.sounds.play('treefall', 0.7)
                    self.hud.pickup('+%d %s' % (n, item_name(k)))
            p.start_swing(hit, 'chop' if tool == 'balta' else 'mine')
            return
        if kind == 'tile':
            plot, ij = obj
            t = plot.get(*ij)
            ripe = t and t[2] and plot.stage(t) >= 3
            if tool == 'tohum' and not ripe and self.seeds.get(self.seed_kind, 0) <= 0:
                self.hud.message('%s tohumun yok. Tüccardan alabilirsin.' % CROPS[self.seed_kind]['name'], 'red')
                return
            if tool == 'kova' and not ripe and self.water <= 0:
                self.hud.message('Sulama kabı boş! Kuyuda ya da gölde E ile doldur.', 'red')
                return
            if tool not in ('capa', 'kova', 'tohum') and not ripe:
                self.hud.message('Çapa (3), sulama kabı (4) ya da tohum (5) kullan.')
                return

            def do():
                res, data = plot.act(ij, tool, self.seed_kind)
                cx, cz = plot.center_local(*ij)
                wp = plot.b.world_point(cx, 0.2, cz)
                if res == 'harvest':
                    n = 1 + (1 if R.random() < 0.35 else 0)
                    self.add_item(data, n)
                    self.sounds.play('harvest', 0.6)
                    self.particles.emit(wp + Vec3(0, 0.3, 0), color.rgb(*CROPS[data]['color']), 10, 2, 3, 0.7, 0.06)
                    self.hud.pickup('+%d %s' % (n, CROPS[data]['name']))
                elif res == 'tilled':
                    self.sounds.play('hoe', 0.55)
                    self.particles.emit(wp, color.rgb(0.35, 0.25, 0.15), 10, 1.5, 2.5, 0.6, 0.05)
                elif res == 'watered':
                    self.water -= 1
                    self.sounds.play('water', 0.5)
                    self.particles.emit(wp + Vec3(0, 0.8, 0), color.rgb(0.6, 0.8, 1.0), 16, 0.6, 0.3, 0.6, 0.03)
                elif res == 'planted':
                    self.seeds[self.seed_kind] -= 1
                    self.sounds.play('plant', 0.5)
                elif res == 'untilled':
                    self.hud.message('Önce çapa ile toprağı sür.')
                elif res == 'busy':
                    self.hud.message('Burada zaten ekin var.')
            p.start_swing(do, 'hoe' if tool == 'capa' else ('water' if tool == 'kova' else 'small'), 2.0)
            return
        if kind == 'building' and not obj.complete and tool == 'cekic':
            p.start_swing(lambda: self.add_to_building(obj), 'build', 1.5)
            return
        p.start_swing(lambda: None, 'chop' if tool in ('balta', 'kazma', 'cekic') else 'small', 1.0)

    def interact(self):
        tg = self.target
        p = self.player
        if tg is None:
            return
        kind, obj, dist = tg
        if kind == 'res' and obj.kind == 'bush':
            res, data = self.nature.hit(obj, None)
            if res == 'picked':
                k, n = data
                self.add_item(k, n)
                self.sounds.play('pickup', 0.5)
                self.hud.pickup('+%d Böğürtlen' % n)
            return
        if kind == 'building':
            b = obj
            if not b.complete:
                self.add_to_building(b)
                return
            pt = self._nearest_point(b)
            if pt == 'sleep':
                self.sleep()
            elif pt == 'water':
                self.refill()
            elif pt == 'cook':
                self.cook()
            elif pt == 'trade':
                self.open_window(self.ui.trade, stall=True)
            elif pt == 'garage':
                self.current_garage = b
                self.open_window(self.ui.garage)
            return
        if kind == 'animal':
            prod = obj.collect()
            if prod:
                self.add_item(prod)
                self.sounds.play('pickup', 0.5)
                self.hud.pickup('+1 %s' % PRODUCTS[prod]['name'])
            return
        if kind == 'merchant':
            self.open_window(self.ui.trade, stall=False)
            return
        if kind == 'car':
            self.enter_car()
            return
        if kind == 'water':
            self.refill()

    def refill(self):
        if self.water >= WATER_MAX:
            self.hud.message('Sulama kabı zaten dolu.')
            return
        self.water = WATER_MAX
        self.sounds.play('water', 0.6, 0.8)
        self.hud.message('Sulama kabı doldu.', 'blue', 1.6)

    def cook(self):
        veg = [k for k in ('havuc', 'domates', 'misir', 'balkabagi', 'bugday') if self.inventory.get(k, 0) > 0]
        total = sum(self.inventory.get(k, 0) for k in veg)
        if total < 2:
            self.hud.message('Çorba için 2 sebze gerekiyor.', 'red')
            return
        used = 0
        for k in veg:
            while self.inventory.get(k, 0) > 0 and used < 2:
                self.inventory[k] -= 1
                used += 1
        self.add_item('corba')
        self.sounds.play('water', 0.4, 0.6)
        self.hud.pickup('+1 Sebze Çorbası (+45 tokluk)')

    def sleep(self):
        e = self.env
        p = self.player
        late = e.time > 0.82 or e.time < 0.24
        if not late and p.energy > 40:
            self.hud.message('Henüz uyku vakti değil (akşam 20:00 sonrası).')
            return
        self.fade_t = 2.4
        self._sleep_pending = True
        self.lock_mouse(False)

    def _finish_sleep(self):
        e = self.env
        p = self.player
        e.skip_to_morning()
        p.energy = 100
        p.hunger = max(0, p.hunger - 18)
        self.lock_mouse(True)
        self.hud.message('Günaydın! Gün %d başladı.' % e.day, 'gold', 4)
        self.save_game()

    def enter_car(self):
        c = self.car
        self.player.in_vehicle = c
        c.engine_on = True
        c.cam_yaw = c.heading
        self.sounds.play('click', 0.4)
        self.hud.message('W/S: gaz/fren  A/D: direksiyon  Boşluk: el freni  V: kamera  E: in', 'info', 5)

    def exit_car(self):
        c = self.car
        x, z = c.exit_point()
        x, z = self.resolve_collision(x, z, c.root.y, 0.4)
        p = self.player
        p.pos = Vec3(x, self.ground_height(x, z, c.root.y + 1), z)
        p.yaw = c.heading
        p.in_vehicle = None
        c.engine_on = False
        c.speed *= 0.3
        self.sounds.set_loop('engine', 0)

    def open_window(self, w, stall=False):
        self.trade_is_stall = stall
        self.ui.close_windows()
        w.show()
        self.lock_mouse(False)

    def close_windows(self):
        self.ui.close_windows()
        if self.state == 'play':
            self.lock_mouse(True)

    def mode_text(self):
        if self.builds.preview is not None:
            name = SPECS[self.builds.preview_kind][0]
            return '%s  •  Sol Tık: yerleştir   R/Tekerlek: döndür   Shift+Tık: çoklu   Sağ Tık: vazgeç' % name
        return ''

    # ------------------------------------------------------------------ ışıklar
    def _update_lights(self):
        n = self.env.night
        p = self.player
        cand = []
        if n > 0.05:
            if p.in_vehicle is None:
                cand.append((p.eye + p.look_dir() * 0.6, 1.4 * n))
            for b in self.builds.buildings:
                if abs(b.x - p.pos.x) > 60 or abs(b.z - p.pos.z) > 60:
                    continue
                for wp, w in b.lights_world():
                    cand.append((wp, w * (1.0 if b.kind == 'ates' else n)))
            mp = self.merchant_lamp.world_position
            cand.append((mp, 2.0 * n))
        else:
            for b in self.builds.buildings:
                if b.kind == 'ates' and b.complete:
                    for wp, w in b.lights_world():
                        cand.append((wp, w * 0.5))
        if self.car is not None and self.car.engine_on and n > 0.05:
            a = math.radians(self.car.heading)
            for d in (5.0, 10.0):
                cand.append((Vec3(self.car.root.x + math.sin(a) * d, self.car.root.y + 1.0, self.car.root.z + math.cos(a) * d), 2.2 * n))
        cand.sort(key=lambda c: (c[0] - p.pos).length())
        from panda3d.core import LVecBase4f
        for i in range(8):
            if i < len(cand):
                wp, w = cand[i]
                self.env.lamps.set_element(i, LVecBase4f(wp.x, wp.y, wp.z, w))
            else:
                self.env.lamps.set_element(i, LVecBase4f(0, -1000, 0, 0))
        self.merchant_lamp.set_shader_input('glow', 0.8 + 2.5 * n)
        for b in self.builds.buildings:
            for e in b.fire_entities:
                if b.kind == 'ates':
                    e.set_shader_input('glow', 1.6 + 0.5 * math.sin(self.t * 13 + b.uid) + 0.3 * math.sin(self.t * 31))
                    e.scale_y = 1 + 0.18 * math.sin(self.t * 9 + b.uid) + 0.08 * math.sin(self.t * 23)
                    e.scale_x = 1 + 0.08 * math.sin(self.t * 7 + 1)
                    e.rotation_y += 40 * utime.dt
                    if R.random() < 0.15 and abs(b.x - p.pos.x) < 40:
                        self.particles.emit(e.world_position + Vec3(0, 0.3, 0), color.rgb(1.0, 0.6, 0.2), 1, 0.4, 2.2, 1.2, 0.02, -0.5)
                else:
                    e.set_shader_input('glow', 0.6 + 3.0 * n)

    # ------------------------------------------------------------------ döngü
    def update(self):
        dt = min(utime.dt, 0.05)
        self.t += dt
        t = self.t
        w, h = window.size
        if w and h:
            camera.set_shader_input('texel', Vec2(1.0 / w, 1.0 / h))
        if self.state == 'menu':
            self.menu_angle += dt * 2.5
            a = math.radians(self.menu_angle)
            cp = Vec3(math.sin(a) * 34 + 4, 0, math.cos(a) * 34 - 8)
            cp.y = self.terrain.ground_y(cp.x, cp.z) + 7.5
            tgt = Vec3(-6, 3, 12)
            d = tgt - cp
            camera.position = cp
            camera.rotation = (math.degrees(math.atan2(-d.y, math.hypot(d.x, d.z))), math.degrees(math.atan2(d.x, d.z)), 0)
            camera.fov = 62
            self.env.time = 0.70 if not AUTOTEST else self.env.time
            self.env.update(0, Vec3(cp.x, 0, cp.z))
            self.env.time = 0.70 if not AUTOTEST else self.env.time
            self.grass.update(cp.x, cp.z, 2)
            self.nature.update(dt)
            self._update_lights()
            return
        if self.state == 'paused':
            return
        p = self.player
        # uyku geçişi
        if self.fade_t > 0:
            self.fade_t -= dt
            k = 1 - abs(self.fade_t - 1.2) / 1.2
            self.fade.color = color.rgba(0, 0, 0, max(0, min(1, k * 1.3)))
            if getattr(self, '_sleep_pending', False) and self.fade_t < 1.2:
                self._sleep_pending = False
                self._finish_sleep()
            if self.fade_t <= 0:
                self.fade.color = color.rgba(0, 0, 0, 0)
        car = p.in_vehicle
        if car is not None:
            p.vm_root.hide()
            car.update(dt, True)
            p.pos = Vec3(car.root.x, car.root.y, car.root.z)
            mdx = mouse.velocity[0] * 140 * abs(p.sens) if mouse.locked else 0
            car.camera_update(dt, mdx, 0)
            self.sounds.set_loop('engine', 0.18 + min(0.25, abs(car.speed) / 60), 0.7 + abs(car.speed) / 22)
        else:
            p.update(dt, t)
        if self.car is not None and self.car is not car:
            self.car.update(dt, False)
        new_day = self.env.update(dt, p.pos)
        if new_day:
            self.hud.message('Gün %d' % self.env.day, 'gold', 3)
        self.nature.update(dt)
        self.nature.tick_regrow(dt)
        self.grass.update(p.pos.x, p.pos.z, 2)
        for plot in self.plots.values():
            plot.update(dt, self.env.rain > 0.5)
        for a in self.animals:
            if abs(a.root.x - p.pos.x) < 90 and abs(a.root.z - p.pos.z) < 90:
                a.update(dt, t, self.animals, self.sounds, (p.pos.x, p.pos.z))
        self.particles.update(dt)
        self.fireflies.recenter(p.pos.x, p.pos.z)
        self.fireflies.update(dt, t, self.env.night, self.env.rain)
        self._update_lights()
        # hedef ve inşaat önizlemesi
        if car is None and not self.ui.any_window():
            if self.builds.preview is not None:
                self.builds.update_preview(p.eye, p.look_dir())
                self.target, self.target_info = None, None
            else:
                self.compute_target()
                if held_keys['e'] and self.target and self.target[0] == 'building' and not self.target[1].complete:
                    self.hold_e += dt
                    if self.hold_e > 0.35:
                        self.hold_e = 0.22
                        self.add_to_building(self.target[1])
                else:
                    self.hold_e = 0
        else:
            self.target, self.target_info = None, None
        self.hud.update(dt)
        # ortam sesleri
        n = self.env.night
        self.sounds.set_loop('rain', self.env.rain * 0.5)
        self.sounds.set_loop('crickets', n * (1 - self.env.rain) * 0.22)
        self.sounds.set_loop('wind', 0.06 + self.env.rain * 0.12)
        if car is None:
            self.sounds.set_loop('engine', 0)
        self.bird_cd -= dt
        if self.bird_cd <= 0:
            self.bird_cd = R.uniform(3, 9)
            if n < 0.3 and self.env.rain < 0.3:
                self.sounds.play('bird%d' % R.integers(1, 4), R.uniform(0.06, 0.16), R.uniform(0.9, 1.2))
        self.autosave_t -= dt
        if self.autosave_t <= 0:
            self.autosave_t = 180
            self.save_game()
        # açlık uyarısı
        if p.hunger < 15 and int(t) % 30 == 0 and int(t - dt) % 30 != 0:
            self.hud.message('Çok açsın! F tuşu ile bir şeyler ye.', 'red', 3)

    def input(self, key):
        if self.state == 'menu':
            if key == 'escape' and self.ui.settings.root.enabled:
                self.ui.settings.hide()
            return
        if key == 'escape':
            if self.ui.settings.root.enabled:
                self.ui.settings.hide()
                return
            if self.ui.any_window():
                self.close_windows()
                return
            if self.builds.preview is not None:
                self.builds.cancel_preview()
                return
            self.set_paused(self.state != 'paused')
            return
        if self.state != 'play' or self.fade_t > 0:
            return
        p = self.player
        if key == 'tab':
            if self.ui.inventory.open:
                self.close_windows()
            else:
                self.open_window(self.ui.inventory)
            return
        if key == 'b' and p.in_vehicle is None:
            if self.ui.book.open:
                self.close_windows()
            else:
                self.builds.cancel_preview()
                self.open_window(self.ui.book)
            return
        if self.ui.any_window():
            return
        if p.in_vehicle is not None:
            if key == 'e':
                self.exit_car()
            elif key == 'v':
                p.in_vehicle.cam_mode = 1 - p.in_vehicle.cam_mode
            return
        if key in '123456' and len(key) == 1:
            self.select_slot(int(key) - 1)
        elif key == 'scroll up' or key == 'scroll down':
            if self.builds.preview is not None:
                self.builds.rotate_preview(15 if key == 'scroll up' else -15)
            else:
                self.select_slot((self.slot + (-1 if key == 'scroll up' else 1)) % len(SLOTS))
        elif key == 'r':
            if self.builds.preview is not None:
                self.builds.rotate_preview(45)
            elif SLOTS[self.slot] == 'tohum':
                self.cycle_seed()
        elif key == 'left mouse down':
            if not mouse.locked:
                self.lock_mouse(True)
                return
            if self.builds.preview is not None:
                self.place_blueprint()
            else:
                self.use_tool()
        elif key == 'right mouse down':
            if self.builds.preview is not None:
                self.builds.cancel_preview()
        elif key == 'e':
            self.hold_e = 0
            self.interact()
        elif key == 'f':
            self.eat()
        elif key == 'x':
            tg = self.target
            if tg and tg[0] == 'building' and not tg[1].complete:
                b = tg[1]
                for k, n in b.refund().items():
                    self.add_item(k, n)
                money = SPECS[b.kind][4]
                self.money += money
                self.builds.remove(b)
                self.hud.message('Plan kaldırıldı, malzemeler iade edildi.', 'info')
        elif key == 'f5':
            self.save_game()


game = Game()
game.current_garage = None


def update():
    game.update()


def input(key):
    game.input(key)


if AUTOTEST:
    import importlib; autotest = importlib.import_module(os.environ.get("VADI_TESTMOD", "autotest"))
    autotest.install(game)

app.run()

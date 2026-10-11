"""Modern arayüz: ana menü, ayarlar, oyun içi HUD, envanter, inşaat defteri, ticaret ve garaj pencereleri."""
import math
import numpy as np
from PIL import Image
from ursina import (Entity, Text, Button, camera, color, window, Vec2, Vec3, Quad, destroy, mouse, Texture, held_keys)

from config import *
import icons
import settings as S
from buildings import SPECS, CATEGORIES, cost_of

BG = color.rgba(0.045, 0.05, 0.06, 0.86)
BG2 = color.rgba(0.11, 0.12, 0.14, 0.92)
BG3 = color.rgba(0.17, 0.18, 0.21, 0.95)
ACC = color.rgb(0.96, 0.70, 0.28)
TXT = color.rgb(0.95, 0.95, 0.93)
MUT = color.rgba(1, 1, 1, 0.55)
GOOD = color.rgb(0.50, 0.88, 0.48)
BAD = color.rgb(0.98, 0.45, 0.38)
BLUE = color.rgb(0.45, 0.72, 1.0)

RES_NAMES = {'kutuk': 'Kütük', 'tas': 'Taş', 'demir': 'Demir', 'bogurtlen': 'Böğürtlen', 'corba': 'Sebze Çorbası'}
ITEM_ICON = {'kutuk': 'log', 'tas': 'stone', 'demir': 'iron', 'bogurtlen': 'berries', 'bugday': 'wheat', 'havuc': 'carrot',
             'domates': 'tomato', 'misir': 'corn', 'balkabagi': 'pumpkin', 'yumurta': 'egg', 'yun': 'wool', 'sut': 'milk',
             'corba': 'soup'}
TOOL_ICON = {'balta': 'axe', 'kazma': 'pickaxe', 'capa': 'hoe', 'kova': 'can', 'tohum': None, 'cekic': 'hammer'}
TOOL_NAME = {'balta': 'Balta', 'kazma': 'Kazma', 'capa': 'Çapa', 'kova': 'Sulama Kabı', 'tohum': 'Tohum', 'cekic': 'Çekiç'}
BUILD_ICON = {'cadir': 'house_icon', 'kulube': 'house_icon', 'kutuk_ev': 'house_icon', 'tas_ev': 'house_icon',
              'ciftlik_evi': 'house_icon', 'yatak': 'house_icon', 'tarla': 'wheat', 'kumes': 'chicken', 'ahir': 'cow',
              'kuyu': 'can', 'cit': 'log', 'pazar': 'coin', 'ates': 'soup', 'fener': 'bolt', 'yol': 'stone', 'garaj': 'car'}
FOOD = {'bogurtlen': RES['bogurtlen']['food'], 'corba': 45}
FOOD.update({k: v['food'] for k, v in CROPS.items() if v['food']})
FOOD.update({k: v['food'] for k, v in PRODUCTS.items() if v['food']})


def item_name(k):
    if k in RES_NAMES:
        return RES_NAMES[k]
    if k in CROPS:
        return CROPS[k]['name']
    if k in PRODUCTS:
        return PRODUCTS[k]['name']
    return k


def item_price(k):
    if k in RES:
        return RES[k]['sell']
    if k in CROPS:
        return CROPS[k]['sell']
    if k in PRODUCTS:
        return PRODUCTS[k]['sell']
    if k == 'corba':
        return 60
    return 0


def ico(name, *args):
    return icons.get(name, *args)


def gradient_tex():
    w = 512
    a = np.linspace(0.92, 0.0, w) ** 1.4
    arr = np.zeros((4, w, 4), dtype=np.uint8)
    arr[..., 3] = (a * 255).astype(np.uint8)
    arr[..., 0] = 5; arr[..., 1] = 8; arr[..., 2] = 10
    return Texture(Image.fromarray(arr, 'RGBA'))


def panel(parent, w, h, pos=(0, 0), col=BG, radius=0.04, z=0):
    return Entity(parent=parent, model=Quad(radius=radius / min(w, h) if min(w, h) > 0 else 0.1, aspect=w / h),
                  color=col, scale=(w, h), position=(pos[0], pos[1], z))


def label(parent, text, pos, scale=1.0, col=TXT, origin=(-0.5, 0), z=-0.01):
    return Text(text, parent=parent, position=(pos[0], pos[1], z), scale=scale, color=col, origin=origin)


def icon_e(parent, tex, pos, s=0.05, z=-0.01, col=color.white):
    return Entity(parent=parent, model='quad', texture=tex, scale=s, position=(pos[0], pos[1], z), color=col)


class FlatButton(Button):
    def __init__(self, text='', parent=camera.ui, w=0.3, h=0.06, pos=(0, 0), on_click=None, col=None, hover=None,
                 text_col=TXT, size=0.9, align='center', enabled_=True, z=-0.02, radius=0.25):
        c = col if col is not None else BG3
        super().__init__(parent=parent, text=text, model=Quad(radius=radius, aspect=w / h), scale=(w, h),
                         position=(pos[0], pos[1], z), color=c, highlight_color=hover or c.tint(0.12),
                         pressed_color=c.tint(-0.08), text_size=size, text_color=text_col if enabled_ else MUT)
        if align == 'left':
            self.text_entity.origin = (-0.5, 0)
            self.text_entity.x = -0.45
        self.enabled_ = enabled_
        if on_click is not None:
            self.on_click = on_click if enabled_ else (lambda: None)


class MenuItem(Button):
    """Ana menüdeki sade, sola yaslı metin düğmesi (üzerine gelince vurgu çubuğu)."""

    def __init__(self, parent, text, pos, on_click, size=1.6):
        super().__init__(parent=parent, text=text, model='quad', scale=(0.46, 0.065), position=(pos[0] + 0.23, pos[1], -0.02),
                         color=color.rgba(1, 1, 1, 0.0), highlight_color=color.rgba(1, 1, 1, 0.06),
                         pressed_color=color.rgba(1, 1, 1, 0.1), text_size=size, text_color=TXT)
        self.text_entity.origin = (-0.5, 0)
        self.text_entity.x = -0.44
        self.bar = Entity(parent=self, model='quad', color=ACC, scale=(0.008, 0.7), x=-0.5, z=-0.01, enabled=False)
        self.on_click = on_click

    def on_mouse_enter(self):
        super().on_mouse_enter()
        self.bar.enabled = True
        self.text_entity.color = ACC

    def on_mouse_exit(self):
        super().on_mouse_exit()
        self.bar.enabled = False
        self.text_entity.color = TXT


class SliderRow(Entity):
    def __init__(self, parent, y, title, lo, hi, value, on_change, fmt='{:.0f}', step=None):
        super().__init__(parent=parent, y=y)
        self.lo, self.hi, self.on_change, self.fmt, self.step = lo, hi, on_change, fmt, step
        label(self, title, (-0.48, 0), 1.0)
        self.track = Button(parent=self, model=Quad(radius=0.5, aspect=0.32 / 0.012), scale=(0.32, 0.012), x=0.18,
                            color=color.rgba(1, 1, 1, 0.15), highlight_color=color.rgba(1, 1, 1, 0.22), z=-0.01)
        self.track.on_click = self._begin
        self.fill = Entity(parent=self, model=Quad(radius=0.5, aspect=1), color=ACC, scale=(0.0, 0.012), x=0.02, z=-0.02,
                           origin=(-0.5, 0))
        self.knob = Entity(parent=self, model=Quad(radius=0.5), color=TXT, scale=0.022, x=0.18, z=-0.03)
        self.val_t = label(self, '', (0.42, 0), 0.95, MUT, origin=(0.5, 0))
        self.dragging = False
        self.set(value, notify=False)

    def _begin(self):
        self.dragging = True
        self._from_mouse()

    def _from_mouse(self):
        rel = (mouse.x - (self.world_x + 0.02)) / 0.32
        v = self.lo + max(0, min(1, rel)) * (self.hi - self.lo)
        if self.step:
            v = round(v / self.step) * self.step
        self.set(v)

    def set(self, v, notify=True):
        self.value = max(self.lo, min(self.hi, v))
        k = (self.value - self.lo) / (self.hi - self.lo)
        self.fill.scale_x = max(0.0001, 0.32 * k)
        self.knob.x = 0.02 + 0.32 * k
        self.val_t.text = self.fmt.format(self.value)
        if notify:
            self.on_change(self.value)

    def update(self):
        if self.dragging:
            if mouse.left:
                self._from_mouse()
            else:
                self.dragging = False


class OptionRow(Entity):
    def __init__(self, parent, y, title, options, index, on_change):
        super().__init__(parent=parent, y=y)
        self.options, self.index, self.on_change = options, index, on_change
        label(self, title, (-0.48, 0), 1.0)
        FlatButton('‹', self, 0.04, 0.042, (0.06, 0), self.prev, size=1.3)
        FlatButton('›', self, 0.04, 0.042, (0.42, 0), self.next, size=1.3)
        self.val = label(self, '', (0.24, 0), 1.0, ACC, origin=(0, 0))
        self.dots = []
        for i in range(len(options)):
            d = Entity(parent=self, model=Quad(radius=0.5), scale=(0.022, 0.005), x=0.24 + (i - (len(options) - 1) / 2) * 0.03,
                       y=-0.022, color=MUT, z=-0.01)
            self.dots.append(d)
        self.refresh()

    def refresh(self):
        self.val.text = self.options[self.index]
        for i, d in enumerate(self.dots):
            d.color = ACC if i == self.index else color.rgba(1, 1, 1, 0.2)

    def prev(self):
        self.index = (self.index - 1) % len(self.options)
        self.refresh()
        self.on_change(self.index)

    def next(self):
        self.index = (self.index + 1) % len(self.options)
        self.refresh()
        self.on_change(self.index)


# =========================================================================== ana menü
class MainMenu:
    def __init__(self, game):
        self.game = game
        self.root = Entity(parent=camera.ui, z=-0.5)
        self.grad = Entity(parent=self.root, model='quad', texture=gradient_tex(), scale=(window.aspect_ratio * 0.75, 1),
                           x=window.left.x + window.aspect_ratio * 0.375, z=0.01)
        L = window.left.x + 0.1
        label(self.root, 'YEŞİL VADİ', (L, 0.27), 5.2, TXT)
        Entity(parent=self.root, model='quad', color=ACC, scale=(0.09, 0.006), position=(L + 0.045, 0.205, -0.01))
        label(self.root, 'hayatta kal  •  inşa et  •  çiftçilik yap', (L, 0.17), 1.15, MUT)
        self.items = []
        self.info = label(self.root, '', (L, -0.44), 0.85, MUT)
        self.build()

    def build(self):
        for it in self.items:
            destroy(it)
        self.items = []
        L = window.left.x + 0.08
        g = self.game
        entries = []
        if g.has_save():
            entries.append(('Devam Et', g.continue_game))
        entries += [('Yeni Oyun', g.new_game), ('Ayarlar', lambda: g.ui.settings.show(self.root)), ('Çıkış', g.quit_game)]
        for i, (t, cb) in enumerate(entries):
            self.items.append(MenuItem(self.root, t, (L, 0.04 - i * 0.08), cb))
        self.info.text = 'Tüm grafikler, modeller ve sesler kodla üretilmiştir.  •  Sürüm 2.0'

    def show(self, v=True):
        self.root.enabled = v
        if v:
            self.build()


# =========================================================================== ayarlar
class SettingsMenu:
    def __init__(self, game):
        self.game = game
        self.root = Entity(parent=camera.ui, enabled=False, z=-0.7)
        self.back_to = None
        self.tab = 'Grafik'
        self.content = None

    def show(self, back_to=None):
        self.back_to = back_to
        if back_to is not None:
            back_to.enabled = False
        self.root.enabled = True
        self.build()

    def hide(self):
        self.root.enabled = False
        S.save(self.game.settings)
        if self.back_to is not None:
            self.back_to.enabled = True

    def build(self):
        for c in list(self.root.children):
            destroy(c)
        Entity(parent=self.root, model='quad', color=color.rgba(0, 0, 0, 0.55), scale=(4, 2), z=0.02)
        W, H = 1.12, 0.82
        panel(self.root, W, H, (0, 0), BG, 0.03)
        label(self.root, 'Ayarlar', (-W / 2 + 0.05, H / 2 - 0.06), 2.0)
        FlatButton('Kapat', self.root, 0.13, 0.05, (W / 2 - 0.1, H / 2 - 0.06), self.hide)
        tabs = ['Grafik', 'Kontroller', 'Ses']
        for i, t in enumerate(tabs):
            active = t == self.tab
            b = FlatButton(t, self.root, 0.17, 0.05, (-W / 2 + 0.135 + i * 0.185, H / 2 - 0.14),
                           (lambda t=t: self._tab(t)), col=ACC if active else BG3,
                           text_col=color.rgb(0.1, 0.1, 0.1) if active else TXT)
        self.content = Entity(parent=self.root, y=H / 2 - 0.25)
        st = self.game.settings
        y = 0
        rows = []
        if self.tab == 'Grafik':
            rows = [('opt', 'Grafik Kalitesi', 'quality'), ('opt', 'Gölgeler', 'shadows'), ('opt', 'Çim Yoğunluğu', 'grass'),
                    ('opt', 'Görüş Mesafesi', 'view'), ('opt', 'Ortam Gölgelemesi (AO)', 'ao'), ('opt', 'Parlama (Bloom)', 'bloom'),
                    ('opt', 'Kenar Yumuşatma (FXAA)', 'fxaa'), ('opt', 'Tam Ekran', 'fullscreen'), ('opt', 'FPS Göster', 'show_fps')]
        elif self.tab == 'Kontroller':
            rows = [('sld', 'Fare Hassasiyeti', 'sens', 0.2, 3.0, '{:.2f}', 0.05), ('opt', 'Fareyi Ters Çevir', 'invert'),
                    ('sld', 'Görüş Alanı (FOV)', 'fov', 60, 100, '{:.0f}°', 1), ('opt', 'Kafa Sallanması', 'head_bob')]
        else:
            rows = [('sld', 'Ana Ses', 'volume', 0, 100, '{:.0f}%', 1)]
        for r in rows:
            if r[0] == 'opt':
                key = r[2]
                OptionRow(self.content, y, r[1], S.OPTIONS[key], int(st[key]), (lambda v, key=key: self._set(key, v)))
            else:
                key = r[2]
                SliderRow(self.content, y, r[1], r[3], r[4], st[key], (lambda v, key=key: self._set(key, v)), r[5], r[6])
            y -= 0.062
        if self.tab == 'Kontroller':
            keys = ('W A S D: Yürü    Shift: Koş    Boşluk: Zıpla    Fare: Bak\n'
                    'Sol Tık: Aleti kullan    E: Etkileşim / Malzeme ekle    1-6: Alet seç\n'
                    'B: İnşaat defteri    R: Döndür / Tohum değiştir    Tab: Envanter    F: Ye\n'
                    'X: Planı kaldır    V: Araba kamerası    Esc: Menü')
            label(self.content, keys, (-0.48, y - 0.04), 0.85, MUT)
        label(self.root, 'Bazı grafik ayarları (çim, gölge) anında uygulanır; tam ekran ve VSync yeniden başlatmayla tamamen etkinleşir.',
              (-W / 2 + 0.05, -H / 2 + 0.04), 0.72, MUT)

    def _tab(self, t):
        self.tab = t
        self.build()

    def _set(self, key, v):
        st = self.game.settings
        st[key] = v
        if key == 'quality':
            st.update(S.PRESETS[int(v)])
            self.game.apply_settings()
            self.build()
            return
        self.game.apply_settings()


# =========================================================================== HUD
class HUD:
    def __init__(self, game):
        self.game = game
        self.root = Entity(parent=camera.ui)
        r = self.root
        # nişangâh
        self.cross = Entity(parent=r, model=Quad(radius=0.5), color=color.rgba(1, 1, 1, 0.85), scale=0.006)
        self.ring = Entity(parent=r, model=Quad(radius=0.5, mode='line', thickness=2), color=ACC, scale=0.03, enabled=False)
        self.target_t = label(r, '', (0, -0.055), 1.05, TXT, origin=(0, 0))
        self.prompt_t = label(r, '', (0, -0.09), 0.95, ACC, origin=(0, 0))
        self.hpbar_bg = Entity(parent=r, model=Quad(radius=0.5, aspect=20), color=color.rgba(0, 0, 0, 0.5), scale=(0.12, 0.006), y=-0.072)
        self.hpbar = Entity(parent=r, model=Quad(radius=0.5, aspect=20), color=ACC, scale=(0.12, 0.006), y=-0.072, x=-0.06,
                            origin=(-0.5, 0), z=-0.01)
        # üst sol: zaman / para
        tl = window.top_left
        self.info_bg = panel(r, 0.36, 0.09, (tl.x + 0.2, tl.y - 0.065), BG, 0.02)
        self.w_icon = icon_e(r, ico('sun_icon'), (tl.x + 0.055, tl.y - 0.065), 0.05)
        self.time_t = label(r, '', (tl.x + 0.09, tl.y - 0.048), 1.05)
        self.day_t = label(r, '', (tl.x + 0.09, tl.y - 0.083), 0.8, MUT)
        icon_e(r, ico('coin'), (tl.x + 0.29, tl.y - 0.048), 0.03)
        self.money_t = label(r, '', (tl.x + 0.31, tl.y - 0.048), 0.95, ACC)
        # üst sağ: kaynaklar
        tr = window.top_right
        self.res_bg = panel(r, 0.33, 0.06, (tr.x - 0.185, tr.y - 0.05), BG, 0.02)
        self.res_t = {}
        for i, k in enumerate(('kutuk', 'tas', 'demir')):
            x = tr.x - 0.33 + i * 0.105
            icon_e(r, ico(ITEM_ICON[k]), (x, tr.y - 0.05), 0.04)
            self.res_t[k] = label(r, '0', (x + 0.025, tr.y - 0.05), 0.95)
        self.fps_t = label(r, '', (tr.x - 0.02, tr.y - 0.1), 0.8, MUT, origin=(0.5, 0))
        # alt sol: açlık / enerji
        bl = window.bottom_left
        self.stat_bg = panel(r, 0.32, 0.1, (bl.x + 0.18, bl.y + 0.08), BG, 0.02)
        self.bars = {}
        for i, (k, ic, c) in enumerate((('hunger', 'food_icon', color.rgb(0.95, 0.55, 0.3)), ('energy', 'bolt', color.rgb(0.98, 0.85, 0.3)))):
            y = bl.y + 0.1 - i * 0.04
            icon_e(r, ico(ic), (bl.x + 0.045, y), 0.03)
            Entity(parent=r, model=Quad(radius=0.5, aspect=30), color=color.rgba(1, 1, 1, 0.12), scale=(0.24, 0.012),
                   position=(bl.x + 0.19, y))
            b = Entity(parent=r, model=Quad(radius=0.5, aspect=30), color=c, scale=(0.24, 0.012), origin=(-0.5, 0),
                       position=(bl.x + 0.07, y, -0.01))
            self.bars[k] = b
        # alt orta: alet çubuğu
        self.slots = []
        self.slot_keys = ['balta', 'kazma', 'capa', 'kova', 'tohum', 'cekic']
        n = len(self.slot_keys)
        gap = 0.085
        x0 = -gap * (n - 1) / 2
        y = window.bottom.y + 0.065
        panel(r, gap * n + 0.02, 0.1, (0, y), BG, 0.02)
        self.sel = Entity(parent=r, model=Quad(radius=0.18, mode='line', thickness=2), color=ACC, scale=0.078, y=y, z=-0.03)
        for i, k in enumerate(self.slot_keys):
            x = x0 + i * gap
            b = FlatButton('', r, 0.075, 0.075, (x, y), (lambda i=i: game.select_slot(i)), col=BG3, radius=0.18)
            tex = ico(TOOL_ICON[k]) if TOOL_ICON[k] else ico('seed_bag', CROPS['bugday']['color'])
            ic = icon_e(r, tex, (x, y + 0.003), 0.06, -0.03)
            kt = label(r, str(i + 1), (x - 0.03, y + 0.028, -0.04), 0.65, MUT)
            ct = label(r, '', (x + 0.033, y - 0.026), 0.75, TXT, origin=(0.5, 0), z=-0.04)
            self.slots.append((b, ic, kt, ct))
        self.water_bar = Entity(parent=r, model='quad', color=BLUE, scale=(0.06, 0.005), origin=(-0.5, 0), z=-0.04)
        self.tool_t = label(r, '', (0, y + 0.075), 0.95, TXT, origin=(0, 0))
        self.tool_t_time = 0
        # mesajlar
        self.msgs = []
        # mod çubuğu (inşaat / araba)
        self.mode_t = label(r, '', (0, window.bottom.y + 0.15), 0.9, TXT, origin=(0, 0))
        self.mode_bg = panel(r, 0.7, 0.04, (0, window.bottom.y + 0.15), BG, 0.02)
        self.speed_t = label(r, '', (window.bottom_right.x - 0.06, window.bottom.y + 0.08), 2.4, TXT, origin=(0.5, 0))
        self.speed_u = label(r, 'km/s', (window.bottom_right.x - 0.06, window.bottom.y + 0.04), 0.9, MUT, origin=(0.5, 0))
        self.floaters = []

    def message(self, text, kind='info', dur=3.0):
        col = {'info': TXT, 'good': GOOD, 'red': BAD, 'gold': ACC, 'blue': BLUE}.get(kind, TXT)
        bg = panel(self.root, max(0.2, len(text) * 0.0105 + 0.05), 0.045, (0, 0.33), BG, 0.02)
        t = label(self.root, text, (0, 0.33), 1.0, col, origin=(0, 0), z=-0.02)
        self.msgs.insert(0, [bg, t, dur])
        while len(self.msgs) > 4:
            b, tt, _ = self.msgs.pop()
            destroy(b); destroy(tt)

    def pickup(self, text):
        self.message(text, 'good', 2.0)

    def set_tool(self, i, name):
        b = self.slots[i][0]
        self.sel.x = b.x
        self.tool_t.text = name
        self.tool_t_time = 1.6

    def update(self, dt):
        g = self.game
        p = g.player
        env = g.env
        self.time_t.text = env.clock
        w = 'Yağmurlu' if env.raining else ('Gece' if env.is_night() else 'Açık')
        self.day_t.text = 'Gün %d  •  %s' % (env.day, w)
        tex = ico('rain_icon') if env.raining else (ico('moon_icon') if env.is_night() else ico('sun_icon'))
        if self.w_icon.texture != tex:
            self.w_icon.texture = tex
        self.money_t.text = str(g.money)
        for k, t in self.res_t.items():
            t.text = str(g.inventory.get(k, 0))
        self.bars['hunger'].scale_x = 0.24 * max(0.001, p.hunger / 100)
        self.bars['energy'].scale_x = 0.24 * max(0.001, p.energy / 100)
        self.bars['hunger'].color = BAD if p.hunger < 20 and (int(g.t * 3) % 2) else color.rgb(0.95, 0.55, 0.3)
        # alet çubuğu
        for i, k in enumerate(self.slot_keys):
            b, ic, kt, ct = self.slots[i]
            if k == 'tohum':
                sk = g.seed_kind
                n = g.seeds.get(sk, 0)
                ct.text = str(n)
                ct.color = TXT if n else BAD
                ic.texture = ico('seed_bag', CROPS[sk]['color'])
            elif k == 'kova':
                self.water_bar.position = (b.x - 0.03, b.y - 0.032, -0.04)
                self.water_bar.scale_x = 0.06 * max(0.001, g.water / 12)
            else:
                ct.text = ''
        if self.tool_t_time > 0:
            self.tool_t_time -= dt
            self.tool_t.alpha = min(1, self.tool_t_time * 2)
        # hedef
        tgt = g.target_info
        if tgt:
            self.target_t.text = tgt.get('name', '')
            self.prompt_t.text = tgt.get('prompt', '')
            hp = tgt.get('hp')
            self.hpbar.enabled = self.hpbar_bg.enabled = hp is not None
            if hp is not None:
                self.hpbar.scale_x = 0.12 * max(0.001, hp)
                self.hpbar.color = tgt.get('bar_col', ACC)
            self.ring.enabled = True
            self.ring.scale = 0.03 + math.sin(g.t * 6) * 0.002
        else:
            self.target_t.text = ''
            self.prompt_t.text = ''
            self.hpbar.enabled = self.hpbar_bg.enabled = False
            self.ring.enabled = False
        # mod çubuğu
        mt = g.mode_text()
        self.mode_t.text = mt
        self.mode_bg.enabled = bool(mt)
        if mt:
            self.mode_bg.scale_x = len(mt) * 0.0098 + 0.05
            self.mode_bg.model = Quad(radius=0.3, aspect=self.mode_bg.scale_x / 0.04)
        car = g.player.in_vehicle
        self.speed_t.enabled = self.speed_u.enabled = car is not None
        if car is not None:
            self.speed_t.text = '%d' % int(abs(car.speed) * 3.6)
        # mesajlar
        y = 0.33
        keep = []
        for m in self.msgs:
            bg, t, life = m
            life -= dt
            m[2] = life
            if life <= 0:
                destroy(bg); destroy(t)
                continue
            a = min(1, life * 2)
            bg.y = t.y = y
            bg.alpha = 0.86 * a
            t.alpha = a
            y -= 0.052
            keep.append(m)
        self.msgs = keep
        # uçan yazılar
        keep = []
        for f in self.floaters:
            t, wp, life = f
            life -= dt
            f[2] = life
            if life <= 0:
                destroy(t)
                continue
            wp.y += dt * 0.8
            sp = world_to_screen(wp)
            if sp is None:
                t.enabled = False
            else:
                t.enabled = True
                t.position = (sp[0], sp[1])
                t.alpha = min(1, life * 1.5)
            keep.append(f)
        self.floaters = keep
        if g.settings.get('show_fps'):
            self.fps_t.text = '%d FPS' % int(1 / max(dt, 1e-4))
        else:
            self.fps_t.text = ''

    def float_text(self, text, wp, col=ACC):
        t = label(self.root, text, (0, 0), 1.1, col, origin=(0, 0))
        self.floaters.append([t, Vec3(*wp), 1.3])


def world_to_screen(wp):
    from panda3d.core import Point3
    from ursina import scene
    p = camera.getRelativePoint(scene, Point3(*wp))
    if p.z <= 0.1:
        return None
    proj = camera.lens.getProjectionMat().xform((p.x, p.y, p.z, 1))
    if proj[3] <= 0:
        return None
    return (proj[0] / proj[3] * camera.aspect_ratio / 2, proj[1] / proj[3] / 2)


# =========================================================================== pencereler
class Window:
    def __init__(self, game, title, w=1.2, h=0.78):
        self.game = game
        self.title = title
        self.w, self.h = w, h
        self.root = Entity(parent=camera.ui, enabled=False, z=-0.3)
        self.items = []

    @property
    def open(self):
        return self.root.enabled

    def show(self):
        self.root.enabled = True
        self.build()

    def hide(self):
        self.root.enabled = False

    def clear(self):
        for c in list(self.root.children):
            destroy(c)

    def frame(self, subtitle=''):
        self.clear()
        Entity(parent=self.root, model='quad', color=color.rgba(0, 0, 0, 0.45), scale=(4, 2), z=0.02)
        panel(self.root, self.w, self.h, (0, 0), BG, 0.03)
        label(self.root, self.title, (-self.w / 2 + 0.04, self.h / 2 - 0.055), 1.8)
        if subtitle:
            label(self.root, subtitle, (-self.w / 2 + 0.04, self.h / 2 - 0.1), 0.85, MUT)
        FlatButton('Kapat  [Esc]', self.root, 0.17, 0.045, (self.w / 2 - 0.115, self.h / 2 - 0.055), self.game.close_windows)
        icon_e(self.root, ico('coin'), (self.w / 2 - 0.36, self.h / 2 - 0.055), 0.032)
        label(self.root, str(self.game.money), (self.w / 2 - 0.34, self.h / 2 - 0.055), 1.05, ACC)

    def build(self):
        self.frame()


class Inventory(Window):
    def __init__(self, game):
        super().__init__(game, 'Envanter', 1.0, 0.7)

    def build(self):
        g = self.game
        self.frame('Yiyeceklere tıklayarak yiyebilirsin.  Açlık: %d / 100' % int(g.player.hunger))
        items = [(k, v) for k, v in g.inventory.items() if v > 0]
        cols = 5
        cw, ch = 0.17, 0.14
        x0 = -self.w / 2 + 0.04 + cw / 2
        y0 = self.h / 2 - 0.2
        for i, (k, v) in enumerate(sorted(items, key=lambda kv: list(ITEM_ICON).index(kv[0]) if kv[0] in ITEM_ICON else 99)):
            x = x0 + (i % cols) * (cw + 0.015)
            y = y0 - (i // cols) * (ch + 0.015)
            food = FOOD.get(k, 0)
            b = FlatButton('', self.root, cw, ch, (x, y), (lambda k=k: (g.eat(k), self.build())) if food else None,
                           col=BG3 if not food else color.rgba(0.2, 0.22, 0.18, 0.95), radius=0.1)
            icon_e(self.root, ico(ITEM_ICON.get(k, 'coin')), (x, y + 0.02), 0.065, -0.03)
            label(self.root, item_name(k), (x, y - 0.042), 0.78, TXT, origin=(0, 0), z=-0.03)
            label(self.root, 'x%d' % v, (x + cw / 2 - 0.012, y + ch / 2 - 0.02), 0.8, ACC, origin=(0.5, 0), z=-0.03)
            if food:
                label(self.root, '+%d' % food, (x - cw / 2 + 0.012, y + ch / 2 - 0.02), 0.72, GOOD, origin=(-0.5, 0), z=-0.03)
        if not items:
            label(self.root, 'Envanterin boş. Ağaç kes, taş topla, hasat yap!', (0, 0), 1.0, MUT, origin=(0, 0))
        y = -self.h / 2 + 0.06
        label(self.root, 'Tohumlar: ' + '   '.join('%s %d' % (CROPS[k]['name'], g.seeds.get(k, 0)) for k in CROP_ORDER),
              (-self.w / 2 + 0.04, y), 0.85, MUT)


class BuildBook(Window):
    def __init__(self, game):
        super().__init__(game, 'İnşaat Defteri', 1.3, 0.8)
        self.cat = CATEGORIES[0]

    def build(self):
        g = self.game
        self.frame('Bir yapı seç, yere yerleştir, sonra malzemeleri ekleyerek inşa et.')
        for i, c in enumerate(CATEGORIES):
            active = c == self.cat
            FlatButton(c, self.root, 0.17, 0.045, (-self.w / 2 + 0.125 + i * 0.185, self.h / 2 - 0.16),
                       (lambda c=c: self._cat(c)), col=ACC if active else BG3,
                       text_col=color.rgb(0.1, 0.1, 0.1) if active else TXT)
        kinds = [k for k, v in SPECS.items() if v[1] == self.cat]
        cols = 3
        cw, ch = 0.39, 0.16
        x0 = -self.w / 2 + 0.04 + cw / 2
        y0 = self.h / 2 - 0.29
        for i, k in enumerate(kinds):
            name, cat, fn, desc, money = SPECS[k]
            x = x0 + (i % cols) * (cw + 0.02)
            y = y0 - (i // cols) * (ch + 0.02)
            FlatButton('', self.root, cw, ch, (x, y), (lambda k=k: g.choose_blueprint(k)), col=BG3, radius=0.08)
            icon_e(self.root, ico(BUILD_ICON.get(k, 'house_icon')), (x - cw / 2 + 0.05, y + 0.02), 0.07, -0.03)
            label(self.root, name, (x - cw / 2 + 0.1, y + 0.045), 1.0, TXT, z=-0.03)
            label(self.root, desc, (x - cw / 2 + 0.1, y + 0.012), 0.68, MUT, z=-0.03)
            cost = cost_of(k)
            parts = []
            for rk in ('kutuk', 'tas', 'demir'):
                if cost.get(rk):
                    have = g.inventory.get(rk, 0)
                    parts.append((rk, cost[rk], have))
            cx = x - cw / 2 + 0.1
            for rk, n, have in parts:
                icon_e(self.root, ico(ITEM_ICON[rk]), (cx + 0.01, y - 0.04), 0.025, -0.03)
                label(self.root, str(n), (cx + 0.027, y - 0.04), 0.8, GOOD if have >= n else TXT, z=-0.03)
                cx += 0.07
            if money:
                icon_e(self.root, ico('coin'), (cx + 0.01, y - 0.04), 0.022, -0.03)
                label(self.root, str(money), (cx + 0.027, y - 0.04), 0.8, GOOD if g.money >= money else BAD, z=-0.03)
        label(self.root, 'Yerleştirirken:  Sol Tık = yerleştir   R / Fare tekerleği = döndür   Sağ Tık / Esc = vazgeç',
              (-self.w / 2 + 0.04, -self.h / 2 + 0.04), 0.8, MUT)

    def _cat(self, c):
        self.cat = c
        self.build()


class Trade(Window):
    def __init__(self, game):
        super().__init__(game, 'Tüccar', 1.3, 0.82)
        self.tab = 'Al'

    def build(self):
        g = self.game
        self.frame('Gezgin tüccar ile alım-satım yap.' if not g.trade_is_stall else 'Kendi pazar tezgâhın.')
        for i, t in enumerate(('Al', 'Sat')):
            active = t == self.tab
            FlatButton(t, self.root, 0.15, 0.045, (-self.w / 2 + 0.115 + i * 0.165, self.h / 2 - 0.16),
                       (lambda t=t: self._tab(t)), col=ACC if active else BG3,
                       text_col=color.rgb(0.1, 0.1, 0.1) if active else TXT)
        top = self.h / 2 - 0.26
        L = -self.w / 2 + 0.05
        if self.tab == 'Al':
            label(self.root, 'Tohumlar', (L, top), 1.05, ACC)
            for i, k in enumerate(CROP_ORDER):
                c = CROPS[k]
                y = top - 0.06 - i * 0.068
                icon_e(self.root, ico('seed_bag', c['color']), (L + 0.025, y), 0.045)
                label(self.root, c['name'], (L + 0.06, y + 0.012), 0.95)
                label(self.root, '%d altın  •  %d dk' % (c['seed'], int(c['time'] / 60)), (L + 0.06, y - 0.016), 0.72, MUT)
                FlatButton('+1', self.root, 0.06, 0.04, (L + 0.33, y), (lambda k=k: (g.buy_seed(k, 1), self.build())),
                           enabled_=g.money >= c['seed'])
                FlatButton('+10', self.root, 0.07, 0.04, (L + 0.405, y), (lambda k=k: (g.buy_seed(k, 10), self.build())),
                           enabled_=g.money >= c['seed'] * 10)
            M = -0.02
            label(self.root, 'Hayvanlar', (M, top), 1.05, ACC)
            for i, (k, a) in enumerate(ANIMALS.items()):
                y = top - 0.06 - i * 0.075
                icon_e(self.root, ico({'tavuk': 'chicken', 'koyun': 'sheep', 'inek': 'cow'}[k]), (M + 0.025, y), 0.05)
                cap, have = g.animal_capacity(k)
                label(self.root, '%s  (%d/%d yer)' % (a['name'], have, cap), (M + 0.065, y + 0.013), 0.95)
                need = 'Kümes' if a['home'] == 'kumes' else 'Ahır'
                label(self.root, '%d altın  •  %s gerekir' % (a['price'], need), (M + 0.065, y - 0.016), 0.72, MUT)
                FlatButton('Al', self.root, 0.07, 0.04, (M + 0.39, y), (lambda k=k: (g.buy_animal(k), self.build())),
                           enabled_=g.money >= a['price'] and have < cap)
            y = top - 0.06 - 3 * 0.075 - 0.04
            label(self.root, 'Araç', (M, y + 0.03), 1.05, ACC)
            y -= 0.03
            icon_e(self.root, ico('car'), (M + 0.025, y), 0.05)
            label(self.root, 'Kamyonet', (M + 0.065, y + 0.013), 0.95)
            label(self.root, '%d altın  •  ya da garajda üret' % CAR_PRICE, (M + 0.065, y - 0.016), 0.72, MUT)
            FlatButton('Al', self.root, 0.07, 0.04, (M + 0.39, y), (lambda: (g.buy_car(), self.build())),
                       enabled_=g.money >= CAR_PRICE and g.car is None)
        else:
            items = [(k, v) for k, v in g.inventory.items() if v > 0 and item_price(k) > 0]
            total = sum(v * item_price(k) for k, v in items)
            for i, (k, v) in enumerate(items):
                col_i = i % 2
                row = i // 2
                x = L + col_i * 0.62
                y = top - row * 0.07
                icon_e(self.root, ico(ITEM_ICON.get(k, 'coin')), (x + 0.025, y), 0.045)
                label(self.root, '%s  x%d' % (item_name(k), v), (x + 0.06, y + 0.012), 0.95)
                label(self.root, 'tanesi %d altın' % item_price(k), (x + 0.06, y - 0.016), 0.72, MUT)
                FlatButton('1 Sat', self.root, 0.09, 0.04, (x + 0.42, y), (lambda k=k: (g.sell(k, 1), self.build())))
                FlatButton('Tümü', self.root, 0.09, 0.04, (x + 0.52, y), (lambda k=k: (g.sell(k, None), self.build())),
                           col=color.rgb(0.55, 0.40, 0.14))
            if not items:
                label(self.root, 'Satacak bir şeyin yok.', (0, 0), 1.0, MUT, origin=(0, 0))
            else:
                FlatButton('Her Şeyi Sat  (+%d altın)' % total, self.root, 0.4, 0.055, (0, -self.h / 2 + 0.06),
                           (lambda: (g.sell_all(), self.build())), col=color.rgb(0.62, 0.45, 0.14))

    def _tab(self, t):
        self.tab = t
        self.build()


class Garage(Window):
    def __init__(self, game):
        super().__init__(game, 'Garaj Tezgâhı', 0.95, 0.66)

    def build(self):
        g = self.game
        self.frame('Parçaları üret ve kamyonetini birleştir.')
        L = -self.w / 2 + 0.05
        top = self.h / 2 - 0.2
        icons_ = {'sasi': 'chassis', 'motor': 'gear', 'tekerlek': 'wheel_icon'}
        for i, (k, p) in enumerate(CAR_PARTS.items()):
            y = top - i * 0.11
            done = k in g.car_parts
            panel(self.root, self.w - 0.08, 0.095, (0, y), BG2, 0.02)
            icon_e(self.root, ico(icons_[k]), (L + 0.03, y), 0.06)
            label(self.root, p['name'], (L + 0.08, y + 0.018), 1.05)
            cx = L + 0.08
            for rk, n in p['cost'].items():
                icon_e(self.root, ico(ITEM_ICON[rk]), (cx + 0.01, y - 0.02), 0.024)
                label(self.root, '%d/%d' % (min(g.inventory.get(rk, 0), n), n), (cx + 0.027, y - 0.02), 0.78,
                      GOOD if g.inventory.get(rk, 0) >= n else BAD)
                cx += 0.09
            if p['money']:
                icon_e(self.root, ico('coin'), (cx + 0.01, y - 0.02), 0.022)
                label(self.root, str(p['money']), (cx + 0.027, y - 0.02), 0.78, GOOD if g.money >= p['money'] else BAD)
            if done:
                label(self.root, 'Hazır', (self.w / 2 - 0.1, y), 1.0, GOOD, origin=(0, 0))
            else:
                ok = all(g.inventory.get(rk, 0) >= n for rk, n in p['cost'].items()) and g.money >= p['money']
                FlatButton('Üret', self.root, 0.11, 0.045, (self.w / 2 - 0.1, y), (lambda k=k: (g.craft_part(k), self.build())),
                           enabled_=ok)
        all_done = len(g.car_parts) == len(CAR_PARTS)
        FlatButton('Kamyoneti Birleştir' if g.car is None else 'Kamyonetin zaten var', self.root, 0.42, 0.06,
                   (0, -self.h / 2 + 0.07), (lambda: (g.assemble_car(), self.build())),
                   col=ACC if all_done and g.car is None else BG3, text_col=color.rgb(0.1, 0.1, 0.1) if all_done else MUT,
                   enabled_=all_done and g.car is None)


class PauseMenu:
    def __init__(self, game):
        self.game = game
        self.root = Entity(parent=camera.ui, enabled=False, z=-0.6)

    def show(self, v=True):
        self.root.enabled = v
        for c in list(self.root.children):
            destroy(c)
        if not v:
            return
        g = self.game
        Entity(parent=self.root, model='quad', texture=gradient_tex(), scale=(window.aspect_ratio * 0.75, 1),
               x=window.left.x + window.aspect_ratio * 0.375, z=0.01)
        Entity(parent=self.root, model='quad', color=color.rgba(0, 0, 0, 0.25), scale=(4, 2), z=0.02)
        L = window.left.x + 0.1
        label(self.root, 'Duraklatıldı', (L, 0.22), 3.2)
        Entity(parent=self.root, model='quad', color=ACC, scale=(0.09, 0.006), position=(L + 0.045, 0.17, -0.01))
        for i, (t, cb) in enumerate([('Devam Et', lambda: g.set_paused(False)), ('Ayarlar', lambda: g.ui.settings.show(self.root)),
                                     ('Kaydet', g.save_game), ('Ana Menü', g.to_main_menu), ('Kaydet ve Çık', g.save_and_quit)]):
            MenuItem(self.root, t, (L - 0.02, 0.06 - i * 0.075), cb, 1.4)


class UI:
    def __init__(self, game):
        self.game = game
        self.hud = HUD(game)
        self.menu = MainMenu(game)
        self.settings = SettingsMenu(game)
        self.pause = PauseMenu(game)
        self.inventory = Inventory(game)
        self.book = BuildBook(game)
        self.trade = Trade(game)
        self.garage = Garage(game)
        self.windows = [self.inventory, self.book, self.trade, self.garage]

    def any_window(self):
        return any(w.open for w in self.windows)

    def close_windows(self):
        for w in self.windows:
            w.hide()

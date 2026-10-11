"""Arayüz ikonları: PIL ile kodla çizilir (hiçbir görsel dosya kullanılmaz)."""
import math
from PIL import Image, ImageDraw, ImageFilter
from ursina import Texture

S = 256      # çizim çözünürlüğü (sonra 96 piksele küçültülerek yumuşatılır)
OUT = 96


def _new():
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _finish(img, shadow=True):
    if shadow:
        a = img.split()[3].filter(ImageFilter.GaussianBlur(6))
        sh = Image.new('RGBA', img.size, (0, 0, 0, 0))
        sh.putalpha(a.point(lambda v: int(v * 0.45)))
        base = Image.new('RGBA', img.size, (0, 0, 0, 0))
        base.alpha_composite(sh, (6, 8))
        base.alpha_composite(img)
        img = base
    return img.resize((OUT, OUT), Image.LANCZOS)


def _shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c[:3]) + ((c[3],) if len(c) == 4 else (255,))


def hoe():
    img, d = _new()
    d.line([(60, 215), (185, 60)], fill=(120, 78, 42), width=22)
    d.line([(64, 210), (183, 64)], fill=(160, 108, 60), width=8)
    d.polygon([(160, 40), (225, 70), (215, 100), (170, 78)], fill=(150, 155, 165))
    d.polygon([(160, 40), (225, 70), (222, 78), (165, 52)], fill=(205, 210, 220))
    return _finish(img)


def can():
    img, d = _new()
    d.rounded_rectangle([60, 95, 175, 210], 18, fill=(70, 140, 200))
    d.rounded_rectangle([60, 95, 100, 210], 18, fill=(95, 170, 230))
    d.polygon([(170, 140), (230, 85), (240, 98), (175, 165)], fill=(60, 120, 180))
    d.ellipse([222, 70, 252, 102], fill=(80, 150, 210))
    d.arc([70, 40, 165, 140], 180, 360, fill=(50, 105, 160), width=16)
    for i in range(3):
        d.ellipse([236 - i * 6, 112 + i * 22, 248 - i * 6, 128 + i * 22], fill=(150, 210, 255))
    return _finish(img)


def sickle():
    img, d = _new()
    d.line([(70, 225), (120, 150)], fill=(120, 78, 42), width=24)
    d.arc([95, 30, 235, 170], 120, 380, fill=(185, 190, 200), width=24)
    d.arc([103, 38, 227, 162], 130, 370, fill=(225, 230, 238), width=8)
    return _finish(img)


def seed_bag(col):
    img, d = _new()
    bag = (214, 180, 120)
    d.polygon([(70, 95), (186, 95), (205, 220), (51, 220)], fill=bag)
    d.polygon([(70, 95), (110, 95), (95, 220), (51, 220)], fill=_shade(bag, 1.1))
    d.ellipse([60, 205, 196, 235], fill=_shade(bag, 0.85))
    d.polygon([(80, 95), (176, 95), (160, 55), (96, 55)], fill=_shade(bag, 0.95))
    d.rectangle([84, 88, 172, 100], fill=(140, 90, 50))
    d.ellipse([92, 122, 164, 194], fill=tuple(int(c * 255) for c in col))
    d.ellipse([104, 132, 130, 156], fill=(255, 255, 255, 120))
    for i, (x, y) in enumerate([(108, 45), (128, 32), (148, 46)]):
        d.ellipse([x, y, x + 16, y + 22], fill=(150, 110, 60))
    return _finish(img)


def coin():
    img, d = _new()
    d.ellipse([28, 28, 228, 228], fill=(200, 140, 30))
    d.ellipse([40, 36, 220, 216], fill=(250, 200, 60))
    d.ellipse([62, 58, 198, 194], outline=(210, 150, 30), width=10)
    d.polygon([(128, 80), (142, 115), (180, 118), (150, 140), (160, 178), (128, 156), (96, 178), (106, 140), (76, 118), (114, 115)],
              fill=(215, 155, 35))
    d.ellipse([70, 60, 110, 90], fill=(255, 245, 190, 160))
    return _finish(img)


def wheat():
    img, d = _new()
    for k, ang in enumerate((-20, 0, 20)):
        a = math.radians(ang)
        x0, y0 = 128, 230
        x1, y1 = 128 + math.sin(a) * 170, 230 - math.cos(a) * 170
        d.line([(x0, y0), (x1, y1)], fill=(170, 140, 60), width=8)
        for i in range(6):
            t = 0.55 + i * 0.08
            cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            for s in (-1, 1):
                d.ellipse([cx + s * 14 - 11, cy - 14, cx + s * 14 + 11, cy + 14], fill=(240, 200, 90))
        d.ellipse([x1 - 10, y1 - 16, x1 + 10, y1 + 16], fill=(245, 210, 100))
    d.rectangle([108, 170, 148, 186], fill=(180, 60, 50))
    return _finish(img)


def carrot():
    img, d = _new()
    d.polygon([(95, 85), (165, 85), (138, 238), (125, 238)], fill=(245, 130, 30))
    d.polygon([(95, 85), (125, 85), (128, 238), (125, 238)], fill=(255, 160, 60))
    for y in (120, 150, 185):
        d.line([(110, y), (128, y + 4)], fill=(200, 95, 20), width=4)
    for ang in (-40, -15, 10, 35):
        a = math.radians(ang)
        d.line([(130, 88), (130 + math.sin(a) * 75, 88 - math.cos(a) * 75)], fill=(70, 150, 50), width=14)
    return _finish(img)


def tomato():
    img, d = _new()
    d.ellipse([40, 60, 216, 220], fill=(215, 40, 35))
    d.ellipse([60, 75, 130, 135], fill=(245, 110, 90))
    for ang in range(0, 360, 72):
        a = math.radians(ang)
        d.polygon([(128, 72), (128 + math.cos(a) * 50, 72 + math.sin(a) * 22 - 6), (128 + math.cos(a + 0.4) * 14, 72 + math.sin(a + 0.4) * 10)],
                  fill=(60, 140, 50))
    d.line([(128, 72), (135, 35)], fill=(60, 120, 40), width=10)
    return _finish(img)


def sunflower():
    img, d = _new()
    d.line([(128, 140), (128, 250)], fill=(70, 140, 50), width=14)
    for i in range(14):
        a = i / 14 * 2 * math.pi
        cx, cy = 128 + math.cos(a) * 62, 110 + math.sin(a) * 62
        d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], fill=(255, 205, 30))
    d.ellipse([78, 60, 178, 160], fill=(110, 70, 30))
    for i in range(30):
        a = i * 2.4
        r = 6 + i * 1.4
        d.ellipse([128 + math.cos(a) * r - 4, 110 + math.sin(a) * r - 4, 128 + math.cos(a) * r + 4, 110 + math.sin(a) * r + 4],
                  fill=(80, 50, 20))
    return _finish(img)


def pumpkin():
    img, d = _new()
    for i, (x0, x1) in enumerate([(30, 130), (126, 226), (60, 196)]):
        c = (235, 125, 25) if i < 2 else (250, 145, 40)
        d.ellipse([x0, 80, x1, 225], fill=c)
    d.line([(128, 90), (128, 220)], fill=(210, 100, 20), width=5)
    d.rounded_rectangle([116, 45, 140, 95], 8, fill=(100, 140, 50))
    d.ellipse([70, 100, 105, 140], fill=(255, 190, 110, 140))
    return _finish(img)


def egg():
    img, d = _new()
    d.ellipse([70, 40, 186, 220], fill=(250, 240, 220))
    d.ellipse([90, 70, 130, 120], fill=(255, 255, 255))
    d.ellipse([70, 40, 186, 220], outline=(220, 200, 170), width=6)
    return _finish(img)


def wool():
    img, d = _new()
    for cx, cy, r in [(90, 140, 55), (160, 140, 55), (125, 100, 55), (125, 170, 50), (80, 105, 35), (175, 105, 35)]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 235, 240))
        d.ellipse([cx - r + 10, cy - r + 6, cx + 5, cy - 5], fill=(255, 255, 255))
    return _finish(img)


def milk():
    img, d = _new()
    d.rounded_rectangle([80, 90, 176, 230], 24, fill=(245, 248, 255))
    d.rectangle([100, 40, 156, 95], fill=(235, 240, 250))
    d.rectangle([94, 30, 162, 52], fill=(70, 130, 210))
    d.rectangle([80, 140, 176, 185], fill=(70, 130, 210))
    d.rounded_rectangle([92, 100, 110, 220], 8, fill=(255, 255, 255))
    return _finish(img)


def chicken():
    img, d = _new()
    d.ellipse([50, 90, 200, 210], fill=(250, 248, 240))
    d.ellipse([140, 50, 210, 120], fill=(250, 248, 240))
    d.polygon([(205, 85), (240, 92), (205, 100)], fill=(250, 180, 40))
    d.ellipse([155, 30, 195, 65], fill=(220, 40, 40))
    d.ellipse([175, 70, 187, 82], fill=(30, 30, 30))
    d.polygon([(55, 120), (20, 80), (45, 150)], fill=(235, 232, 225))
    d.line([(110, 205), (105, 240)], fill=(240, 160, 40), width=8)
    d.line([(145, 205), (150, 240)], fill=(240, 160, 40), width=8)
    return _finish(img)


def cow():
    img, d = _new()
    d.rounded_rectangle([40, 80, 200, 180], 30, fill=(250, 250, 250))
    d.ellipse([70, 95, 120, 140], fill=(40, 40, 40))
    d.ellipse([140, 120, 180, 165], fill=(40, 40, 40))
    d.rounded_rectangle([170, 60, 240, 130], 20, fill=(250, 250, 250))
    d.rounded_rectangle([200, 100, 245, 135], 14, fill=(240, 170, 170))
    d.polygon([(185, 62), (178, 35), (195, 58)], fill=(230, 220, 190))
    d.polygon([(225, 62), (235, 35), (215, 58)], fill=(230, 220, 190))
    for x in (60, 95, 150, 180):
        d.rectangle([x, 175, x + 18, 230], fill=(235, 235, 235))
    return _finish(img)


def sheep():
    img, d = _new()
    for cx, cy, r in [(80, 130, 50), (130, 115, 55), (170, 135, 48), (120, 160, 50)]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(240, 240, 235))
    d.ellipse([180, 80, 240, 140], fill=(50, 45, 45))
    d.ellipse([205, 100, 215, 110], fill=(255, 255, 255))
    for x in (80, 110, 140, 165):
        d.rectangle([x, 185, x + 14, 235], fill=(50, 45, 45))
    return _finish(img)


def expand():
    img, d = _new()
    d.rounded_rectangle([40, 40, 216, 216], 20, fill=(140, 95, 55))
    for i in range(4):
        d.rectangle([52, 58 + i * 40, 204, 76 + i * 40], fill=(110, 70, 40))
    d.ellipse([150, 150, 240, 240], fill=(80, 190, 80))
    d.rectangle([188, 165, 202, 225], fill=(255, 255, 255))
    d.rectangle([165, 188, 225, 202], fill=(255, 255, 255))
    return _finish(img)


def sun_icon():
    img, d = _new()
    for i in range(12):
        a = i / 12 * 2 * math.pi
        d.line([(128 + math.cos(a) * 70, 128 + math.sin(a) * 70), (128 + math.cos(a) * 110, 128 + math.sin(a) * 110)],
               fill=(255, 200, 40), width=16)
    d.ellipse([68, 68, 188, 188], fill=(255, 215, 60))
    return _finish(img)


def moon_icon():
    img, d = _new()
    d.ellipse([50, 40, 210, 200], fill=(235, 235, 210))
    d.ellipse([95, 20, 245, 170], fill=(0, 0, 0, 0))
    return _finish(img)


def rain_icon():
    img, d = _new()
    for cx, cy, r in [(90, 110, 45), (140, 90, 55), (185, 115, 40)]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(170, 180, 195))
    d.rounded_rectangle([50, 105, 220, 150], 22, fill=(170, 180, 195))
    for x in (80, 125, 170):
        d.line([(x, 170), (x - 12, 215)], fill=(90, 160, 240), width=10)
    return _finish(img)


_cache = {}


def get(name, *args):
    key = (name,) + args
    if key not in _cache:
        img = globals()[name](*args)
        _cache[key] = Texture(img)
    return _cache[key]


# ----------------------------------------------------------------------------- yeni ikonlar
def axe():
    img, d = _new()
    d.line([(70, 225), (165, 45)], fill=(120, 78, 42), width=20)
    d.line([(74, 220), (163, 50)], fill=(165, 112, 62), width=7)
    d.polygon([(140, 40), (215, 55), (225, 120), (170, 105), (150, 85)], fill=(150, 156, 166))
    d.polygon([(205, 52), (215, 55), (225, 120), (214, 117)], fill=(225, 230, 238))
    return _finish(img)


def pickaxe():
    img, d = _new()
    d.line([(80, 230), (150, 70)], fill=(120, 78, 42), width=20)
    d.arc([40, 30, 240, 170], 200, 340, fill=(140, 144, 150), width=26)
    d.arc([48, 38, 232, 162], 205, 335, fill=(195, 200, 208), width=8)
    return _finish(img)


def hammer():
    img, d = _new()
    d.line([(85, 230), (150, 95)], fill=(120, 78, 42), width=22)
    d.polygon([(95, 60), (200, 105), (185, 140), (80, 95)], fill=(140, 144, 150))
    d.polygon([(95, 60), (200, 105), (196, 114), (91, 70)], fill=(205, 210, 218))
    return _finish(img)


def log():
    img, d = _new()
    d.rounded_rectangle([40, 90, 210, 170], 40, fill=(120, 80, 45))
    for i in range(5):
        d.line([(60, 100 + i * 15), (190, 100 + i * 15)], fill=(95, 62, 34), width=4)
    d.ellipse([170, 88, 236, 172], fill=(210, 170, 110))
    for r in (25, 16, 8):
        d.ellipse([203 - r, 130 - r * 1.25, 203 + r, 130 + r * 1.25], outline=(165, 120, 70), width=3)
    return _finish(img)


def stone():
    img, d = _new()
    d.polygon([(50, 170), (70, 100), (130, 70), (195, 95), (215, 165), (160, 200), (85, 200)], fill=(130, 130, 128))
    d.polygon([(70, 100), (130, 70), (195, 95), (150, 120), (95, 125)], fill=(175, 175, 172))
    d.line([(110, 140), (150, 170)], fill=(100, 100, 98), width=4)
    return _finish(img)


def iron():
    img, d = _new()
    for i, (x, y) in enumerate([(70, 140), (130, 110), (110, 175)]):
        d.polygon([(x - 45, y), (x - 25, y - 30), (x + 35, y - 30), (x + 50, y), (x + 35, y + 25), (x - 30, y + 25)],
                  fill=(120, 125, 135))
        d.polygon([(x - 25, y - 30), (x + 35, y - 30), (x + 25, y - 12), (x - 15, y - 12)], fill=(185, 190, 200))
    return _finish(img)


def berries():
    img, d = _new()
    for x, y in [(100, 140), (140, 130), (120, 170), (160, 165), (80, 175), (125, 105)]:
        d.ellipse([x - 26, y - 26, x + 26, y + 26], fill=(90, 15, 40))
        d.ellipse([x - 14, y - 16, x - 2, y - 4], fill=(200, 120, 150))
    d.polygon([(120, 80), (180, 40), (175, 75)], fill=(60, 130, 50))
    return _finish(img)


def corn():
    img, d = _new()
    d.ellipse([90, 40, 170, 220], fill=(245, 205, 60))
    for y in range(60, 210, 16):
        for x in range(100, 165, 16):
            d.ellipse([x, y, x + 12, y + 12], fill=(250, 220, 90))
    d.polygon([(90, 120), (40, 230), (120, 210)], fill=(110, 160, 60))
    d.polygon([(170, 120), (215, 230), (140, 210)], fill=(90, 140, 50))
    return _finish(img)


def soup():
    img, d = _new()
    d.ellipse([40, 110, 216, 170], fill=(200, 90, 40))
    d.pieslice([40, 60, 216, 230], 0, 180, fill=(140, 90, 60))
    d.ellipse([40, 110, 216, 170], fill=(210, 110, 45))
    d.ellipse([80, 125, 110, 145], fill=(250, 160, 60))
    d.ellipse([140, 128, 170, 148], fill=(120, 170, 70))
    for x in (100, 130, 160):
        d.arc([x - 15, 40, x + 15, 100], 200, 340, fill=(230, 230, 230), width=6)
    return _finish(img)


def car():
    img, d = _new()
    d.rounded_rectangle([20, 120, 236, 185], 20, fill=(170, 30, 25))
    d.polygon([(70, 120), (95, 70), (165, 70), (175, 120)], fill=(170, 30, 25))
    d.polygon([(85, 115), (102, 80), (155, 80), (163, 115)], fill=(80, 110, 130))
    for x in (65, 190):
        d.ellipse([x - 28, 160, x + 28, 216], fill=(30, 30, 30))
        d.ellipse([x - 13, 175, x + 13, 201], fill=(180, 180, 185))
    return _finish(img)


def gear():
    img, d = _new()
    for i in range(10):
        a = i / 10 * 2 * math.pi
        d.polygon([(128 + math.cos(a - 0.17) * 80, 128 + math.sin(a - 0.17) * 80),
                   (128 + math.cos(a - 0.12) * 108, 128 + math.sin(a - 0.12) * 108),
                   (128 + math.cos(a + 0.12) * 108, 128 + math.sin(a + 0.12) * 108),
                   (128 + math.cos(a + 0.17) * 80, 128 + math.sin(a + 0.17) * 80)], fill=(150, 155, 165))
    d.ellipse([45, 45, 211, 211], fill=(150, 155, 165))
    d.ellipse([95, 95, 161, 161], fill=(0, 0, 0, 0))
    return _finish(img)


def wheel_icon():
    img, d = _new()
    d.ellipse([30, 30, 226, 226], fill=(35, 35, 35))
    d.ellipse([75, 75, 181, 181], fill=(180, 182, 188))
    d.ellipse([110, 110, 146, 146], fill=(110, 110, 115))
    return _finish(img)


def chassis():
    img, d = _new()
    d.rectangle([30, 110, 226, 140], fill=(120, 125, 135))
    for x in (60, 196):
        d.rectangle([x - 8, 80, x + 8, 170], fill=(100, 105, 115))
    d.rectangle([110, 90, 146, 160], fill=(150, 80, 50))
    return _finish(img)


def heart():
    img, d = _new()
    d.ellipse([40, 60, 135, 155], fill=(230, 70, 60))
    d.ellipse([121, 60, 216, 155], fill=(230, 70, 60))
    d.polygon([(45, 125), (128, 225), (211, 125)], fill=(230, 70, 60))
    return _finish(img)


def food_icon():
    img, d = _new()
    d.ellipse([50, 60, 206, 216], fill=(210, 60, 50))
    d.ellipse([75, 85, 120, 130], fill=(250, 150, 140))
    d.line([(128, 65), (140, 30)], fill=(100, 70, 40), width=10)
    d.ellipse([140, 20, 200, 55], fill=(80, 150, 60))
    return _finish(img)


def bolt():
    img, d = _new()
    d.polygon([(145, 20), (60, 145), (120, 145), (100, 236), (196, 100), (135, 100)], fill=(255, 200, 50))
    return _finish(img)


def house_icon():
    img, d = _new()
    d.polygon([(30, 125), (128, 40), (226, 125)], fill=(150, 70, 45))
    d.rectangle([55, 120, 201, 220], fill=(160, 115, 70))
    for i in range(5):
        d.line([(55, 135 + i * 18), (201, 135 + i * 18)], fill=(125, 85, 50), width=4)
    d.rectangle([110, 160, 146, 220], fill=(80, 50, 30))
    d.rectangle([160, 145, 190, 175], fill=(140, 180, 210))
    return _finish(img)


def book():
    img, d = _new()
    d.rounded_rectangle([50, 40, 210, 220], 12, fill=(120, 70, 40))
    d.rectangle([65, 50, 200, 210], fill=(235, 225, 200))
    d.rectangle([50, 40, 75, 220], fill=(95, 55, 30))
    for y in range(80, 200, 22):
        d.line([(90, y), (185, y)], fill=(170, 160, 140), width=4)
    return _finish(img)

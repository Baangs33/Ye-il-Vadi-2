"""Genel ayarlar, harita, kaynaklar, ekinler, hayvanlar ve araba tarifleri."""

WORLD = 200.0               # harita yarı genişliği (400 x 400 m)
TERRAIN_STEP = 2.0
WATER_Y = -1.2
LAKE = (-72.0, 48.0, 34.0)  # x, z, yarıçap
SPAWN = (2.0, -6.0)
MERCHANT = (9.0, -16.0)
ROAD = [(14.0, -215.0), (20.0, -150.0), (6.0, -95.0), (-6.0, -55.0), (4.0, -26.0), (8.0, -12.0)]

DAY_LENGTH = 16 * 60.0      # bir oyun günü (saniye)
START_TIME = 0.32

# --- hayatta kalma
HUNGER_TIME = DAY_LENGTH * 1.3      # tok -> aç süresi
ENERGY_REGEN = 14.0
SPRINT_COST = 9.0
JUMP_COST = 6.0
SWING_COST = 3.5

# --- kaynaklar
RES = {
    'kutuk':   dict(name='Kütük',      sell=3),
    'tas':     dict(name='Taş',        sell=2),
    'demir':   dict(name='Demir',      sell=8),
    'bogurtlen': dict(name='Böğürtlen', sell=2, food=7),
}

CROPS = {
    'bugday':    dict(name='Buğday',    seed=4,  sell=11, time=300.0, food=0,  color=(0.95, 0.80, 0.36)),
    'havuc':     dict(name='Havuç',     seed=6,  sell=16, time=360.0, food=12, color=(0.98, 0.52, 0.12)),
    'domates':   dict(name='Domates',   seed=9,  sell=24, time=480.0, food=10, color=(0.90, 0.16, 0.12)),
    'misir':     dict(name='Mısır',     seed=10, sell=28, time=540.0, food=16, color=(0.98, 0.85, 0.25)),
    'balkabagi': dict(name='Balkabağı', seed=16, sell=55, time=720.0, food=35, color=(0.97, 0.53, 0.10)),
}
CROP_ORDER = ['bugday', 'havuc', 'domates', 'misir', 'balkabagi']

ANIMALS = {
    'tavuk': dict(name='Tavuk', price=50,  product='yumurta', time=180.0, home='kumes', per_home=6),
    'koyun': dict(name='Koyun', price=140, product='yun',     time=300.0, home='ahir',  per_home=4),
    'inek':  dict(name='İnek',  price=240, product='sut',     time=260.0, home='ahir',  per_home=3),
}
PRODUCTS = {
    'yumurta': dict(name='Yumurta', sell=12, food=8),
    'yun':     dict(name='Yün',     sell=45, food=0),
    'sut':     dict(name='Süt',     sell=30, food=14),
}

# --- araba
CAR_PRICE = 2400
CAR_PARTS = {
    'sasi':     dict(name='Şasi ve Kasa', cost={'kutuk': 20, 'demir': 14}, money=0),
    'motor':    dict(name='Motor',        cost={'demir': 22},              money=250),
    'tekerlek': dict(name='4 Tekerlek',   cost={'demir': 10, 'kutuk': 6},  money=60),
}

START_MONEY = 60
SAVE_FILE = 'vadi_kayit.json'
SETTINGS_FILE = 'vadi_ayarlar.json'

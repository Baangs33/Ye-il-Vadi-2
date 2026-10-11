"""Oyun ayarları: kayıt, ön ayarlar ve seçenek tanımları."""
import json
import os
from config import SETTINGS_FILE

DEFAULTS = {
    'quality': 2,        # 0 Düşük, 1 Orta, 2 Yüksek, 3 Ultra
    'shadows': 2,        # 0 Kapalı, 1 Düşük, 2 Orta, 3 Yüksek
    'grass': 2,          # 0 Kapalı ... 3 Çok
    'view': 1,           # 0 Kısa, 1 Orta, 2 Uzak
    'ao': 0,
    'bloom': 1,
    'fxaa': 1,
    'fov': 75,
    'fullscreen': 0,
    'vsync': 1,
    'show_fps': 0,
    'sens': 1.0,
    'invert': 0,
    'volume': 80,
    'head_bob': 1,
}

OPTIONS = {
    'quality': ['Düşük', 'Orta', 'Yüksek', 'Ultra'],
    'shadows': ['Kapalı', 'Düşük', 'Orta', 'Yüksek'],
    'grass': ['Kapalı', 'Az', 'Orta', 'Çok'],
    'view': ['Kısa', 'Orta', 'Uzak'],
    'ao': ['Kapalı', 'Açık'],
    'bloom': ['Kapalı', 'Açık'],
    'fxaa': ['Kapalı', 'Açık'],
    'fullscreen': ['Kapalı', 'Açık'],
    'vsync': ['Kapalı', 'Açık'],
    'show_fps': ['Kapalı', 'Açık'],
    'invert': ['Kapalı', 'Açık'],
    'head_bob': ['Kapalı', 'Açık'],
}

PRESETS = {
    0: dict(shadows=1, grass=1, view=0, ao=0, bloom=0, fxaa=1),
    1: dict(shadows=2, grass=1, view=1, ao=0, bloom=1, fxaa=1),
    2: dict(shadows=2, grass=2, view=1, ao=0, bloom=1, fxaa=1),
    3: dict(shadows=3, grass=3, view=2, ao=1, bloom=1, fxaa=1),
}

SHADOW_RES = [0, 1024, 2048, 4096]
GRASS = [(0.0, 0), (0.45, 36), (0.8, 52), (1.2, 66)]   # (yoğunluk, yarıçap)
VIEW = [(0.75, 450), (1.0, 700), (1.35, 900)]          # (sis çarpanı, uzak kırpma)


def load():
    s = dict(DEFAULTS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding='utf-8') as f:
                s.update(json.load(f))
        except Exception:
            pass
    return s


def save(s):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
    except Exception:
        pass

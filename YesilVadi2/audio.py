"""Tüm ses efektleri numpy ile sentezlenir ve ilk açılışta 'sesler' klasörüne yazılır."""
import os
import wave
import random
import numpy as np

SR = 22050
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sesler')


def _env(n, a=0.01, r=0.2):
    t = np.linspace(0, 1, n)
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / max(r, 1e-4) * 3)
    return e


def _lowpass(x, k):
    y = np.zeros_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += (x[i] - acc) * k
        y[i] = acc
    return y


def _lp_fast(x, k):
    # birinci dereceden alçak geçiren (vektörel yaklaşım: hareketli ortalama zinciri)
    w = max(1, int(1 / k))
    ker = np.ones(w) / w
    return np.convolve(np.convolve(x, ker, 'same'), ker, 'same')


def _t(sec):
    return np.arange(int(SR * sec)) / SR


def s_hoe():
    t = _t(0.28)
    thud = np.sin(2 * np.pi * (90 - 60 * t) * t) * np.exp(-t * 25)
    noise = _lp_fast(np.random.uniform(-1, 1, len(t)), 0.15) * np.exp(-t * 18)
    return thud * 0.9 + noise * 1.6


def s_water():
    t = _t(0.7)
    n = np.random.uniform(-1, 1, len(t))
    hp = n - _lp_fast(n, 0.08)
    env = np.sin(np.pi * np.clip(t / 0.7, 0, 1)) ** 0.7
    bub = sum(np.sin(2 * np.pi * f * t) * np.exp(-((t - s) * 40) ** 2) for f, s in [(900, .1), (1200, .25), (800, .4), (1400, .5)])
    return hp * env * 0.5 + bub * 0.25


def s_harvest():
    t = _t(0.22)
    f = 300 + 900 * t / 0.22
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14) * 0.8


def s_plant():
    t = _t(0.15)
    return (np.sin(2 * np.pi * 520 * t) + 0.5 * np.sin(2 * np.pi * 780 * t)) * np.exp(-t * 35) * 0.6


def s_coin():
    t = _t(0.55)
    a = np.sin(2 * np.pi * 1318 * t) * np.exp(-t * 9) * (t < 0.08)
    b = np.sin(2 * np.pi * 1760 * t) * np.exp(-(t - 0.08).clip(0) * 6) * (t >= 0.08)
    return (a + b + 0.3 * np.sin(2 * np.pi * 3520 * t) * np.exp(-t * 12)) * 0.5


def s_click():
    t = _t(0.06)
    return np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 80) * 0.4


def s_error():
    t = _t(0.25)
    return np.sign(np.sin(2 * np.pi * 150 * t)) * np.exp(-t * 10) * 0.25


def s_bird():
    out = []
    for _ in range(random.randint(2, 5)):
        dur = random.uniform(0.06, 0.14)
        t = _t(dur)
        f0, f1 = random.uniform(2500, 4200), random.uniform(2000, 4800)
        f = f0 + (f1 - f0) * t / dur
        out.append(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / dur) * 0.35)
        out.append(np.zeros(int(SR * random.uniform(0.03, 0.12))))
    return np.concatenate(out)


def s_cluck():
    out = []
    for _ in range(3):
        t = _t(0.07)
        n = np.random.uniform(-1, 1, len(t))
        tone = np.sin(2 * np.pi * (700 + 300 * np.sin(t * 90)) * t)
        out.append((tone * 0.6 + n * 0.3) * np.sin(np.pi * t / 0.07) * 0.5)
        out.append(np.zeros(int(SR * 0.05)))
    return np.concatenate(out)


def s_moo():
    t = _t(1.2)
    f = 115 + 25 * np.sin(np.pi * t / 1.2) - 15 * t
    ph = 2 * np.pi * np.cumsum(f) / SR
    saw = sum(np.sin(ph * k) / k for k in range(1, 12))
    env = np.minimum(1, t / 0.15) * np.minimum(1, (1.2 - t) / 0.3)
    return _lp_fast(saw, 0.25) * env * 0.5


def s_baa():
    t = _t(0.8)
    f = 330 * (1 + 0.04 * np.sin(2 * np.pi * 22 * t)) - 60 * t
    ph = 2 * np.pi * np.cumsum(f) / SR
    v = sum(np.sin(ph * k) / (k ** 1.3) for k in range(1, 8))
    env = np.minimum(1, t / 0.05) * np.minimum(1, (0.8 - t) / 0.3)
    return v * env * 0.35


def s_rain():
    t = _t(4.0)
    n = np.random.uniform(-1, 1, len(t))
    x = _lp_fast(n, 0.35) * 0.5
    drops = np.zeros_like(t)
    for _ in range(160):
        i = random.randint(0, len(t) - 500)
        drops[i:i + 300] += np.random.uniform(-1, 1, 300) * np.exp(-np.arange(300) / 40) * 0.6
    y = x + drops
    fade = int(SR * 0.2)                                  # döngü eklemi yumuşatılır
    y[:fade] = y[:fade] * np.linspace(0, 1, fade) + y[-fade:] * np.linspace(1, 0, fade)
    return y[:-fade] * 0.5


def s_crickets():
    t = _t(3.0)
    out = np.zeros_like(t)
    for start in np.arange(0, 3.0, 0.55):
        for k in range(3):
            s = start + k * 0.05 + random.uniform(0, 0.02)
            m = (t > s) & (t < s + 0.035)
            out[m] += np.sin(2 * np.pi * 4300 * t[m]) * 0.25
    return out


def s_wind():
    t = _t(5.0)
    n = np.random.uniform(-1, 1, len(t))
    x = _lp_fast(_lp_fast(n, 0.03), 0.05)
    mod = 0.6 + 0.4 * np.sin(2 * np.pi * t / 5.0)
    return x * mod * 2.5


def s_buy():
    t = _t(0.3)
    return (np.sin(2 * np.pi * 880 * t) * (t < 0.1) + np.sin(2 * np.pi * 1320 * t) * (t >= 0.1)) * np.exp(-t * 8) * 0.4


def s_chop():
    t = _t(0.35)
    thunk = np.sin(2 * np.pi * (180 - 80 * t) * t) * np.exp(-t * 22)
    crack = np.random.uniform(-1, 1, len(t)) * np.exp(-t * 40)
    wood = np.sin(2 * np.pi * 420 * t) * np.exp(-t * 30) * 0.4
    return thunk + crack * 0.7 + wood


def s_mine():
    t = _t(0.4)
    ring = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, d in ((1850, 18), (2630, 22), (3900, 30)))
    hit = np.random.uniform(-1, 1, len(t)) * np.exp(-t * 60)
    return ring * 0.35 + hit * 0.8


def s_treefall():
    t = _t(2.4)
    creak = np.sin(2 * np.pi * (90 + 40 * np.sin(t * 3)) * t) * np.exp(-((t - 0.5) * 2) ** 2) * 0.4
    n = np.random.uniform(-1, 1, len(t))
    rustle = _lp_fast(n, 0.3) * np.exp(-((t - 1.4) * 1.8) ** 2) * 1.2
    thud = np.sin(2 * np.pi * (55 - 20 * (t - 1.7).clip(0)) * t) * np.exp(-(t - 1.7).clip(0) * 6) * (t > 1.7) * 1.5
    return creak + rustle + thud


def s_build():
    out = []
    for _ in range(2):
        t = _t(0.12)
        out.append((np.sin(2 * np.pi * 900 * t) * 0.5 + np.random.uniform(-1, 1, len(t)) * 0.6) * np.exp(-t * 45))
        out.append(np.zeros(int(SR * 0.08)))
    return np.concatenate(out)


def s_step():
    t = _t(0.12)
    n = np.random.uniform(-1, 1, len(t))
    return _lp_fast(n, 0.25) * np.exp(-t * 30) * 0.8


def s_jump():
    t = _t(0.15)
    return _lp_fast(np.random.uniform(-1, 1, len(t)), 0.2) * np.exp(-t * 20) * 0.6


def s_land():
    t = _t(0.25)
    return (np.sin(2 * np.pi * 70 * t) + _lp_fast(np.random.uniform(-1, 1, len(t)), 0.2)) * np.exp(-t * 18)


def s_eat():
    out = []
    for _ in range(3):
        t = _t(0.08)
        out.append(np.random.uniform(-1, 1, len(t)) * np.exp(-t * 35) * 0.6)
        out.append(np.zeros(int(SR * 0.07)))
    return np.concatenate(out)


def s_engine():
    t = _t(2.0)
    f = 45.0
    ph = 2 * np.pi * f * t
    v = sum(np.sin(ph * k + k) / k for k in range(1, 9)) + 0.3 * np.sign(np.sin(ph * 0.5))
    v += _lp_fast(np.random.uniform(-1, 1, len(t)), 0.3) * 0.3
    return v * 0.4


def s_crash():
    t = _t(0.6)
    return (_lp_fast(np.random.uniform(-1, 1, len(t)), 0.4) + np.sin(2 * np.pi * 60 * t)) * np.exp(-t * 7)


def s_pickup():
    t = _t(0.18)
    return np.sin(2 * np.pi * (500 + 900 * t / 0.18) * t) * np.exp(-t * 12) * 0.5


SOUNDS = {
    'hoe': s_hoe, 'water': s_water, 'harvest': s_harvest, 'plant': s_plant, 'coin': s_coin, 'click': s_click,
    'error': s_error, 'bird1': s_bird, 'bird2': s_bird, 'bird3': s_bird, 'cluck': s_cluck, 'moo': s_moo, 'baa': s_baa,
    'rain': s_rain, 'crickets': s_crickets, 'wind': s_wind, 'buy': s_buy,
    'chop': s_chop, 'mine': s_mine, 'treefall': s_treefall, 'build': s_build, 'step': s_step, 'jump': s_jump,
    'land': s_land, 'eat': s_eat, 'engine': s_engine, 'crash': s_crash, 'pickup': s_pickup,
}


def _write(path, data):
    data = np.asarray(data, dtype=np.float64)
    peak = np.max(np.abs(data)) or 1
    data = data / max(peak, 1.0) * 0.9
    pcm = (data * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def generate_all():
    os.makedirs(DIR, exist_ok=True)
    random.seed(3)
    np.random.seed(3)
    for name, fn in SOUNDS.items():
        p = os.path.join(DIR, name + '.wav')
        if not os.path.exists(p):
            _write(p, fn())


class SoundBank:
    def __init__(self):
        generate_all()
        from ursina import Audio
        self.Audio = Audio
        self.loops = {}
        self.master = 1.0
        for name in ('rain', 'crickets', 'wind', 'engine'):
            try:
                a = Audio(os.path.join('sesler', name + '.wav'), loop=True, autoplay=True, volume=0)
                self.loops[name] = a
            except Exception:
                pass

    def play(self, name, volume=0.6, pitch=None):
        if self.master <= 0.001:
            return
        try:
            a = self.Audio(os.path.join('sesler', name + '.wav'), autoplay=False, auto_destroy=True,
                           volume=volume * self.master)
            if pitch is None:
                pitch = random.uniform(0.92, 1.08)
            a.pitch = pitch
            a.play()
        except Exception:
            pass

    def set_loop(self, name, volume, pitch=None):
        a = self.loops.get(name)
        if a is not None:
            try:
                a.volume = volume * self.master
                if pitch is not None:
                    a.pitch = pitch
            except Exception:
                pass

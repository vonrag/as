#!/usr/bin/env python3
"""
Зинедин Зидан — 30-секундный ролик в стиле футбольных карточек (walkout + карточка).

Всё рисуется кодом (PIL + numpy), звук синтезируется, сборка — ffmpeg.
    python3 render.py                 -> zidane_card.mp4
    python3 render.py --preview 4 11  -> PNG-кадры на 4 с и 11 с в preview/
    python3 render.py --lang en       -> zidane_card_en.mp4 (английская версия)
    python3 render.py --player ronaldo --lang en -> ronaldo_card_en.mp4
Данные игроков — в players.py.
"""
import math
import os
import random
import subprocess
import sys
import wave
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from players import PLAYERS

W, H = 1080, 1920
FPS = 30
DUR = 30.0
N = int(FPS * DUR)


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


LANG = "en" if _arg("--lang", "ru") == "en" else "ru"
P = PLAYERS[_arg("--player", "zidane")]
if LANG == "ru" and "ru" in P:  # русские надписи для игроков, чьи данные записаны по-английски
    P = {**P, **P["ru"]}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   P["out"] + ("_en.mp4" if LANG == "en" else ".mp4"))

# Все надписи пишутся по-русски; для английской версии подменяются здесь.
EN = {
    "ЦАП": "CAM", "СКР": "PAC", "УДР": "SHO", "ПАС": "PAS", "ДРБ": "DRI", "ЗЩТ": "DEF", "ФИЗ": "PHY",
    "ЛЕГЕНДА": "LEGEND", "ВЫХОДИТ НА ПОЛЕ": "WALKS OUT", "ФРАНЦИЯ": "FRANCE",
    "АТАКУЮЩИЙ ПОЛУЗАЩИТНИК": "ATTACKING MIDFIELDER", "РЕАЛ": "REAL", "МАДРИД": "MADRID",
    "ЮВЕНТУС · РЕАЛ · СБОРНАЯ ФРАНЦИИ": "JUVENTUS · REAL MADRID · FRANCE",
    "ЗИНЕДИН ЗИДАН": "ZINEDINE ZIDANE", "ХАРАКТЕРИСТИКИ": "ATTRIBUTES",
    "СКОРОСТЬ": "PACE", "УДАРЫ": "SHOOTING", "ПАСЫ": "PASSING", "ДРИБЛИНГ": "DRIBBLING",
    "ЗАЩИТА": "DEFENDING", "ФИЗИКА": "PHYSICAL",
    "КАРЬЕРА": "CAREER", "МАТЧИ И ГОЛЫ В ЧЕМПИОНАТАХ": "LEAGUE APPEARANCES & GOALS",
    "КАНН": "CANNES", "БОРДО": "BORDEAUX", "ЮВЕНТУС": "JUVENTUS", "РЕАЛ МАДРИД": "REAL MADRID",
    "ИГР": "APPS", "ГОЛОВ": "GOALS", "СБОРНАЯ ФРАНЦИИ: 108 МАТЧЕЙ · 31 ГОЛ": "FRANCE: 108 CAPS · 31 GOALS",
    "ТРОФЕИ": "TROPHIES", "ЛЕГЕНДА · 96": "LEGEND · 96",
}


def tr(text):
    return EN.get(text, text) if LANG == "en" else text

FONT_DIR = "/usr/share/fonts/opentype/inter/"
BLACK = "InterDisplay-Black.otf"
XB = "InterDisplay-ExtraBold.otf"
BOLD = "InterDisplay-Bold.otf"
SEMI = "InterDisplay-SemiBold.otf"

GOLD = (240, 200, 105)
GOLD_L = (255, 232, 160)
WHITE = (255, 255, 255)
INK = (52, 36, 8)

# --------------------------------------------------------------------------- helpers


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def prog(t, a, b):
    return clamp((t - a) / (b - a))


def ease_out(x):
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_back(x, s=1.7):
    x -= 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def win(t, a, b, fin=0.25, fout=0.25):
    """Альфа окна [a,b] с плавным входом/выходом."""
    if t < a or t > b:
        return 0.0
    v = 1.0
    if fin > 0:
        v = min(v, (t - a) / fin)
    if fout > 0:
        v = min(v, (b - t) / fout)
    return clamp(v)


@lru_cache(None)
def font(name, size):
    return ImageFont.truetype(FONT_DIR + name, size)


@lru_cache(None)
def timg(text, fname, size, fill=WHITE, tracking=0, stroke=0, sfill=None, glow=0, gcol=None, galpha=1.0):
    text = tr(text)
    f = font(fname, size)
    widths = [f.getlength(c) for c in text]
    tw = int(sum(widths) + tracking * max(0, len(text) - 1))
    asc, desc = f.getmetrics()
    pad = glow * 3 + stroke + 6
    w, h = tw + 2 * pad, asc + desc + 2 * pad

    def draw(img, col, sw=0, sc=None):
        d = ImageDraw.Draw(img)
        x = pad
        for c, cw in zip(text, widths):
            d.text((x, pad), c, font=f, fill=col, stroke_width=sw, stroke_fill=sc)
            x += cw + tracking

    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw(im, fill, stroke, sfill)
    if glow:
        g = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw(g, gcol or fill, stroke + 2, gcol or fill)
        g = g.filter(ImageFilter.GaussianBlur(glow))
        if galpha != 1.0:
            g.putalpha(g.getchannel("A").point(lambda v: int(min(255, v * galpha))))
        g2 = g.copy()
        g.alpha_composite(g2)
        g.alpha_composite(im)
        im = g
    return im


def with_alpha(im, alpha):
    if alpha >= 0.999:
        return im
    im = im.copy()
    im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    return im


def blit(canvas, im, cx, cy, scale=1.0, alpha=1.0, angle=0.0, sx=None, anchor="c", hq=False):
    if alpha <= 0.004:
        return
    sxx = scale if sx is None else sx
    if abs(sxx - 1) > 1e-3 or abs(scale - 1) > 1e-3:
        nw, nh = max(1, int(im.width * sxx)), max(1, int(im.height * scale))
        im = im.resize((nw, nh), Image.LANCZOS if hq else Image.BILINEAR)
    if angle:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    im = with_alpha(im, alpha)
    if anchor == "c":
        x, y = cx - im.width / 2, cy - im.height / 2
    elif anchor == "l":
        x, y = cx, cy - im.height / 2
    else:  # "r"
        x, y = cx - im.width, cy - im.height / 2
    canvas.paste(im, (int(round(x)), int(round(y))), im)


def make_blob(size, color, power=3.0):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    a = np.clip(1 - r, 0, 1) ** power
    arr = np.zeros((size, size, 4), np.uint8)
    arr[..., :3] = color
    arr[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def star_pts(cx, cy, r1, r2, n=5, rot=-90):
    pts = []
    for i in range(n * 2):
        r = r1 if i % 2 == 0 else r2
        a = math.radians(rot + i * 180 / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


# --------------------------------------------------------------------------- background

BW, BH = W // 2, H // 2
_yy, _xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
_dx, _dy = _xx - BW / 2, _yy - BH * 0.46
_R = np.hypot(_dx, _dy) / (BH * 0.55)
_ANG = np.arctan2(_dy, _dx)
_VIG = np.clip(1.15 - 0.55 * (np.hypot(_xx / BW - 0.5, _yy / BH - 0.5) * 1.6) ** 2, 0.25, 1)[..., None]
_CORE = np.exp(-(_R ** 2) * 1.8)[..., None]
_RAYFALL = np.exp(-_R * 1.2)


def background(t, tint, ray_int, out_col=(6, 8, 22)):
    rays = (0.5 + 0.5 * np.sin(_ANG * 12 + t * 0.35)) ** 6 * _RAYFALL * ray_int
    rays2 = (0.5 + 0.5 * np.sin(_ANG * 7 - t * 0.22 + 1.3)) ** 10 * _RAYFALL * ray_int * 0.6
    base = np.array(out_col, np.float32) + (np.array(tint, np.float32) - np.array(out_col, np.float32)) * _CORE
    base = base + (rays + rays2)[..., None] * np.array(tint, np.float32) * 1.4
    base *= _VIG
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    return img.resize((W, H), Image.BILINEAR)


random.seed(7)
PARTICLES = [
    (random.uniform(0, W), random.uniform(0, H), random.uniform(20, 120), random.uniform(1.5, 4.5), random.uniform(0, 6.28))
    for _ in range(140)
]


def particles(canvas, t, amount, color=GOLD):
    if amount <= 0:
        return
    d = ImageDraw.Draw(canvas)
    for x0, y0, sp, sz, ph in PARTICLES:
        y = (y0 - sp * t) % H
        x = x0 + 18 * math.sin(t * 0.8 + ph)
        tw = (0.5 + 0.5 * math.sin(t * 3 + ph * 3)) * amount
        if tw < 0.05:
            continue
        c = tuple(int(v * tw) for v in color)
        r = sz * (0.6 + 0.4 * tw)
        d.ellipse((x - r, y - r, x + r, y + r), fill=c)


# --------------------------------------------------------------------------- card

CW, CH = 760, 1110


def shape(w, h, m=0):
    rel = [(0.10, 0.0), (0.36, 0.0), (0.5, 0.032), (0.64, 0.0), (0.90, 0.0), (1.0, 0.075),
           (1.0, 0.84), (0.5, 1.0), (0.0, 0.84), (0.0, 0.075)]
    return [(m + x * (w - 2 * m), m + y * (h - 2 * m)) for x, y in rel]


def card_mask(w, h, m=0, ss=3):
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).polygon([(x * ss, y * ss) for x, y in shape(w, h, m)], fill=255)
    return im.resize((w, h), Image.LANCZOS)


def gradient(w, h, stops, diag=0.3):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = np.clip((yy / h) * (1 - diag) + (xx / w) * diag, 0, 1)
    pos = np.array([p for p, _ in stops], np.float32)
    out = np.zeros((h, w, 3), np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(s, pos, np.array([c[ch] for _, c in stops], np.float32))
    return out, xx, yy


def draw_jersey(img, cx, cy, s):
    u = [(-0.33, -1.0), (-0.95, -0.80), (-1.30, -0.22), (-0.95, 0.0), (-0.76, -0.30), (-0.76, 1.0),
         (0.76, 1.0), (0.76, -0.30), (0.95, 0.0), (1.30, -0.22), (0.95, -0.80), (0.33, -1.0), (0.0, -0.72)]
    pts = [(cx + x * s, cy + y * s) for x, y in u]
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).polygon([(x + 10, y + 16) for x, y in pts], fill=(60, 35, 0, 120))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(img)
    J = P["jersey"]
    d.polygon(pts, fill=J["body"])
    if J.get("stripes"):  # вертикальные полосы клубной формы
        m = Image.new("L", img.size, 0)
        ImageDraw.Draw(m).polygon(pts, fill=255)
        st = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(st)
        sw = s * 0.26
        x = cx - 1.4 * s
        k = 0
        while x < cx + 1.4 * s:
            sd.rectangle((x, cy - 1.1 * s, x + sw, cy + 1.1 * s), fill=J["stripes"][k % len(J["stripes"])])
            x += sw
            k += 1
        img.paste(st, (0, 0), Image.fromarray(np.minimum(np.array(m), np.array(st.getchannel("A")))))
        d = ImageDraw.Draw(img)
    # манжеты и ворот
    d.line([(cx - 1.30 * s, cy - 0.22 * s), (cx - 0.95 * s, cy)], fill=J["trim"], width=int(s * 0.07))
    d.line([(cx + 1.30 * s, cy - 0.22 * s), (cx + 0.95 * s, cy)], fill=J["trim"], width=int(s * 0.07))
    d.line([(cx - 0.33 * s, cy - 1.0 * s), (cx, cy - 0.72 * s), (cx + 0.33 * s, cy - 1.0 * s)],
           fill=J["trim"], width=int(s * 0.06))
    # полоска цветов флага на груди
    by = cy - 0.42 * s
    for i, col in enumerate(J["stripe"] or []):
        d.rectangle((cx - 0.76 * s + i * 0.507 * s, by, cx - 0.76 * s + (i + 1) * 0.507 * s, by + 0.07 * s), fill=col)
    num = timg(J["number"], BLACK, int(s * 0.95), fill=J["num"], stroke=3, sfill=J["num_stroke"])
    img.alpha_composite(num, (int(cx - num.width / 2), int(cy + 0.25 * s - num.height / 2)))


def build_front():
    rgb, xx, yy = gradient(CW, CH, [(0.0, (255, 240, 175)), (0.30, (238, 199, 102)),
                                     (0.62, (205, 152, 55)), (1.0, (140, 92, 26))])
    stripes = (np.sin((xx + yy) / 11.0) > 0.86).astype(np.float32) * 14
    hi = np.exp(-(((xx - CW * 0.25) / 260) ** 2 + ((yy - CH * 0.12) / 240) ** 2)) * 45
    rgb = rgb + stripes[..., None] + hi[..., None]
    a = np.array(card_mask(CW, CH))
    arr = np.dstack([np.clip(rgb, 0, 255), a]).astype(np.uint8)
    img = Image.fromarray(arr, "RGBA")

    d = ImageDraw.Draw(img)
    d.polygon(shape(CW, CH, 22), outline=(120, 80, 20), width=5)
    d.polygon(shape(CW, CH, 32), outline=(255, 240, 190), width=2)

    draw_jersey(img, CW * 0.63, CH * 0.30, 165)

    lx = CW * 0.20
    blit_on = lambda im, x, y: img.alpha_composite(im, (int(x - im.width / 2), int(y - im.height / 2)))
    blit_on(timg(P["rating"], BLACK, 170, fill=INK), lx, CH * 0.165)
    blit_on(timg(P["pos"], XB, 56, fill=INK), lx, CH * 0.255)
    d.line([(lx - 55, CH * 0.295), (lx + 55, CH * 0.295)], fill=INK, width=3)
    fx, fy, fw, fh = lx - 52, CH * 0.315, 104, 68
    mini = Image.fromarray(np.clip(P["flag"](fw, fh), 0, 255).astype(np.uint8), "RGB")
    img.paste(mini, (int(fx), int(fy)))
    d.rectangle((fx, fy, fx + fw, fy + fh), outline=INK, width=2)
    d.line([(lx - 55, CH * 0.395), (lx + 55, CH * 0.395)], fill=INK, width=3)
    d.ellipse((lx - 46, CH * 0.415, lx + 46, CH * 0.415 + 92), fill=INK)
    d.polygon(star_pts(lx, CH * 0.415 + 46, 34, 14), fill=GOLD_L)

    ny = CH * 0.585
    blit_on(timg(P["card_name"], BLACK, 108, fill=INK, tracking=4), CW / 2, ny)
    d.line([(CW * 0.14, CH * 0.645), (CW * 0.86, CH * 0.645)], fill=INK, width=3)
    stats = P["card_stats"]
    for i, (v, lab) in enumerate(stats):
        col, row = i // 3, i % 3
        x = CW * (0.17 if col == 0 else 0.575)
        y = CH * (0.70 + row * 0.062)
        vi = timg(v, BLACK, 62, fill=INK)
        li = timg(lab, SEMI, 46, fill=INK)
        img.alpha_composite(vi, (int(x), int(y - vi.height / 2)))
        img.alpha_composite(li, (int(x + vi.width + 2), int(y - li.height / 2 + 4)))
    d.line([(CW * 0.5, CH * 0.675), (CW * 0.5, CH * 0.86)], fill=INK, width=3)
    blit_on(timg("ЛЕГЕНДА", XB, 34, fill=INK, tracking=8), CW / 2, CH * 0.905)
    return img


def build_back():
    rgb, xx, yy = gradient(CW, CH, [(0.0, (40, 36, 70)), (0.5, (16, 16, 38)), (1.0, (8, 8, 20))])
    a = np.array(card_mask(CW, CH))
    img = Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(img)
    d.polygon(shape(CW, CH, 22), outline=GOLD, width=6)
    z = timg(P["back_letter"], BLACK, 520, fill=GOLD, glow=10, gcol=(255, 190, 80), galpha=0.6)
    img.alpha_composite(z, (int(CW / 2 - z.width / 2), int(CH * 0.45 - z.height / 2)))
    return img


def build_glow(front):
    pad = 120
    g = Image.new("RGBA", (CW + 2 * pad, CH + 2 * pad), (0, 0, 0, 0))
    solid = Image.new("RGBA", front.size, (255, 200, 90, 255))
    solid.putalpha(front.getchannel("A"))
    g.alpha_composite(solid, (pad, pad))
    return g.filter(ImageFilter.GaussianBlur(45))


FRONT = build_front()
BACK = build_back()
GLOW = build_glow(FRONT)
FRONT_ARR = np.array(FRONT).astype(np.float32)
_cy, _cx = np.mgrid[0:CH, 0:CW].astype(np.float32)
SHEEN_U = _cx * 0.85 + _cy * 0.45
SHEEN_MAX = float(SHEEN_U.max())


def front_with_sheen(p):
    """p in [0,1] — положение блика; вне диапазона — без блика."""
    if p <= 0 or p >= 1:
        return FRONT
    pos = -150 + p * (SHEEN_MAX + 300)
    band = np.exp(-(((SHEEN_U - pos) / 55.0) ** 2)) * 150 + np.exp(-(((SHEEN_U - pos + 120) / 18.0) ** 2)) * 90
    arr = FRONT_ARR.copy()
    arr[..., :3] = np.clip(arr[..., :3] + band[..., None], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def sheen_at(t, starts, dur=0.9):
    for s in starts:
        if s <= t <= s + dur:
            return ease_in_out((t - s) / dur)
    return 0.0


def draw_card(canvas, cx, cy, scale, theta=0.0, alpha=1.0, sheen=0.0, glow=0.0):
    c = math.cos(theta)
    face = front_with_sheen(sheen) if c >= 0 else BACK
    sx = scale * max(0.02, abs(c))
    if glow > 0:
        blit(canvas, GLOW, cx, cy, scale=scale * 1.02, sx=sx * 1.02, alpha=alpha * glow)
    blit(canvas, face, cx, cy, scale=scale, sx=sx, alpha=alpha, hq=True)


# --------------------------------------------------------------------------- scene assets

BLOB_W = make_blob(900, (255, 255, 255), 2.2)
BLOB_G = make_blob(900, (255, 196, 90), 2.0)


FLAG_BASE = P["flag"](760, 500)


def flag_frame(t, w=760, h=500):
    xx = np.arange(w, dtype=np.float32)
    dy = 26 * np.sin(xx / 95 - t * 6) * (0.25 + xx / w)
    shade = 1 + 0.22 * np.cos(xx / 95 - t * 6)
    hh = h + 80
    yy = np.arange(hh, dtype=np.float32)[:, None] - 40 - dy[None, :]
    inside = (yy >= 0) & (yy < h)
    src = FLAG_BASE
    sy = np.clip(yy, 0, h - 1).astype(np.int32)
    arr = np.zeros((hh, w, 4), np.float32)
    arr[..., :3] = src[sy, np.arange(w)[None, :]] * shade[None, :, None]
    arr[..., 3] = inside * 255
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


def rounded_plate(w, h, fill=(18, 22, 48, 225), accent=GOLD, r=26):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), r, fill=fill, outline=(240, 200, 105, 110), width=2)
    d.rounded_rectangle((0, 0, 16, h - 1), 8, fill=accent)
    return im


CLUBS = P["clubs"]


def build_club_plate(years, name, games, goals):
    im = rounded_plate(940, 210)
    im.alpha_composite(timg(years, XB, 44, fill=GOLD, tracking=2), (46, 22))
    size = 74  # длинные названия уменьшаются, чтобы не наезжать на цифры
    while size > 44 and timg(name, BLACK, size, fill=WHITE).width > 560:
        size -= 2
    nm = timg(name, BLACK, size, fill=WHITE)
    im.alpha_composite(nm, (44, int(82 + (timg(name, BLACK, 74, fill=WHITE).height - nm.height) / 2)))
    g = timg(games, BLACK, 66, fill=WHITE)
    gl = timg("ИГР", SEMI, 28, fill=(190, 190, 210), tracking=3)
    s = timg(goals, BLACK, 66, fill=GOLD)
    sl = timg("ГОЛОВ", SEMI, 28, fill=(190, 190, 210), tracking=3)
    im.alpha_composite(g, (int(720 - g.width / 2), 30))
    im.alpha_composite(gl, (int(720 - gl.width / 2), 128))
    im.alpha_composite(s, (int(855 - s.width / 2), 30))
    im.alpha_composite(sl, (int(855 - sl.width / 2), 128))
    return im


CLUB_PLATES = [build_club_plate(*c) for c in CLUBS]

TROPHIES = P["trophies_ru"] if LANG == "ru" and P["trophies_ru"] else P["trophies_en"]


def build_trophy(big, l1, l2):
    w, h = 460, 330
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 30, fill=(20, 22, 50, 230), outline=(240, 200, 105, 150), width=3)
    d.polygon(star_pts(w / 2, 62, 36, 15), fill=GOLD_L)
    b = timg(big, BLACK, 92, fill=GOLD, glow=6, gcol=(255, 180, 60), galpha=0.5)
    im.alpha_composite(b, (int(w / 2 - b.width / 2), int(160 - b.height / 2)))
    t1 = timg(l1, XB, 36, fill=WHITE, tracking=2)
    t2 = timg(l2, XB, 36, fill=WHITE, tracking=2)
    im.alpha_composite(t1, (int(w / 2 - t1.width / 2), 222))
    im.alpha_composite(t2, (int(w / 2 - t2.width / 2), 264))
    return im


TROPHY_BADGES = [build_trophy(*x) for x in TROPHIES]

STATS = P["stats"]

# --------------------------------------------------------------------------- timeline

FLASHES = [(3.5, 1.0, 0.45), (5.3, 0.7, 0.35), (7.1, 0.7, 0.35), (9.0, 0.5, 0.3),
           (10.5, 0.9, 0.5), (20.0, 0.35, 0.3), (23.6, 0.35, 0.3), (27.0, 0.8, 0.5)]


def render(t):
    # ---- фон: оттенок по сцене
    if t < 3.5:
        tint, ray = (90, 70, 30), 0.25 + 0.6 * prog(t, 0, 3.5)
    elif t < 5.3:
        tint, ray = P["country_tint"], 0.9
    elif t < 7.1:
        tint, ray = (120, 90, 30), 0.9
    elif t < 9.0:
        tint, ray = P["club_tint"], 0.9
    else:
        tint, ray = (95, 72, 32), 0.75
    canvas = background(t, tint, ray)
    particles(canvas, t, 0.35 + 0.65 * prog(t, 0.5, 3.0))

    # ---- 1. Интро 0–3.5
    a = win(t, 0.2, 3.45, 0.6, 0.15)
    if a:
        p = ease_out(prog(t, 0.2, 1.4))
        blit(canvas, timg("ICON PACK", XB, 40, fill=(220, 220, 235), tracking=14), W / 2, 760, alpha=a * p)
        blit(canvas, timg("ЛЕГЕНДА", BLACK, 190, fill=GOLD_L, glow=18, gcol=(255, 170, 40), galpha=0.8),
             W / 2, 920, scale=1.25 - 0.25 * p, alpha=a * p)
        blit(canvas, timg("ВЫХОДИТ НА ПОЛЕ", BOLD, 46, fill=WHITE, tracking=12), W / 2, 1060,
             alpha=a * ease_out(prog(t, 0.9, 1.8)))
    if 2.2 < t < 3.6:
        p = prog(t, 2.2, 3.5)
        blit(canvas, BLOB_G, W / 2, 940, scale=0.3 + 2.4 * p ** 2, alpha=clamp(p * 1.3))
        blit(canvas, BLOB_W, W / 2, 940, scale=0.1 + 1.4 * p ** 3, alpha=clamp(p * 1.5))

    # ---- 2. Walkout: флаг / позиция / клуб
    a = win(t, 3.5, 5.3, 0.2, 0.2)
    if a:
        p = ease_out(prog(t, 3.5, 4.3))
        blit(canvas, BLOB_W, W / 2, 900, scale=1.6, alpha=0.18 * a)
        blit(canvas, flag_frame(t), W / 2, 860, scale=1.25 - 0.25 * p, alpha=a)
        blit(canvas, timg(P["country"], BLACK, 140, fill=WHITE, glow=14, gcol=P["country_glow"], galpha=0.7),
             W / 2, 1290, scale=1.15 - 0.15 * p, alpha=a)
    a = win(t, 5.3, 7.1, 0.2, 0.2)
    if a:
        p = ease_out(prog(t, 5.3, 6.1))
        blit(canvas, timg(P["pos"], BLACK, 330, fill=GOLD_L, glow=22, gcol=(255, 160, 30), galpha=0.8),
             W / 2, 880, scale=1.3 - 0.3 * p, alpha=a)
        blit(canvas, timg(P["pos_long"], BOLD, 42, fill=WHITE, tracking=8), W / 2, 1110,
             alpha=a * ease_out(prog(t, 5.6, 6.3)))
    a = win(t, 7.1, 9.0, 0.2, 0.25)
    if a:
        p = ease_out(prog(t, 7.1, 7.9))
        s = 1.25 - 0.25 * p
        for txt, size, y in P["club_lines"]:
            blit(canvas, timg(txt, BLACK, size, fill=WHITE, glow=18, gcol=P["club_glow"], galpha=0.7),
                 W / 2, y, scale=s, alpha=a)
        blit(canvas, timg(P["club_sub"], BOLD, 34, fill=GOLD, tracking=5), W / 2, 1170,
             alpha=a * ease_out(prog(t, 7.4, 8.1)))

    # ---- 3–4. Карточка: вылет, представление, характеристики
    if 9.0 <= t < 20.3:
        if t < 10.5:
            p = ease_out(prog(t, 9.0, 10.5))
            scale, cy, theta, alpha = 0.12 + 1.03 * p, 1350 - 390 * p, (1 - p) * 5 * math.pi, clamp((t - 9) * 4)
        elif t < 13.0:
            scale, cy, theta, alpha = 1.15, 960 + 10 * math.sin((t - 10.5) * 2.2), 0.0, 1.0
        else:
            p = ease_in_out(prog(t, 13.0, 13.8))
            scale, cy, theta, alpha = 1.15 - 0.53 * p, 960 + 10 * math.sin((t - 10.5) * 2.2) * (1 - p) - 400 * p, 0.0, 1.0
            if t > 19.6:
                q = ease_in_out(prog(t, 19.6, 20.3))
                cy -= 1100 * q
        glow = 0.55 + 0.25 * math.sin(t * 3)
        if 10.5 <= t < 11.3:
            glow += 1.2 * (1 - prog(t, 10.5, 11.3))
        draw_card(canvas, W / 2, cy, scale, theta, alpha, sheen_at(t, [10.6, 12.2, 15.0, 18.0]), glow)

        # ударная волна
        if 10.5 <= t < 11.4:
            p = prog(t, 10.5, 11.4)
            ring = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            r = 200 + 900 * ease_out(p)
            ImageDraw.Draw(ring).ellipse((W / 2 - r, 960 - r, W / 2 + r, 960 + r),
                                         outline=(255, 220, 140, int(220 * (1 - p))), width=int(4 + 24 * (1 - p)))
            canvas.paste(ring, (0, 0), ring)

        a = win(t, 11.0, 13.4, 0.5, 0.4)
        if a:
            blit(canvas, timg(P["full_name"], BLACK, 84, fill=WHITE, tracking=4, glow=10, gcol=(255, 180, 60), galpha=0.5),
                 W / 2, 190, alpha=a, scale=0.9 + 0.1 * ease_out(prog(t, 11.0, 11.7)))
            blit(canvas, timg(f'{P["nickname"]} · {P["rating"]} OVR', XB, 48, fill=GOLD, tracking=10), W / 2, 1745,
                 alpha=a * ease_out(prog(t, 11.4, 12.0)))

        # характеристики
        a = win(t, 13.5, 20.2, 0.4, 0.5)
        if a:
            drop = 0 if t < 19.6 else 300 * ease_in_out(prog(t, 19.6, 20.2))
            blit(canvas, timg("ХАРАКТЕРИСТИКИ", XB, 46, fill=GOLD, tracking=10), W / 2, 1000 + drop, alpha=a)
            d = ImageDraw.Draw(canvas)
            for i, (lab, val) in enumerate(STATS):
                st = 13.9 + i * 0.5
                if t < st:
                    continue
                pa = ease_out(prog(t, st, st + 0.35)) * a
                fill = ease_out(prog(t, st + 0.1, st + 0.9))
                y = 1100 + i * 128 + drop + (1 - pa) * 40
                blit(canvas, timg(lab, BOLD, 44, fill=WHITE, tracking=3), 90, y, alpha=pa, anchor="l")
                num = int(round(val * fill))
                col = (90, 230, 120) if val >= 90 else (200, 230, 90) if val >= 80 else (240, 200, 80)
                blit(canvas, timg(str(num), BLACK, 64, fill=col), 990, y - 4, alpha=pa, anchor="r")
                bx0, bx1, byy = 90, 990, y + 44
                k = pa
                d.rounded_rectangle((bx0, byy, bx1, byy + 16), 8, fill=tuple(int(v * k) for v in (45, 48, 75)))
                fx = bx0 + (bx1 - bx0) * val / 100 * fill
                if fx > bx0 + 16:
                    d.rounded_rectangle((bx0, byy, fx, byy + 16), 8, fill=tuple(int(v * k) for v in col))

    # ---- 5a. Карьера по клубам 20–23.6
    a = win(t, 20.0, 23.6, 0.3, 0.3)
    if a:
        blit(canvas, timg("КАРЬЕРА", BLACK, 120, fill=GOLD_L, glow=14, gcol=(255, 170, 40), galpha=0.6),
             W / 2, 270, alpha=a, scale=0.92 + 0.08 * ease_out(prog(t, 20, 20.6)))
        blit(canvas, timg("МАТЧИ И ГОЛЫ В ЧЕМПИОНАТАХ", BOLD, 32, fill=(200, 200, 220), tracking=6), W / 2, 380, alpha=a)
        for i, pl in enumerate(CLUB_PLATES):
            st = 20.3 + i * 0.35
            if t < st:
                continue
            p = ease_back(prog(t, st, st + 0.55), 1.2)
            x = W / 2 + (1 - p) * 1100
            blit(canvas, pl, x, 560 + i * 270, alpha=a * clamp(prog(t, st, st + 0.3)))
        blit(canvas, timg(P["national"], XB, 38, fill=WHITE, tracking=3), W / 2, 1700,
             alpha=a * ease_out(prog(t, 21.9, 22.5)))

    # ---- 5b. Трофеи 23.6–27
    a = win(t, 23.6, 27.0, 0.3, 0.25)
    if a:
        blit(canvas, timg("ТРОФЕИ", BLACK, 130, fill=GOLD_L, glow=14, gcol=(255, 170, 40), galpha=0.6),
             W / 2, 280, alpha=a, scale=0.92 + 0.08 * ease_out(prog(t, 23.6, 24.2)))
        for i, b in enumerate(TROPHY_BADGES):
            st = 23.9 + i * 0.3
            if t < st:
                continue
            p = ease_back(prog(t, st, st + 0.5))
            col, row = i % 2, i // 2
            blit(canvas, b, W / 2 + (-245 if col == 0 else 245), 620 + row * 370,
                 scale=max(0.01, p), alpha=a * clamp(prog(t, st, st + 0.2)))

    # ---- 6. Финал 27–30
    if t >= 27.0:
        p = ease_out(prog(t, 27.0, 28.0))
        blit(canvas, timg(P["nickname"], BLACK, 300, fill=(255, 210, 120), tracking=6), W / 2, 880,
             scale=1.15 - 0.1 * p, alpha=0.13 * p)
        draw_card(canvas, W / 2, 860 + 12 * math.sin(t * 2.2), 0.6 + 0.42 * p, (1 - p) * 2 * math.pi,
                  1.0, sheen_at(t, [27.9, 29.0], 0.8), 0.7 + 0.2 * math.sin(t * 3))
        blit(canvas, timg(P["full_name"], BLACK, 78, fill=WHITE, tracking=4), W / 2, 1520,
             alpha=ease_out(prog(t, 27.6, 28.3)))
        blit(canvas, timg(tr("ЛЕГЕНДА") + " · " + P["rating"], XB, 54, fill=GOLD, tracking=12, glow=8, gcol=(255, 170, 40), galpha=0.5),
             W / 2, 1615, alpha=ease_out(prog(t, 27.9, 28.6)))

    # ---- пост: вспышки, затемнение
    flash = 0.0
    for s, amp, dur in FLASHES:
        if s <= t < s + dur:
            flash = max(flash, amp * (1 - (t - s) / dur) ** 2)
    if 3.2 <= t < 3.5:
        flash = max(flash, prog(t, 3.2, 3.5))
    fade = min(prog(t, 0, 0.4), 1 - prog(t, 29.2, 30.0))
    if flash > 0.003 or fade < 0.999:
        arr = np.asarray(canvas).astype(np.float32)
        arr = arr * (1 - flash) + 255 * flash
        arr *= fade
        canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return canvas


# --------------------------------------------------------------------------- audio

SR = 44100


def lowpass(x, k):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def make_audio(path):
    n = int(SR * DUR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    rng = np.random.default_rng(3)

    def add(sig, start, gain=1.0):
        i = int(start * SR)
        if i >= n:
            return
        m = min(len(sig), n - i)
        out[i:i + m] += sig[:m] * gain

    def boom(d=1.6):
        tt = np.arange(int(d * SR)) / SR
        f = 40 + 70 * np.exp(-tt * 6)
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 2.8)
        s += lowpass(rng.standard_normal(len(tt)), 6) * np.exp(-tt * 14) * 0.5
        return s

    def kick():
        tt = np.arange(int(0.35 * SR)) / SR
        f = 45 + 110 * np.exp(-tt * 28)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 10)

    def hat():
        tt = np.arange(int(0.06 * SR)) / SR
        nz = rng.standard_normal(len(tt))
        return (nz - lowpass(nz, 4)) * np.exp(-tt * 70)

    def whoosh(d=0.8, rise=True):
        tt = np.arange(int(d * SR)) / SR
        nz = rng.standard_normal(len(tt))
        env = (tt / d) ** 2 if rise else np.sin(np.pi * tt / d)
        return (lowpass(nz, 3) * 0.6 + lowpass(nz, 12) * 0.8) * env

    def blip(f=1320, d=0.12):
        tt = np.arange(int(d * SR)) / SR
        return np.sin(2 * np.pi * f * tt) * np.exp(-tt * 30)

    # подложка: аккорд ля минор / фа мажор
    def pad_chord(freqs, a, b):
        i0, i1 = int(a * SR), int(b * SR)
        tt = t[i0:i1]
        env = np.clip((tt - a) / 1.2, 0, 1) * np.clip((b - tt) / 0.8, 0, 1)
        s = sum(np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 2.003 * f * tt) +
                0.15 * np.sin(2 * np.pi * 3.01 * f * tt) for f in freqs)
        out[i0:i1] += s * env * 0.035

    pad_chord([110, 164.81, 220, 261.63], 0.0, 9.6)
    pad_chord([87.31, 130.81, 174.61, 220.0], 9.0, 16.2)
    pad_chord([98.0, 146.83, 196.0, 246.94], 16.0, 23.2)
    pad_chord([110, 164.81, 220, 261.63, 329.63], 23.0, 30.0)

    # райзер в интро
    tt = t[: int(3.5 * SR)]
    sweep = np.sin(2 * np.pi * np.cumsum(200 + 900 * (tt / 3.5) ** 2) / SR) * (tt / 3.5) ** 3 * 0.18
    out[: len(tt)] += sweep
    add(whoosh(3.4), 0.1, 0.25)

    for s in [3.5, 5.3, 7.1, 10.5, 27.0]:
        add(boom(), s, 0.9)
    for s in [5.0, 6.8, 8.6, 20.0 - 0.4, 23.6 - 0.4, 26.6]:
        add(whoosh(0.5), s, 0.3)
    add(whoosh(1.4), 9.1, 0.25)

    # бит 120 BPM: 10.5–27
    bass_notes = {0: 43.65, 1: 43.65, 2: 49.0, 3: 55.0}
    beat = 0.5
    k = 0
    tb = 10.5
    while tb < 27.0:
        add(kick(), tb, 0.85)
        add(hat(), tb + beat / 2, 0.22)
        bar = int((tb - 10.5) // 2) % 4
        f = bass_notes[bar]
        ttb = np.arange(int(0.45 * SR)) / SR
        add(np.sin(2 * np.pi * f * 2 * ttb) * np.exp(-ttb * 5) * 0.5 + np.sin(2 * np.pi * f * 4 * ttb) * np.exp(-ttb * 8) * 0.15, tb, 0.6)
        k += 1
        tb += beat

    # арпеджио 13–27
    arp = [440, 523.25, 659.25, 880, 659.25, 523.25]
    i = 0
    ta = 13.0
    while ta < 27.0:
        tt = np.arange(int(0.2 * SR)) / SR
        f = arp[i % len(arp)]
        add((np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt)) * np.exp(-tt * 14), ta, 0.05)
        i += 1
        ta += 0.125

    for i in range(len(STATS)):
        add(blip(990 + i * 110), 13.9 + i * 0.5, 0.2)
    for i in range(4):
        add(blip(700 + i * 90, 0.15), 20.3 + i * 0.35, 0.18)
    for i in range(6):
        add(blip(1200 + i * 80, 0.15), 23.9 + i * 0.3, 0.18)

    # финальный аккорд-звон
    tt = np.arange(int(3.0 * SR)) / SR
    bell = sum(np.sin(2 * np.pi * f * tt) * np.exp(-tt * (1.5 + j * 0.3)) for j, f in enumerate([880, 1318.5, 1760, 2217]))
    add(bell, 27.0, 0.06)

    out *= np.clip((DUR - t) / 1.2, 0, 1)
    out = np.tanh(out * 1.3)
    out /= np.max(np.abs(out)) + 1e-9
    out *= 0.89
    pcm = (out * 32767).astype(np.int16)
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(stereo.tobytes())


# --------------------------------------------------------------------------- main


def main():
    if "--preview" in sys.argv:
        args = sys.argv[sys.argv.index("--preview") + 1:]
        times = [a for a in args if not a.startswith("--") and a not in ("ru", "en")]
        os.makedirs("preview", exist_ok=True)
        for s in times:
            render(float(s)).save(f"preview/{LANG}_frame_{float(s):05.2f}.png")
        return
    wav = OUT.replace(".mp4", ".wav")
    make_audio(wav)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        proc.stdin.write(render(i / FPS).tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{N}", flush=True)
    proc.stdin.close()
    proc.wait()
    os.remove(wav)
    print("done:", OUT)


if __name__ == "__main__":
    main()

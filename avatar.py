#!/usr/bin/env python3
"""Аватарки для Instagram (1080×1080) в стиле ролика: золотая карточка на тёмном фоне с лучами.

    python3 avatar.py  -> avatar/avatar_star.png, avatar_10.png, avatar_96.png
Содержимое держится в центральном круге — Instagram обрезает аватар кругом.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from render import (BLACK, GOLD_L, INK, XB, card_mask, gradient, make_blob, shape, star_pts, timg)

S = 1080
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avatar")


def background():
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dx, dy = xx - S / 2, yy - S / 2
    r = np.hypot(dx, dy) / (S * 0.5)
    ang = np.arctan2(dy, dx)
    out_c = np.array((8, 9, 24), np.float32)
    tint = np.array((120, 88, 34), np.float32)
    base = out_c + (tint - out_c) * np.exp(-(r ** 2) * 2.2)[..., None]
    rays = (0.5 + 0.5 * np.sin(ang * 12 + 0.4)) ** 6 * np.exp(-r * 1.3)
    base += rays[..., None] * tint * 1.3
    base *= np.clip(1.1 - 0.5 * r ** 2, 0.3, 1)[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(11)
    for _ in range(70):
        a, rr = rng.uniform(0, 2 * math.pi), rng.uniform(0.25, 0.48) * S
        x, y, sz = S / 2 + rr * math.cos(a), S / 2 + rr * math.sin(a), rng.uniform(2, 5)
        k = rng.uniform(0.4, 1)
        d.ellipse((x - sz, y - sz, x + sz, y + sz), fill=(int(255 * k), int(205 * k), int(110 * k)))
    return img


def gold_card(w, h):
    rgb, xx, yy = gradient(w, h, [(0.0, (255, 240, 175)), (0.30, (238, 199, 102)),
                                   (0.62, (205, 152, 55)), (1.0, (140, 92, 26))])
    stripes = (np.sin((xx + yy) / 9.0) > 0.86).astype(np.float32) * 14
    hi = np.exp(-(((xx - w * 0.25) / (w * 0.35)) ** 2 + ((yy - h * 0.12) / (h * 0.22)) ** 2)) * 50
    rgb = rgb + stripes[..., None] + hi[..., None]
    a = np.array(card_mask(w, h))
    img = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a]).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(img)
    d.polygon(shape(w, h, 18), outline=(120, 80, 20), width=6)
    d.polygon(shape(w, h, 28), outline=(255, 240, 190), width=3)
    return img


def compose(content):
    cw, ch = 500, 730
    card = gold_card(cw, ch)
    content(card, cw, ch)
    # диагональный блик
    arr = np.array(card).astype(np.float32)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    u = xx * 0.85 + yy * 0.45
    band = np.exp(-(((u - cw * 0.55) / 40.0) ** 2)) * 70
    arr[..., :3] = np.clip(arr[..., :3] + band[..., None], 0, 255)
    card = Image.fromarray(arr.astype(np.uint8), "RGBA")

    img = background()
    blob = make_blob(1000, (255, 196, 90), 2.0)
    img.paste(blob, (40, 40), blob)
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    solid = Image.new("RGBA", card.size, (255, 200, 90, 255))
    solid.putalpha(card.getchannel("A"))
    glow.paste(solid, ((S - cw) // 2, (S - ch) // 2 + 8), solid)
    glow = glow.filter(ImageFilter.GaussianBlur(40))
    img.paste(glow, (0, 0), glow)
    img.paste(glow, (0, 0), glow)
    img.paste(card, ((S - cw) // 2, (S - ch) // 2 + 8), card)
    return img


def centered(card, im, x, y):
    card.alpha_composite(im, (int(x - im.width / 2), int(y - im.height / 2)))


def c_star(card, w, h):
    d = ImageDraw.Draw(card)
    d.polygon(star_pts(w / 2, h * 0.43, 175, 72), fill=INK)
    d.polygon(star_pts(w / 2, h * 0.43, 140, 57), fill=GOLD_L)
    centered(card, timg("ЛЕГЕНДЫ", XB, 40, fill=INK, tracking=8), w / 2, h * 0.8)


def c_ten(card, w, h):
    centered(card, timg("10", BLACK, 330, fill=INK), w / 2, h * 0.42)
    ImageDraw.Draw(card).polygon(star_pts(w / 2, h * 0.79, 40, 16), fill=INK)


def c_96(card, w, h):
    centered(card, timg("96", BLACK, 230, fill=INK), w / 2, h * 0.30)
    d = ImageDraw.Draw(card)
    d.line([(w * 0.25, h * 0.50), (w * 0.75, h * 0.50)], fill=INK, width=5)
    d.ellipse((w / 2 - 80, h * 0.56, w / 2 + 80, h * 0.56 + 160), fill=INK)
    d.polygon(star_pts(w / 2, h * 0.56 + 80, 60, 24), fill=GOLD_L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    variants = {"star": c_star, "10": c_ten, "96": c_96}
    imgs = {}
    for name, fn in variants.items():
        im = compose(fn)
        im.save(os.path.join(OUT_DIR, f"avatar_{name}.png"))
        imgs[name] = im

    # превью: как выглядит в кружке Instagram (крупно и мелко)
    sheet = Image.new("RGB", (3 * 420, 560), (250, 250, 250))
    mask = Image.new("L", (360, 360), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, 359, 359), fill=255)
    small_mask = Image.new("L", (110, 110), 0)
    ImageDraw.Draw(small_mask).ellipse((0, 0, 109, 109), fill=255)
    for i, im in enumerate(imgs.values()):
        sheet.paste(im.resize((360, 360), Image.LANCZOS), (30 + i * 420, 30), mask)
        sheet.paste(im.resize((110, 110), Image.LANCZOS), (155 + i * 420, 420), small_mask)
    sheet.save(os.path.join(OUT_DIR, "preview_circle.png"))


if __name__ == "__main__":
    main()

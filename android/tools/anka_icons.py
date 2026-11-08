#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 EddizEge
# SPDX-License-Identifier: MIT
"""Writes Anka Native's icon resources from the Anka mark (anka_logo.py): the adaptive launcher icon's foreground and monochrome layers (drawable-<density>/
anka_fg.png, anka_mono.png), the legacy mipmaps (mipmap-<density>/ic_launcher.png, ic_launcher_round.png), the in-app mark (drawable-nodpi/anka_mark.png: the
header badge and the main screen's drifting background), the notification icon (drawable-nodpi/anka_notify.png, white) and android/images/logo.png.
Needs only Pillow. Usage: python3 android/tools/anka_icons.py (from anywhere; it writes into this checkout)."""
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anka_logo  # noqa: E402

RES = os.path.join(HERE, '..', 'app', 'src', 'main', 'res')
DENSITIES = {'mdpi': 1.0, 'hdpi': 1.5, 'xhdpi': 2.0, 'xxhdpi': 3.0, 'xxxhdpi': 4.0}
SAFE = 66 / 108     # the adaptive icon's safe circle: the bird stays inside it


def save(image, *parts):
    path = os.path.join(RES, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, optimize=True)
    return path


def layer(size, bird):
    """A 108 dp adaptive layer of `size` px with the bird (no tile) centred in the safe circle"""
    canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    inner = round(size * SAFE * 1.06)
    canvas.alpha_composite(bird(inner), ((size - inner) // 2, (size - inner) // 2))
    return canvas


def white(image):
    """The same shape in white (its alpha kept): monochrome and notification icons"""
    out = Image.new('RGBA', image.size, (255, 255, 255, 0))
    out.putalpha(image.getchannel('A'))
    return out


def round_icon(image):
    mask = Image.new('L', image.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, image.size[0] - 1, image.size[1] - 1), fill=255)
    tile = Image.new('RGBA', image.size, anka_logo.TILE_TOP + (255,))
    inner = round(image.size[0] * 0.8)   # inside the circle
    off = (image.size[0] - inner) // 2
    tile.alpha_composite(anka_logo.render(inner, tile=False), (off, off))
    out = Image.new('RGBA', image.size, (0, 0, 0, 0))
    out.paste(tile, (0, 0), mask)
    return out


def main():
    bird = lambda n: anka_logo.render(n, tile=False)   # noqa: E731
    for name, k in DENSITIES.items():
        fg = layer(round(108 * k), bird)
        print(save(fg, f'drawable-{name}', 'anka_fg.png'))
        print(save(white(fg), f'drawable-{name}', 'anka_mono.png'))
        n = round(48 * k)
        print(save(anka_logo.render(n), f'mipmap-{name}', 'ic_launcher.png'))
        print(save(round_icon(Image.new('RGBA', (n, n))), f'mipmap-{name}', 'ic_launcher_round.png'))
    print(save(bird(256), 'drawable-nodpi', 'anka_mark.png'))
    print(save(white(bird(96)), 'drawable-nodpi', 'anka_notify.png'))
    logo = os.path.join(HERE, '..', 'images', 'logo.png')
    anka_logo.render(256).save(logo, optimize=True)
    print(logo)


if __name__ == '__main__':
    main()

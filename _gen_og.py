#!/usr/bin/env python3
"""Render KernelDF and DataKernelBench Open Graph images with IBM Plex."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SANS = Path("/Library/Fonts/IBM-Plex-Sans")
MONO = Path("/Library/Fonts/IBM-Plex-Mono")

W, H = 1200, 630
SCALE = 2
SW, SH = W * SCALE, H * SCALE

PAPER = (247, 247, 242, 255)
INK = (17, 23, 19, 255)
MUTED = (94, 103, 97, 255)
LINE = (216, 222, 217, 255)
ACCENT = (8, 127, 91, 255)
GRID = (17, 23, 19, 10)


def px(value):
    return int(round(value * SCALE))


def load_font(path, size):
    return ImageFont.truetype(str(path), px(size))


def draw_tracked(draw, xy, text, typeface, fill, tracking_em=0):
    x, y = xy
    tracking = tracking_em * typeface.size
    for ch in text:
        draw.text((x, y), ch, font=typeface, fill=fill)
        x += draw.textlength(ch, font=typeface) + tracking
    return x


def text_width(draw, text, typeface, tracking_em=0):
    if not text:
        return 0
    tracking = tracking_em * typeface.size
    return sum(draw.textlength(ch, font=typeface) for ch in text) + tracking * (len(text) - 1)


def draw_grid(img):
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    step = px(28)
    for x in range(0, img.width, step):
        d.line([(x, 0), (x, img.height)], fill=GRID, width=SCALE)
    for y in range(0, img.height, step):
        d.line([(0, y), (img.width, y)], fill=GRID, width=SCALE)
    return Image.alpha_composite(img, overlay)


def draw_mark(draw, origin, size):
    ox, oy = origin
    s = size / 248.0

    def p(x, y):
        return (ox + x * s, oy + y * s)

    def r(v):
        return v * s

    for x, y in ((0, 0), (88, 0), (0, 88), (0, 176), (88, 176)):
        draw.rounded_rectangle([p(x, y), p(x + 72, y + 72)], radius=r(14), fill=INK)

    draw.polygon([p(176, 0), p(248, 0), p(176, 72)], fill=INK)
    draw.polygon([p(248, 88), p(248, 160), p(176, 124)], fill=INK)
    draw.polygon([p(176, 176), p(248, 248), p(176, 248)], fill=INK)

    cx, cy = p(124, 124)
    draw.ellipse([cx - r(36), cy - r(36), cx + r(36), cy + r(36)], fill=ACCENT)


def canvas():
    img = Image.new("RGBA", (SW, SH), PAPER)
    img = draw_grid(img)
    return img, ImageDraw.Draw(img)


def save(img, path):
    out = img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    out.save(path, "PNG", optimize=True)
    print(f"wrote {path} {out.size}")


def render_kerneldf():
    img, draw = canvas()
    sans_bold = load_font(SANS / "IBMPlexSans-Bold.otf", 124)
    sans_medium = load_font(SANS / "IBMPlexSans-Medium.otf", 32)
    mono = load_font(MONO / "IBMPlexMono-Medium.otf", 15)

    pad = px(80)
    mark_size = px(176)
    draw_mark(draw, (SW - pad - mark_size, (SH - mark_size) // 2), mark_size)

    y = px(168)
    draw_tracked(draw, (pad, y), "AI × DATA SYSTEMS × GPU KERNELS", mono, ACCENT, 0.1)
    y += px(46)
    draw_tracked(draw, (pad, y), "KernelDF", sans_bold, INK, -0.07)
    y += px(152)
    draw_tracked(draw, (pad, y), "Data processing, down to the kernel.", sans_medium, INK, -0.03)
    save(img, ROOT / "og.png")


def render_datakernelbench():
    img, draw = canvas()
    sans_bold = load_font(SANS / "IBMPlexSans-Bold.otf", 96)
    sans_question = load_font(SANS / "IBMPlexSans-Medium.otf", 36)

    pad = px(80)
    mark_size = px(168)
    draw_mark(draw, (SW - pad - mark_size, (SH - mark_size) // 2), mark_size)

    y = px(208)
    draw_tracked(draw, (pad, y), "DataKernelBench", sans_bold, INK, -0.07)
    y += px(128)
    draw_tracked(
        draw,
        (pad, y),
        "Can LLMs optimize database queries on GPUs?",
        sans_question,
        INK,
        -0.035,
    )
    save(img, ROOT / "datakernelbench" / "og.png")


if __name__ == "__main__":
    render_kerneldf()
    render_datakernelbench()

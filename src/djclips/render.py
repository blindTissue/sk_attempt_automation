"""Compose the still: cover A × cover B, title and artist beneath each."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .recognize import Track

FORMATS = {"9x16": (1080, 1920), "4x5": (1080, 1350)}  # Reels / feed portrait
COVER = 420
RADIUS = 28
TEXT_BLOCK = 280  # room reserved under the covers for title + artist

# (path, index) candidates per style; the first that loads wins.
_LATIN = {
    "bold": [("/System/Library/Fonts/HelveticaNeue.ttc", 1)],
    "regular": [("/System/Library/Fonts/HelveticaNeue.ttc", 0)],
    "thin": [("/System/Library/Fonts/HelveticaNeue.ttc", 12)],
}
_WIDE = {  # covers Hangul / CJK / other scripts Helvetica lacks
    "bold": [("/System/Library/Fonts/AppleSDGothicNeo.ttc", 6), ("/Library/Fonts/Arial Unicode.ttf", 0)],
    "regular": [("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0), ("/Library/Fonts/Arial Unicode.ttf", 0)],
}
_LINUX = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


@lru_cache(maxsize=None)
def font(style: str, size: int, wide: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (_WIDE.get(style, []) if wide else []) + _LATIN.get(style, []) + [(p, 0) for p in _LINUX]
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _needs_wide(text: str) -> bool:
    return any(ord(c) > 0x24F for c in text)


def _wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int, max_lines: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = word
        while draw.textlength(cur, font=fnt) > width:  # single word too long (or unspaced CJK)
            cut = len(cur)
            while cut > 1 and draw.textlength(cur[:cut], font=fnt) > width:
                cut -= 1
            lines.append(cur[:cut])
            cur = cur[cut:]
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(last + "…", font=fnt) > width:
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"
    return lines


def _placeholder(track: Track) -> Image.Image:
    img = Image.linear_gradient("L").resize((COVER, COVER))
    img = ImageOps.colorize(img, "#3a2e5c", "#c0587e")
    d = ImageDraw.Draw(img)
    initials = "".join(w[0] for w in track.title.split()[:2]).upper() or "?"
    f = font("bold", 160, _needs_wide(initials))
    d.text((COVER / 2, COVER / 2), initials, font=f, fill="white", anchor="mm")
    return img


def _rounded(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    img = ImageOps.fit(img, (COVER, COVER), Image.LANCZOS)
    mask = Image.new("L", (COVER, COVER), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, COVER - 1, COVER - 1), RADIUS, fill=255)
    return img, mask


def _background(a: Image.Image, b: Image.Image, W: int, H: int) -> Image.Image:
    left = ImageOps.fit(a, (W, H)).filter(ImageFilter.GaussianBlur(70))
    right = ImageOps.fit(b, (W, H)).filter(ImageFilter.GaussianBlur(70))
    blend = Image.linear_gradient("L").rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT).resize((W, H))
    bg = Image.composite(right, left, blend)
    shade = Image.new("RGB", (W, H), (8, 8, 12))
    return Image.blend(bg, shade, 0.62)


def render_card(a: Track, b: Track, cover_a: Image.Image | None, cover_b: Image.Image | None,
                out_path: Path, size: tuple[int, int] = FORMATS["9x16"]) -> Path:
    W, H = size
    margin = (W - 2 * COVER - 140) // 2
    top = (H - COVER - TEXT_BLOCK) // 2  # centre covers + text vertically
    cover_a = cover_a or _placeholder(a)
    cover_b = cover_b or _placeholder(b)
    canvas = _background(cover_a, cover_b, W, H).convert("RGBA")

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    xs = [margin, W - margin - COVER]
    for x in xs:
        sd.rounded_rectangle((x, top + 18, x + COVER, top + COVER + 18), RADIUS, fill=(0, 0, 0, 170))
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(24)))

    for x, cover in zip(xs, (cover_a, cover_b)):
        img, mask = _rounded(cover)
        canvas.paste(img, (x, top), mask)

    d = ImageDraw.Draw(canvas)
    d.text((W / 2, top + COVER / 2), "×", font=font("thin", 150), fill="white", anchor="mm")

    for x, track in zip(xs, (a, b)):
        cx, y = x + COVER / 2, top + COVER + 60
        tf = font("bold", 46, _needs_wide(track.title))
        for line in _wrap(d, track.title, tf, COVER, 3):
            d.text((cx, y), line, font=tf, fill="white", anchor="ma")
            y += 58
        y += 10
        af = font("regular", 36, _needs_wide(track.artist))
        for line in _wrap(d, track.artist, af, COVER, 2):
            d.text((cx, y), line, font=af, fill=(255, 255, 255, 175), anchor="ma")
            y += 46

    canvas.convert("RGB").save(out_path)
    return out_path

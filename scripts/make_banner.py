"""Generate the profile banner: particles "denoise" from scattered noise into my name.

Samples a dot grid inside the rendered name, gives every dot a random noisy
starting offset, and animates it home with plain CSS (no JS, so GitHub plays it).
A small counter ticks t = 1000 -> 0 like a diffusion sampler.

    python scripts/make_banner.py   # writes assets/banner-light.svg, assets/banner-dark.svg
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAME = "Ethan Jin"
FONT = ("/System/Library/Fonts/Avenir Next.ttc", 0)  # Avenir Next Bold
W, H = 1200, 300
FONT_SIZE = 168
STEP = 5.6       # dot grid spacing (viewBox units)
RADIUS = 2.0
T_TOTAL = 3.2    # seconds until everything has landed
SEED = 8

PALETTES = {
    "light": {"ink": "#1f2328", "noise": "#6b8cff", "muted": "#8b949e"},
    "dark": {"ink": "#e6edf3", "noise": "#5b7cfa", "muted": "#7d8590"},
}

OUT = Path(__file__).resolve().parent.parent / "assets"


def sample_targets():
    """Hex-offset dot grid clipped to the name's glyphs."""
    scale = 4
    font = ImageFont.truetype(FONT[0], FONT_SIZE * scale, index=FONT[1])
    mask = Image.new("L", (W * scale, H * scale), 0)
    draw = ImageDraw.Draw(mask)
    left, top, right, bottom = draw.textbbox((0, 0), NAME, font=font)
    x0 = (W * scale - (right - left)) / 2 - left
    y0 = (H * scale - (bottom - top)) / 2 - top - 8 * scale
    draw.text((x0, y0), NAME, font=font, fill=255)
    m = np.asarray(mask) > 127

    pts = []
    row_h = STEP * np.sqrt(3) / 2
    for r, y in enumerate(np.arange(0, H, row_h)):
        offset = STEP / 2 if r % 2 else 0
        for x in np.arange(offset, W, STEP):
            if m[int(y * scale), int(x * scale)]:
                pts.append((x, y))
    return np.array(pts)


def build(targets, rng):
    center = np.array([W / 2, H / 2 - 8])
    # x_T ~ pure noise: a wide gaussian cloud around the banner's center
    noise = rng.normal(0, 1, targets.shape) * np.array([W * 0.42, H * 0.55])
    starts = np.clip(center + noise, [-40, -40], [W + 40, H + 40])
    offsets = starts - targets

    dist = np.linalg.norm(offsets, axis=1)
    far = dist / dist.max()
    # outer particles travel longer and land last
    delays = rng.uniform(0, 0.35, len(targets)) + far * 0.25
    durations = 1.7 + far * 1.2
    durations = np.minimum(durations, T_TOTAL - delays)
    jitter = rng.normal(0, 7, targets.shape)
    return offsets, jitter, delays, durations


def counter_frames():
    """Diffusion-style timestep readout: ticks fast at first, slows near t = 0."""
    n = 24
    u = np.linspace(0, 1, n + 1)
    times = T_TOTAL * (1 - (1 - u) ** 1.6)
    values = np.round(1000 * (1 - u) / 10).astype(int) * 10
    return [(float(times[i]), float(times[i + 1] - times[i]), int(values[i])) for i in range(n)], float(times[-1])


def render(theme, targets, offsets, jitter, delays, durations):
    p = PALETTES[theme]
    frames, t_end = counter_frames()

    dots = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{RADIUS}" style="--x:{dx:.0f}px;--y:{dy:.0f}px;'
        f'--jx:{jx:.0f}px;--jy:{jy:.0f}px;--d:{d:.2f}s;--t:{t:.2f}s"/>'
        for (x, y), (dx, dy), (jx, jy), d, t in zip(targets, offsets, jitter, delays, durations)
    )
    ticks = "\n".join(
        f'<text class="k" style="--i:{start:.3f}s;--s:{dur:.3f}s">t = {v}</text>'
        for start, dur, v in frames
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{NAME}">
<title>{NAME}</title>
<style>
circle {{
  fill: {p["ink"]};
  animation: denoise var(--t) cubic-bezier(.22,.61,.25,1) var(--d) both;
}}
@keyframes denoise {{
  0% {{ transform: translate(var(--x), var(--y)); fill: {p["noise"]}; opacity: .28; }}
  55% {{ transform: translate(calc(var(--x) * .14 + var(--jx)), calc(var(--y) * .14 + var(--jy))); opacity: .75; }}
  100% {{ transform: translate(0, 0); fill: {p["ink"]}; opacity: 1; }}
}}
text {{
  font: 500 15px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  fill: {p["muted"]};
  text-anchor: end;
}}
.k {{ opacity: 0; animation: tick var(--s) steps(1) var(--i); }}
.done {{ opacity: 0; animation: tick .01s {t_end:.2f}s forwards; }}
@keyframes tick {{ 0%, 100% {{ opacity: 1; }} }}
@media (prefers-reduced-motion: reduce) {{
  circle, .done {{ animation: none; opacity: 1; }}
  .k {{ display: none; }}
}}
</style>
<g>
{dots}
</g>
<g transform="translate({W - 24} {H - 22})">
{ticks}
<text class="done">t = 0</text>
</g>
</svg>
"""


def main():
    rng = np.random.default_rng(SEED)
    targets = sample_targets()
    params = build(targets, rng)
    OUT.mkdir(exist_ok=True)
    for theme in PALETTES:
        path = OUT / f"banner-{theme}.svg"
        path.write_text(render(theme, targets, *params))
        print(f"{path.name}: {len(targets)} particles, {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

"""Hero banners for README headers: the project's own data rendered as glowing 'data art'."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

THEMES = {
    "sunset":  ("#0f0c29", "#302b63", "#24243e", ["#ff6a88", "#ffb86c", "#ffd36e", "#f78fb3"]),
    "ocean":   ("#03111f", "#0b3a5b", "#062238", ["#3ee8ff", "#4facfe", "#00f2a9", "#a0f1ff"]),
    "aurora":  ("#0b0f1f", "#1b1446", "#0d2b3a", ["#7cffcb", "#b388ff", "#53d8fb", "#ff9de2"]),
    "ember":   ("#140a05", "#3a1606", "#1f0d14", ["#ff7b00", "#ffb703", "#fb5607", "#ffd166"]),
    "neon":    ("#06040f", "#1a0638", "#04202a", ["#00f5d4", "#f15bb5", "#fee440", "#9b5de5"]),
}


def glow_line(ax, x, y, color, lw=2.0, layers=6, alpha=0.9, **kw):
    for i in range(layers, 0, -1):
        ax.plot(x, y, color=color, lw=lw + i * 2.2, alpha=0.035 * alpha, solid_capstyle="round", **kw)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round", **kw)


def glow_scatter(ax, x, y, color, s=6, alpha=0.6):
    ax.scatter(x, y, s=s * 9, color=color, alpha=alpha * 0.06, lw=0)
    ax.scatter(x, y, s=s * 3, color=color, alpha=alpha * 0.15, lw=0)
    ax.scatter(x, y, s=s, color=color, alpha=alpha, lw=0)


def make_banner(path, title, subtitle, tags, art, theme="aurora", emoji_free_kicker=None, size=(16, 5)):
    top, mid, bottom, accents = THEMES[theme]
    fig = plt.figure(figsize=size, dpi=110)
    bg = fig.add_axes([0, 0, 1, 1])
    grad = np.linspace(0, 1, 512)[None, :] * 0.7 + np.linspace(0, 1, 160)[:, None] * 0.3
    bg.imshow(grad, aspect="auto", cmap=LinearSegmentedColormap.from_list("bg", [top, mid, bottom]), extent=[0, 1, 0, 1])
    # soft colour blobs
    yy, xx = np.mgrid[0:1:160j, 0:1:512j]
    for (cx, cy, r), c in zip([(0.82, 0.75, 0.25), (0.62, 0.15, 0.22), (0.98, 0.2, 0.18)], accents):
        blob = np.exp(-(((xx - cx) * 3.2) ** 2 + ((yy - cy) / r * 0.35) ** 2) * 2.2)
        rgba = np.zeros((*blob.shape, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(c)
        rgba[..., 3] = blob * 0.22
        bg.imshow(rgba, aspect="auto", extent=[0, 1, 0, 1], origin="lower")
    bg.set_axis_off()

    art_ax = fig.add_axes([0.52, 0.07, 0.46, 0.86])
    art_ax.set_axis_off()
    art(art_ax, accents)

    # left text panel with a fade so the art never fights the title
    fade = np.linspace(1, 0, 256)[None, :] ** 1.6
    rgba = np.zeros((1, 256, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(top)
    rgba[..., 3] = fade * 0.92
    shade = fig.add_axes([0, 0, 0.66, 1])
    shade.imshow(rgba, aspect="auto", extent=[0, 1, 0, 1])
    shade.set_axis_off()

    if emoji_free_kicker:
        fig.text(0.045, 0.80, emoji_free_kicker.upper(), color=accents[0], fontsize=12, fontweight="bold", family="DejaVu Sans")
    fig.text(0.045, 0.56, title, color="white", fontsize=36, fontweight="bold", family="DejaVu Sans", va="center")
    fig.text(0.045, 0.39, subtitle, color="#d9dcef", fontsize=15, family="DejaVu Sans", va="center")
    x = 0.045
    renderer = fig.canvas.get_renderer()
    for i, t in enumerate(tags):
        txt = fig.text(x + 0.012, 0.175, t, color="white", fontsize=12, va="center", fontweight="bold")
        w = txt.get_window_extent(renderer).width / fig.bbox.width + 0.024
        fig.patches.append(FancyBboxPatch((x, 0.13), w, 0.09, boxstyle="round,pad=0,rounding_size=0.04",
                                          transform=fig.transFigure, facecolor=accents[i % len(accents)], alpha=0.18,
                                          edgecolor=accents[i % len(accents)], lw=1.3))
        x += w + 0.012
    fig.savefig(path, dpi=110)
    plt.close(fig)


AI_PROMPT_NOTE = """If you prefer an AI-generated illustration, generate a 1600x500 image and save it as `assets/banner.png`."""

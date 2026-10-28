"""Shared plotting style and figure export."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text

from . import config as cfg

# Colour-vision-deficiency-checked categorical order (fixed, never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
NEUTRAL = "#b9b8b0"         # de-emphasised marks
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# Single-hue sequential ramp (light = low, dark = high) for rates and magnitudes.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
PLOTLY_SEQ = [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)]

# Guard against leftover draft section labels in published figures.
_BANNED = re.compile(r"\b(module|enhancement|insight\s*\d)\b", re.I)


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "font.family": "DejaVu Sans",
    })


def suptitle(fig, title: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)


def save_fig(fig, name: str, dpi: int = 150) -> Path:
    """Save to ``images/<name>.png`` after checking no draft labels slipped in."""
    for artist in fig.findobj(Text):
        if _BANNED.search(artist.get_text() or ""):
            raise ValueError(f"Figure text contains a draft label: {artist.get_text()!r}")
    if _BANNED.search(name):
        raise ValueError(f"File name contains a draft label: {name!r}")
    cfg.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.IMAGES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved {path.relative_to(cfg.ROOT_DIR)}")
    return path

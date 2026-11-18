"""Shared figure style: Okabe-Ito colours (colour-blind safe) and light axes."""

import matplotlib.pyplot as plt

COLORS = {"none": "#333333", "BES": "#0072B2", "CAA": "#E69F00", "prompt": "#009E73",
          "LEACE": "#CC79A7", "mean-diff": "#56B4E9", "supervised": "#D55E00", "oracle": "#999999"}
INK = "#333333"


def style(ax, grid_axis="both"):
    ax.grid(alpha=0.25, lw=0.6, axis=grid_axis)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def subplots(*args, **kwargs):
    kwargs.setdefault("dpi", 140)
    return plt.subplots(*args, **kwargs)

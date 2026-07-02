"""
_ts_helpers.py
─────────────────────────────────────────────────────────────────────────────
Shared plotting helpers for time series components.

Provides
────────
- ``draw_eo_line``    : vertical EO cutoff line with annotation
- ``shade_pre_post``  : pre/post EO ``axvspan`` shading

These helpers exist because the TS notebook cells repeat the same
EO-line + shading pattern. Keep this module private (leading underscore)
since it is an implementation detail of the TS components.
─────────────────────────────────────────────────────────────────────────────
"""
from typing import Optional

import pandas as pd

from constants import EO_DATE
from src.tufte_style import COLORS


def draw_eo_line(
    ax,
    eo_date: pd.Timestamp = EO_DATE,
    label: str = "EO (Jul 23 2025)",
    annotation: str = "  EO signed\n  Jul 23 2025",
    annotation_va: str = "top",
) -> None:
    """
    Draw a dashed vertical line at the EO date with a text annotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    eo_date : pandas.Timestamp, optional
        The Executive Order date. Defaults to the project-wide ``EO_DATE``.
    label : str, optional
        Legend label for the vertical line.
    annotation : str, optional
        Text to render next to the line.
    annotation_va : {"top", "bottom"}, optional
        Vertical alignment of the annotation. ``"top"`` anchors near the
        top of the axes; ``"bottom"`` anchors near the bottom (useful for
        cumulative plots where the upper area is busy).

    Returns
    -------
    None
        Mutates ``ax`` in place.
    """
    ax.axvline(
        eo_date,
        color=COLORS["highlight"],
        linestyle="--",
        linewidth=1.5,
        label=label,
        zorder=5,
    )

    y_lo, y_hi = ax.get_ylim()
    y_text = y_hi * 0.95 if annotation_va == "top" else y_lo + (y_hi - y_lo) * 0.05

    ax.text(
        eo_date,
        y_text,
        annotation,
        color=COLORS["highlight"],
        fontsize=9,
        va=annotation_va,
    )


def shade_pre_post(
    ax,
    start: pd.Timestamp,
    end: pd.Timestamp,
    eo_date: pd.Timestamp = EO_DATE,
    alpha: float = 0.05,
) -> None:
    """
    Shade pre- and post-EO regions on a time series axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    start : pandas.Timestamp
        Left edge of the pre-EO shaded region (typically the data minimum).
    end : pandas.Timestamp
        Right edge of the post-EO shaded region (typically the data maximum).
    eo_date : pandas.Timestamp, optional
        The Executive Order date dividing the two regions.
    alpha : float, optional
        Shading transparency.

    Returns
    -------
    None
        Mutates ``ax`` in place.
    """
    ax.axvspan(start, eo_date, alpha=alpha, color=COLORS["pre_eo"])
    ax.axvspan(eo_date, end,   alpha=alpha, color=COLORS["post_eo"])

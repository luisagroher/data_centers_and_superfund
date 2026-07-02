"""
dc_near_sf_thresholds.py
─────────────────────────────────────────────────────────────────────────────
Component 2a — Number of data centers within 0.5, 1, 3, 5 miles of any
Superfund site. Mirrors Dashboard notebook (04) Component 2a.
─────────────────────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import streamlit as st

from constants import DISTANCE_THRESHOLDS_MI
from src.tufte_style import COLORS


def render(dc_gdf) -> None:
    """
    Render a bar chart of DC counts within each distance threshold.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``near_sf_{X}mi`` binary columns
        for each threshold in ``DISTANCE_THRESHOLDS_MI`` (e.g.
        ``near_sf_0_5mi``, ``near_sf_1mi``, …).

    Returns
    -------
    None
        Writes the chart directly into the current Streamlit page.
    """
    threshold_cols = [
        f"near_sf_{str(t).replace('.', '_')}mi"
        for t in DISTANCE_THRESHOLDS_MI
    ]
    labels = [
        f"{t} mile{'s' if t > 1 else ''}"
        for t in DISTANCE_THRESHOLDS_MI
    ]
    counts = [int(dc_gdf[col].sum()) for col in threshold_cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        labels,
        counts,
        color=COLORS["proposed"],
        alpha=0.85,
        edgecolor="white",
        width=0.5,
    )
    ax.set_title(
        "Data Centers Within X Miles of a Superfund Site",
        fontsize=14, pad=12,
    )
    ax.set_xlabel("Distance Threshold")
    ax.set_ylabel("Number of Data Centers")

    fig.tight_layout()
    st.pyplot(fig)

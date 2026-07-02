"""
state_counts.py
─────────────────────────────────────────────────────────────────────────────
Component 1 — Count of Superfund sites and data centers by state.

Grouped bar chart, one pair of bars per state, restricted to states that
appear in both datasets. Mirrors Dashboard notebook (04) Component 1.
─────────────────────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.tufte_style import COLORS


def render(dc_gdf, sf_gdf) -> None:
    """
    Render a grouped bar chart of DC and Superfund site counts by state.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain a ``state`` column (2-letter code).
    sf_gdf : geopandas.GeoDataFrame
        Superfund site polygons. Must contain a ``STATE_CODE`` column
        (whitespace-stripped by ``data_loader``).

    Returns
    -------
    None
        Writes the chart directly into the current Streamlit page.
    """
    dc_by_state = dc_gdf["state"].value_counts().rename("Data Centers")
    sf_by_state = sf_gdf["STATE_CODE"].value_counts().rename("Superfund Sites")

    state_df = pd.DataFrame({
        "Data Centers":    dc_by_state,
        "Superfund Sites": sf_by_state,
    }).fillna(0).astype(int)

    state_df = state_df[
        (state_df["Data Centers"]    > 0) &
        (state_df["Superfund Sites"] > 0)
    ].sort_values("Data Centers", ascending=False)

    fig, ax = plt.subplots(figsize=(18, 6))
    x     = np.arange(len(state_df))
    width = 0.4

    ax.bar(
        x - width / 2,
        state_df["Data Centers"],
        width,
        label="Data Centers",
        color=COLORS["proposed"],
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        state_df["Superfund Sites"],
        width,
        label="Superfund Sites",
        color=COLORS["superfund"],
        alpha=0.85,
    )

    ax.set_title("Data Centers & Superfund Sites by State", fontsize=14, pad=12)
    ax.set_xlabel("State")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(state_df.index, rotation=90, fontsize=8)
    ax.legend(loc="upper right")

    fig.tight_layout()
    st.pyplot(fig)

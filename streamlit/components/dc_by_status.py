"""
dc_by_status.py
─────────────────────────────────────────────────────────────────────────────
Component 7 — Number of data centers in each dashboard status category.

This component IS the status breakdown, so no status toggle is provided.
Bars are colored using ``DASHBOARD_STATUS_PALETTE`` and ordered by the
palette's canonical sequence (Proposed → Under Construction → Operational
→ Expanding → Inactive).
─────────────────────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import streamlit as st

from constants import DASHBOARD_STATUS_PALETTE


def render(dc_gdf) -> None:
    """
    Render a horizontal bar chart of data center counts by dashboard status.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain a ``dashboard_status`` column.

    Returns
    -------
    None
        Writes the chart directly into the current Streamlit page.
    """
    present = set(dc_gdf["dashboard_status"].dropna().unique())
    ordered = [s for s in DASHBOARD_STATUS_PALETTE.keys() if s in present]

    counts = (
        dc_gdf["dashboard_status"]
        .value_counts()
        .reindex(ordered)
        .fillna(0)
        .astype(int)
    )
    colors = [DASHBOARD_STATUS_PALETTE[s] for s in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        counts.index, counts.values,
        color=colors, alpha=0.85, edgecolor="white",
    )

    ax.set_title("Number of Data Centers by Status", fontsize=14, pad=12)
    ax.set_xlabel("Number of Data Centers")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)

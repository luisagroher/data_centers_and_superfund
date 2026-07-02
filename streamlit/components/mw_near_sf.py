"""
mw_near_sf.py
─────────────────────────────────────────────────────────────────────────────
Component 6 — Total MW capacity of data centers within 1 mile of each
Superfund site, grouped by ``nearest_sf_name``.

When status is ``"All"``, all statuses contribute (single-color bars).
When a specific status is selected, only that slice is shown, colored
from ``DASHBOARD_STATUS_PALETTE``. Only DCs with ``near_sf_1mi == 1`` and
non-null ``mw`` are included.

Note
────
Distance threshold is hardcoded to 1 mile to match the notebook. Could be
parameterized later if reviewers want it user-toggleable.
─────────────────────────────────────────────────────────────────────────────
"""
from typing import Optional

import matplotlib.pyplot as plt
import streamlit as st

from constants import DASHBOARD_STATUS_PALETTE
from src.tufte_style import COLORS

ALL_STATUSES = "All"


def render(dc_gdf, status: Optional[str] = None) -> None:
    """
    Render a horizontal bar chart of total MW per Superfund site, for
    data centers within 1 mile of that site.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``near_sf_1mi``, ``mw``,
        ``nearest_sf_name``, and ``dashboard_status`` columns.
    status : str, optional
        Filter status. ``"All"`` includes all statuses; any individual
        status in ``DASHBOARD_STATUS_PALETTE`` shows only that slice.
        If ``None``, an interactive radio selector is shown.

    Returns
    -------
    None
        Writes the chart directly into the current Streamlit page.
    """
    status_options = [ALL_STATUSES] + list(DASHBOARD_STATUS_PALETTE.keys())

    if status is None:
        status = st.radio(
            "Status",
            status_options,
            horizontal=True,
            key="mw_near_sf",
        )

    mask = (dc_gdf["near_sf_1mi"] == 1) & (dc_gdf["mw"].notnull())
    if status != ALL_STATUSES:
        mask &= (dc_gdf["dashboard_status"] == status)

    near_sf_mw = (
        dc_gdf.loc[mask]
        .groupby("nearest_sf_name")["mw"]
        .sum()
        .sort_values(ascending=True)
    )

    if near_sf_mw.empty:
        st.info(
            f"No data centers within 1 mile of a Superfund site "
            f"for status: {status}"
        )
        return

    color = (
        DASHBOARD_STATUS_PALETTE[status]
        if status != ALL_STATUSES
        else COLORS["proposed"]
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(
        near_sf_mw.index, near_sf_mw.values,
        color=color, alpha=0.85, edgecolor="white",
    )

    ax.set_title(
        "Total MW of Data Centers Within 1 Mile of Superfund Sites",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Total MW Capacity")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)

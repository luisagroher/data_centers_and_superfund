"""
top_sf_by_nearby_dc.py
─────────────────────────────────────────────────────────────────────────────
Component 4 — Top N Superfund sites by number of data centers within 1
mile, stacked by status group or filtered to a single status group.

Status taxonomy
───────────────
This component uses the pre-computed ``pipeline_count_within_1mi`` and
``operating_count_within_1mi`` columns on the Superfund GeoDataFrame.
These are based on the upstream ``status_group`` mapping
(In Pipeline / Operating / Inactive), a coarser taxonomy than the
``dashboard_status`` used elsewhere in the app. ``"In Pipeline"`` collapses
Proposed, Under Construction, and Expanding. Once the upstream pipeline is
updated to match ``dashboard_status``, the taxonomy split goes away.
─────────────────────────────────────────────────────────────────────────────
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.tufte_style import COLORS

ALL_STATUSES = "All"
IN_PIPELINE  = "In Pipeline"
OPERATING    = "Operating"

STATUS_GROUP_COLORS = {
    IN_PIPELINE: COLORS["proposed"],
    OPERATING:   COLORS["operating"],
}


def render(sf_gdf, status: Optional[str] = None, top_n: int = 15) -> None:
    """
    Render a horizontal bar chart of the top-N Superfund sites by nearby
    data center count (within 1 mile).

    Parameters
    ----------
    sf_gdf : geopandas.GeoDataFrame
        Superfund site polygons. Must contain ``SITE_NAME``, ``STATE_CODE``,
        ``dc_count_within_1mi``, ``pipeline_count_within_1mi``, and
        ``operating_count_within_1mi`` columns.
    status : str, optional
        Filter status group. ``"All"`` stacks In Pipeline + Operating;
        ``"In Pipeline"`` and ``"Operating"`` show single-color bars.
        If ``None``, an interactive radio selector is shown.
    top_n : int, default 15
        Number of top sites to display.

    Returns
    -------
    None
        Writes the chart directly into the current Streamlit page.
    """
    status_options = [ALL_STATUSES, IN_PIPELINE, OPERATING]

    if status is None:
        status = st.radio(
            "Status group",
            status_options,
            horizontal=True,
            key="top_sf_by_nearby_dc",
        )

    sites = sf_gdf.loc[sf_gdf["dc_count_within_1mi"] > 0].copy()
    if sites.empty:
        st.info("No Superfund sites have any data center within 1 mile.")
        return

    sites = sites.sort_values("dc_count_within_1mi", ascending=True).tail(top_n)
    sites["site_label"] = (
        sites["SITE_NAME"].str[:35]
        + " (" + sites["STATE_CODE"].str.strip() + ")"
    )

    y     = np.arange(len(sites))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    if status == ALL_STATUSES:
        ax.barh(
            y, sites["pipeline_count_within_1mi"], width,
            label=f"{IN_PIPELINE} (Proposed + Under Construction + Expanding)",
            color=STATUS_GROUP_COLORS[IN_PIPELINE],
            alpha=0.85, edgecolor="white",
        )
        ax.barh(
            y, sites["operating_count_within_1mi"], width,
            left=sites["pipeline_count_within_1mi"],
            label=OPERATING,
            color=STATUS_GROUP_COLORS[OPERATING],
            alpha=0.85, edgecolor="white",
        )
        ax.legend(loc="lower right")
    elif status == IN_PIPELINE:
        ax.barh(
            y, sites["pipeline_count_within_1mi"], width,
            color=STATUS_GROUP_COLORS[IN_PIPELINE],
            alpha=0.85, edgecolor="white",
        )
    else:  # OPERATING
        ax.barh(
            y, sites["operating_count_within_1mi"], width,
            color=STATUS_GROUP_COLORS[OPERATING],
            alpha=0.85, edgecolor="white",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(sites["site_label"], fontsize=9)
    ax.set_title(
        f"Top {top_n} Superfund Sites by Nearby Data Center Count "
        f"(within 1 mile)",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Number of Data Centers")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)

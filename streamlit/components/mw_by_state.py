"""
mw_by_state.py
─────────────────────────────────────────────────────────────────────────────
Component 5 — Top N states by total data center MW capacity.

When status is ``"All"``, bars are stacked by ``dashboard_status``. When a
specific status is selected, bars are single-colored. Only rows with
non-null ``mw`` are included.

Fixes a notebook bug where the stacking accumulator referenced
``top_cities`` instead of ``top_states``.
─────────────────────────────────────────────────────────────────────────────
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from constants import DASHBOARD_STATUS_PALETTE

ALL_STATUSES = "All"


def render(
    dc_gdf,
    status: Optional[str] = None,
    top_n: int = 20,
) -> None:
    """
    Render a horizontal bar chart of the top-N states by total MW.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``state``, ``dashboard_status``,
        and ``mw`` columns.
    status : str, optional
        Filter status. ``"All"`` stacks bars by status; any individual
        status in ``DASHBOARD_STATUS_PALETTE`` shows only that slice.
        If ``None``, an interactive radio selector is shown.
    top_n : int, default 20
        Number of top states to display.

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
            key="mw_by_state",
        )

    df = dc_gdf.loc[~dc_gdf["mw"].isna()]
    fig, ax = plt.subplots(figsize=(8, 6))

    if status == ALL_STATUSES:
        states = (
            df.groupby(["state", "dashboard_status"])["mw"]
            .sum()
            .unstack(fill_value=0)
        )
        ordered = [
            c for c in DASHBOARD_STATUS_PALETTE.keys() if c in states.columns
        ]
        states = states[ordered]
        states["__total"] = states.sum(axis=1)
        states = (
            states.sort_values("__total", ascending=True)
            .drop(columns="__total")
            .tail(top_n)
        )

        left = np.zeros(len(states))
        for group in ordered:
            ax.barh(
                states.index, states[group], left=left, label=group,
                color=DASHBOARD_STATUS_PALETTE[group],
                alpha=0.85, edgecolor="white",
            )
            left += states[group].values
        ax.legend(loc="lower right")

    else:
        filtered = df.loc[df["dashboard_status"] == status]
        sums = (
            filtered.groupby("state")["mw"].sum()
            .sort_values(ascending=True)
            .tail(top_n)
        )
        sums = sums[sums > 0]
        if sums.empty:
            st.info(f"No data centers with MW data for status: {status}")
            return
        ax.barh(
            sums.index, sums.values,
            color=DASHBOARD_STATUS_PALETTE[status],
            alpha=0.85, edgecolor="white",
        )

    ax.set_title(
        f"Top {top_n} States by Data Center MW Capacity",
        fontsize=14, pad=12,
    )
    ax.set_xlabel("Total MW Capacity")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)

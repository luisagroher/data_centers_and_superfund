"""
top_cities_by_dc.py
─────────────────────────────────────────────────────────────────────────────
Component 3 — Top N cities by data center count.

When status is ``"All"``, bars are stacked by ``dashboard_status`` using
``DASHBOARD_STATUS_PALETTE``. When a specific status is selected, bars are
single-colored and show only that slice.
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
    Render a horizontal bar chart of the top-N cities by data center count.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``city``, ``state``, and
        ``dashboard_status`` columns.
    status : str, optional
        Filter status. ``"All"`` stacks bars by status; any individual
        status in ``DASHBOARD_STATUS_PALETTE`` shows only that slice.
        If ``None``, an interactive radio selector is shown.
    top_n : int, default 20
        Number of top cities to display.

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
            key="top_cities_by_dc",
        )

    df = dc_gdf.copy()
    df["city_state"] = (
        df["city"].str.strip() + ", " + df["state"].str.strip()
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    if status == ALL_STATUSES:
        cities = (
            df.groupby("city_state")["dashboard_status"]
            .value_counts()
            .unstack(fill_value=0)
        )
        ordered = [
            c for c in DASHBOARD_STATUS_PALETTE.keys() if c in cities.columns
        ]
        cities = cities[ordered]
        cities["__total"] = cities.sum(axis=1)
        cities = (
            cities.sort_values("__total", ascending=True)
            .drop(columns="__total")
            .tail(top_n)
        )

        left = np.zeros(len(cities))
        for group in ordered:
            ax.barh(
                cities.index, cities[group], left=left, label=group,
                color=DASHBOARD_STATUS_PALETTE[group],
                alpha=0.85, edgecolor="white",
            )
            left += cities[group].values

        for i, (_, row) in enumerate(cities.iterrows()):
            ax.text(
                row.sum() + 0.3, i, f"{int(row.sum()):,}",
                va="center", fontsize=9,
            )
        ax.legend(loc="lower right")

    else:
        filtered = df.loc[df["dashboard_status"] == status]
        counts = (
            filtered["city_state"].value_counts()
            .sort_values(ascending=True)
            .tail(top_n)
        )
        if counts.empty:
            st.info(f"No data centers found with status: {status}")
            return
        ax.barh(
            counts.index, counts.values,
            color=DASHBOARD_STATUS_PALETTE[status],
            alpha=0.85, edgecolor="white",
        )

    ax.set_title(
        f"Top {top_n} Cities by Data Center Count",
        fontsize=14, pad=12,
    )
    ax.set_xlabel("Number of Data Centers")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)

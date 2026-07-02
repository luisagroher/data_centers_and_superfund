"""
top_states_by_status.py
─────────────────────────────────────────────────────────────────────────────
Components 9 / 10 / 10b — Top N states by data center count, filtered to a
single dashboard status. Parameterized version of three near-identical
notebook cells (Proposed, Operational, Under Construction).

The user toggles status via a Streamlit radio. Any status present in
``DASHBOARD_STATUS_PALETTE`` is selectable; bar color follows the palette.
─────────────────────────────────────────────────────────────────────────────
"""
from typing import Optional

import matplotlib.pyplot as plt
import streamlit as st

from constants import DASHBOARD_STATUS_PALETTE
from src.tufte_style import COLORS


def render(
    dc_gdf,
    status: Optional[str] = None,
    top_n: int = 15,
) -> None:
    """
    Render a horizontal bar chart of the top-N states by DC count for a
    single dashboard status.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``state`` and ``dashboard_status``
        columns.
    status : str, optional
        Dashboard status to filter on (e.g. ``"Proposed"``,
        ``"Operational"``, ``"Under Construction"``). If ``None``, an
        interactive radio selector is shown and the user picks one.
    top_n : int, default 15
        Number of top states to display.

    Returns
    -------
    None
        Writes the chart directly into the current Streamlit page.
    """
    status_options = list(DASHBOARD_STATUS_PALETTE.keys())

    if status is None:
        status = st.radio(
            "Status",
            status_options,
            horizontal=True,
            key="top_states_by_status",
        )

    filtered = dc_gdf.loc[dc_gdf["dashboard_status"] == status, "state"]
    counts   = filtered.value_counts().sort_values(ascending=True).tail(top_n)

    if counts.empty:
        st.info(f"No data centers found with status: {status}")
        return

    color = DASHBOARD_STATUS_PALETTE.get(status, COLORS["proposed"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        counts.index,
        counts.values,
        color=color,
        alpha=0.85,
        edgecolor="white",
    )

    ax.set_title(
        f"Top {top_n} States — {status} Data Centers",
        fontsize=14, pad=12,
    )
    ax.set_xlabel("Number of Data Centers")
    ax.set_ylabel("")

    fig.tight_layout()
    st.pyplot(fig)

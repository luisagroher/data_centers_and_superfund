"""
proposals_over_time.py
─────────────────────────────────────────────────────────────────────────────
TS Component 1 / 1b — Monthly data center counts over time with the EO
cutoff marked.

Merged version of two near-identical notebook cells:
- "All" shows grouped bars broken out by dashboard status.
- A specific status collapses to a single-color bar chart (matches TS 1b
  when "Proposed" is selected).
─────────────────────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from components._ts_helpers import draw_eo_line, shade_pre_post
from constants import DASHBOARD_STATUS_PALETTE


def render(dc_gdf) -> None:
    """
    Render monthly data center counts over time with the EO cutoff marked.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``date_created`` and
        ``dashboard_status`` columns.

    Returns
    -------
    None
        Writes the chart directly into the current Streamlit page.
    """
    status_options = ["All"] + list(DASHBOARD_STATUS_PALETTE.keys())
    status = st.radio(
        "Status",
        status_options,
        horizontal=True,
        key="proposals_over_time",
    )

    dc = dc_gdf.dropna(subset=["date_created"]).copy()

    monthly = (
        dc.groupby([dc["date_created"].dt.to_period("M"), "dashboard_status"])
        .size()
        .unstack(fill_value=0)
    )
    ordered_cols = [c for c in DASHBOARD_STATUS_PALETTE.keys() if c in monthly.columns]
    monthly = monthly[ordered_cols]
    monthly.index = monthly.index.to_timestamp()

    if monthly.empty:
        st.info("No data centers with valid creation dates.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    if status == "All":
        n_status  = len(ordered_cols)
        bar_width = 15  # days
        offsets   = np.linspace(
            -(n_status - 1) * bar_width / 2,
              (n_status - 1) * bar_width / 2,
            n_status,
        )
        for s, offset in zip(ordered_cols, offsets):
            ax.bar(
                monthly.index + pd.Timedelta(days=offset),
                monthly[s],
                width=bar_width,
                label=s,
                color=DASHBOARD_STATUS_PALETTE[s],
                alpha=0.85,
                edgecolor="none",
            )
        title = "Data Center Proposals Over Time"
    else:
        series = monthly[status] if status in monthly.columns else None
        if series is None or series.sum() == 0:
            st.info(f"No data centers found with status: {status}")
            plt.close(fig)
            return
        ax.bar(
            series.index,
            series.values,
            width=20,
            color=DASHBOARD_STATUS_PALETTE[status],
            alpha=0.85,
            edgecolor="none",
            label=status,
        )
        title = f"{status} Data Centers Over Time"

    draw_eo_line(ax)
    shade_pre_post(ax, monthly.index.min(), monthly.index.max())

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Data Centers")
    ax.legend(loc="upper left", fontsize=9)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

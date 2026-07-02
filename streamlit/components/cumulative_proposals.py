"""
cumulative_proposals.py
─────────────────────────────────────────────────────────────────────────────
TS Component 4 — Cumulative data center proposals over time, with one
line per dashboard status. The EO cutoff is annotated near the bottom so
it doesn't overlap the upper-left legend.
─────────────────────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import streamlit as st

from components._ts_helpers import draw_eo_line, shade_pre_post
from constants import DASHBOARD_STATUS_PALETTE


def render(dc_gdf) -> None:
    """
    Render cumulative data center proposals over time by dashboard status.

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
    dc = dc_gdf.dropna(subset=["date_created"]).copy()

    cumulative = (
        dc.groupby([dc["date_created"].dt.to_period("M"), "dashboard_status"])
        .size()
        .unstack(fill_value=0)
    )
    ordered_cols = [c for c in DASHBOARD_STATUS_PALETTE.keys() if c in cumulative.columns]
    cumulative = cumulative[ordered_cols].cumsum()
    cumulative.index = cumulative.index.to_timestamp()

    if cumulative.empty:
        st.info("No data centers with valid creation dates.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    for status in cumulative.columns:
        ax.plot(
            cumulative.index,
            cumulative[status],
            label=status,
            color=DASHBOARD_STATUS_PALETTE[status],
            linewidth=2,
            alpha=0.85,
        )

    draw_eo_line(ax, annotation_va="bottom")
    shade_pre_post(ax, cumulative.index.min(), cumulative.index.max())

    ax.set_title(
        "Cumulative Data Center Proposals Over Time by Status",
        fontsize=14, pad=12,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Count")
    ax.legend(loc="upper left", fontsize=9)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

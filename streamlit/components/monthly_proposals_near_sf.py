"""
monthly_proposals_near_sf.py
─────────────────────────────────────────────────────────────────────────────
TS Component 2 — Monthly share of Proposed data centers within a given
distance threshold of a Superfund site, with bars colored pre/post EO.

Months with fewer than ``min_proposals`` proposals are dropped to avoid
noise from sparse months.
─────────────────────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Patch

from components._ts_helpers import draw_eo_line
from constants import DEFAULT_RADIUS_MI, DISTANCE_THRESHOLDS_MI, EO_DATE
from src.tufte_style import COLORS

_THRESHOLD_COL = {
    0.5: "near_sf_0_5mi",
    1:   "near_sf_1mi",
    3:   "near_sf_3mi",
    5:   "near_sf_5mi",
}


def render(dc_gdf, min_proposals: int = 5) -> None:
    """
    Render monthly share of Proposed DCs within X miles of a Superfund site.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``date_created``,
        ``dashboard_status``, and the ``near_sf_*mi`` flag columns.
    min_proposals : int, default 5
        Minimum number of proposals in a month to include that month.

    Returns
    -------
    None
        Writes the chart directly into the current Streamlit page.
    """
    radius_mi = st.select_slider(
        "Radius (miles)",
        options=DISTANCE_THRESHOLDS_MI,
        value=DEFAULT_RADIUS_MI,
        key="monthly_proposals_near_sf_radius",
    )
    near_col = _THRESHOLD_COL[radius_mi]

    proposed = dc_gdf[dc_gdf["dashboard_status"] == "Proposed"].copy()
    proposed = proposed.dropna(subset=["date_created"])
    proposed["month"] = proposed["date_created"].dt.to_period("M")

    monthly_pct = (
        proposed.groupby("month")
        .agg(total=(near_col, "count"), near_sf=(near_col, "sum"))
        .assign(pct_near=lambda x: x["near_sf"] / x["total"] * 100)
    )
    monthly_pct = monthly_pct[monthly_pct["total"] >= min_proposals]
    monthly_pct.index = monthly_pct.index.to_timestamp()

    if monthly_pct.empty:
        st.info(
            f"No months with at least {min_proposals} proposals at the "
            f"{radius_mi}-mile threshold."
        )
        return

    colors = [
        COLORS["post_eo"] if idx >= EO_DATE else COLORS["pre_eo"]
        for idx in monthly_pct.index
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        monthly_pct.index,
        monthly_pct["pct_near"],
        width=20,
        color=colors,
        alpha=0.85,
        edgecolor="none",
    )

    draw_eo_line(ax)

    legend_elements = [
        Patch(facecolor=COLORS["pre_eo"],  label="Pre-EO"),
        Patch(facecolor=COLORS["post_eo"], label="Post-EO"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title(
        f"Monthly % of Proposed Data Centers Within {radius_mi} Mile(s) "
        f"of a Superfund Site\n(months with ≥{min_proposals} proposals only)",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel(f"% of Proposed DCs Within {radius_mi} mi of SF")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

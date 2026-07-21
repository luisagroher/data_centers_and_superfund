"""
cumulative_proposals.py
─────────────────────────────────────────────────────────────────────────────
TS Component 4 — Cumulative data center proposals over time, filtered to a
user-selected distance from a Superfund site, with one line per dashboard
status (or a single line when a specific status is chosen).

The radius slider controls which ``near_sf_*mi`` flag column defines
"nearby" — ``"5+"`` is the complement of ``near_sf_5mi`` (farther than 5
miles from any Superfund site). The EO cutoff is annotated near the
bottom so it doesn't overlap the upper-left legend.
─────────────────────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import streamlit as st

from components._ts_helpers import draw_eo_line, shade_pre_post
from constants import DASHBOARD_STATUS_PALETTE, DEFAULT_RADIUS_MI, DISTANCE_THRESHOLDS_MI

ALL_STATUSES = "All"

_THRESHOLD_COL = {
    0.5: "near_sf_0_5mi",
    1:   "near_sf_1mi",
    3:   "near_sf_3mi",
    5:   "near_sf_5mi",
}

_RADIUS_OPTIONS = [*DISTANCE_THRESHOLDS_MI, "5+"]


def _near_sf_mask(dc, radius_mi):
    """Boolean mask for rows within (or, for ``"5+"``, beyond) the radius."""
    if radius_mi == "5+":
        return dc["near_sf_5mi"] == 0
    return dc[_THRESHOLD_COL[radius_mi]] == 1


def _radius_label(radius_mi) -> str:
    if radius_mi == "5+":
        return "More Than 5 Miles"
    return f"{radius_mi} Mile{'s' if radius_mi != 1 else ''}"


def _radius_phrase(radius_mi) -> str:
    """Preposition-aware phrase for chart titles, e.g. 'Within 1 Mile of' vs
    'More Than 5 Miles from'."""
    if radius_mi == "5+":
        return "More Than 5 Miles from"
    return f"Within {_radius_label(radius_mi)} of"


def render(dc_gdf) -> None:
    """
    Render cumulative data center counts over time, filtered to a chosen
    distance from a Superfund site and (optionally) a single status.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``date_created``,
        ``dashboard_status``, and the ``near_sf_*mi`` flag columns.

    Returns
    -------
    None
        Writes the chart directly into the current Streamlit page.
    """
    radius_mi = st.select_slider(
        "Radius (miles)",
        options=_RADIUS_OPTIONS,
        value=DEFAULT_RADIUS_MI,
        key="cumulative_proposals_radius",
    )
    status_options = [ALL_STATUSES] + list(DASHBOARD_STATUS_PALETTE.keys())
    status = st.radio(
        "Status",
        status_options,
        horizontal=True,
        key="cumulative_proposals_status",
    )

    dc = dc_gdf.dropna(subset=["date_created"]).copy()
    dc = dc[_near_sf_mask(dc, radius_mi)]
    if status != ALL_STATUSES:
        dc = dc[dc["dashboard_status"] == status]

    cumulative = (
        dc.groupby([dc["date_created"].dt.to_period("M"), "dashboard_status"])
        .size()
        .unstack(fill_value=0)
    )
    ordered_cols = [c for c in DASHBOARD_STATUS_PALETTE.keys() if c in cumulative.columns]
    cumulative = cumulative[ordered_cols].cumsum()
    cumulative.index = cumulative.index.to_timestamp()

    dist_label = _radius_label(radius_mi)

    if cumulative.empty:
        st.info(
            f"No data centers within {dist_label} of a Superfund site "
            f"for status: {status}."
        )
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    for col in cumulative.columns:
        ax.plot(
            cumulative.index,
            cumulative[col],
            label=col,
            color=DASHBOARD_STATUS_PALETTE[col],
            linewidth=2,
            alpha=0.85,
        )

    draw_eo_line(ax, annotation_va="bottom")
    shade_pre_post(ax, cumulative.index.min(), cumulative.index.max())

    phrase = _radius_phrase(radius_mi)
    title = (
        f"Cumulative Data Center Proposals {phrase} a Superfund Site"
        if status == ALL_STATUSES
        else f"Cumulative {status} Data Centers {phrase} a Superfund Site"
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Count")
    ax.legend(loc="upper left", fontsize=9)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

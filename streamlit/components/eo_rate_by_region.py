"""
eo_rate_by_region.py
─────────────────────────────────────────────────────────────────────────────
TS Component 3 + companion rate table — Pre vs post-EO proposal rate
(proposals per day) by census region, filtered to a user-selected distance
from a Superfund site and (optionally) a single dashboard status, with a
per-region multiplier annotation. The full rate table is rendered beneath
the chart.

The radius slider controls which ``near_sf_*mi`` flag column defines
"nearby" — ``"5+"`` is the complement of ``near_sf_5mi`` (farther than 5
miles from any Superfund site).
─────────────────────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from constants import DASHBOARD_STATUS_PALETTE, DEFAULT_RADIUS_MI, DISTANCE_THRESHOLDS_MI, EO_DATE
from src.tufte_style import COLORS

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
    Render pre/post-EO proposal rates by census region, plus a rate table,
    filtered to a chosen distance from a Superfund site and (optionally) a
    single status.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``date_created``,
        ``dashboard_status``, ``census_region``, and the ``near_sf_*mi``
        flag columns.

    Returns
    -------
    None
        Writes the chart and table directly into the current Streamlit page.
    """
    radius_mi = st.select_slider(
        "Radius (miles)",
        options=_RADIUS_OPTIONS,
        value=DEFAULT_RADIUS_MI,
        key="eo_rate_by_region_radius",
    )
    status_options = [ALL_STATUSES] + list(DASHBOARD_STATUS_PALETTE.keys())
    status = st.radio(
        "Status",
        status_options,
        horizontal=True,
        key="eo_rate_by_region_status",
    )

    dc = dc_gdf.dropna(subset=["date_created", "census_region"]).copy()
    dc = dc[_near_sf_mask(dc, radius_mi)]
    if status != ALL_STATUSES:
        dc = dc[dc["dashboard_status"] == status]

    dist_label = _radius_label(radius_mi)

    if dc.empty:
        st.info(
            f"No data centers within {dist_label} of a Superfund site "
            f"for status: {status}."
        )
        return

    pre_eo_days  = max((EO_DATE - dc["date_created"].min()).days, 1)
    post_eo_days = max((dc["date_created"].max() - EO_DATE).days, 1)

    region_rates = (
        dc.groupby("census_region")
        .apply(lambda x: pd.Series({
            "Pre-EO":  (x["date_created"] < EO_DATE).sum()  / pre_eo_days,
            "Post-EO": (x["date_created"] >= EO_DATE).sum() / post_eo_days,
        }))
        .sort_values("Post-EO", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x     = np.arange(len(region_rates))
    width = 0.35

    bars_pre = ax.bar(
        x - width / 2, region_rates["Pre-EO"], width,
        label="Pre-EO", color=COLORS["pre_eo"], alpha=0.85, edgecolor="none",
    )
    bars_post = ax.bar(
        x + width / 2, region_rates["Post-EO"], width,
        label="Post-EO", color=COLORS["post_eo"], alpha=0.85, edgecolor="none",
    )

    for bar in [*bars_pre, *bars_post]:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.2f}",
            ha="center", fontsize=8,
        )

    for i, region in enumerate(region_rates.index):
        pre  = region_rates.loc[region, "Pre-EO"]
        post = region_rates.loc[region, "Post-EO"]
        if pre > 0:
            ax.text(
                i, max(pre, post) + 0.015,
                f"{post / pre:.1f}x",
                ha="center", fontsize=9,
                color=COLORS["highlight"], fontweight="bold",
            )

    phrase = _radius_phrase(radius_mi)
    title = (
        f"Data Center Rate Pre vs Post EO by Region\n{phrase} a Superfund Site (per day)"
        if status == ALL_STATUSES
        else f"{status} Data Center Rate Pre vs Post EO by Region\n{phrase} a Superfund Site (per day)"
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Census Region")
    ax.set_ylabel("Proposals per Day")
    ax.set_xticks(x)
    ax.set_xticklabels(region_rates.index)
    ax.legend(loc="upper right")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

    rate_table = (
        dc.groupby("census_region")
        .apply(lambda x: pd.Series({
            "Pre-EO Count"    : int((x["date_created"] < EO_DATE).sum()),
            "Post-EO Count"   : int((x["date_created"] >= EO_DATE).sum()),
            "Pre-EO Rate/Day" : (x["date_created"] < EO_DATE).sum()  / pre_eo_days,
            "Post-EO Rate/Day": (x["date_created"] >= EO_DATE).sum() / post_eo_days,
        }))
    )
    rate_table["Multiplier"] = (
        rate_table["Post-EO Rate/Day"] / rate_table["Pre-EO Rate/Day"]
    ).round(1)

    total_pre  = int(rate_table["Pre-EO Count"].sum())
    total_post = int(rate_table["Post-EO Count"].sum())
    total = pd.Series({
        "Pre-EO Count"    : total_pre,
        "Post-EO Count"   : total_post,
        "Pre-EO Rate/Day" : total_pre  / pre_eo_days,
        "Post-EO Rate/Day": total_post / post_eo_days,
        "Multiplier"      : round(
            (total_post / post_eo_days) / (total_pre / pre_eo_days), 1
        ) if total_pre > 0 else float("nan"),
    }, name="Total")

    rate_table = pd.concat([rate_table, total.to_frame().T])
    rate_table["Pre-EO Rate/Day"]  = rate_table["Pre-EO Rate/Day"].round(3)
    rate_table["Post-EO Rate/Day"] = rate_table["Post-EO Rate/Day"].round(3)

    st.subheader("Rate table")
    st.dataframe(rate_table, use_container_width=True)

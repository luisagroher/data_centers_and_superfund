"""
eo_rate_by_region.py
─────────────────────────────────────────────────────────────────────────────
TS Component 3 + companion rate table — Pre vs post-EO proposal rate
(proposals per day) by census region, with a per-region multiplier
annotation. The full rate table is rendered beneath the chart.
─────────────────────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from constants import EO_DATE
from src.tufte_style import COLORS


def render(dc_gdf) -> None:
    """
    Render pre/post-EO proposal rates by census region, plus a rate table.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``date_created``,
        ``dashboard_status``, and ``census_region`` columns.

    Returns
    -------
    None
        Writes the chart and table directly into the current Streamlit page.
    """
    proposed = dc_gdf[dc_gdf["dashboard_status"] == "Proposed"].copy()
    proposed = proposed.dropna(subset=["date_created", "census_region"])

    if proposed.empty:
        st.info("No proposed data centers with valid dates and regions.")
        return

    pre_eo_days  = max((EO_DATE - proposed["date_created"].min()).days, 1)
    post_eo_days = max((proposed["date_created"].max() - EO_DATE).days, 1)

    region_rates = (
        proposed.groupby("census_region")
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

    ax.set_title(
        "Proposed Data Center Rate Pre vs Post EO by Region\n"
        "(proposals per day)",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Census Region")
    ax.set_ylabel("Proposals per Day")
    ax.set_xticks(x)
    ax.set_xticklabels(region_rates.index)
    ax.legend(loc="upper right")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

    rate_table = (
        proposed.groupby("census_region")
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

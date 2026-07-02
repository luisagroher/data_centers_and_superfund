"""
status_summary_header.py
─────────────────────────────────────────────────────────────────────────────
Decorative status summary strip — one colored box per dashboard status
(Proposed, Under Construction, Operational, Expanding, Inactive) showing
the total number of data centers in that category.

Intended to sit at the top of any dashboard page as a high-level summary.
Box colors follow ``DASHBOARD_STATUS_PALETTE`` (defined via ``tufte_style``).
─────────────────────────────────────────────────────────────────────────────
"""
import streamlit as st

from constants import DASHBOARD_STATUS_PALETTE


def render(dc_gdf) -> None:
    """
    Render a row of evenly spaced, colored count boxes — one per dashboard
    status — at the top of the page.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain a ``dashboard_status`` column.

    Returns
    -------
    None
        Writes the boxes directly into the current Streamlit page.
    """
    counts = (
        dc_gdf["dashboard_status"]
        .value_counts()
        .reindex(DASHBOARD_STATUS_PALETTE.keys())
        .fillna(0)
        .astype(int)
    )

    columns = st.columns(len(DASHBOARD_STATUS_PALETTE))
    for col, (status, color) in zip(columns, DASHBOARD_STATUS_PALETTE.items()):
        count = int(counts.loc[status])
        col.markdown(
            f"""
            <div style="
                background-color: {color};
                color: white;
                padding: 14px 10px;
                border-radius: 6px;
                text-align: center;
                line-height: 1.2;
            ">
                <div style="font-size: 1.6rem; font-weight: 600;">{count:,}</div>
                <div style="font-size: 0.85rem; opacity: 0.95;">{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

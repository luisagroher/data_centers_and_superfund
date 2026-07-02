"""
sf_site_labels.py
─────────────────────────────────────────────────────────────────────────────
Decorative label grid — one card per Superfund site showing the site name,
state abbreviation, and nearby data center count at a user-selected radius.

Cards use a cream background with forest-green text and are arranged in a
responsive grid (≤3 columns on a wide browser).
─────────────────────────────────────────────────────────────────────────────
"""
import streamlit as st

from constants import DEFAULT_RADIUS_MI, DISTANCE_THRESHOLDS_MI


_CREAM        = "#FFF8E7"
_FOREST_GREEN = "#228B22"
_BORDER       = "#D6C89A"

_RADIUS_TO_COL = {
    0.5: "dc_count_within_0_5mi",
    1:   "dc_count_within_1mi",
    3:   "dc_count_within_3mi",
    5:   "dc_count_within_5mi",
}


def _card(site_name: str, state: str, count: int) -> str:
    return f"""
    <div style="
        background:      {_CREAM};
        border:          1px solid {_BORDER};
        border-radius:   8px;
        padding:         20px 24px;
        display:         flex;
        align-items:     center;
        justify-content: space-between;
    ">
      <div>
        <div style="
            font-size:   1.1rem;
            font-weight: 700;
            color:       {_FOREST_GREEN};
            line-height: 1.2;
        ">{site_name}</div>
        <div style="
            font-size:   0.85rem;
            font-weight: 500;
            color:       {_FOREST_GREEN};
            opacity:     0.75;
            line-height: 1.4;
        ">{state}</div>
      </div>
      <div style="
          font-size:   1.6rem;
          font-weight: 600;
          color:       {_FOREST_GREEN};
          margin-left: 16px;
      ">{count:,}</div>
    </div>"""


def render(sf_gdf, top_n: int = 30) -> None:
    """
    Render a grid of decorative Superfund site labels ranked by nearby DC count.

    Parameters
    ----------
    sf_gdf : geopandas.GeoDataFrame
        Superfund site polygons. Must contain ``SITE_NAME``, ``STATE_CODE``,
        and ``dc_count_within_*`` columns for each threshold in
        ``DISTANCE_THRESHOLDS_MI``.
    top_n : int, default 30
        Number of top sites to display.

    Returns
    -------
    None
        Writes the label grid directly into the current Streamlit page.
    """
    radius_mi = st.select_slider(
        "Radius (miles)",
        options=DISTANCE_THRESHOLDS_MI,
        value=DEFAULT_RADIUS_MI,
        key="sf_site_labels_radius",
    )

    count_col = _RADIUS_TO_COL[radius_mi]

    top_sites = (
        sf_gdf[sf_gdf[count_col] > 0]
        [["SITE_NAME", "STATE_CODE", count_col]]
        .sort_values(count_col, ascending=False)
        .head(top_n)
    )

    if top_sites.empty:
        st.info(f"No Superfund sites have any data center within {radius_mi:g} mile(s).")
        return

    cards_html = "".join(
        _card(row["SITE_NAME"], row["STATE_CODE"].strip(), int(row[count_col]))
        for _, row in top_sites.iterrows()
    )

    st.subheader(f"Top {top_n} Superfund Sites by Nearby Data Centers (within {radius_mi:g} mi)")
    st.markdown(
        f"""
        <div style="
            display:               grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap:                   10px;
            padding:               4px 0 12px;
        ">
          {cards_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

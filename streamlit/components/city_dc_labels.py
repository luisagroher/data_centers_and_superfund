"""
city_dc_labels.py
─────────────────────────────────────────────────────────────────────────────
Decorative label grid — one card per city showing the city name, state
abbreviation, and data center count.  Cards use a cream background with
forest-green text and are arranged in a responsive flexbox grid (≤3 columns).
─────────────────────────────────────────────────────────────────────────────
"""
import streamlit as st


_CREAM        = "#FFF8E7"
_FOREST_GREEN = "#228B22"
_BORDER       = "#D6C89A"


def _card(city: str, state: str, count: int) -> str:
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
        ">{city}</div>
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


def render(dc_gdf, top_n: int = 30) -> None:
    """
    Render a grid of decorative city-count labels.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain ``city`` and ``state`` columns.
    top_n : int, default 30
        Number of top cities to display.

    Returns
    -------
    None
        Writes the label grid directly into the current Streamlit page.
    """
    df = dc_gdf.dropna(subset=["city", "state"]).copy()
    df["city"]  = df["city"].str.strip()
    df["state"] = df["state"].str.strip()

    counts = (
        df.groupby(["city", "state"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )

    cards_html = "".join(
        _card(row["city"], row["state"], row["count"])
        for _, row in counts.iterrows()
    )

    st.subheader(f"Top {top_n} Cities by Data Center Count")
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

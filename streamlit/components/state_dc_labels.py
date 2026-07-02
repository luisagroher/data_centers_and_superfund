"""
state_dc_labels.py
─────────────────────────────────────────────────────────────────────────────
Decorative label grid — one card per state showing the state abbreviation
and its data center count.  Cards use a cream background with forest-green
text and are arranged in a responsive flexbox row.
─────────────────────────────────────────────────────────────────────────────
"""
import streamlit as st


_CREAM        = "#FFF8E7"
_FOREST_GREEN = "#228B22"
_BORDER       = "#D6C89A"


def _card(state: str, count: int) -> str:
    return f"""
    <div style="
        background:    {_CREAM};
        border:        1px solid {_BORDER};
        border-radius: 8px;
        padding:       20px 24px;
        display:       flex;
        align-items:   center;
        justify-content: space-between;
    ">
      <div style="
          font-size:   1.2rem;
          font-weight: 700;
          color:       {_FOREST_GREEN};
      ">{state}</div>
      <div style="
          font-size:   1.6rem;
          font-weight: 600;
          color:       {_FOREST_GREEN};
      ">{count:,}</div>
    </div>"""


def render(dc_gdf) -> None:
    """
    Render a grid of decorative state-count labels.

    Parameters
    ----------
    dc_gdf : geopandas.GeoDataFrame
        Data center points. Must contain a ``state`` column (2-letter code).

    Returns
    -------
    None
        Writes the label grid directly into the current Streamlit page.
    """
    counts = (
        dc_gdf["state"]
        .dropna()
        .value_counts()
        .sort_values(ascending=False)
    )

    cards_html = "".join(_card(state, count) for state, count in counts.items())

    st.subheader("Data Centers by State")
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

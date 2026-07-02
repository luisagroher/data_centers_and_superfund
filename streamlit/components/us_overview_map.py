"""
us_overview_map.py
─────────────────────────────────────────────────────────────────────────────
National overview map: all Superfund site polygons colored in blue and all
data center points as bubbles colored by dashboard status.

Renders a Leaflet map via folium / streamlit-folium.
─────────────────────────────────────────────────────────────────────────────
"""
import folium
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

from constants import DASHBOARD_STATUS_PALETTE, SUPERFUND_COLOR


def _legend_html(status_palette: dict, superfund_color: str | None) -> str:
    """Build an HTML legend for the map. Pass superfund_color=None to omit it."""
    items = [
        f"""
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
          <div style="width:14px;height:14px;border-radius:50%;
                      background:{color};opacity:0.85;flex-shrink:0;"></div>
          <span style="font-size:12px;">{label}</span>
        </div>"""
        for label, color in status_palette.items()
    ]
    sf_item = (
        f"""
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
          <div style="width:14px;height:14px;
                      background:{superfund_color};opacity:0.45;flex-shrink:0;"></div>
          <span style="font-size:12px;">Superfund site</span>
        </div>"""
        if superfund_color else ""
    )

    return f"""
    <div style="
        position: fixed;
        bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 10px 14px;
        border: 1px solid #ccc; border-radius: 6px;
        font-family: sans-serif; line-height: 1.6;
        box-shadow: 0 2px 6px rgba(0,0,0,.15);
    ">
      <div style="font-size:13px;font-weight:600;margin-bottom:6px;">Legend</div>
      {sf_item}
      {''.join(items)}
    </div>"""


def render(sf_gdf: gpd.GeoDataFrame, dc_gdf: gpd.GeoDataFrame) -> None:
    """
    Render the national overview map.

    Parameters
    ----------
    sf_gdf : geopandas.GeoDataFrame
        Superfund site polygons in EPSG:4326. Must contain ``SITE_NAME``
        and ``STATE_CODE``.
    dc_gdf : geopandas.GeoDataFrame
        Data center points in EPSG:4326. Must contain ``dashboard_status``,
        ``facility_name``, ``city``, and ``state``.

    Returns
    -------
    None
        Writes the map directly into the current Streamlit page.
    """
    st.subheader("Data Centers & Superfund Sites — National Overview")

    radio_options = ["All"] + list(DASHBOARD_STATUS_PALETTE.keys()) + ["Superfund only"]
    view_filter = st.radio(
        "Filter",
        radio_options,
        horizontal=True,
        key="us_overview_filter",
    )

    if view_filter == "All":
        dc_filtered    = dc_gdf
        active_palette = DASHBOARD_STATUS_PALETTE
        show_superfund = True
    elif view_filter == "Superfund only":
        dc_filtered    = dc_gdf.iloc[:0]
        active_palette = {}
        show_superfund = True
    else:
        dc_filtered    = dc_gdf[dc_gdf["dashboard_status"] == view_filter]
        active_palette = {view_filter: DASHBOARD_STATUS_PALETTE[view_filter]}
        show_superfund = False

    m = folium.Map(
        location=[38.5, -96.5],
        zoom_start=4,
        tiles="CartoDB positron",
    )

    if show_superfund:
        folium.GeoJson(
            sf_gdf.__geo_interface__,
            name="Superfund sites",
            style_function=lambda _: {
                "fillColor":   SUPERFUND_COLOR,
                "color":       SUPERFUND_COLOR,
                "weight":      0.5,
                "fillOpacity": 0.35,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["SITE_NAME", "STATE_CODE"],
                aliases=["Site:", "State:"],
                localize=True,
            ),
        ).add_to(m)

    for _, row in dc_filtered.iterrows():
        if row.geometry is None:
            continue
        color  = DASHBOARD_STATUS_PALETTE.get(row["dashboard_status"], "#888888")
        name   = row.get("facility_name") or ""
        city   = row.get("city") or ""
        state  = row.get("state") or ""
        status = row.get("dashboard_status") or ""
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            color=color,
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{name}<br>{city}, {state}<br>{status}",
        ).add_to(m)

    m.get_root().html.add_child(
        folium.Element(_legend_html(active_palette, SUPERFUND_COLOR if show_superfund else None))
    )

    sf_count = len(sf_gdf) if show_superfund else 0
    st.caption(
        f"{sf_count:,} Superfund sites · {len(dc_filtered):,} data centers"
    )
    st_folium(m, width=None, height=650, returned_objects=[])

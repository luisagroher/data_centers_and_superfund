"""
map_view.py
─────────────────────────────────────────────────────────────────────────────
Combined drill-down map: merges us_overview_map.py (national) and
state_site_map.py (single-site) into one flow — US -> state ->
(city | Superfund site).

Superfund site polygons are always drawn, at every granularity; there is
no "Superfund only" toggle. A dashboard-status radio (All / Proposed /
Under Construction / Operational / Expanding / Inactive) is always
available and filters the data center layer at every granularity.

Prototype for the Story 1 map flow — not yet wired into a full page
layout, just previewable via the app.py gallery.
─────────────────────────────────────────────────────────────────────────────
"""
from typing import Optional

import folium
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

from constants import (
    CRS_METRIC,
    CRS_WGS84,
    DASHBOARD_STATUS_PALETTE,
    DEFAULT_RADIUS_MI,
    DISTANCE_THRESHOLDS_MI,
    METERS_PER_MILE,
    SUPERFUND_COLOR,
)

ALL_STATUSES = "All"
_DRILL_OPTIONS = ["State overview", "City", "Superfund site"]


def _legend_html(status_palette: dict) -> str:
    """Build an HTML legend. Superfund is always shown, so it's not optional here."""
    items = [
        f"""
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
          <div style="width:14px;height:14px;border-radius:50%;
                      background:{color};opacity:0.85;flex-shrink:0;"></div>
          <span style="font-size:12px;">{label}</span>
        </div>"""
        for label, color in status_palette.items()
    ]
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
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
        <div style="width:14px;height:14px;
                    background:{SUPERFUND_COLOR};opacity:0.45;flex-shrink:0;"></div>
        <span style="font-size:12px;">Superfund site</span>
      </div>
      {''.join(items)}
    </div>"""


def _sf_style_function(selected_epa_id: Optional[str]):
    """Style Superfund polygons, highlighting ``selected_epa_id`` if set."""

    def style_function(feature):
        is_selected = (
            selected_epa_id is not None
            and feature["properties"].get("EPA_ID") == selected_epa_id
        )
        return {
            "fillColor":   SUPERFUND_COLOR,
            "color":       SUPERFUND_COLOR,
            "weight":      2.5 if is_selected else 0.5,
            "fillOpacity": 0.55 if is_selected else 0.3,
        }

    return style_function


def _dcs_within_radius(
    site_row, dc_gdf: gpd.GeoDataFrame, radius_mi: float
) -> gpd.GeoDataFrame:
    """Subset ``dc_gdf`` to points within ``radius_mi`` miles of a site polygon."""
    site_metric = (
        gpd.GeoSeries([site_row.geometry], crs=CRS_WGS84).to_crs(CRS_METRIC).iloc[0]
    )
    buffer    = site_metric.buffer(radius_mi * METERS_PER_MILE)
    dc_metric = dc_gdf.to_crs(CRS_METRIC)
    mask      = dc_metric.geometry.intersects(buffer)
    return dc_gdf.loc[mask]


def _fit_bounds(m: folium.Map, frames: list, pad: float = 0.05) -> None:
    """Fit the map viewport to the combined bounds of one or more GeoDataFrames."""
    frames = [g for g in frames if g is not None and len(g) > 0]
    if not frames:
        return
    minx = min(g.total_bounds[0] for g in frames)
    miny = min(g.total_bounds[1] for g in frames)
    maxx = max(g.total_bounds[2] for g in frames)
    maxy = max(g.total_bounds[3] for g in frames)
    if minx == maxx and miny == maxy:
        minx, maxx = minx - pad, maxx + pad
        miny, maxy = miny - pad, maxy + pad
    m.fit_bounds([[miny, minx], [maxy, maxx]])


def render(sf_gdf: gpd.GeoDataFrame, dc_gdf: gpd.GeoDataFrame) -> None:
    """
    Render the combined drill-down map: US -> state -> (city | Superfund site).

    Parameters
    ----------
    sf_gdf : geopandas.GeoDataFrame
        Superfund site polygons in EPSG:4326. Must contain ``EPA_ID``,
        ``SITE_NAME``, ``STATE_CODE``, ``CITY_NAME``.
    dc_gdf : geopandas.GeoDataFrame
        Data center points in EPSG:4326. Must contain ``dashboard_status``,
        ``facility_name``, ``city``, ``state``.

    Returns
    -------
    None
        Writes the map directly into the current Streamlit page.
    """
    st.subheader("Data Centers & Superfund Sites")

    status_options = [ALL_STATUSES] + list(DASHBOARD_STATUS_PALETTE.keys())
    status_choice = st.radio(
        "Status", status_options, horizontal=True, key="map_view_status",
    )
    if status_choice == ALL_STATUSES:
        dc_status      = dc_gdf
        active_palette = DASHBOARD_STATUS_PALETTE
    else:
        dc_status      = dc_gdf[dc_gdf["dashboard_status"] == status_choice]
        active_palette = {status_choice: DASHBOARD_STATUS_PALETTE[status_choice]}

    states = sorted(set(dc_gdf["state"].dropna()) | set(sf_gdf["STATE_CODE"].dropna()))
    state_choice = st.selectbox("State", ["All states"] + states, key="map_view_state")

    selected_epa_id: Optional[str] = None
    site_row  = None
    city_choice: Optional[str] = None
    radius_mi = DEFAULT_RADIUS_MI

    if state_choice == "All states":
        granularity = "national"
        dc_view  = dc_status
        sf_scope = sf_gdf
    else:
        sf_scope = sf_gdf[sf_gdf["STATE_CODE"] == state_choice]
        dc_state = dc_status[dc_status["state"] == state_choice]

        drill_choice = st.selectbox(
            "Drill into", _DRILL_OPTIONS, key="map_view_drill",
        )

        if drill_choice == "City":
            cities = sorted(dc_state["city"].dropna().unique())
            if not cities:
                st.info(f"No data centers of this status recorded in {state_choice}.")
                granularity = "state"
                dc_view = dc_state
            else:
                city_choice = st.selectbox("City", cities, key="map_view_city")
                granularity = "city"
                dc_view = dc_state[dc_state["city"] == city_choice]

        elif drill_choice == "Superfund site":
            sites_in_state = sf_scope[["EPA_ID", "SITE_NAME"]].sort_values("SITE_NAME")
            if sites_in_state.empty:
                st.info(f"No Superfund sites recorded in {state_choice}.")
                granularity = "state"
                dc_view = dc_state
            else:
                site_name = st.selectbox(
                    "Superfund site", sites_in_state["SITE_NAME"].tolist(), key="map_view_site",
                )
                selected_epa_id = sites_in_state.loc[
                    sites_in_state["SITE_NAME"] == site_name, "EPA_ID"
                ].iloc[0]
                radius_mi = st.select_slider(
                    "Radius (miles)", options=DISTANCE_THRESHOLDS_MI,
                    value=DEFAULT_RADIUS_MI, key="map_view_radius",
                )
                site_row = sf_gdf.loc[sf_gdf["EPA_ID"] == selected_epa_id].iloc[0]
                granularity = "site"
                dc_view = _dcs_within_radius(site_row, dc_state, radius_mi)

        else:
            granularity = "state"
            dc_view = dc_state

    m = folium.Map(location=[38.5, -96.5], zoom_start=4, tiles="CartoDB positron")

    folium.GeoJson(
        sf_gdf.__geo_interface__,
        name="Superfund sites",
        style_function=_sf_style_function(selected_epa_id),
        tooltip=folium.GeoJsonTooltip(
            fields=["SITE_NAME", "STATE_CODE"], aliases=["Site:", "State:"], localize=True,
        ),
    ).add_to(m)

    marker_radius = 10 if granularity in ("city", "site") else 6
    marker_weight = 1 if granularity in ("city", "site") else 0.5
    for _, row in dc_view.iterrows():
        if row.geometry is None:
            continue
        color = DASHBOARD_STATUS_PALETTE.get(row["dashboard_status"], "#888888")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=marker_radius,
            color=color,
            weight=marker_weight,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=(
                f"{row.get('facility_name', '')}<br>"
                f"{row.get('city', '')}, {row.get('state', '')}<br>"
                f"{row.get('dashboard_status', '')}"
            ),
        ).add_to(m)

    m.get_root().html.add_child(folium.Element(_legend_html(active_palette)))

    if granularity == "state":
        _fit_bounds(m, [dc_view, sf_scope])
    elif granularity == "city":
        sf_city = (
            sf_scope[sf_scope["CITY_NAME"].astype(str).str.strip().str.lower()
                     == city_choice.strip().lower()]
            if "CITY_NAME" in sf_scope.columns else sf_scope.iloc[0:0]
        )
        frames = [f for f in [dc_view, sf_city] if len(f) > 0]
        _fit_bounds(m, frames if frames else [dc_state, sf_scope])
    elif granularity == "site":
        site_gdf = gpd.GeoDataFrame(geometry=[site_row.geometry], crs=CRS_WGS84)
        _fit_bounds(m, [site_gdf, dc_view])
    # national: keep the default US-wide view

    if granularity == "national":
        st.caption(f"{len(sf_gdf):,} Superfund sites · {len(dc_view):,} data centers ({status_choice})")
    elif granularity == "state":
        st.caption(
            f"{state_choice}: {len(sf_scope):,} Superfund sites · "
            f"{len(dc_view):,} data centers ({status_choice})"
        )
    elif granularity == "city":
        st.caption(f"{city_choice}, {state_choice}: {len(dc_view):,} data centers ({status_choice})")
    elif granularity == "site":
        st.caption(
            f"{len(dc_view):,} data center(s) within {radius_mi:g} mi of "
            f"**{site_row['SITE_NAME']}** ({state_choice}), status: {status_choice}"
        )

    st_folium(m, width=None, height=650, returned_objects=[])

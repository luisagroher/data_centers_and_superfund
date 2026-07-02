"""
state_site_map.py
─────────────────────────────────────────────────────────────────────────────
Map view: a single Superfund site polygon and the data centers within a
user-specified radius, color-coded by dashboard status.

Renders a Leaflet map via folium / streamlit-folium.
─────────────────────────────────────────────────────────────────────────────
"""
import folium
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

from constants import (
    CRS_METRIC,
    CRS_WGS84,
    DASHBOARD_STATUS_PALETTE,
    METERS_PER_MILE,
    SUPERFUND_COLOR,
)


def _dcs_within_radius(
    site_row,
    dc_gdf: gpd.GeoDataFrame,
    radius_mi: float,
) -> gpd.GeoDataFrame:
    """
    Subset ``dc_gdf`` to data centers whose point geometry falls within
    ``radius_mi`` miles of the Superfund site polygon.

    Parameters
    ----------
    site_row : geopandas.GeoSeries or pandas.Series
        Single row from the Superfund GeoDataFrame. Must expose a polygon
        ``geometry`` in EPSG:4326.
    dc_gdf : geopandas.GeoDataFrame
        Data center points in EPSG:4326.
    radius_mi : float
        Buffer radius in miles, applied to the site polygon.

    Returns
    -------
    geopandas.GeoDataFrame
        Rows from ``dc_gdf`` (original CRS preserved) that intersect the
        buffered polygon.
    """
    site_metric = (
        gpd.GeoSeries([site_row.geometry], crs=CRS_WGS84)
        .to_crs(CRS_METRIC)
        .iloc[0]
    )
    buffer    = site_metric.buffer(radius_mi * METERS_PER_MILE)
    dc_metric = dc_gdf.to_crs(CRS_METRIC)
    mask      = dc_metric.geometry.intersects(buffer)
    return dc_gdf.loc[mask]


def render(
    sf_gdf: gpd.GeoDataFrame,
    dc_gdf: gpd.GeoDataFrame,
    site_epa_id: str,
    radius_mi: float = 1.0,
) -> None:
    """
    Render the Superfund site map with surrounding data centers.

    Parameters
    ----------
    sf_gdf : geopandas.GeoDataFrame
        Superfund site polygons in EPSG:4326. Must contain ``EPA_ID``,
        ``SITE_NAME``, ``STATE_CODE`` and a polygon ``geometry``.
    dc_gdf : geopandas.GeoDataFrame
        Data center points in EPSG:4326. Must contain ``dashboard_status``
        and a point ``geometry``.
    site_epa_id : str
        EPA_ID of the site to display.
    radius_mi : float, default 1.0
        Search radius in miles used to find data centers around the site.

    Returns
    -------
    None
        Writes the map directly into the current Streamlit page.
    """
    site_row    = sf_gdf.loc[sf_gdf["EPA_ID"] == site_epa_id].iloc[0]
    nearby_dcs  = _dcs_within_radius(site_row, dc_gdf, radius_mi)
    centroid    = site_row.geometry.centroid

    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    folium.GeoJson(
        site_row.geometry.__geo_interface__,
        name=site_row["SITE_NAME"],
        style_function=lambda _: {
            "fillColor":   SUPERFUND_COLOR,
            "color":       SUPERFUND_COLOR,
            "weight":      2,
            "fillOpacity": 0.35,
        },
        tooltip=site_row["SITE_NAME"],
    ).add_to(m)

    for _, row in nearby_dcs.iterrows():
        color = DASHBOARD_STATUS_PALETTE.get(row["dashboard_status"], "#888888")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=10,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=(
                f"{row.get('city', '')}, {row.get('state', '')} — "
                f"{row['dashboard_status']}"
            ),
        ).add_to(m)

    st.caption(
        f"Showing {len(nearby_dcs)} data center(s) within {radius_mi:g} mi of "
        f"**{site_row['SITE_NAME']}** ({site_row['STATE_CODE']})"
    )
    st_folium(m, width=None, height=600, returned_objects=[])
